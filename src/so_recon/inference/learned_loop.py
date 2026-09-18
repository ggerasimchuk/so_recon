"""E03 Task 07 — the same-target learned SMC loop (plan §0.2 items 2-3, §5.4, §12).

`run_physical_smc` starts every run from the prior: it builds `PhysicalTarget(prior,
prior, ...)` and hands the prior to `infer`/`continue_inference` as the sampling law. This
module is that same E02 engine with ONE seam changed: the sampling law handed to BOTH the
target construction and `infer`/`continue_inference` is the defensive mixture

    r = (1-epsilon) q_phi + epsilon p0,        epsilon fixed before inference,

where `q_phi` is a Task 06 exported checkpoint bound to one run world through the Task 05
freezing machinery. Because the E02 engine already treats the proposal argument as `r` —
beta-step increments `log p0 + log L - log r`, the full bridge MH ratio, CESS/ESS and the
kernel table — reusing it unchanged IS the §5.4 mathematics: nothing here reimplements
sampling, weighting or resampling.

Identity discipline (§4.3): the run payload carries BOTH identities — the strict E02
runtime fingerprint (proposal-inclusive, the only one a checkpoint resume accepts) and the
new proposal-free scientific target identity from `validation.e03_protocol`. Claim
discipline (§4.4, ledger ruling): a beta<1 or incomplete output is labelled a diagnostic
partial ensemble and never receives a posterior bundle; raw `q` ensembles, if ever
published, must be labelled `raw_proposal`.

Per §12, a native restart of final particles is NOT part of this loop: the run's evidence
needs the state outputs at months 0/12/24/36, which the ordinary output request already
publishes for every forward, so the separately-accounted restart machinery stays unused.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from so_recon.config.inference import InferenceConfig
from so_recon.config.learning import LearningConfig
from so_recon.config.resources import require_resource_profile
from so_recon.config.schema import ProjectConfig
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.geology.renderer import theta_hash
from so_recon.inference.checkpoint import load_state, save_state
from so_recon.inference.contracts import (
    ObservationBundle,
    PosteriorBundle,
    PriorContext,
    SMCState,
)
from so_recon.inference.proposals import DefensiveMixture
from so_recon.inference.smc import SMCBudgetStop, continue_inference, infer
from so_recon.inference.target import PhysicalTarget, RunFactory
from so_recon.inference.weights import ess
from so_recon.ml.checkpoint import CheckpointHashes, load_checkpoint
from so_recon.ml.context import build_context_batch
from so_recon.ml.contracts import PROPOSAL_MANIFEST_SCHEMA, ProposalManifest
from so_recon.ml.dataset import CorpusDataset
from so_recon.ml.proposal import bind_frozen_proposal
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.simulator.budget import BudgetLedger, BudgetStop
from so_recon.simulator.results import load_forward_result
from so_recon.simulator.schedule import month_edges_s
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.validation.e03_protocol import (
    ensemble_kind,
    scientific_target_identity,
    verify_balance_threshold_config,
)
from so_recon.validation.physical_smc import PHYSICAL_STATE_MONTHS, _family_probabilities

LEARNED_SMC_RUN_SCHEMA = "e03-learned-smc-run-1"
LEARNED_POSTERIOR_BUNDLE_SCHEMA = "e03-learned-posterior-bundle-1"
PARTICLE_DENSITY_DIAGNOSTICS_SCHEMA = "e03-particle-density-diagnostics-1"
CORPUS_MANIFEST_SCHEMA = "e03-corpus-manifest-1"
PARENT_CONTEXT_SCHEMA = "e03-parent-context-1"
ENGINEERING_SMOKE_LABEL = "engineering-smoke"

#: §5.4: the recorded density agreement between the mixture the engine used and the
#: TargetEvaluation it stored. The engine itself enforces this at every evaluation; the
#: diagnostics artifact restates it so a reader can check without the engine.
DENSITY_MATCH_ATOL = 1e-12


class _BudgetAwareTarget:
    """Translate a session resource refusal into a resumable SMC stop."""

    def __init__(self, target: PhysicalTarget) -> None:
        self._target = target
        self.fingerprint = target.fingerprint
        self.checkpoint_hashes = target.checkpoint_hashes

    def evaluate(self, theta: Any) -> Any:
        try:
            return self._target.evaluate(theta)
        except BudgetStop as exc:
            raise SMCBudgetStop(str(exc)) from exc


@dataclass(frozen=True)
class ParentWorld:
    """One corpus parent as an inference problem: prior, data, and nothing from truth/."""

    parent_id: str
    design_id: str
    split: str
    payload: dict[str, Any]
    context: PriorContext
    observations: ObservationBundle
    prior: GaussianConditionalPrior
    context_ref: ArtifactRef
    state_times_s: tuple[float, ...]


@dataclass(frozen=True)
class DefensiveLaw:
    """The bound q and the mixture r built from it, with the identities they froze."""

    q: Any
    mixture: DefensiveMixture
    epsilon: float
    manifest: ProposalManifest
    stream_hashes: dict[str, str]
    spec_hash: str
    weights_sha256: str


def learned_state_times(context: PriorContext) -> tuple[float, ...]:
    """The explicit month 0/12/24/36 output times of §4's run evidence, from the design."""
    design = context.design
    source = design.get("p1_design") or design.get("inverse_design")
    if not isinstance(source, dict) or "start_date" not in source or "n_months" not in source:
        raise ValueError(
            "the run world's design declares no calendar (start_date/n_months): the "
            "explicit month 0/12/24/36 state outputs cannot be derived from it"
        )
    n_months = int(source["n_months"])
    if n_months < max(PHYSICAL_STATE_MONTHS):
        raise ValueError(
            f"a {n_months}-month world cannot publish states at months "
            f"{list(PHYSICAL_STATE_MONTHS)}"
        )
    edges = month_edges_s(date.fromisoformat(str(source["start_date"])), n_months)
    return tuple(float(edges[month]) for month in PHYSICAL_STATE_MONTHS)


def load_parent_world(
    *,
    corpus_path: Path,
    parent_path: Path,
    paths: ProjectPaths,
    ctx: RunContext,
) -> ParentWorld:
    """Read one published run world through its corpus, refusing the evaluation split.

    The parent file is not trusted as bytes from the command line: it must BE the file the
    corpus manifest names for its parent id, verified through the stream index, and the
    split comes from the manifest row. The ledger ruling for this task allows development
    (and train) parents only — `evaluation` is the evaluator's, and the refusal lives here
    so no later step can forget it.
    """
    corpus_source = paths.resolve(paths.relative(Path(corpus_path)))
    if not corpus_source.is_file():
        raise ValueError(f"corpus manifest {paths.relative(corpus_source)} does not exist")
    corpus_ref = register_artifact(
        corpus_source,
        paths,
        schema_version=CORPUS_MANIFEST_SCHEMA,
        producer_run_id=ctx.run_id,
        media_type="application/json",
        now=datetime.now(UTC),
    )
    ctx.update(raw_input_hashes={"corpus_manifest": corpus_ref.sha256})
    dataset = CorpusDataset.open(corpus_ref, paths)
    parent_source = paths.resolve(paths.relative(Path(parent_path)))
    if not parent_source.is_file():
        raise ValueError(f"parent context {paths.relative(parent_source)} does not exist")
    declared = json.loads(parent_source.read_text(encoding="utf-8"))
    if declared.get("schema_version") != PARENT_CONTEXT_SCHEMA:
        raise ValueError("the parent file is not a published e03 parent context")
    parent_id = str(declared.get("parent_id"))
    rows = [row for row in dataset.parents() if row.parent_id == parent_id]
    if len(rows) != 1:
        raise ValueError(f"the corpus names no unique parent {parent_id!r}")
    row = rows[0]
    if row.split == "evaluation":
        raise ValueError(
            f"parent {parent_id!r} belongs to the evaluation split: evaluation worlds are "
            "the evaluator's and are refused by the learned loop (ledger ruling, Task 07)"
        )
    if row.split not in ("train", "development"):
        raise ValueError(f"parent {parent_id!r} declares unknown split {row.split!r}")
    if paths.resolve(row.context_ref.path) != parent_source:
        raise ValueError(
            f"the --parent file is not the context artifact the corpus manifest names for "
            f"{parent_id!r}: a run world comes from its corpus, not from an arbitrary path"
        )
    payload = dataset.load_context(row)
    inference_input = payload["inference_input"]
    context = PriorContext.model_validate(inference_input["context"])
    observations = ObservationBundle.model_validate(inference_input["observations"])
    prior = GaussianConditionalPrior(context)
    return ParentWorld(
        parent_id=parent_id,
        design_id=str(payload["design_id"]),
        split=str(row.split),
        payload=dict(payload),
        context=context,
        observations=observations,
        prior=prior,
        context_ref=row.context_ref,
        state_times_s=learned_state_times(context),
    )


def bind_defensive_law(
    *,
    proposal_manifest_path: Path,
    parent: ParentWorld,
    learning: LearningConfig,
    paths: ProjectPaths,
) -> DefensiveLaw:
    """Bind the frozen q to the run world and arm `r = (1-eps) q + eps p0` (§5.4).

    Every stream is verified against the manifest digests BEFORE the model is rebuilt, and
    `load_checkpoint` re-verifies inside, so an altered checkpoint is refused at this
    boundary rather than discovered in the numbers. `bind_frozen_proposal` then refuses the
    incompatible pairings (spec, layout, basis, architecture) exactly as in Task 05.
    """
    manifest_path = paths.resolve(paths.relative(Path(proposal_manifest_path)))
    if not manifest_path.is_file():
        raise ValueError(f"proposal manifest {paths.relative(manifest_path)} does not exist")
    manifest = ProposalManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.schema_version != PROPOSAL_MANIFEST_SCHEMA:
        raise ValueError("unexpected proposal manifest schema")
    root = manifest_path.parent
    stream_files = {
        "weights_sha256": (root / "weights.safetensors", manifest.weights_ref.path),
        "flow_config_sha256": (root / "flow_config.json", manifest.flow_config_ref.path),
        "scaler_sha256": (root / "scaler.json", manifest.scaler_ref.path),
    }
    stream_hashes: dict[str, str] = {}
    for key, (stream, ref_path) in stream_files.items():
        if not stream.is_file():
            raise ValueError(
                f"the proposal export is incomplete: {stream.name} is missing beside the manifest"
            )
        digest = sha256_file(stream)
        expected = str(getattr(manifest, key))
        if digest != expected:
            raise ValueError(
                f"{stream.name} fails its published digest (got {digest}, the manifest "
                f"vouches for {expected}): the checkpoint was modified after publication"
            )
        if sha256_file(paths.resolve(ref_path)) != expected:
            raise ValueError(
                f"{stream.name} disagrees with the stream the manifest registers at "
                f"{ref_path}: the export and its registered source are different objects"
            )
        stream_hashes[key] = digest
    checkpoint = load_checkpoint(
        root,
        expected=CheckpointHashes(
            weights_sha256=manifest.weights_sha256,
            flow_config_sha256=manifest.flow_config_sha256,
            scaler_sha256=manifest.scaler_sha256,
        ),
    )
    from so_recon.synthetic.inverse_corpus import default_context_spec

    spec = default_context_spec(
        cutoff_s=float(parent.payload["cutoff_s"]),
        max_wells=learning.encoder.max_wells,
        max_months=learning.encoder.max_months,
    )
    batch = build_context_batch(parent.payload, spec=spec)
    q = bind_frozen_proposal(
        checkpoint=checkpoint,
        manifest=manifest,
        spec=spec,
        batch=batch,
        schema=parent.context.density_schema,
    )
    epsilon = float(learning.defensive_epsilon)
    mixture = DefensiveMixture(parent.prior, q, epsilon)
    return DefensiveLaw(
        q=q,
        mixture=mixture,
        epsilon=epsilon,
        manifest=manifest,
        stream_hashes=stream_hashes,
        spec_hash=str(spec.spec_hash),
        weights_sha256=manifest.weights_sha256,
    )


def particle_density_diagnostics(
    *,
    state: SMCState,
    law: DefensiveLaw,
    parent_id: str,
    run_label: str,
) -> dict[str, Any]:
    """§4.4: the versioned density-diagnostics artifact for the final particles.

    `log_q` is the LEARNED component alone; `log_r` is the defensive mixture the engine
    actually sampled and weighted — the two must be recorded separately, and `log_r` must
    agree with the TargetEvaluation the engine stored for the same theta. This artifact
    sits BESIDE `TargetEvaluation` (which keeps its own single `log_r`) and is tied to each
    particle's theta hash.
    """
    particles = []
    for index, particle in enumerate(state.particles):
        evaluation = particle.evaluation
        log_q = float(law.q.log_prob(evaluation.theta))
        log_r = float(law.mixture.log_prob(evaluation.theta))
        if abs(log_r - float(evaluation.log_r)) > DENSITY_MATCH_ATOL:
            raise ValueError(
                f"particle {index} theta {theta_hash(evaluation.theta)}: mixture density "
                f"{log_r!r} disagrees with the engine's stored log_r {evaluation.log_r!r}"
            )
        particles.append(
            {
                "particle_id": particle.particle_id,
                "ancestor_id": particle.ancestor_id,
                "theta_hash": theta_hash(evaluation.theta),
                "log_p0": float(evaluation.log_p0),
                "log_q": log_q,
                "log_r": log_r,
                "log_l": float(evaluation.log_l),
                "log_weight": float(state.log_weights[index]) if state.log_weights else None,
            }
        )
    return {
        "schema_version": PARTICLE_DENSITY_DIAGNOSTICS_SCHEMA,
        "parent_id": parent_id,
        "run_label": run_label,
        "beta": float(state.beta),
        "algorithm_status": state.algorithm_status,
        "proposal_fingerprint": law.q.fingerprint,
        "mixture_fingerprint": law.mixture.fingerprint,
        "epsilon": law.epsilon,
        "weights_sha256": law.weights_sha256,
        "particles": particles,
    }


def _final_forward_refs(
    state: SMCState, state_times_s: tuple[float, ...], paths: ProjectPaths
) -> tuple[ArtifactRef, ...]:
    """The distinct physical forwards behind the final particles, state times verified."""
    refs: dict[str, ArtifactRef] = {}
    for particle in state.particles:
        ref = particle.evaluation.forward_ref
        if ref is None:
            raise ValueError(
                "a physical learned run must score through published forwards: particle "
                f"{particle.particle_id} carries none"
            )
        refs[ref.artifact_id] = ref
    ordered = tuple(sorted(refs.values(), key=lambda item: item.artifact_id))
    for ref in ordered:
        result = load_forward_result(paths.resolve(ref.path), paths)
        missing = [time for time in state_times_s if time not in result.times_s]
        if missing:
            raise ValueError(
                f"forward {ref.path} does not publish the explicit state outputs at "
                f"{missing}: the run's state evidence is incomplete"
            )
    return ordered


def run_learned_smc(
    *,
    ctx: RunContext,
    parent: ParentWorld,
    law: DefensiveLaw,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
    config: InferenceConfig,
    resume_checkpoint_ref: ArtifactRef | None = None,
    resume_payload: dict[str, Any] | None = None,
    run_label: str = ENGINEERING_SMOKE_LABEL,
    stop_requested: Callable[[], bool] = lambda: False,
    adapter_hash: str | None = None,
) -> ArtifactRef:
    """One learned defensive-mixture SMC run: r into the target AND into the engine.

    The mixture is fixed before inference starts and is the one object handed to
    `PhysicalTarget` (its `log_r`, its fingerprint, its checkpoint hashes) and to
    `infer`/`continue_inference` (its samples, its kernel densities). There is no
    prior-fallback path: a run of this function either uses `r` everywhere or fails.
    `adapter_hash` is `PhysicalTarget`'s own seam: None hashes the real forward sources
    (the only mode the CLI uses); unit tests pin it to run without the repository tree.
    """
    paths: ProjectPaths = worker.paths
    thresholds = verify_balance_threshold_config(paths.root)
    physical = PhysicalTarget(
        parent.prior,
        law.mixture,
        parent.context,
        parent.observations,
        worker,
        ledger,
        run_factory,
        parent_run_ids=(ctx.run_id, parent.context_ref.producer_run_id),
        state_times_s=parent.state_times_s,
        adapter_hash=adapter_hash,
    )
    identity = scientific_target_identity(physical)
    if resume_payload is not None:
        previous_identity = str(resume_payload.get("scientific_target_identity"))
        if previous_identity != identity:
            raise ValueError(
                "the resumed run's scientific target identity differs: the world, data or "
                "physics changed between sessions, so this is not the same posterior"
            )
        previous_parent = resume_payload.get("parent", {})
        if previous_parent.get("context_sha256") != parent.context_ref.sha256:
            raise ValueError("the resumed run belongs to another parent context")
    target = _BudgetAwareTarget(physical)
    if resume_checkpoint_ref is None:
        state = infer(
            target,
            law.mixture,
            parent.context.density_schema,
            config,
            ctx.run_dir / "working",
            stop_requested,
        )
    else:
        expected_hashes = {
            "target_hash": physical.fingerprint,
            "proposal_hash": law.mixture.fingerprint,
            "basis_hash": parent.context.density_schema.basis_hash,
            "config_hash": sha256_json(config.model_dump(mode="json")),
            **physical.checkpoint_hashes,
        }
        previous_state = load_state(resume_checkpoint_ref, paths, expected_hashes)
        state = continue_inference(
            previous_state,
            target,
            law.mixture,
            parent.context.density_schema,
            config,
            ctx.run_dir / "working",
            stop_requested,
        )
    checkpoint_ref = save_state(state, ctx.run_dir / "checkpoint/manifest.json", paths, ctx)
    ledger_ref = register_artifact(
        ledger.path,
        paths,
        schema_version=ledger.record.schema_version,
        producer_run_id=ctx.run_id,
        media_type="application/json",
        now=datetime.now(UTC),
    )
    ctx.add_output("budget_ledger", ledger_ref)
    budget = {
        "session": ledger.session_totals().model_dump(mode="json"),
        "cumulative": ledger.cumulative_totals().model_dump(mode="json"),
    }
    diagnostics = particle_density_diagnostics(
        state=state, law=law, parent_id=parent.parent_id, run_label=run_label
    )
    diagnostics_ref = write_json_artifact(
        ctx.run_dir / "particle_density_diagnostics.json",
        diagnostics,
        paths,
        schema_version=PARTICLE_DENSITY_DIAGNOSTICS_SCHEMA,
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )
    ctx.add_output("particle_density_diagnostics", diagnostics_ref)
    weights = np.exp(np.asarray(state.log_weights, dtype=np.float64))
    families = _family_probabilities(state, weights / weights.sum()) if len(weights) else {}
    physical_state_refs: tuple[ArtifactRef, ...] = ()
    state_evidence: dict[str, Any] = {
        "state_times_s": list(parent.state_times_s),
        "note": ("no likelihood was scored (empty observations): no physical forwards to report"),
    }
    if parent.observations.history or parent.observations.logs:
        physical_state_refs = _final_forward_refs(state, parent.state_times_s, paths)
        state_evidence = {
            "state_times_s": list(parent.state_times_s),
            "months": list(PHYSICAL_STATE_MONTHS),
            "forward_refs": [ref.model_dump(mode="json") for ref in physical_state_refs],
        }
    kind = ensemble_kind(state.beta, state.algorithm_status)
    payload = {
        "schema_version": LEARNED_SMC_RUN_SCHEMA,
        "run_label": run_label,
        "engineering_smoke": run_label == ENGINEERING_SMOKE_LABEL,
        "parent": {
            "parent_id": parent.parent_id,
            "design_id": parent.design_id,
            "split": parent.split,
            "context_ref": parent.context_ref.model_dump(mode="json"),
            "context_sha256": parent.context_ref.sha256,
        },
        "proposal": {
            "q_fingerprint": law.q.fingerprint,
            "mixture_fingerprint": law.mixture.fingerprint,
            "epsilon": law.epsilon,
            "weights_sha256": law.weights_sha256,
            "stream_hashes": law.stream_hashes,
            "spec_hash": law.spec_hash,
            "proposal_manifest_schema": law.manifest.schema_version,
        },
        "scientific_target_identity": identity,
        "runtime_target_fingerprint": physical.fingerprint,
        "target_hashes": physical.checkpoint_hashes,
        "balance_thresholds": thresholds,
        "algorithm_status": state.algorithm_status,
        "beta": state.beta,
        "ensemble_kind": kind,
        "posterior_claim": (
            "POSTERIOR" if kind == "posterior" else "REFUSED: beta<1 or incomplete output"
        ),
        "n_particles": config.n_particles,
        "seed": config.seed,
        "log_evidence": state.log_evidence,
        "unique_ancestors": len({particle.ancestor_id for particle in state.particles}),
        "final_ess": ess(np.asarray(state.log_weights, dtype=np.float64))
        if state.log_weights
        else None,
        "family_probabilities": families,
        "beta_history": list(state.diagnostics.get("beta_history", [])),
        "ess_history": list(state.diagnostics.get("ess_history", [])),
        "resampling": list(state.diagnostics.get("resampling", [])),
        "state_evidence": state_evidence,
        "native_restart": {
            "used": False,
            "reason": (
                "state outputs at months 0/12/24/36 come from the ordinary output request; "
                "a separately accounted native restart was not needed for this run's "
                "evidence (plan §12)"
            ),
        },
        "config": config.model_dump(mode="json"),
        "checkpoint": checkpoint_ref.model_dump(mode="json"),
        "budget_ledger_ref": None if ledger_ref is None else ledger_ref.model_dump(mode="json"),
        "resumed_from_checkpoint": (
            None if resume_checkpoint_ref is None else resume_checkpoint_ref.model_dump(mode="json")
        ),
        "budget": budget,
    }
    ref = write_json_artifact(
        ctx.run_dir / "learned_smc.json",
        payload,
        paths,
        schema_version=LEARNED_SMC_RUN_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(
            parent.context_ref.artifact_id,
            checkpoint_ref.artifact_id,
            diagnostics_ref.artifact_id,
        ),
        now=datetime.now(UTC),
    )
    ctx.add_output("learned_smc", ref)
    manifest = json.loads(paths.resolve(checkpoint_ref.path).read_text(encoding="utf-8"))
    particles_ref = ArtifactRef.model_validate(manifest["shards"]["particles"])
    if kind == "posterior":
        posterior = PosteriorBundle(
            state_ref=checkpoint_ref,
            particles_ref=particles_ref,
            diagnostics_ref=ref,
            ledger_ref=ledger_ref,
            algorithm_status=state.algorithm_status,
            beta=state.beta,
            convergence_status="NOT_ASSESSED",
            physical_state_refs=physical_state_refs,
            parent_run_ids=(parent.context_ref.producer_run_id,),
        )
        posterior_ref = write_json_artifact(
            ctx.run_dir / "posterior_bundle.json",
            posterior.model_dump(mode="json"),
            paths,
            schema_version=LEARNED_POSTERIOR_BUNDLE_SCHEMA,
            producer_run_id=ctx.run_id,
            parent_artifact_ids=(
                checkpoint_ref.artifact_id,
                particles_ref.artifact_id,
                ref.artifact_id,
            ),
            now=datetime.now(UTC),
        )
        ctx.add_output("posterior_bundle", posterior_ref)
    return ref


def _resume_inputs(
    resume: Path | None, paths: ProjectPaths
) -> tuple[ArtifactRef | None, dict[str, Any] | None]:
    if resume is None:
        return None, None
    source = paths.resolve(paths.relative(Path(resume)))
    if not source.is_file():
        raise ValueError(f"resume payload {paths.relative(source)} does not exist")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LEARNED_SMC_RUN_SCHEMA:
        raise ValueError("the resume payload is not a learned SMC run")
    if payload.get("ensemble_kind") == "posterior":
        raise ValueError("the resume payload already reached beta=1: nothing to continue")
    return ArtifactRef.model_validate(payload["checkpoint"]), payload


def run_learned_loop_command(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    corpus_path: Path,
    parent_path: Path,
    proposal_manifest_path: Path,
    resume: Path | None = None,
    session_id: str | None = None,
    julia: str | None = None,
    argv: Sequence[str] = (),
) -> RunContext:
    """`run-learned-loop`: one bounded learned session against one published run world."""
    from so_recon.environment.resources import probe_resources
    from so_recon.simulator.julia_bridge import find_julia

    learning = cfg.learning
    if learning is None:
        raise ValueError("run-learned-loop requires the validated learning block")
    inference = cfg.inference
    if inference is None:
        raise ValueError("run-learned-loop requires the validated inference block")
    profile = require_resource_profile(cfg.resources, command="run-learned-loop")

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        resume_checkpoint_ref, resume_payload = _resume_inputs(resume, paths)
        parent = load_parent_world(
            corpus_path=corpus_path, parent_path=parent_path, paths=paths, ctx=ctx
        )
        law = bind_defensive_law(
            proposal_manifest_path=proposal_manifest_path,
            parent=parent,
            learning=learning,
            paths=paths,
        )
        log.info(
            "run world %s (split %s, design %s): q %s, epsilon %.2f",
            parent.parent_id,
            parent.split,
            parent.design_id,
            law.q.fingerprint[:12],
            law.epsilon,
        )
        session = paths.artifacts / (session_id or f"{parent.parent_id}-learned-smc-{ctx.run_id}")
        session.mkdir(parents=True, exist_ok=True)
        config = InferenceConfig(
            n_particles=inference.n_particles,
            cess_fraction=inference.cess_fraction,
            resample_fraction=inference.resample_fraction,
            max_beta_steps=inference.max_beta_steps,
            moves_per_level=inference.moves_per_level,
            seed=inference.seed,
            rw_scale=inference.rw_scale,
            pcn_scale=inference.pcn_scale,
        )

        def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
            return RunContext.start(
                command=command,
                argv=[],
                cfg=cfg,
                paths=paths,
                parent_run_ids=parent_run_ids,
            )

        def probe() -> Any:
            return probe_resources(None, session)

        ledger_path = session / "ledger.json"
        if resume_payload is None:
            ledger = BudgetLedger.start(
                profile=profile,
                path=ledger_path,
                session_id=session.name,
                probe=probe,
            )
        else:
            previous_ledger = resume_payload.get("budget_ledger_ref")
            if not isinstance(previous_ledger, dict):
                raise ValueError("the resume payload records no budget ledger to continue")
            ledger = BudgetLedger.resume(
                parent_path=paths.resolve(str(previous_ledger["path"])),
                path=ledger_path,
                session_id=session.name,
                probe=probe,
            )
        with PersistentJuliaWorker(
            find_julia(julia), paths.julia, session, profile, paths=paths, probe=probe
        ) as worker:
            ref = run_learned_smc(
                ctx=ctx,
                parent=parent,
                law=law,
                worker=worker,
                ledger=ledger,
                run_factory=run_factory,
                config=config,
                resume_checkpoint_ref=resume_checkpoint_ref,
                resume_payload=resume_payload,
                run_label=ENGINEERING_SMOKE_LABEL,
            )
        payload = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
        complete = payload["algorithm_status"] == "COMPLETE" and float(payload["beta"]) == 1.0
        notes = [
            f"parent={parent.parent_id} split={parent.split}",
            f"status={payload['algorithm_status']} beta={payload['beta']}",
            f"ensemble_kind={payload['ensemble_kind']}",
            f"final_ess={payload['final_ess']}",
        ]
        if payload.get("budget"):
            session_totals = payload["budget"]["session"]
            notes.append(
                f"forwards={session_totals.get('forwards')} "
                f"wall_s={session_totals.get('wall_s'):.1f}"
            )
        log.info(
            "learned run finished: status %s beta %s ess %s",
            payload["algorithm_status"],
            payload["beta"],
            payload["final_ess"],
        )
        return ("PASS" if complete else "FAIL"), notes

    return execute_run(
        command="run-learned-loop",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={
            "e03_learned_smc_run": LEARNED_SMC_RUN_SCHEMA,
            "e03_particle_density_diagnostics": PARTICLE_DENSITY_DIAGNOSTICS_SCHEMA,
            "e03_learned_posterior_bundle": LEARNED_POSTERIOR_BUNDLE_SCHEMA,
        },
    )


__all__ = [
    "CORPUS_MANIFEST_SCHEMA",
    "ENGINEERING_SMOKE_LABEL",
    "LEARNED_POSTERIOR_BUNDLE_SCHEMA",
    "LEARNED_SMC_RUN_SCHEMA",
    "PARENT_CONTEXT_SCHEMA",
    "PARTICLE_DENSITY_DIAGNOSTICS_SCHEMA",
    "DefensiveLaw",
    "ParentWorld",
    "bind_defensive_law",
    "learned_state_times",
    "load_parent_world",
    "particle_density_diagnostics",
    "run_learned_loop_command",
    "run_learned_smc",
]
