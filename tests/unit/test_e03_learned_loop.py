"""E03 Task 07 — the same-target learned SMC loop (plan §0.2 items 2-3, §4.3, §5.4, §9).

The rows this suite pins:

* scientific target identity — changing q (or epsilon, or the diagnostic state-request
  times) does not change it; changing physics or data does; the E02 runtime fingerprint
  stays proposal-strict and DOES move with q;
* the config carries the §0.2 item 10 balance law (1e-5 median step, not E02's 1e-4) and
  the runner refuses to start when config and code disagree;
* epsilon=1 is exactly the prior-start path, engine-level, bitwise;
* an altered checkpoint stream is refused at binding, before any inference;
* stop/resume phase boundaries with a learned defensive mixture, including the pending
  draw/MH and the budget stop, mirrored from the E02 engine suite;
* the runner really hands r = (1-eps) q + eps p0 to BOTH the target and the engine, and
  the recorded log_q/log_r are the true densities (logsumexp of the components, never the
  sampled component alone);
* beta<1 outputs are labelled diagnostics and never receive a posterior bundle, and the
  L=1 (empty-data) run restores the prior on an independent reference draw;
* the run world comes from the corpus, the evaluation split is refused, and a foreign
  parent file is refused.

No Julia runs here: physics-free worlds use empty observations (the pure prior/mixture
path of `PhysicalTarget.evaluate`) and a stub worker.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from so_recon.config.inference import InferenceConfig
from so_recon.config.learning import EncoderParams, FlowParams, LearningConfig
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.inference.contracts import (
    DensitySchema,
    ObservationBundle,
    PriorContext,
    TargetEvaluation,
    ThetaRecord,
)
from so_recon.inference.proposals import DefensiveMixture
from so_recon.inference.smc import SMCBudgetStop, continue_inference, infer
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import default_context_spec
from so_recon.validation.e03_protocol import (
    DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
    POSTERIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
    completed_posterior,
    ensemble_kind,
    scientific_target_identity,
    verify_balance_threshold_config,
)

ROOT = Path(__file__).resolve().parents[2]

WELLS = [("I1", 2, 2, 0), ("I2", 2, 13, 0), ("P1", 13, 2, 1), ("P2", 13, 13, 1)]
N_MONTHS = 36
EDGES = month_edges_s(date(2000, 1, 1), N_MONTHS)
CUTOFF_S = float(EDGES[-1])

ADAPTER_HASH = "a" * 64
LOCK_HASH = "1" * 64


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _prior_context(
    seed: int,
    *,
    design: dict[str, Any] | None = None,
    families: tuple[int, ...] = (0, 1),
) -> PriorContext:
    rng = np.random.default_rng(seed)
    n_geology = 12
    mean = 0.3 * rng.standard_normal(n_geology)
    rotation, _ = np.linalg.qr(rng.standard_normal((n_geology, n_geology)))
    rotation = rotation * np.sign(np.diag(rotation))
    root = rotation @ np.diag(np.linspace(0.6, 1.4, n_geology))
    schema = DensitySchema(
        schema_id="p1-conditional-12",
        n_v=11,
        n_residual=4,
        families=families,
        basis_hash=sha256_json({"schema_id": "p1-conditional-12", "seed": seed}),
        transform_version="e02-geology-conditional-1",
    )
    return PriorContext(
        density_schema=schema,
        n_geology=n_geology,
        n_state_residual=0,
        mean=mean,
        chol=root,
        rotation=rotation,
        design={} if design is None else design,
        g_hash=sha256_json({"g": seed}),
        information_hash=sha256_json({"information": seed}),
    )


def _parent_payload(context: PriorContext, seed: int) -> dict[str, Any]:
    """The canonical corpus parent shape (Task 04/06 fixtures) for one world."""
    design = {
        "design_id": "e02-t2-v2",
        "shape": [16, 16, 2],
        "n_months": N_MONTHS,
        "start_date": "2000-01-01",
        "well_columns": [
            {"well_id": name, "column": [i, j], "role": "producer" if role else "injector"}
            for name, i, j, role in WELLS
        ],
    }
    controls = [
        {
            "well_id": name,
            "start_s": EDGES[month],
            "end_s": EDGES[month + 1],
            "role": "producer" if role else "injector",
            "target": "liquid_rate" if role else "water_rate",
            "value": 20.0,
            "bhp_limit_pa": None,
            "connection_open": (True, True),
        }
        for name, _i, _j, role in WELLS
        for month in range(N_MONTHS)
    ]
    history = [
        {
            "well_id": name,
            "month_index": month,
            "raw_value": 0.2 + 0.01 * ((seed + month) % 7),
            "bin_index": 31,
            "quality_group": "watercut-0.01",
            "observed_valid": True,
            "reset": False,
            "sigma_multiplier": 1.0,
        }
        for name, _i, _j, role in WELLS
        if role
        for month in range(N_MONTHS)
    ]
    context_payload = context.model_dump(mode="python")
    for name in ("mean", "chol", "rotation"):
        context_payload[name] = np.asarray(context_payload[name], dtype=np.float64).tolist()
    context_payload["design"] = {"inverse_design": design}
    return {
        "schema_version": "e03-parent-context-1",
        "parent_id": f"e02-t2-v2-{seed:04d}",
        "design_id": "e02-t2-v2",
        "split": "development",
        "cutoff_s": CUTOFF_S,
        "well_ids": ["P1", "P2"],
        "inference_input": {
            "context": context_payload,
            "G": [],
            "U": controls,
            "observations": {
                "history": history,
                "logs": [],
                "bin_edges_by_group": {
                    "watercut-0.01": [round(v, 4) for v in np.linspace(0.0, 1.0, 101)]
                },
                "cutoff_s": CUTOFF_S,
                "information_hash": "a" * 64,
                "observation_hash": "b" * 64,
            },
        },
    }


def _empty_observations(observation_hash: str = "b" * 64) -> ObservationBundle:
    return ObservationBundle(
        history=(),
        logs=(),
        bin_edges_by_group={},
        cutoff_s=CUTOFF_S,
        information_hash="a" * 64,
        observation_hash=observation_hash,
    )


@dataclass
class _StubWorker:
    paths: ProjectPaths
    environment_lock_hash: str = LOCK_HASH


def _refusing_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
    raise AssertionError("the physics-free worlds of this suite must not open child runs")


def _learning() -> LearningConfig:
    from so_recon.config.learning import FamilyPlan, TrainingHyperParams

    return LearningConfig(
        corpus=(FamilyPlan(design_id="e02-t2-v2", train=4, development=2, evaluation=1),),
        encoder=EncoderParams(
            variant="summary", width=8, heads=2, context_dim=8, max_wells=4, max_months=36
        ),
        flow=FlowParams(n_transforms=2, n_bins=4),
        training=TrainingHyperParams(max_epochs=1, patience=1, batch_size=4),
    )


def _export_proposal(paths: ProjectPaths, payload: dict[str, Any]) -> Path:
    """A real two-directory frozen export: streams+manifest in the export root, refs to a
    registered source directory — the Task 06 layout on disk."""
    from so_recon.ml.checkpoint import save_checkpoint
    from so_recon.ml.context import build_context_batch
    from so_recon.ml.contracts import LayoutSupport, ProposalManifest
    from so_recon.ml.flow import (
        DEFAULT_CONDITIONER_HIDDEN_FEATURES,
        ConditionalNSF,
    )
    from so_recon.ml.normalization import FeatureScaler
    from so_recon.ml.proposal import LearnedProposal, flow_config_payload
    from so_recon.synthetic.inverse_corpus import WELL_TIME_FEATURES

    learning = _learning()
    spec = default_context_spec(
        cutoff_s=float(payload["cutoff_s"]),
        max_wells=learning.encoder.max_wells,
        max_months=learning.encoder.max_months,
    )
    # the scaler is fit on train-labelled batches only; the export fixture plays the role
    # of the training stream regardless of the world's own split
    train_payload = dict(payload)
    train_payload["split"] = "train"
    batch = build_context_batch(train_payload, spec=spec)
    from so_recon.ml.encoders import build_encoder

    schema = DensitySchema.model_validate(payload["inference_input"]["context"]["density_schema"])
    encoder = build_encoder(
        "summary",
        learning.encoder,
        len(WELL_TIME_FEATURES),
        len(spec.edge_features),
        len(spec.static_features),
    )
    flow = ConditionalNSF(
        learning.flow,
        n_v=schema.n_v,
        context_dim=learning.encoder.context_dim,
        n_families=2,
        hidden_features=DEFAULT_CONDITIONER_HIDDEN_FEATURES,
    )
    torch.manual_seed(11)
    model = LearnedProposal(encoder, flow, n_families=2)
    scaler = FeatureScaler.fit([batch], spec)
    config = flow_config_payload(
        flow=learning.flow,
        encoder_variant="summary",
        encoder_params=learning.encoder,
        n_well_time_features=len(WELL_TIME_FEATURES),
        n_edge_features=len(spec.edge_features),
        n_static_features=len(spec.static_features),
        n_v=schema.n_v,
        context_dim=learning.encoder.context_dim,
        hidden_features=DEFAULT_CONDITIONER_HIDDEN_FEATURES,
        n_families=2,
    )
    source = paths.artifacts / "test-proposal-src"
    export = paths.artifacts / "test-proposal-export"
    hashes = save_checkpoint(
        source, module=model, flow_config=config, scaler_payload=scaler.payload()
    )
    export.mkdir(parents=True, exist_ok=True)
    for name in ("weights.safetensors", "flow_config.json", "scaler.json"):
        (export / name).write_bytes((source / name).read_bytes())

    def _ref(filename: str, digest: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=digest,
            path=str((source / filename).relative_to(paths.root)).replace("\\", "/"),
            sha256=digest,
            size_bytes=(source / filename).stat().st_size,
            media_type="application/octet-stream",
            schema_version="e03-task07-test",
            producer_run_id="task07-test",
            parent_artifact_ids=[],
            created_at="2026-09-17T00:00:00",
        )

    manifest = ProposalManifest(
        weights_ref=_ref("weights.safetensors", hashes.weights_sha256),
        weights_sha256=hashes.weights_sha256,
        flow_config_ref=_ref("flow_config.json", hashes.flow_config_sha256),
        flow_config_sha256=hashes.flow_config_sha256,
        scaler_ref=_ref("scaler.json", hashes.scaler_sha256),
        scaler_sha256=hashes.scaler_sha256,
        supported_layouts=(
            LayoutSupport(
                schema_id=schema.schema_id,
                n_v=schema.n_v,
                n_residual=schema.n_residual,
                families=schema.families,
                basis_hash=schema.basis_hash,
            ),
        ),
        operational_dtype="float64",
        operational_backend="cpu",
        architecture_version="e03-conditional-nsf-1",
        context_builder_version="e03-context-1",
    )
    (export / "proposal_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export / "proposal_manifest.json"


def _bind_law(paths: ProjectPaths, parent: Any, manifest_path: Path) -> Any:
    from so_recon.inference.learned_loop import bind_defensive_law

    return bind_defensive_law(
        proposal_manifest_path=manifest_path,
        parent=parent,
        learning=_learning(),
        paths=paths,
    )


def _world(paths: ProjectPaths, *, seed: int = 3) -> Any:
    """A ParentWorld with empty observations: likelihood L=1, no forward, no Julia."""
    from so_recon.inference.learned_loop import ParentWorld

    context = _prior_context(seed)
    payload = _parent_payload(context, seed)
    ref = ArtifactRef(
        artifact_id="c" * 64,
        path=f"artifacts/test-worlds/{seed}.json",
        sha256="c" * 64,
        size_bytes=1,
        media_type="application/json",
        schema_version="e03-parent-context-1",
        producer_run_id="task07-test",
        parent_artifact_ids=[],
        created_at="2026-09-17T00:00:00",
    )
    return ParentWorld(
        parent_id=payload["parent_id"],
        design_id="e02-t2-v2",
        split="development",
        payload=payload,
        context=context,
        observations=_empty_observations(),
        prior=GaussianConditionalPrior(context),
        context_ref=ref,
        state_times_s=tuple(float(EDGES[month]) for month in (0, 12, 24, 36)),
    )


def _project(tmp_path: Path) -> ProjectPaths:
    """A minimal project carrying the frozen E03 tolerances config."""
    project = tmp_path / "repo"
    (project / "configs").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (project / "configs" / "e03_tolerances.yml").write_text(
        (ROOT / "configs" / "e03_tolerances.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    paths = ProjectPaths.default(project)
    paths.ensure_dirs()
    return paths


def _ledger(paths: ProjectPaths, name: str) -> BudgetLedger:
    from so_recon.config.resources import P1_LOOP_PROFILE
    from so_recon.environment.resources import probe_resources

    session = paths.artifacts / f"test-session-{name}"
    session.mkdir(parents=True, exist_ok=True)
    return BudgetLedger.start(
        profile=P1_LOOP_PROFILE,
        path=session / "ledger.json",
        session_id=session.name,
        probe=lambda: probe_resources(None, session),
    )


def _config(**overrides: Any) -> InferenceConfig:
    values: dict[str, Any] = {
        "n_particles": 12,
        "cess_fraction": 0.8,
        "resample_fraction": 0.9,
        "max_beta_steps": 8,
        "moves_per_level": 1,
        "seed": 73,
        "rw_scale": 0.7,
        "pcn_scale": 0.3,
    }
    values.update(overrides)
    return InferenceConfig(**values)


def _run(
    paths: ProjectPaths,
    parent: Any,
    law: Any,
    *,
    name: str,
    config: InferenceConfig | None = None,
    stop_requested: Any = lambda: False,
    resume_checkpoint_ref: ArtifactRef | None = None,
    resume_payload: dict[str, Any] | None = None,
) -> tuple[ArtifactRef, dict[str, Any]]:
    from so_recon.inference.learned_loop import run_learned_smc

    ctx = RunContext.start(command=f"test-learned-{name}", argv=[], cfg=None, paths=paths)
    ref = run_learned_smc(
        ctx=ctx,
        parent=parent,
        law=law,
        worker=_StubWorker(paths),
        ledger=_ledger(paths, name),
        run_factory=_refusing_factory,
        config=config or _config(),
        resume_checkpoint_ref=resume_checkpoint_ref,
        resume_payload=resume_payload,
        stop_requested=stop_requested,
        adapter_hash=ADAPTER_HASH,
    )
    payload = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
    return ref, payload


def _state_of(paths: ProjectPaths, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(paths.resolve(payload["checkpoint"]["path"]).read_text(encoding="utf-8"))
    return dict(manifest["state"])


def never_stop() -> bool:
    return False


def stop_on(call: int) -> Any:
    seen = 0

    def requested() -> bool:
        nonlocal seen
        seen += 1
        return seen == call

    return requested


# --------------------------------------------------------------------------------------
# §4.3 scientific target identity
# --------------------------------------------------------------------------------------


class _ToyQ:
    """A second sampler law over the same schema, distinct from the prior."""

    def __init__(self, shift: float) -> None:
        self.shift = shift
        self.fingerprint = sha256_json({"kind": "toy-q", "shift": shift})

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        raise AssertionError("identity tests never sample")

    def log_prob(self, theta: ThetaRecord) -> float:
        schema = _prior_context(3).density_schema
        coordinates = np.asarray(theta.v + theta.z_perp, dtype=np.float64)
        return float(
            norm.logpdf(coordinates[0], self.shift, 1.0)
            - 0.5 * (coordinates[1:] @ coordinates[1:])
            - 0.5 * (len(coordinates) - 1) * math.log(2.0 * math.pi)
            - math.log(len(schema.families))
        )


def _physical(
    paths: ProjectPaths,
    context: PriorContext,
    proposal: Any,
    observations: ObservationBundle,
    *,
    adapter_hash: str = ADAPTER_HASH,
    state_times_s: tuple[float, ...] = (EDGES[12],),
) -> Any:
    from so_recon.inference.target import PhysicalTarget

    return PhysicalTarget(
        GaussianConditionalPrior(context),
        proposal,
        context,
        observations,
        _StubWorker(paths),
        None,  # type: ignore[arg-type]
        _refusing_factory,  # type: ignore[arg-type]
        adapter_hash=adapter_hash,
        state_times_s=state_times_s,
    )


def test_q_change_leaves_scientific_identity_but_moves_the_runtime_fingerprint(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    context = _prior_context(3)
    observations = _empty_observations()
    prior = GaussianConditionalPrior(context)
    q = _ToyQ(shift=1.5)
    mixture = DefensiveMixture(prior, q, 0.1)

    prior_start = _physical(paths, context, prior, observations)
    learned = _physical(paths, context, mixture, observations)

    assert scientific_target_identity(prior_start) == scientific_target_identity(learned)
    # §0.2 item 2: the E02 runtime/checkpoint identity stays strict — it moves with q.
    assert prior_start.fingerprint != learned.fingerprint
    assert prior_start.checkpoint_hashes["proposal_hash"] == prior.fingerprint
    assert learned.checkpoint_hashes["proposal_hash"] == mixture.fingerprint
    # epsilon is a sampler choice: a different defensive weight is still the same posterior
    other_eps = _physical(paths, context, DefensiveMixture(prior, q, 0.3), observations)
    assert scientific_target_identity(other_eps) == scientific_target_identity(learned)
    assert other_eps.fingerprint != learned.fingerprint


def test_physics_and_data_changes_move_the_scientific_identity(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    context = _prior_context(3)
    q = _ToyQ(shift=1.5)
    mixture = DefensiveMixture(GaussianConditionalPrior(context), q, 0.1)
    base = _physical(paths, context, mixture, _empty_observations())
    identity = scientific_target_identity(base)

    # data: another observation contract (values/identity) — a different inverse problem
    other_data = _physical(paths, context, mixture, _empty_observations(observation_hash="d" * 64))
    assert scientific_target_identity(other_data) != identity
    # conditioning: another world (G/information/design) — another prior and renderer input
    other_world = _physical(paths, _prior_context(4), mixture, _empty_observations())
    assert scientific_target_identity(other_world) != identity
    # solver/discretization: another adapter implementation
    other_solver = _physical(paths, context, mixture, _empty_observations(), adapter_hash="e" * 64)
    assert scientific_target_identity(other_solver) != identity

    # §4.3 exclusion: a diagnostic output-request change moves the runtime identity only
    other_outputs = _physical(
        paths, context, mixture, _empty_observations(), state_times_s=(EDGES[24],)
    )
    assert other_outputs.fingerprint != base.fingerprint
    assert (
        other_outputs.checkpoint_hashes["output_request_hash"]
        != base.checkpoint_hashes["output_request_hash"]
    )
    assert scientific_target_identity(other_outputs) == identity


def test_labels_distinguish_posterior_from_partial_and_raw_proposal() -> None:
    assert completed_posterior(1.0, "COMPLETE")
    assert ensemble_kind(1.0, "COMPLETE") == POSTERIOR_ENSEMBLE_KIND
    for beta, status in ((0.5, "COMPLETE"), (1.0, "INCOMPLETE_BUDGET"), (0.0, "INCOMPLETE_BUDGET")):
        assert not completed_posterior(beta, status)
        assert ensemble_kind(beta, status) == DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND
    # §4.4: a raw q ensemble has its own marker, never the posterior's
    assert RAW_PROPOSAL_ENSEMBLE_KIND == "raw_proposal"
    assert RAW_PROPOSAL_ENSEMBLE_KIND != POSTERIOR_ENSEMBLE_KIND


def test_config_carries_the_e03_balance_law_and_a_wrong_config_is_refused(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    thresholds = verify_balance_threshold_config(paths.root)
    assert thresholds["truth_balance_step_median_max"] == 1.0e-5
    assert thresholds["truth_balance_cumulative_max"] == 1.0e-3

    body = (paths.root / "configs" / "e03_tolerances.yml").read_text(encoding="utf-8")
    inherited = body.replace(
        "truth_balance_step_median_max: 1.0e-5", "truth_balance_step_median_max: 1.0e-4"
    )
    (paths.root / "configs" / "e03_tolerances.yml").write_text(inherited, encoding="utf-8")
    with pytest.raises(ValueError, match="1e-05|normative balance law"):
        verify_balance_threshold_config(paths.root)


# --------------------------------------------------------------------------------------
# §5.4 bridge: epsilon=1 and the engine's mixture arithmetic
# --------------------------------------------------------------------------------------


class _GaussianDensity:
    def __init__(self, mean: float, variance: float) -> None:
        self.mean = mean
        self.variance = variance
        self.fingerprint = sha256_json({"kind": "test-gaussian", "mean": mean, "var": variance})

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        values = rng.normal(self.mean, math.sqrt(self.variance), size=n)
        return tuple(_theta_1d(float(value)) for value in values)

    def log_prob(self, theta: ThetaRecord) -> float:
        return float(norm.logpdf(theta.v[0], self.mean, math.sqrt(self.variance)))


SCHEMA_1D = DensitySchema(
    schema_id="gaussian-toy-1",
    n_v=1,
    n_residual=0,
    families=(0,),
    basis_hash="a" * 64,
    transform_version="identity-1",
)


def _theta_1d(value: float) -> ThetaRecord:
    return ThetaRecord(
        schema_id=SCHEMA_1D.schema_id, s=0, v=(value,), z_perp=(), basis_hash=SCHEMA_1D.basis_hash
    )


def test_epsilon_one_is_bitwise_the_prior_start_path(tmp_path: Path) -> None:
    from so_recon.validation.toy_inverse import GaussianToyTarget

    prior = _GaussianDensity(0.0, 1.0)
    q = _GaussianDensity(2.5, 1.0)
    target = GaussianToyTarget(0.0, 1.0, 1.5, 0.7, prior)
    config = _config()
    mixture = DefensiveMixture(prior, q, 1.0)

    learned = infer(target, mixture, SCHEMA_1D, config, tmp_path / "eps-one", never_stop)
    prior_start = infer(target, prior, SCHEMA_1D, config, tmp_path / "prior", never_stop)

    assert learned.algorithm_status == prior_start.algorithm_status == "COMPLETE"
    assert learned.beta == prior_start.beta == 1.0
    assert learned.log_evidence == prior_start.log_evidence
    assert learned.log_weights == prior_start.log_weights
    assert [p.evaluation.theta for p in learned.particles] == [
        p.evaluation.theta for p in prior_start.particles
    ]
    assert learned.diagnostics["beta_history"] == prior_start.diagnostics["beta_history"]
    assert learned.diagnostics["ess_history"] == prior_start.diagnostics["ess_history"]
    assert learned.diagnostics["moves"] == prior_start.diagnostics["moves"]
    # the identities remain honest: the epsilon=1 mixture IS a different declared law
    assert learned.proposal_hash == mixture.fingerprint != prior_start.proposal_hash


class _LatentGaussianTarget:
    """A toy physical target over the 11+4 two-family schema a real q binds to."""

    def __init__(self, prior: GaussianConditionalPrior, mixture: DefensiveMixture) -> None:
        self.prior = prior
        self.mixture = mixture
        self.fingerprint = sha256_json(
            {
                "kind": "latent-gaussian-toy-1",
                "prior": prior.fingerprint,
                "proposal": mixture.fingerprint,
            }
        )
        self.checkpoint_hashes: dict[str, str] = {"proposal_hash": mixture.fingerprint}

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
        log_p0 = self.prior.log_prob(theta)
        log_l = float(norm.logpdf(theta.v[0], 0.4, 0.3))
        return TargetEvaluation(
            theta=theta,
            log_p0=log_p0,
            log_p0_in_support=log_p0 != -math.inf,
            log_l=log_l,
            log_l_in_support=log_l != -math.inf,
            log_r=self.mixture.log_prob(theta),
            log_r_in_support=True,
            forward_ref=None,
            cache_key=sha256_json(theta.model_dump(mode="json")),
        )


def test_stop_and_resume_with_a_learned_mixture_is_the_same_phase_machine(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    schema = parent.context.density_schema
    target = _LatentGaussianTarget(parent.prior, law.mixture)
    config = _config(n_particles=8, moves_per_level=2, max_beta_steps=6)

    uninterrupted = infer(target, law.mixture, schema, config, tmp_path / "full", never_stop)
    assert uninterrupted.algorithm_status == "COMPLETE" and uninterrupted.beta == 1.0

    for boundary in (2, 3, 17, 45):
        work = tmp_path / f"cut-{boundary}"
        interrupted = infer(target, law.mixture, schema, config, work, stop_on(boundary))
        if interrupted.algorithm_status == "COMPLETE":
            continue
        assert interrupted.algorithm_status == "INCOMPLETE_BUDGET"
        resumed = continue_inference(
            interrupted, target, law.mixture, schema, config, work, never_stop
        )
        assert resumed.model_dump(mode="json") == uninterrupted.model_dump(mode="json")


def test_learned_mixture_budget_stop_preserves_pending_work_for_a_later_session(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    schema = parent.context.density_schema
    plain = _LatentGaussianTarget(parent.prior, law.mixture)
    config = _config(n_particles=8)

    class OnceBudgeted:
        fingerprint = plain.fingerprint
        checkpoint_hashes = plain.checkpoint_hashes

        def __init__(self) -> None:
            self.stop = True

        def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
            if self.stop:
                raise SMCBudgetStop("P1 session exhausted")
            return plain.evaluate(theta)

    bounded = OnceBudgeted()
    stopped = infer(bounded, law.mixture, schema, config, tmp_path / "budget", never_stop)
    assert stopped.algorithm_status == "INCOMPLETE_BUDGET"
    assert stopped.pending_proposal is not None
    assert stopped.diagnostics["stop_reason"] == "P1 session exhausted"

    bounded.stop = False
    resumed = continue_inference(
        stopped, bounded, law.mixture, schema, config, tmp_path / "budget", never_stop
    )
    uninterrupted = infer(plain, law.mixture, schema, config, tmp_path / "full", never_stop)
    assert resumed.model_dump(mode="json") == uninterrupted.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# binding: altered checkpoints and foreign layouts are refused
# --------------------------------------------------------------------------------------


def test_binding_refuses_an_altered_checkpoint_stream(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    manifest = _export_proposal(paths, parent.payload)
    weights = paths.artifacts / "test-proposal-export" / "weights.safetensors"
    original = weights.read_bytes()

    weights.write_bytes(original + b"\x00")
    with pytest.raises(ValueError, match="fails its published digest"):
        _bind_law(paths, parent, manifest)

    weights.write_bytes(original)
    source = paths.artifacts / "test-proposal-src" / "scaler.json"
    scaler_bytes = source.read_bytes()
    source.write_bytes(scaler_bytes.replace(b"e03-scaler-1", b"e03-scaler-2", 1))
    with pytest.raises(ValueError, match="disagrees with the stream the manifest registers"):
        _bind_law(paths, parent, manifest)
    source.write_bytes(scaler_bytes)

    law = _bind_law(paths, parent, manifest)
    assert law.epsilon == 0.10
    assert isinstance(law.mixture, DefensiveMixture)
    rng = np.random.default_rng(5)
    theta = law.mixture.sample(4, rng)[0]
    log_q, log_p0 = law.q.log_prob(theta), parent.prior.log_prob(theta)
    assert law.mixture.log_prob(theta) == pytest.approx(
        float(logsumexp([math.log(0.9) + log_q, math.log(0.1) + log_p0]))
    )


def test_binding_refuses_a_schema_of_another_basis(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    manifest = _export_proposal(paths, parent.payload)
    foreign = _world(paths, seed=9)  # same schema_id, another basis hash
    with pytest.raises(ValueError, match="layout|basis"):
        _bind_law(paths, foreign, manifest)


# --------------------------------------------------------------------------------------
# the runner: r into target and engine, true densities, claims, resume
# --------------------------------------------------------------------------------------


def test_runner_hands_r_to_target_and_engine_and_records_true_densities(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    ref, payload = _run(paths, parent, law, name="full")

    assert payload["algorithm_status"] == "COMPLETE"
    assert payload["beta"] == 1.0
    assert payload["ensemble_kind"] == POSTERIOR_ENSEMBLE_KIND

    # §0.2 item 2: the runner's target is the MIXTURE target, never PhysicalTarget(prior,
    # prior). The runtime fingerprint is checked against both constructions.
    from so_recon.inference.target import PhysicalTarget

    mixture_target = PhysicalTarget(
        parent.prior,
        law.mixture,
        parent.context,
        parent.observations,
        _StubWorker(paths),
        None,  # type: ignore[arg-type]
        _refusing_factory,  # type: ignore[arg-type]
        adapter_hash=ADAPTER_HASH,
        state_times_s=parent.state_times_s,
    )
    prior_fallback = PhysicalTarget(
        parent.prior,
        parent.prior,
        parent.context,
        parent.observations,
        _StubWorker(paths),
        None,  # type: ignore[arg-type]
        _refusing_factory,  # type: ignore[arg-type]
        adapter_hash=ADAPTER_HASH,
        state_times_s=parent.state_times_s,
    )
    assert payload["runtime_target_fingerprint"] == mixture_target.fingerprint
    assert payload["runtime_target_fingerprint"] != prior_fallback.fingerprint
    assert payload["target_hashes"]["proposal_hash"] == law.mixture.fingerprint
    assert payload["scientific_target_identity"] == scientific_target_identity(mixture_target)

    # §4.4: the recorded log_q/log_r are the true densities of the FINAL particles
    state = _state_of(paths, payload)
    diagnostics = json.loads(
        (paths.resolve(ref.path).parent / "particle_density_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["schema_version"] == "e03-particle-density-diagnostics-1"
    assert diagnostics["epsilon"] == 0.10
    by_particle = {row["particle_id"]: row for row in diagnostics["particles"]}
    assert len(by_particle) == len(state["particles"])
    for particle in state["particles"]:
        evaluation = particle["evaluation"]
        theta = ThetaRecord.model_validate(evaluation["theta"])
        row = by_particle[particle["particle_id"]]
        assert row["theta_hash"] == sha256_json(theta.model_dump(mode="json"))
        assert row["log_q"] == pytest.approx(law.q.log_prob(theta), abs=1e-12)
        expected_r = float(
            logsumexp(
                [math.log(0.9) + row["log_q"], math.log(0.1) + row["log_p0"]],
            )
        )
        assert row["log_r"] == pytest.approx(expected_r, abs=1e-12)
        assert row["log_r"] == pytest.approx(evaluation["log_r"], abs=1e-12)

    assert (paths.resolve(ref.path).parent / "posterior_bundle.json").is_file()


def test_beta_below_one_is_a_diagnostic_and_never_a_posterior(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    ref, payload = _run(paths, parent, law, name="partial", config=_config(max_beta_steps=1))

    assert payload["algorithm_status"] == "INCOMPLETE_BUDGET"
    assert payload["beta"] < 1.0
    assert payload["ensemble_kind"] == DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND
    assert payload["posterior_claim"].startswith("REFUSED")
    run_dir = paths.resolve(ref.path).parent
    assert not (run_dir / "posterior_bundle.json").exists()


def test_empty_data_run_restores_the_prior_on_an_independent_reference(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    # L = 1 for every theta (no observations): the beta=1 target IS p0, so a learned-q
    # start must recover the prior — the §9 empty-data correction row.
    _ref, payload = _run(
        paths, parent, law, name="empty-data", config=_config(n_particles=192, seed=5)
    )
    assert payload["ensemble_kind"] == POSTERIOR_ENSEMBLE_KIND
    state = _state_of(paths, payload)
    weights = np.exp(np.asarray(state["log_weights"], dtype=np.float64))
    weights /= weights.sum()
    v0 = np.asarray([particle["evaluation"]["theta"]["v"][0] for particle in state["particles"]])
    learned_mean = float(weights @ v0)
    reference_mean = float(
        np.mean([theta.v[0] for theta in parent.prior.sample(200_000, np.random.default_rng(777))])
    )
    # deterministic run, fixed seeds; the tolerance is ~4 MC standard errors of a
    # weighted ensemble with the run's own final ESS and unit prior scale
    effective = max(float(payload["final_ess"]), 1.0)
    assert abs(learned_mean - reference_mean) <= max(0.10, 4.0 / math.sqrt(effective))
    assert sum(payload["family_probabilities"].values()) == pytest.approx(1.0)


def test_runner_stop_resume_and_reload_reproduce_the_uninterrupted_run(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    config = _config(n_particles=10)

    _ref_full, full = _run(paths, parent, law, name="uninterrupted", config=config)
    # boundary 11 falls inside initialization: the resumed session must pick up the
    # persisted pending proposal draw and still reproduce the uninterrupted run
    ref_cut, cut = _run(paths, parent, law, name="cut", config=config, stop_requested=stop_on(11))
    assert cut["algorithm_status"] == "INCOMPLETE_BUDGET"
    assert cut["ensemble_kind"] == DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND

    resume_ref, resumed = _run(
        paths,
        parent,
        law,
        name="resumed",
        config=config,
        resume_checkpoint_ref=ArtifactRef.model_validate(cut["checkpoint"]),
        resume_payload=cut,
    )
    assert resumed["algorithm_status"] == "COMPLETE"
    assert resumed["beta"] == 1.0
    assert resumed["ensemble_kind"] == POSTERIOR_ENSEMBLE_KIND
    assert resumed["beta_history"] == full["beta_history"]
    assert resumed["ess_history"] == full["ess_history"]
    assert resumed["log_evidence"] == full["log_evidence"]
    assert resumed["family_probabilities"] == full["family_probabilities"]
    state_a = _state_of(paths, full)
    state_b = _state_of(paths, resumed)
    thetas_a = [p["evaluation"]["theta"] for p in state_a["particles"]]
    thetas_b = [p["evaluation"]["theta"] for p in state_b["particles"]]
    assert thetas_a == thetas_b
    assert state_a["log_weights"] == state_b["log_weights"]
    assert (paths.resolve(resume_ref.path).parent / "posterior_bundle.json").is_file()


def test_resume_refuses_another_world_and_another_physics(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    parent = _world(paths)
    law = _bind_law(paths, parent, _export_proposal(paths, parent.payload))
    config = _config(n_particles=6)
    _ref, cut = _run(paths, parent, law, name="cut-x", config=config, stop_requested=stop_on(3))
    assert cut["algorithm_status"] == "INCOMPLETE_BUDGET"

    foreign_payload = dict(cut)
    foreign_payload["parent"] = dict(cut["parent"])
    foreign_payload["parent"]["context_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another parent context"):
        _run(
            paths,
            parent,
            law,
            name="refused-parent",
            config=config,
            resume_checkpoint_ref=ArtifactRef.model_validate(cut["checkpoint"]),
            resume_payload=foreign_payload,
        )

    shifted_payload = dict(cut)
    shifted_payload["scientific_target_identity"] = "0" * 64
    with pytest.raises(ValueError, match="scientific target identity"):
        _run(
            paths,
            parent,
            law,
            name="refused-physics",
            config=config,
            resume_checkpoint_ref=ArtifactRef.model_validate(cut["checkpoint"]),
            resume_payload=shifted_payload,
        )


# --------------------------------------------------------------------------------------
# the run world gate: corpus, evaluation split, foreign files, calendars
# --------------------------------------------------------------------------------------


def _publish_corpus(paths: ProjectPaths, root: Path) -> dict[str, Path]:
    """Three parents (train/development/evaluation) in the real corpus disk layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from so_recon.inference.contracts import NoiseTheta
    from so_recon.ml.contracts import CorpusManifest, ParentRow, ViewRow
    from so_recon.registry.hashing import sha256_bytes

    context_dir = root / "context"
    truth_dir = root / "truth"
    context_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "index.json").write_bytes(b'{"files": {}}\n')
    parents: list[ParentRow] = []
    index_files: dict[str, str] = {}
    for index, split in enumerate(("train", "development", "evaluation")):
        context = _prior_context(100 + index)
        payload = _parent_payload(context, 100 + index)
        payload["split"] = split
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        (context_dir / f"{payload['parent_id']}.json").write_bytes(body)
        index_files[f"{payload['parent_id']}.json"] = sha256_bytes(body)
        digest = sha256_bytes(body)
        ref = ArtifactRef(
            artifact_id=digest,
            path=f"artifacts/corpus/{root.name}/context/{payload['parent_id']}.json",
            sha256=digest,
            size_bytes=len(body),
            media_type="application/json",
            schema_version="e03-parent-context-1",
            producer_run_id="task07-test",
            parent_artifact_ids=[],
            created_at="2026-09-17T00:00:00",
        )
        schema = context.density_schema
        parents.append(
            ParentRow(
                parent_id=payload["parent_id"],
                design_id="e02-t2-v2",
                split=split,
                truth_seed=index,
                history_seed=1000 + index,
                theta=_theta_record(np.random.default_rng(900 + index), context),
                noise=NoiseTheta(sigma=0.05, rho=0.3, nu=5.0, log_bias=0.01),
                schema_id=schema.schema_id,
                basis_hash=schema.basis_hash,
                observation_hash="b" * 64,
                model_hash="f" * 64,
                context_ref=ref,
                truth_ref=ref,
                views=(
                    ViewRow(
                        view_id="m36-copy0",
                        prefix_months=36,
                        noise_copy=0,
                        observation_hash="b" * 64,
                        weight=1.0,
                    ),
                ),
            )
        )
    labels = root / "labels.parquet"
    pq.write_table(
        pa.table(
            {
                "parent_id": [p.parent_id for p in parents],
                "split": [str(p.split) for p in parents],
            }
        ),
        labels,
    )
    labels_ref = register_artifact(
        labels,
        paths,
        schema_version="e03-labels-1",
        producer_run_id="task07-test",
        media_type="application/vnd.apache.parquet",
        now=datetime.now(UTC),
    )
    index_body = json.dumps({"files": index_files}, indent=2, sort_keys=True).encode("utf-8")
    (context_dir / "index.json").write_bytes(index_body)
    context_ref = register_artifact(
        context_dir / "index.json",
        paths,
        schema_version="e03-context-index-1",
        producer_run_id="task07-test",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    truth_ref = register_artifact(
        truth_dir / "index.json",
        paths,
        schema_version="e03-truth-index-1",
        producer_run_id="task07-test",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    manifest = CorpusManifest(
        experiment_id=root.name,
        config_version="e03-learning-1",
        namespace="thin_slice",
        learning_config_hash="a" * 64,
        design_distribution={"e02-t2-v2": 3},
        totals={"train": 1, "development": 1, "evaluation": 1},
        expected=3,
        complete=3,
        failed=0,
        parents=tuple(parents),
        labels_ref=labels_ref,
        context_dir_ref=context_ref,
        truth_dir_ref=truth_ref,
        schemas={p.schema_id: p.basis_hash for p in parents},
        basis_hashes={p.schema_id: p.basis_hash for p in parents},
        source_commit="test",
        source_dirty=False,
    )
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "train": context_dir / f"{parents[0].parent_id}.json",
        "development": context_dir / f"{parents[1].parent_id}.json",
        "evaluation": context_dir / f"{parents[2].parent_id}.json",
    }


def _theta_record(rng: np.random.Generator, context: PriorContext) -> ThetaRecord:
    schema = context.density_schema
    coordinates = rng.standard_normal(schema.n_v + schema.n_residual)
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=int(rng.integers(0, len(schema.families))),
        v=tuple(coordinates[: schema.n_v].tolist()),
        z_perp=tuple(coordinates[schema.n_v :].tolist()),
        basis_hash=schema.basis_hash,
    )


def test_run_worlds_come_from_the_corpus_and_evaluation_is_refused(tmp_path: Path) -> None:
    from so_recon.inference.learned_loop import load_parent_world

    paths = _project(tmp_path)
    published = _publish_corpus(paths, paths.artifacts / "corpus" / "tiny-3")
    ctx = RunContext.start(command="test-load-world", argv=[], cfg=None, paths=paths)

    world = load_parent_world(
        corpus_path=published["manifest"],
        parent_path=published["development"],
        paths=paths,
        ctx=ctx,
    )
    assert world.split == "development"
    assert world.parent_id in published["development"].name
    assert world.state_times_s == (
        EDGES[0],
        EDGES[12],
        EDGES[24],
        EDGES[36],
    )

    with pytest.raises(ValueError, match="evaluation split"):
        load_parent_world(
            corpus_path=published["manifest"],
            parent_path=published["evaluation"],
            paths=paths,
            ctx=ctx,
        )

    foreign = paths.artifacts / "foreign.json"
    foreign.write_text(
        json.dumps({**json.loads(published["development"].read_text(encoding="utf-8"))}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="names no unique parent|not the context artifact"):
        load_parent_world(
            corpus_path=published["manifest"],
            parent_path=foreign,
            paths=paths,
            ctx=ctx,
        )


def test_state_times_require_a_calendar_in_the_design() -> None:
    from so_recon.inference.learned_loop import learned_state_times

    calendar = _prior_context(3, design={"p1_design": {"start_date": "2000-01-01", "n_months": 24}})
    with pytest.raises(ValueError, match="cannot publish states"):
        learned_state_times(calendar)
    with pytest.raises(ValueError, match="no calendar"):
        learned_state_times(_prior_context(3, design={}))
