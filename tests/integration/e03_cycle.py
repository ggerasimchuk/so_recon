"""E03 Task 09 Part A — the composition of the first thin full cycle (plan §10.1, §12).

This module is the CODE of the cycle; executing it is a separate, explicitly deferred
human decision (Part B). The julia-marked entry point is
`tests.integration.test_e03_learned_loop.test_first_thin_cycle`; everything here exists
so that entry point stays a short, readable wiring of the published machinery:

* the WORLD is one NEW T1 parent of a freshly generated micro-corpus (`build_learning_corpus`
  with a cycle-only `LearningConfig`): real Julia truth forwards under the E03 1e-5
  balance law, real generative history — never a corpus parent of any earlier namespace
  and never a toy model (§9);
* the PROPOSAL is the honest blocker this composition had to solve: a frozen checkpoint
  binds only to latent layouts it was trained for (`_matched_layout` refuses a foreign
  basis), so a checkpoint that never saw the new world cannot serve it. The cycle
  therefore trains its own bounded smoke checkpoint on the micro-corpus (max_epochs from
  `configs/e03_smoke.yml`) and re-exports it through the Task 06 export command. This is
  an ENGINEERING smoke only — q has seen the run world's (theta, Y) — which is exactly why
  the scientific status below is INCONCLUSIVE and no PROMISING label exists to award. An
  externally supplied manifest (env `E03_CYCLE_PROPOSAL_MANIFEST`) is honoured when its
  layouts cover the run world, and refused by the binder when they do not;
* B1 is the EXISTING prior-start runner (`run_physical_smc`) over an experiment payload
  assembled from the corpus parent's own published artifacts — not the learned runner
  with epsilon=1;
* B0 reuses B1's prior-start initial draws through the Task 08 provenance proof
  (`log_r == log_p0`), so B0 costs no new forwards;
* Q draws raw q thetas and evaluates each through `PhysicalTarget(prior, q, ...)`: real
  physical realisations of the proposal, published as `ensemble_kind=raw_proposal` with
  the posterior claim refused at construction (§4.4);
* M is `run_learned_smc` with the defensive mixture r = 0.9 q + 0.1 p0 (epsilon 0.10).

Every stage runs in its OWN `BudgetLedger` session bounded by the approved
`P1_E03_SMOKE` preset: a stage over cap stops INCOMPLETE_BUDGET and nothing here resumes
it automatically (§12). The evaluator side is Task 08's: operational products without
truth, truth-conditional scores, the six mandated maps, per-method comparison rows and a
per-method cost block (wall/forwards/RSS/disk). The report's scientific status is
INCONCLUSIVE by construction (§1.3): a smoke this small earns no PROMISING/PASS label for
merely finishing, so no such label can be produced.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from so_recon.config.inference import InferenceConfig
from so_recon.config.learning import FamilyPlan, LearningConfig
from so_recon.config.resources import ResourceProfile, require_resource_profile
from so_recon.config.schema import ProjectConfig
from so_recon.inference.contracts import ObservationBundle, PriorContext, ThetaRecord
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.validation.e03_protocol import (
    E03_BALANCE_THRESHOLDS,
    POSTERIOR_ENSEMBLE_KIND,
    PRIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
)
from so_recon.validation.ensemble_states import (
    INFERENCE_GRID_EXTENT_M,
    INFERENCE_GRID_SHAPE,
    EnsembleStateProducts,
    EnsembleStateScores,
    ParticleZones,
    ZoneSupport,
    aggregate_particle_zones,
    ensemble_state_products,
    primary_quadrant_support,
)
from so_recon.validation.learned_comparison import (
    B0_PROVENANCE,
    RunDiagnostics,
    b0_from_prior_start_checkpoint,
    bundle_ensemble,
    comparison_row,
    load_particle_states,
    render_ensemble_state_maps,
    run_diagnostics_from_payloads,
)

CYCLE_REPORT_SCHEMA = "e03-cycle-report-1"
RAW_Q_ENSEMBLE_SCHEMA = "e03-raw-proposal-ensemble-1"

#: Why the raw-q ensemble's `posterior_claim` is False. The note is prose and carries no
#: machine meaning; the claim itself is the boolean beside it.
RAW_Q_POSTERIOR_CLAIM_NOTE = "a raw proposal ensemble is never a posterior (plan §4.4)"

CYCLE_EXPERIMENT_ID = "e03-cycle-smoke-1"
CYCLE_RUN_LABEL = "e03-cycle-smoke-1"
CYCLE_DESIGN_ID = "e02-t1-v1"

#: The micro-corpus of the cycle: FIVE new T1 worlds (4 train + 1 development, no
#: evaluation parent — nothing here is blind). The corpus seed is deliberately outside
#: every earlier namespace: parent seeds are pure functions of (corpus_seed, design,
#: index, namespace), so 6203 can never reproduce a thin-slice (3117) or scientific
#: parent. The RUN WORLD is parent index 0 of the train split: a world that did not exist
#: before this cycle ran.
CYCLE_CORPUS_SEED = 6203
CYCLE_CORPUS_TRAIN = 4
CYCLE_CORPUS_DEVELOPMENT = 1
CYCLE_RUN_PARENT_INDEX = 0

#: The raw-q draw stream of the cycle, apart from every inference seed (11/12) and from
#: the smoke SMC seed (1301).
CYCLE_Q_SEED = 137

#: Env var honouring an externally trained/exported proposal manifest (e.g. the Task 06
#: export). It is used as-is; when its supported layouts do not cover the run world the
#: binder's own refusal is the honest outcome.
PROPOSAL_MANIFEST_ENV = "E03_CYCLE_PROPOSAL_MANIFEST"

SCIENTIFIC_STATUS_INCONCLUSIVE = "INCONCLUSIVE"
SCIENTIFIC_STATUS_REASON = (
    "first thin smoke cycle (plan §1.3): one world, N16, a smoke-trained proposal that "
    "saw the run world in training — technical completion earns no PROMISING_STATE, "
    "PROMISING_ML or PASS label, and none exists in this report"
)

METHOD_IDS: tuple[str, ...] = ("B0", "B1", "Q", "M")


# --------------------------------------------------------------------------------------
# knobs: the cycle's LearningConfig and InferenceConfig, from the smoke yaml
# --------------------------------------------------------------------------------------


def cycle_learning(base: LearningConfig) -> LearningConfig:
    """The cycle's micro-corpus namespace: the frozen learning plan, re-seeded.

    Everything that defines the model and its training (encoder, flow, optimizer,
    defensive epsilon, training seed) is the config's own; only the corpus namespace is
    the cycle's: five new T1 parents from a fresh master seed. `thin=True` at build time
    selects exactly this `thin_slice` plan.
    """
    return base.model_copy(
        update={
            "corpus_seed": CYCLE_CORPUS_SEED,
            "thin_slice": (
                FamilyPlan(
                    design_id=CYCLE_DESIGN_ID,
                    train=CYCLE_CORPUS_TRAIN,
                    development=CYCLE_CORPUS_DEVELOPMENT,
                    evaluation=0,
                ),
            ),
        }
    )


def smoke_inference_config(cfg: ProjectConfig) -> InferenceConfig:
    """The cycle's SMC knobs, straight from the smoke yaml's `inference:` block."""
    inference = cfg.inference
    if inference is None:
        raise ValueError("the cycle requires the smoke config's inference block")
    return InferenceConfig(
        n_particles=inference.n_particles,
        cess_fraction=inference.cess_fraction,
        resample_fraction=inference.resample_fraction,
        max_beta_steps=inference.max_beta_steps,
        moves_per_level=inference.moves_per_level,
        seed=inference.seed,
        rw_scale=inference.rw_scale,
        pcn_scale=inference.pcn_scale,
    )


def cycle_session_profile(cfg: ProjectConfig) -> ResourceProfile:
    """The approved per-stage session budget (§12): P1_E03_SMOKE from the smoke yaml."""
    return require_resource_profile(cfg.resources, command="e03-cycle-smoke")


# --------------------------------------------------------------------------------------
# B1's experiment: the E02 runner's input, assembled from the corpus parent's artifacts
# --------------------------------------------------------------------------------------


def experiment_payload_from_parent(
    *,
    parent_context_payload: Mapping[str, Any],
    parent_truth_payload: Mapping[str, Any],
    experiment_id: str,
    state_times_s: Sequence[float],
    corpus_manifest_ref: ArtifactRef,
) -> dict[str, Any]:
    """Assemble the `e02-physical-experiment-1` input of `run_physical_smc` for one
    corpus parent.

    Every field is a published artifact of the corpus build: the context and
    observations of the parent's canonical inference input, and the parent's own truth
    forward — the one the corpus ran under the E03 1e-5 balance law. Nothing is
    re-simulated and nothing is invented; a parent whose physical checks did not pass
    is refused here rather than handed to B1 as a passing experiment.
    """
    checks = dict(parent_truth_payload.get("physical_checks", {}))
    if checks.get("status") != "PASS":
        raise ValueError(
            f"parent {parent_context_payload.get('parent_id')!r} failed its corpus "
            f"physical checks ({checks.get('status')!r}): it cannot become B1's world"
        )
    inference_input = dict(parent_context_payload["inference_input"])
    return {
        "schema_version": "e02-physical-experiment-1",
        "status": "PASS",
        "experiment_id": experiment_id,
        "design_id": str(parent_context_payload["design_id"]),
        "truth_seed": int(parent_truth_payload["truth_seed"]),
        "history_seed": int(parent_truth_payload["history_seed"]),
        "context": dict(inference_input["context"]),
        "observations": dict(inference_input["observations"]),
        "truth": {
            "generator": dict(parent_truth_payload["generator"]),
            "forward": dict(parent_truth_payload["forward"]),
            "checks": checks,
        },
        "closed_preflight": None,
        "pair": None,
        "state_times_s": [float(value) for value in state_times_s],
        "assembled_by_cycle": {
            "schema": "e03-cycle-experiment-assembly-1",
            "corpus_manifest": corpus_manifest_ref.model_dump(mode="json"),
            "parent_id": str(parent_context_payload["parent_id"]),
            "note": (
                "assembled from the corpus parent's published artifacts so B1 and M see "
                "one and the same world; prepare_physical_experiment is not re-run"
            ),
        },
    }


# --------------------------------------------------------------------------------------
# Q: raw proposal draws, physically realised, never a posterior
# --------------------------------------------------------------------------------------


def raw_q_evaluations(
    *,
    target: Any,
    q: Any,
    n_particles: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Draw n thetas from q and evaluate each through `PhysicalTarget(prior, q, ...)`.

    The target's sampling law IS q, so every recorded `log_r` is the learned density of
    the draw that produced it. Each evaluation carries its own physical forward; a draw
    whose forward fails raises, because a raw ensemble with missing states is not a
    state ensemble.
    """
    rng = np.random.default_rng(seed)
    thetas: tuple[ThetaRecord, ...] = q.sample(n_particles, rng)
    evaluations: list[dict[str, Any]] = []
    for index, theta in enumerate(thetas):
        evaluation = target.evaluate(theta)
        if evaluation.forward_ref is None:
            raise ValueError(
                f"raw q draw {index} published no physical forward: the raw ensemble is "
                "a state ensemble or it is nothing"
            )
        evaluations.append(evaluation.model_dump(mode="json"))
    return tuple(evaluations)


def raw_q_ensemble_payload(
    *,
    evaluations: Sequence[Mapping[str, Any]],
    law: Any,
    n_particles: int,
    seed: int,
    parent_id: str,
) -> dict[str, Any]:
    """The versioned `raw_proposal` ensemble artifact (§4.4): the claim is refused on
    the artifact's face, not in the prose around it."""
    return {
        "schema_version": RAW_Q_ENSEMBLE_SCHEMA,
        "parent_id": parent_id,
        "ensemble_kind": RAW_PROPOSAL_ENSEMBLE_KIND,
        # The claim is a BOOLEAN, machine-checkable field; the prose lives beside it and
        # never in it, so no reader can mistake an explanation for a value (§4.4).
        "posterior_claim": False,
        "posterior_claim_note": RAW_Q_POSTERIOR_CLAIM_NOTE,
        "q_fingerprint": law.q.fingerprint,
        "mixture_fingerprint": law.mixture.fingerprint,
        "epsilon": float(law.epsilon),
        "weights_sha256": law.weights_sha256,
        "n_particles": n_particles,
        "seed": seed,
        "weights": [1.0 / n_particles] * n_particles,
        "evaluations": [dict(evaluation) for evaluation in evaluations],
    }


# --------------------------------------------------------------------------------------
# evaluator wiring: zones, products, scores per method
# --------------------------------------------------------------------------------------


def _refs_and_labels(
    evaluations: Sequence[Mapping[str, Any]],
) -> tuple[list[ArtifactRef], list[int]]:
    refs = [ArtifactRef.model_validate(evaluation["forward_ref"]) for evaluation in evaluations]
    labels = [int(evaluation["theta"]["s"]) for evaluation in evaluations]
    return refs, labels


def zones_from_evaluations(
    *,
    evaluations: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    paths: ProjectPaths,
    support: ZoneSupport,
    time_index: int,
    admissible_s: Sequence[int],
) -> tuple[ParticleZones, EnsembleStateProducts]:
    """Zones and truth-free products of a raw evaluation list (B0, Q)."""
    refs, labels = _refs_and_labels(evaluations)
    so, pv, bo = load_particle_states(refs, paths, time_index)
    zones = aggregate_particle_zones(so=so, pv=pv, bo=bo, support=support)
    products = ensemble_state_products(
        zones, weights, support=support, s_labels=labels, admissible_s=admissible_s
    )
    return zones, products


def select_run_parent(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The cycle's ONE run world, chosen from what the corpus manifest actually publishes.

    `ParentRow` is a strict model with no ordinal field, so the choice cannot be made on
    a position the manifest never wrote down. It is made on the identity it does write:
    the `CYCLE_RUN_PARENT_INDEX`-th TRAIN parent of the cycle design, in parent-id order.
    Parent ids are pure functions of the corpus seed, so this names the same world on
    every rebuild of the same corpus — and never an `evaluation` parent.
    """
    candidates = sorted(
        (
            dict(row)
            for row in parent_rows
            if row["design_id"] == CYCLE_DESIGN_ID and row["split"] == "train"
        ),
        key=lambda row: str(row["parent_id"]),
    )
    if len(candidates) <= CYCLE_RUN_PARENT_INDEX:
        raise ValueError(
            f"the cycle corpus publishes {len(candidates)} train parents of "
            f"{CYCLE_DESIGN_ID}, so it names no run world at index {CYCLE_RUN_PARENT_INDEX}"
        )
    return candidates[CYCLE_RUN_PARENT_INDEX]


@dataclass(frozen=True)
class MethodEvaluation:
    """One method's evaluator output: the row plus everything the report re-states."""

    method_id: str
    ensemble_kind: str
    posterior_claim: bool
    products: EnsembleStateProducts
    scores: EnsembleStateScores
    zones: ParticleZones
    row: dict[str, Any]
    run: RunDiagnostics | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def evaluate_method(
    *,
    method_id: str,
    parent_id: str,
    inference_seed: int,
    paths: ProjectPaths,
    support: ZoneSupport,
    time_index: int,
    admissible_s: Sequence[int],
    truth_so: np.ndarray,
    truth_pv: np.ndarray,
    smc_payload: Mapping[str, Any] | None = None,
    checkpoint_manifest: Mapping[str, Any] | None = None,
    posterior_bundle: Mapping[str, Any] | None = None,
    evaluations: Sequence[Mapping[str, Any]] | None = None,
) -> MethodEvaluation:
    """Wire one method into the Task 08 evaluator.

    SMC methods (B1, M) enter through their published posterior bundle and checkpoint:
    `bundle_ensemble` is the loader that refuses anything but a COMPLETE beta=1 bundle
    and a particle forward the bundle published. Its ONE reading of the physical states
    and its ONE weight vector feed both the operational products and the score, so the
    two can never rest on independently derived weights. Draw-set methods (B0, Q) enter
    through their evaluations with uniform weights. Truth enters exactly once, in the
    score.
    """
    from so_recon.validation.ensemble_states import score_ensemble_states

    run: RunDiagnostics | None = None
    beta: float | None = None
    algorithm_status: str | None = None
    if smc_payload is not None and checkpoint_manifest is not None:
        run = run_diagnostics_from_payloads(smc_payload, checkpoint_manifest)
        ensemble = bundle_ensemble(
            dict(posterior_bundle or {}),
            dict(checkpoint_manifest),
            paths,
            support=support,
            time_index=time_index,
        )
        zones, weights = ensemble.zones, ensemble.weights
        products = ensemble_state_products(
            zones,
            weights,
            support=support,
            s_labels=ensemble.s_labels,
            admissible_s=admissible_s,
        )
        kind = POSTERIOR_ENSEMBLE_KIND
        claim = True
        # `bundle_ensemble` has already refused anything but a COMPLETE beta=1 bundle, so
        # the row publishes the bundle's OWN beta and status beside the label they imply.
        bundle = dict(posterior_bundle or {})
        beta = float(bundle["beta"])
        algorithm_status = str(bundle["algorithm_status"])
    else:
        if evaluations is None:
            raise ValueError(f"method {method_id} carries neither a bundle nor evaluations")
        n = len(evaluations)
        weights = np.full(n, 1.0 / n, dtype=np.float64)
        zones, products = zones_from_evaluations(
            evaluations=evaluations,
            weights=weights,
            paths=paths,
            support=support,
            time_index=time_index,
            admissible_s=admissible_s,
        )
        # A draw set is never a posterior: q's draws are the raw proposal, B0's are the
        # uncorrected static prior, and neither carries a claim (§4.4).
        kind = RAW_PROPOSAL_ENSEMBLE_KIND if method_id == "Q" else PRIOR_ENSEMBLE_KIND
        claim = False
    scores = score_ensemble_states(
        zones, weights, truth_so=truth_so, truth_pv=truth_pv, support=support
    )
    row = comparison_row(
        parent_id=parent_id,
        method_id=method_id,
        inference_seed=inference_seed,
        ensemble_kind=kind,
        posterior_claim=claim,
        beta=beta,
        algorithm_status=algorithm_status,
        products=products,
        scores=scores,
        run=run,
    )
    return MethodEvaluation(
        method_id=method_id,
        ensemble_kind=kind,
        posterior_claim=claim,
        products=products,
        scores=scores,
        zones=zones,
        row=row,
        run=run,
    )


def truth_rows(
    truth_ref: ArtifactRef, paths: ProjectPaths, support: ZoneSupport, time_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth so/pv cell rows and the truth's own zonal So on the primary support.

    The truth is aggregated with the truth's own PV (§5.5) and never enters any method's
    operational products.
    """
    so, pv, bo = load_particle_states([truth_ref], paths, time_index)
    zones = aggregate_particle_zones(so=so, pv=pv, bo=bo, support=support)
    truth_zone_so = np.asarray(zones.so[0], dtype=np.float64)
    if not np.isfinite(truth_zone_so).all():
        raise ValueError("the truth forward leaves an undefined zonal So on the primary support")
    return np.asarray(so[0], dtype=np.float64), np.asarray(pv[0], dtype=np.float64), truth_zone_so


def render_cycle_figures(
    *,
    evaluation: MethodEvaluation,
    support: ZoneSupport,
    output_dir: Path,
    truth_zone_so: np.ndarray,
) -> tuple[Path, ...]:
    """The six mandated maps (truth, error, mean, median, width, representative)."""
    return render_ensemble_state_maps(
        evaluation.products,
        support,
        shape=INFERENCE_GRID_SHAPE,
        extent_m=INFERENCE_GRID_EXTENT_M,
        output_dir=output_dir,
        truth_zone_so=truth_zone_so,
    )


# --------------------------------------------------------------------------------------
# cost accounting (§10.4/§12): per-method wall/forwards/RSS/disk from the ledgers
# --------------------------------------------------------------------------------------


def method_cost(record: Mapping[str, Any], *, note: str | None = None) -> dict[str, Any]:
    """One method's session cost from its `SessionLedger` record (JSON shape)."""
    entries = list(record.get("entries", ()))
    costs = [entry["cost"] for entry in entries if entry.get("cost") is not None]
    peak_rss = [int(cost["peak_rss_bytes"]) for cost in costs if cost.get("peak_rss_bytes")]
    cost: dict[str, Any] = {
        "forwards": len(entries),
        "wall_s": float(sum(cost["wall_s"] for cost in costs)),
        "cpu_s": float(sum(cost["cpu_s"] for cost in costs)),
        "output_bytes": int(sum(cost["output_bytes"] for cost in costs)),
        "peak_rss_bytes": max(peak_rss) if peak_rss else None,
        "session_id": record.get("session_id"),
    }
    if note is not None:
        cost["note"] = note
    return cost


B0_COST_NOTE = (
    "reuses B1's prior-start initial draws (proven log_r == log_p0): their forwards are "
    "paid inside B1's session, and B0 spends none of its own (§10.1)"
)


# --------------------------------------------------------------------------------------
# the verdict: pure checks over published payloads, and the INCONCLUSIVE report
# --------------------------------------------------------------------------------------


def cycle_checks(
    *,
    corpus: Mapping[str, Any],
    truth_checks: Mapping[str, Any],
    b1: Mapping[str, Any],
    m: Mapping[str, Any],
    b0: Mapping[str, Any],
    q: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Mapping[str, Any]],
    epsilon: float,
    same_target_identity: bool,
) -> dict[str, bool]:
    """The cycle's machine checks, pure over published payload shapes.

    These restate what the runners already asserted internally (balance, controls,
    beta=1) as facts about the artifacts they published, plus the composition's own
    laws: distinct method ids, recorded costs, one scientific target for B1 and M.
    """
    balance_ok = dict(m.get("balance_thresholds", {})) == dict(E03_BALANCE_THRESHOLDS)
    method_ids = sorted(str(row["method_id"]) for row in rows)
    checks = {
        "corpus_complete": (
            int(corpus.get("failed", 1)) == 0
            and int(corpus.get("complete", 0)) == int(corpus.get("expected", -1))
        ),
        "truth_checks_pass": truth_checks.get("status") == "PASS",
        "b1_beta_one_complete": b1.get("algorithm_status") == "COMPLETE"
        and float(b1.get("beta", 0.0)) == 1.0,
        "m_beta_one_complete": m.get("algorithm_status") == "COMPLETE"
        and float(m.get("beta", 0.0)) == 1.0,
        "b0_provenance_proven": b0.get("provenance") == B0_PROVENANCE,
        "b0_is_the_prior_start_ensemble": int(b0.get("n_draws", 0)) == int(b0.get("expected", -1)),
        "b0_never_claims_posterior": b0.get("ensemble_kind") == PRIOR_ENSEMBLE_KIND
        and b0.get("posterior_claim") is False,
        "q_never_claims_posterior": q.get("ensemble_kind") == RAW_PROPOSAL_ENSEMBLE_KIND
        and q.get("posterior_claim") is False,
        "method_ids_distinct": method_ids == sorted(METHOD_IDS),
        "every_method_cost_recorded": all(
            method in costs
            and costs[method].get("wall_s") is not None
            and costs[method].get("forwards") is not None
            and costs[method].get("output_bytes") is not None
            for method in METHOD_IDS
        ),
        "balance_law_recorded_by_m": balance_ok,
        "defensive_epsilon_is_declared": float(m.get("proposal", {}).get("epsilon", -1.0))
        == float(epsilon),
        "b1_and_m_share_the_scientific_target": bool(same_target_identity),
    }
    return checks


def assemble_cycle_report(
    *,
    world: Mapping[str, Any],
    knobs: Mapping[str, Any],
    proposal_provenance: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    figures: Mapping[str, Sequence[str]],
    costs: Mapping[str, Mapping[str, Any]],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    """The `e03-cycle-report-1` payload. The scientific status is not an input.

    §1.3 fixes it: a first thin smoke earns INCONCLUSIVE, so the label is written here,
    once, and no caller can pass a better one in.
    """
    return {
        "schema_version": CYCLE_REPORT_SCHEMA,
        "scientific_status": SCIENTIFIC_STATUS_INCONCLUSIVE,
        "scientific_status_reason": SCIENTIFIC_STATUS_REASON,
        "world": dict(world),
        "knobs": dict(knobs),
        "proposal_provenance": dict(proposal_provenance),
        "methods": {name: dict(payload) for name, payload in methods.items()},
        "rows": [dict(row) for row in rows],
        "figures": {name: list(paths_) for name, paths_ in figures.items()},
        "costs": {name: dict(cost) for name, cost in costs.items()},
        "checks": dict(checks),
        "checks_pass": all(bool(value) for value in checks.values()),
    }


# --------------------------------------------------------------------------------------
# the driver: one Julia worker, one bounded session per stage
# --------------------------------------------------------------------------------------


@dataclass
class CycleOutcome:
    """What the cycle hands its caller: the report, its checks and where it all lives."""

    report: dict[str, Any]
    session_dir: Path
    report_ref: ArtifactRef


def _stage_ledger(
    paths: ProjectPaths, session: Path, stage: str, profile: ResourceProfile
) -> BudgetLedger:
    from so_recon.environment.resources import probe_resources

    stage_dir = session / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return BudgetLedger.start(
        profile=profile,
        path=stage_dir / "ledger.json",
        session_id=f"{session.name}-{stage}",
        probe=lambda: probe_resources(None, stage_dir),
    )


def _declared_output(ctx: RunContext, name: str, paths: ProjectPaths) -> Path:
    """The file of a run's NAMED output — the contract, not a guessed sibling filename.

    Both SMC runners register their posterior bundle as `posterior_bundle`; a run that
    did not declare it is refused here rather than silently read from a path that
    happens to exist.
    """
    ref = ctx.record.outputs.get(name)
    if ref is None:
        raise ValueError(
            f"run {ctx.run_id} declared no {name!r} output: "
            f"it published {sorted(ctx.record.outputs)}"
        )
    return paths.resolve(ref.path)


def _register_png(path: Path, ctx: RunContext, paths: ProjectPaths) -> ArtifactRef:
    return register_artifact(
        path,
        paths,
        schema_version="e03-cycle-figure-1",
        producer_run_id=ctx.run_id,
        media_type="image/png",
        now=datetime.now(UTC),
    )


def run_e03_cycle_smoke(
    *,
    paths: ProjectPaths,
    cfg: ProjectConfig,
    julia: Any,
    session_id: str | None = None,
) -> CycleOutcome:
    """Drive the whole cycle. Julia-marked, heavy, and never run on import or in CI.

    Stages: micro-corpus (world) → smoke training + frozen export → B1 → B0 → Q → M →
    evaluator → figures → rows → costs → report. Each stage has its own bounded ledger
    session; nothing is resumed anywhere.
    """
    import json

    from so_recon.geology.density import GaussianConditionalPrior
    from so_recon.inference.learned_loop import (
        bind_defensive_law,
        learned_state_times,
        load_parent_world,
        run_learned_smc,
    )
    from so_recon.inference.target import PhysicalTarget
    from so_recon.ml.commands import run_export_proposal, run_train_proposal
    from so_recon.simulator.worker import PersistentJuliaWorker
    from so_recon.synthetic.inverse_corpus import build_learning_corpus
    from so_recon.validation.e03_protocol import scientific_target_identity
    from so_recon.validation.physical_smc import run_physical_smc

    learning = cfg.learning
    if learning is None:
        raise ValueError("the cycle requires the smoke config's learning block")
    profile = cycle_session_profile(cfg)
    inference = smoke_inference_config(cfg)
    cycle_config = cycle_learning(learning)

    main = RunContext.start(
        command="e03-cycle-smoke",
        argv=[],
        cfg=cfg,
        paths=paths,
        schema_versions={
            "e03_cycle_report": CYCLE_REPORT_SCHEMA,
            "e03_raw_q_ensemble": RAW_Q_ENSEMBLE_SCHEMA,
        },
    )
    session = paths.artifacts / (session_id or f"e03-cycle-smoke-{main.run_id}")
    session.mkdir(parents=True, exist_ok=True)

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command, argv=[], cfg=cfg, paths=paths, parent_run_ids=parent_run_ids
        )

    try:
        with PersistentJuliaWorker(
            julia, paths.julia, session / "worker", profile, paths=paths
        ) as worker:
            # --- stage W: the micro-corpus; the run world is its first train parent ---
            corpus_ctx = run_factory("e03-cycle-corpus", (main.run_id,))
            corpus_ledger = _stage_ledger(paths, session, "corpus", profile)
            corpus_ref = build_learning_corpus(
                ctx=corpus_ctx,
                learning=cycle_config,
                paths=paths,
                worker=worker,
                ledger=corpus_ledger,
                run_factory=run_factory,
                thin=True,
                experiment_id=CYCLE_EXPERIMENT_ID,
            )
            corpus_payload = json.loads(paths.resolve(corpus_ref.path).read_text(encoding="utf-8"))
            parent_row = select_run_parent(corpus_payload["parents"])
            run_parent_id = str(parent_row["parent_id"])
            context_path = paths.resolve(str(parent_row["context_ref"]["path"]))
            truth_path = paths.resolve(str(parent_row["truth_ref"]["path"]))
            parent_context_payload = json.loads(context_path.read_text(encoding="utf-8"))
            parent_truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
            corpus_ctx.finish(
                "PASS",
                notes=[f"corpus={CYCLE_EXPERIMENT_ID} run_world={run_parent_id}"],
            )

            # --- stage T: the smoke proposal (see the module docstring for why the
            # on-disk Task 06 checkpoint cannot bind to a world it never saw) ---
            external_manifest = os.environ.get(PROPOSAL_MANIFEST_ENV)
            if external_manifest:
                external_path = Path(external_manifest)
                proposal_manifest_path = (
                    external_path
                    if external_path.is_absolute()
                    else paths.resolve(external_manifest)
                )
                proposal_provenance = {
                    "source": "external",
                    "manifest": external_manifest,
                    "training_costs": None,
                }
            else:
                train_ctx = run_train_proposal(
                    cfg, paths, corpus_path=paths.resolve(corpus_ref.path), argv=[]
                )
                training_path = _declared_output(train_ctx, "training_manifest", paths)
                training_payload = json.loads(training_path.read_text(encoding="utf-8"))
                export_ctx = run_export_proposal(
                    cfg,
                    paths,
                    training_manifest_path=training_path,
                    out=session / "proposal",
                    argv=[],
                )
                proposal_manifest_path = _declared_output(export_ctx, "proposal_manifest", paths)
                proposal_provenance = {
                    "source": "cycle-smoke-training",
                    "training_run": train_ctx.run_id,
                    "export_run": export_ctx.run_id,
                    "epochs": training_payload.get("epochs_run"),
                    "training_costs": training_payload.get("costs"),
                    "note": (
                        "engineering smoke only: q trained on a micro-corpus that "
                        "CONTAINS the run world, because a frozen checkpoint refuses a "
                        "basis it never saw; no scientific claim may be built on it"
                    ),
                }

            # --- stage B1: the prior-start baseline on the SAME world ---
            b1_ctx = run_factory("e03-cycle-b1", (main.run_id,))
            b1_ledger = _stage_ledger(paths, session, "b1", profile)
            context = PriorContext.model_validate(
                parent_context_payload["inference_input"]["context"]
            )
            observations = ObservationBundle.model_validate(
                parent_context_payload["inference_input"]["observations"]
            )
            # The corpus truth payload publishes no state times (it is not part of its
            # contract), so B1's are derived from the run world's OWN design calendar —
            # the very function the learned runner uses, so B1 and M report at the same
            # months.
            state_times_s = tuple(learned_state_times(context))
            experiment_payload = experiment_payload_from_parent(
                parent_context_payload=parent_context_payload,
                parent_truth_payload=parent_truth_payload,
                experiment_id=CYCLE_EXPERIMENT_ID,
                state_times_s=state_times_s,
                corpus_manifest_ref=corpus_ref,
            )
            experiment_ref = write_json_artifact(
                b1_ctx.run_dir / "physical_experiment.json",
                experiment_payload,
                paths,
                schema_version="e02-physical-experiment-1",
                producer_run_id=b1_ctx.run_id,
                parent_artifact_ids=(
                    ArtifactRef.model_validate(parent_row["context_ref"]).artifact_id,
                    ArtifactRef.model_validate(parent_row["truth_ref"]).artifact_id,
                ),
                now=datetime.now(UTC),
            )
            b1_ctx.add_output("physical_experiment", experiment_ref)
            b1_ref = run_physical_smc(
                ctx=b1_ctx,
                experiment_ref=experiment_ref,
                worker=worker,
                ledger=b1_ledger,
                run_factory=run_factory,
                config=inference,
            )
            b1_payload = json.loads(paths.resolve(b1_ref.path).read_text(encoding="utf-8"))
            b1_bundle = json.loads(
                _declared_output(b1_ctx, "posterior_bundle", paths).read_text(encoding="utf-8")
            )
            b1_checkpoint = json.loads(
                paths.resolve(b1_payload["checkpoint"]["path"]).read_text(encoding="utf-8")
            )
            b1_ctx.finish(
                "PASS",
                notes=[
                    f"status={b1_payload['algorithm_status']} beta={b1_payload['beta']}",
                ],
            )

            # --- stage B0: the proven prior-start draws of B1 itself ---
            b0 = b0_from_prior_start_checkpoint(b1_checkpoint)

            # --- stage M: the learned defensive-mixture run on the same world ---
            m_ctx = run_factory("e03-cycle-m", (main.run_id,))
            m_ledger = _stage_ledger(paths, session, "m", profile)
            parent = load_parent_world(
                corpus_path=paths.resolve(corpus_ref.path),
                parent_path=context_path,
                paths=paths,
                ctx=m_ctx,
            )
            if parent.parent_id != run_parent_id:
                raise ValueError(
                    f"the loaded run world {parent.parent_id} is not the cycle's "
                    f"chosen parent {run_parent_id}"
                )
            law = bind_defensive_law(
                proposal_manifest_path=proposal_manifest_path,
                parent=parent,
                learning=learning,
                paths=paths,
            )
            m_ref = run_learned_smc(
                ctx=m_ctx,
                parent=parent,
                law=law,
                worker=worker,
                ledger=m_ledger,
                run_factory=run_factory,
                config=inference,
                run_label=CYCLE_RUN_LABEL,
            )
            m_payload = json.loads(paths.resolve(m_ref.path).read_text(encoding="utf-8"))
            m_bundle = json.loads(
                _declared_output(m_ctx, "posterior_bundle", paths).read_text(encoding="utf-8")
            )
            m_checkpoint = json.loads(
                paths.resolve(m_payload["checkpoint"]["path"]).read_text(encoding="utf-8")
            )
            m_ctx.finish(
                "PASS",
                notes=[
                    f"status={m_payload['algorithm_status']} beta={m_payload['beta']}",
                    f"epsilon={law.epsilon}",
                ],
            )

            # --- stage Q: raw q draws, physically realised ---
            q_ctx = run_factory("e03-cycle-q", (main.run_id,))
            q_ledger = _stage_ledger(paths, session, "q", profile)
            q_target = PhysicalTarget(
                parent.prior,
                law.q,
                parent.context,
                parent.observations,
                worker,
                q_ledger,
                run_factory,
                parent_run_ids=(q_ctx.run_id, parent.context_ref.producer_run_id),
                state_times_s=parent.state_times_s,
            )
            q_evaluations = raw_q_evaluations(
                target=q_target,
                q=law.q,
                n_particles=inference.n_particles,
                seed=CYCLE_Q_SEED,
            )
            q_payload = raw_q_ensemble_payload(
                evaluations=q_evaluations,
                law=law,
                n_particles=inference.n_particles,
                seed=CYCLE_Q_SEED,
                parent_id=run_parent_id,
            )
            q_ref = write_json_artifact(
                q_ctx.run_dir / "raw_proposal_ensemble.json",
                q_payload,
                paths,
                schema_version=RAW_Q_ENSEMBLE_SCHEMA,
                producer_run_id=q_ctx.run_id,
                now=datetime.now(UTC),
            )
            q_ctx.add_output("raw_proposal_ensemble", q_ref)
            q_ctx.finish("PASS", notes=[f"draws={len(q_evaluations)}"])

            # --- §4.3: B1 and M must be the SAME mathematical posterior target ---
            b1_identity_target = PhysicalTarget(
                GaussianConditionalPrior(context),
                GaussianConditionalPrior(context),
                context,
                observations,
                worker,
                b1_ledger,
                run_factory,
                parent_run_ids=(b1_ctx.run_id,),
                state_times_s=state_times_s,
            )
            same_target = (
                scientific_target_identity(b1_identity_target)
                == m_payload["scientific_target_identity"]
            )

            # --- evaluator: products without truth, scores with it ---
            support = primary_quadrant_support()
            admissible_s = tuple(int(s) for s in context.density_schema.families)
            truth_ref = ArtifactRef.model_validate(parent_truth_payload["forward"])
            truth_so, truth_pv, truth_zone_so = truth_rows(truth_ref, paths, support, -1)

            evaluations = {
                "B0": evaluate_method(
                    method_id="B0",
                    parent_id=run_parent_id,
                    inference_seed=inference.seed,
                    paths=paths,
                    support=support,
                    time_index=-1,
                    admissible_s=admissible_s,
                    truth_so=truth_so,
                    truth_pv=truth_pv,
                    evaluations=b0.evaluations,
                ),
                "B1": evaluate_method(
                    method_id="B1",
                    parent_id=run_parent_id,
                    inference_seed=inference.seed,
                    paths=paths,
                    support=support,
                    time_index=-1,
                    admissible_s=admissible_s,
                    truth_so=truth_so,
                    truth_pv=truth_pv,
                    smc_payload=b1_payload,
                    checkpoint_manifest=b1_checkpoint,
                    posterior_bundle=b1_bundle,
                ),
                "Q": evaluate_method(
                    method_id="Q",
                    parent_id=run_parent_id,
                    inference_seed=inference.seed,
                    paths=paths,
                    support=support,
                    time_index=-1,
                    admissible_s=admissible_s,
                    truth_so=truth_so,
                    truth_pv=truth_pv,
                    evaluations=q_evaluations,
                ),
                "M": evaluate_method(
                    method_id="M",
                    parent_id=run_parent_id,
                    inference_seed=inference.seed,
                    paths=paths,
                    support=support,
                    time_index=-1,
                    admissible_s=admissible_s,
                    truth_so=truth_so,
                    truth_pv=truth_pv,
                    smc_payload=m_payload,
                    checkpoint_manifest=m_checkpoint,
                    posterior_bundle=m_bundle,
                ),
            }
            rows = [evaluations[name].row for name in METHOD_IDS]

            figures: dict[str, list[str]] = {}
            for name in METHOD_IDS:
                figure_dir = session / "figures" / name
                paths_ = render_cycle_figures(
                    evaluation=evaluations[name],
                    support=support,
                    output_dir=figure_dir,
                    truth_zone_so=truth_zone_so,
                )
                for path in paths_:
                    ref = _register_png(path, main, paths)
                    main.add_output(f"figure.{name}.{path.name}", ref)
                figures[name] = [str(paths.relative(path)) for path in paths_]

            costs = {
                "world": method_cost(json.loads(corpus_ledger.path.read_text(encoding="utf-8"))),
                "B0": method_cost({"entries": (), "session_id": None}, note=B0_COST_NOTE),
                "B1": method_cost(json.loads(b1_ledger.path.read_text(encoding="utf-8"))),
                "Q": method_cost(json.loads(q_ledger.path.read_text(encoding="utf-8"))),
                "M": method_cost(json.loads(m_ledger.path.read_text(encoding="utf-8"))),
            }

            checks = cycle_checks(
                corpus=corpus_payload,
                truth_checks=parent_truth_payload["physical_checks"],
                b1=b1_payload,
                m=m_payload,
                b0={
                    "provenance": b0.provenance,
                    "n_draws": b0.n_draws,
                    "expected": inference.n_particles,
                    "ensemble_kind": evaluations["B0"].ensemble_kind,
                    "posterior_claim": evaluations["B0"].posterior_claim,
                },
                q=q_payload,
                rows=rows,
                costs=costs,
                epsilon=law.epsilon,
                same_target_identity=same_target,
            )
            report = assemble_cycle_report(
                world={
                    "design_id": CYCLE_DESIGN_ID,
                    "experiment_id": CYCLE_EXPERIMENT_ID,
                    "corpus_manifest": corpus_ref.model_dump(mode="json"),
                    "parent_id": run_parent_id,
                    "split": parent_row["split"],
                    "truth_seed": parent_truth_payload["truth_seed"],
                    "history_seed": parent_truth_payload["history_seed"],
                    "state_times_s": list(state_times_s),
                    "truth_checks": parent_truth_payload["physical_checks"],
                    "experiment_ref": experiment_ref.model_dump(mode="json"),
                },
                knobs={
                    "inference": inference.model_dump(mode="json"),
                    "cycle_corpus_seed": CYCLE_CORPUS_SEED,
                    "cycle_corpus_plan": {
                        "train": CYCLE_CORPUS_TRAIN,
                        "development": CYCLE_CORPUS_DEVELOPMENT,
                        "evaluation": 0,
                    },
                    "q_seed": CYCLE_Q_SEED,
                    "profile": profile.model_dump(mode="json"),
                    "config_version": cfg.config_version,
                },
                proposal_provenance=proposal_provenance,
                methods={
                    "B0": {
                        "provenance": b0.provenance,
                        "n_draws": b0.n_draws,
                        "ensemble_kind": evaluations["B0"].ensemble_kind,
                        "posterior_claim": evaluations["B0"].posterior_claim,
                    },
                    "B1": {
                        "run_ref": b1_ref.model_dump(mode="json"),
                        "algorithm_status": b1_payload["algorithm_status"],
                        "beta": b1_payload["beta"],
                        "final_state_summary": bool(b1_payload.get("state")),
                    },
                    "Q": {"run_ref": q_ref.model_dump(mode="json")},
                    "M": {
                        "run_ref": m_ref.model_dump(mode="json"),
                        "algorithm_status": m_payload["algorithm_status"],
                        "beta": m_payload["beta"],
                        "epsilon": m_payload["proposal"]["epsilon"],
                        "ensemble_kind": m_payload["ensemble_kind"],
                        "balance_thresholds": m_payload["balance_thresholds"],
                    },
                },
                rows=rows,
                figures=figures,
                costs=costs,
                checks=checks,
            )
            report_ref = write_json_artifact(
                session / "cycle_report.json",
                report,
                paths,
                schema_version=CYCLE_REPORT_SCHEMA,
                producer_run_id=main.run_id,
                parent_artifact_ids=(
                    corpus_ref.artifact_id,
                    b1_ref.artifact_id,
                    q_ref.artifact_id,
                    m_ref.artifact_id,
                ),
                now=datetime.now(UTC),
            )
            main.add_output("cycle_report", report_ref)
            main.finish("PASS" if report["checks_pass"] else "FAIL")
            return CycleOutcome(report=report, session_dir=session, report_ref=report_ref)
    except BaseException as exc:
        if main.record.status == "RUNNING":
            main.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
        raise


__all__ = [
    "B0_COST_NOTE",
    "CYCLE_CORPUS_DEVELOPMENT",
    "CYCLE_CORPUS_SEED",
    "CYCLE_CORPUS_TRAIN",
    "CYCLE_DESIGN_ID",
    "CYCLE_EXPERIMENT_ID",
    "CYCLE_Q_SEED",
    "CYCLE_REPORT_SCHEMA",
    "CYCLE_RUN_LABEL",
    "CYCLE_RUN_PARENT_INDEX",
    "CycleOutcome",
    "METHOD_IDS",
    "MethodEvaluation",
    "PROPOSAL_MANIFEST_ENV",
    "RAW_Q_ENSEMBLE_SCHEMA",
    "RAW_Q_POSTERIOR_CLAIM_NOTE",
    "SCIENTIFIC_STATUS_INCONCLUSIVE",
    "SCIENTIFIC_STATUS_REASON",
    "assemble_cycle_report",
    "cycle_checks",
    "cycle_learning",
    "cycle_session_profile",
    "evaluate_method",
    "experiment_payload_from_parent",
    "method_cost",
    "raw_q_ensemble_payload",
    "raw_q_evaluations",
    "render_cycle_figures",
    "run_e03_cycle_smoke",
    "select_run_parent",
    "smoke_inference_config",
    "truth_rows",
    "zones_from_evaluations",
]
