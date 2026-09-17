"""The E03 leakage-safe learned-inverse corpus (plan E03 §6).

One parent is one independent world, and the three data streams of plan §4.1 never meet
on disk:

* **inference input** — the conditional context, the sparse G it was conditioned on, the
  controls U and the observed history — published per parent under `context/`;
* **training labels** — the theta that GENERATED that history, in `labels.parquet`;
* **evaluation truth** — the forward artifact, physical checks and the full theta, under
  `truth/`, read by the evaluator only.

The corpus theta is a NEW draw from `GaussianConditionalPrior(context)` (plan §6.1 step
3): the world builders condition the prior on G and publish their own lineage, but their
E01-style truth is not an amortized training label. The history is generated with
`noise=rendered.noise` — the noise law OF THAT SAME THETA — passed explicitly, because
`generate_dynamic_history` would otherwise default to the fixed diagnostic law and the
labels would disagree with the data (plan §0.2 item 1).

Failures are recorded, never discarded and never retried with a fresh «convenient» seed:
a failed parent consumes its slot in the manifest exactly as it consumed its compute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so_recon.config.learning import FamilyPlan, LearningConfig, SplitName, corpus_totals
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.geology.renderer import (
    RenderedParameters,
    build_inverse_case,
    geology_coefficients,
    render_theta,
    theta_hash,
)
from so_recon.inference.contracts import (
    N_P1_GEOLOGY_IN_V,
    ModelObservations,
    ObservationBundle,
    PriorContext,
    ThetaRecord,
)
from so_recon.ml.contracts import (
    CORPUS_MANIFEST_SCHEMA,
    LABELS_SCHEMA,
    ContextSpec,
    CorpusManifest,
    FailureRow,
    ParentRow,
    ViewRow,
)
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.contracts import (
    SECONDS_PER_DAY,
    CaseBundle,
    ControlSegment,
    OutputRequest,
)
from so_recon.simulator.forward import (
    BASE_MAX_NONLINEAR_ITERATIONS,
    DEFAULT_MAX_TIMESTEP_DAYS,
    SolverConfig,
    simulate,
)
from so_recon.simulator.results import load_forward_result, write_forward_result
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.simulator.budget import BudgetLedger
from so_recon.synthetic.inverse_worlds import (
    generate_dynamic_history,
    inference_payload,
    make_inverse_world,
)
from so_recon.synthetic.loop_designs import make_t3_world
from so_recon.validation.physical_smc import PHYSICAL_STATE_MONTHS, _history_template, _so_states

#: Stream tags. Distinct from everything the world builders derive from the same parent
#: seed, so a corpus theta can never accidentally repeat a world's internal truth draw.
CORPUS_TRUTH_TAG = 1_990_301
CORPUS_HISTORY_TAG = 1_990_302

CONTEXT_SCHEMA = "e03-parent-context-1"
TRUTH_SCHEMA = "e03-parent-truth-1"
CORPUS_COMMAND = "e03-build-corpus"

#: The E03 normative truth-balance thresholds (plan §0.2 item 10): the SPEC smooth
#: verification number, not the historical E02 runner's 1e-4.
TRUTH_BALANCE_CUMULATIVE_MAX = 1.0e-3
TRUTH_BALANCE_STEP_MEDIAN_MAX = 1.0e-5

#: The controls gate of the corpus truth checks (plan §6.1 «проверить ... controls»).
#: The same number `configs/e01_tolerances.yml` fixes as `rate_control_relative_max`,
#: restated here because the corpus truth payload cites its thresholds explicitly.
TRUTH_RATE_CONTROL_RELATIVE_MAX = 1.0e-4

RunFactory = Callable[[str, tuple[str, ...]], RunContext]

#: Well-time features the first context builder publishes (plan §7.1). The typed
#: allowlist lives with the encoder; the corpus only guarantees these channels exist.
WELL_TIME_FEATURES: tuple[str, ...] = (
    "control_kind_bhp",
    "control_kind_liquid_rate",
    "control_kind_water_rate",
    "control_kind_disabled",
    "control_value_normalized",
    "observed_valid",
    "observed_bin_center",
    "observed_bin_width",
    "reset",
    "month_index_scaled",
    "elapsed_month_scaled",
    "upper_connection_open",
    "lower_connection_open",
    "padding",
)


class CorpusForwardError(RuntimeError):
    """A corpus parent's forward did not complete. Recorded, never converted to -inf."""


@dataclass(frozen=True)
class ParentForwardOutcome:
    """What the evaluator needs of a parent's forward: its artifact and identity."""

    forward_ref: ArtifactRef
    model_hash: str
    checks: dict[str, Any]


def _derived_seed(parent_seed: int, tag: int) -> int:
    state = np.random.SeedSequence([parent_seed, tag]).generate_state(1, dtype=np.uint32)[0]
    return int(state) % (2**31)


def truth_latent_rng(parent_seed: int) -> np.random.Generator:
    """The independent truth-latent stream of the CORPUS draw (plan §6.1 step 3)."""
    return np.random.default_rng(np.random.SeedSequence([parent_seed, CORPUS_TRUTH_TAG]))


def history_seed(parent_seed: int) -> int:
    """The separate history seed: noise is drawn after F, from its own stream."""
    return _derived_seed(parent_seed, CORPUS_HISTORY_TAG)


# --------------------------------------------------------------------------------------
# physics seams (integration runs the real ones; unit tests stub these two)
# --------------------------------------------------------------------------------------


def run_parent_forward(
    rendered: RenderedParameters,
    case: CaseBundle,
    *,
    ctx: RunContext,
    worker: PersistentJuliaWorker | None,
    ledger: BudgetLedger | None,
    paths: ProjectPaths,
    run_factory: RunFactory,
) -> ParentForwardOutcome:
    """One physical forward of the corpus theta, with the E03 balance thresholds."""
    edges = case.report_edges_s
    request = OutputRequest(
        state_times_s=tuple(
            float(edges[min(month, len(edges) - 1)]) for month in PHYSICAL_STATE_MONTHS
        ),
        keep_native_restart=False,
    )
    forward_ctx = run_factory("e03-corpus-forward", (ctx.run_id,))
    result = simulate(
        case,
        request,
        worker=worker,  # type: ignore[arg-type]
        ctx=forward_ctx,
        ledger=ledger,  # type: ignore[arg-type]
        solver_config=SolverConfig(
            max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
            max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
        ),
    )
    if result.status != "COMPLETE":
        forward_ctx.finish("FAIL", notes=[f"corpus forward {result.status}"])
        raise CorpusForwardError(
            f"corpus forward {result.job_id} ended {result.status}: {result.reason}"
        )
    checks = _corpus_truth_checks(result, case, paths)
    if checks["status"] != "PASS":
        forward_ctx.finish("FAIL", notes=["e03 corpus truth checks failed"])
        raise CorpusForwardError(f"corpus forward failed its physical checks: {checks}")
    forward_ref = _register_forward_result(result, forward_ctx, paths)
    forward_ctx.finish("PASS", notes=["e03 corpus parent forward"])
    return ParentForwardOutcome(
        forward_ref=forward_ref,
        model_hash=result.model_hash,
        checks=checks,
    )


def _corpus_truth_checks(result: Any, case: CaseBundle, paths: ProjectPaths) -> dict[str, Any]:
    """The E02 truth checks under the E03 NORMATIVE step threshold of 1e-5.

    The historical E02 runner allowed a 1e-4 median step balance for its own scope; plan
    §0.2 item 10 fixes 1e-5 for E03 unless a documented exception exists for a task
    class. The cumulative threshold is unchanged. Plan §6.1 also names the CONTROLS, so
    the rate targets the case declared are verified against the published monthly table
    here (`simulate`'s own request checks cover only the time axis and the restart).
    """
    states = _so_states(result, paths)
    if result.balances_path is None:
        raise CorpusForwardError("complete corpus truth has no balance artifact")
    balances = pq.read_table(paths.resolve(result.balances_path)).to_pylist()
    if not balances:
        # An empty table must not die inside max() as an unexplained ValueError: it is a
        # forward that published no balance statement, and it is recorded as one.
        raise CorpusForwardError(
            f"corpus forward {result.job_id} published an empty balance table"
        )
    cumulative = max(float(row["cumulative_relative"]) for row in balances)
    step = max(float(row["median_step_relative"]) for row in balances)
    controls = _corpus_controls_check(case.controls, result, paths)
    checks = {
        "complete": result.status == "COMPLETE",
        "finite_bounded_so": bool(
            np.isfinite(states).all() and np.all((states >= 0) & (states <= 1))
        ),
        "balance_cumulative": cumulative <= TRUTH_BALANCE_CUMULATIVE_MAX,
        "balance_step_median": step <= TRUTH_BALANCE_STEP_MEDIAN_MAX,
        "controls_rate": controls["rate_control_relative"] <= TRUTH_RATE_CONTROL_RELATIVE_MAX,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "balance_cumulative_relative_max": cumulative,
        "balance_step_median_relative_max": step,
        "so_min": float(states.min()),
        "so_max": float(states.max()),
        **controls,
    }


#: Which published monthly column carries the volume each rate target prescribes.
_RATE_TARGET_COLUMNS: dict[str, str] = {
    "water_rate": "water_inj_m3_sc",
    "liquid_rate": "liquid_prod_m3_sc",
}


def _corpus_controls_check(
    controls: Sequence[ControlSegment], result: Any, paths: ProjectPaths
) -> dict[str, Any]:
    """Verify the case's controls against the monthly table the forward published.

    A rate-target segment makes a claim the published volumes can falsify: the segment's
    own day rate integrated over its overlap with each month (the same target
    `_rate_control_relative` scores in the E01 acceptance stack, `value * overlap /
    SECONDS_PER_DAY`). A `bhp` segment makes no rate claim — the solver is free to flow
    whatever rate that pressure produces — so it is recorded as pressure-controlled and
    verified only for schedule coverage: its wells and months must all be present. The
    month edges are taken from the published table itself, so the check scores the
    schedule that actually ran against the schedule the case declared.
    """
    monthly = pq.read_table(paths.resolve(result.monthly_path)).to_pylist()
    rows = {(str(row["well_id"]), int(row["month_index"])): row for row in monthly}
    edges = {
        int(row["month_index"]): (float(row["start_s"]), float(row["end_s"]))
        for row in monthly
    }
    n_months = len(edges)
    if not monthly or sorted(edges) != list(range(n_months)):
        raise CorpusForwardError(
            f"corpus forward {result.job_id} published a monthly table whose months are not "
            "0..n-1 without gaps or repeats"
        )
    control_wells = {segment.well_id for segment in controls}
    if {well for well, _month in rows} != control_wells:
        raise CorpusForwardError(
            f"corpus forward {result.job_id} published monthly rows for wells "
            f"{sorted({well for well, _ in rows})} but the case controls {sorted(control_wells)}"
        )
    targets: dict[tuple[str, int], tuple[str, float]] = {}
    bhp_wells: set[str] = set()
    for segment in controls:
        if segment.target == "bhp":
            bhp_wells.add(segment.well_id)
            continue
        column = _RATE_TARGET_COLUMNS.get(segment.target)
        if column is None:
            raise CorpusForwardError(
                f"corpus forward {result.job_id} controls well {segment.well_id!r} on "
                f"{segment.target!r}, which no published monthly column can verify"
            )
        for month in range(n_months):
            start_s, end_s = edges[month]
            overlap = min(segment.end_s, end_s) - max(segment.start_s, start_s)
            if overlap > 0.0:
                key = (segment.well_id, month)
                _kind, total = targets.get(key, (column, 0.0))
                targets[key] = (column, total + segment.value * overlap / SECONDS_PER_DAY)
    worst = 0.0
    for (well, month), (column, target) in sorted(targets.items()):
        row = rows.get((well, month))
        if row is None:
            raise CorpusForwardError(
                f"corpus forward {result.job_id} published no monthly row for well {well!r} "
                f"month {month}, which its own controls schedule"
            )
        actual = float(row[column])
        worst = max(worst, abs(actual - target) / max(abs(target), 1.0e-6))
    return {
        "controls_months": n_months,
        "rate_controlled_wells": sorted({well for well, _ in targets}),
        "bhp_controlled_wells": sorted(bhp_wells),
        "n_rate_targets": len(targets),
        "rate_control_relative": worst,
        "rate_control_relative_max": TRUTH_RATE_CONTROL_RELATIVE_MAX,
    }


def _register_forward_result(
    result: Any, forward_ctx: RunContext, paths: ProjectPaths
) -> ArtifactRef:
    path = forward_ctx.run_dir / "forward_result.json"
    write_forward_result(result, path)
    ref = register_artifact(
        path,
        paths,
        schema_version=result.schema_version,
        producer_run_id=forward_ctx.run_id,
        media_type="application/json",
        now=datetime.now(UTC),
    )
    forward_ctx.add_output("forward_result", ref)
    return ref


def history_prediction(
    truth: ParentForwardOutcome, template: ObservationBundle, paths: ProjectPaths
) -> ModelObservations:
    """The model prediction at the places the history lives."""
    from so_recon.observation.predict import predict_observations

    result = load_forward_result(paths.resolve(truth.forward_ref.path), paths)
    return predict_observations(result, template, paths)


# --------------------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------------------


def _world_for(design_id: str, seed: int, paths: ProjectPaths, ctx: RunContext):
    if design_id == "e03-t3-v1":
        return make_t3_world(design_id, seed, paths, ctx)
    return make_inverse_world(design_id, seed, paths, ctx)


def _prior_context_payload(context: PriorContext) -> dict[str, Any]:
    payload = context.model_dump(mode="python")
    for name in ("mean", "chol", "rotation"):
        payload[name] = np.asarray(payload[name], dtype=np.float64).tolist()
    return payload


def build_learning_corpus(
    *,
    ctx: RunContext,
    learning: LearningConfig,
    paths: ProjectPaths,
    worker: PersistentJuliaWorker | None = None,
    ledger: BudgetLedger | None = None,
    run_factory: RunFactory,
    thin: bool,
    design_filter: Sequence[str] | None = None,
    max_parents: int | None = None,
    experiment_id: str | None = None,
) -> ArtifactRef:
    """Build and publish one corpus namespace (scientific or one-shot thin slice)."""
    plans = learning.plans(thin=thin)
    if design_filter is not None:
        plans = tuple(plan for plan in plans if plan.design_id in set(design_filter))
    if not plans:
        raise ValueError("no family plan selected for this corpus build")
    namespace = "thin_slice" if thin else "scientific"
    experiment = experiment_id or f"e03-corpus-{'thin' if thin else 'main'}-1"
    root = paths.artifacts / "corpus" / experiment
    context_dir = root / "context"
    truth_dir = root / "truth"
    context_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    totals = corpus_totals(plans)
    expected = sum(totals.values())
    parents: list[ParentRow] = []
    failures: list[FailureRow] = []
    labels_rows: list[dict[str, Any]] = []
    schemas: dict[str, str] = {}
    basis_hashes: dict[str, str] = {}
    design_distribution: dict[str, int] = {}

    iterator: list[tuple[FamilyPlan, int]] = [
        (plan, index) for plan in plans for index in range(plan.n_parents)
    ]
    if max_parents is not None:
        iterator = iterator[:max_parents]

    for plan, index in iterator:
        parent_id = f"{plan.design_id}-{index:04d}"
        split = learning.split_of(plan, index)
        seed = learning.parent_seed(plan, index, thin=thin)
        design_distribution[plan.design_id] = design_distribution.get(plan.design_id, 0) + 1
        try:
            parent = _build_one_parent(
                ctx=ctx,
                learning=learning,
                paths=paths,
                worker=worker,
                ledger=ledger,
                run_factory=run_factory,
                design_id=plan.design_id,
                parent_id=parent_id,
                split=split,
                seed=seed,
                context_dir=context_dir,
                truth_dir=truth_dir,
            )
        except CorpusForwardError as error:
            failures.append(
                FailureRow(
                    parent_id=parent_id,
                    design_id=plan.design_id,
                    split=split,
                    truth_seed=seed,
                    stage="forward",
                    reason=str(error)[:500],
                    attempts=2,
                )
            )
            continue
        except _CorpusStageError as error:
            failures.append(
                FailureRow(
                    parent_id=parent_id,
                    design_id=plan.design_id,
                    split=split,
                    truth_seed=seed,
                    stage=error.stage,
                    reason=str(error)[:500],
                    attempts=1,
                )
            )
            continue
        parents.append(parent)
        schemas[parent.schema_id] = parent.basis_hash
        basis_hashes[parent.schema_id] = parent.basis_hash
        theta = parent.theta
        labels_rows.append(
            {
                "parent_id": parent.parent_id,
                "design_id": parent.design_id,
                "split": parent.split,
                "truth_seed": parent.truth_seed,
                "history_seed": parent.history_seed,
                "schema_id": parent.schema_id,
                "basis_hash": parent.basis_hash,
                "s": theta.s,
                "v": list(theta.v),
                "z_perp": list(theta.z_perp),
                "noise_sigma": parent.noise.sigma,
                "noise_rho": parent.noise.rho,
                "noise_log_bias": parent.noise.log_bias,
                "observation_hash": parent.observation_hash,
                "model_hash": parent.model_hash,
            }
        )

    labels_path = root / "labels.parquet"
    labels_ref = _write_labels(labels_path, labels_rows, paths, ctx)
    context_ref = _register_stream_dir(context_dir, paths, ctx, schema=CONTEXT_SCHEMA)
    truth_ref = _register_stream_dir(truth_dir, paths, ctx, schema=TRUTH_SCHEMA)
    # Plan §6.1: before the manifest certifies them, re-read every published stream
    # from disk and prove it against its refs. A mismatch here is a build-integrity
    # violation, not a parent failure: it aborts the build rather than publishing a
    # manifest that vouches for bytes that are not on disk.
    _verify_published_streams(
        root,
        parents,
        labels_ref=labels_ref,
        context_ref=context_ref,
        truth_ref=truth_ref,
        paths=paths,
    )
    source_commit, source_dirty = _source_state(paths)
    manifest = CorpusManifest(
        experiment_id=experiment,
        config_version=learning.config_version,
        namespace=namespace,
        learning_config_hash=sha256_json(learning.model_dump(mode="json")),
        design_distribution=design_distribution,
        totals=totals,
        expected=expected,
        complete=len(parents),
        failed=len(failures),
        parents=tuple(parents),
        failures=tuple(failures),
        labels_ref=labels_ref,
        context_dir_ref=context_ref,
        truth_dir_ref=truth_ref,
        schemas=schemas,
        basis_hashes=basis_hashes,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    ref = write_json_artifact(
        root / "corpus_manifest.json",
        manifest.model_dump(mode="json"),
        paths,
        schema_version=CORPUS_MANIFEST_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(labels_ref.artifact_id, context_ref.artifact_id, truth_ref.artifact_id),
        now=datetime.now(UTC),
    )
    ctx.add_output("corpus_manifest", ref)
    return ref


class _CorpusStageError(RuntimeError):
    """A non-forward parent failure, with the stage it happened in."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage


def _stage(stage: str, call: Callable[[], Any]):
    """Run one parent stage so that whatever it raises becomes a recorded failure.

    Nothing here retries and nothing re-draws with a fresh seed (plan §6.1): the stage
    either completes or the parent is published as a `FailureRow` naming the stage. A
    `CorpusForwardError` passes through untouched because the builder records it with
    the forward's own attempt accounting.
    """
    try:
        return call()
    except (_CorpusStageError, CorpusForwardError):
        raise
    except Exception as error:
        raise _CorpusStageError(stage, f"{type(error).__name__}: {error}") from error


#: The build-time theta→physical round-trip tolerance (plan §6.1): the renderer is
#: deterministic, so the label must reproduce its arrays to float round-off.
ROUND_TRIP_ATOL = 1.0e-12


def _render_round_trip(
    theta: ThetaRecord, context: PriorContext, rendered: RenderedParameters
) -> dict[str, Any]:
    """Prove the recorded label and the forwarded arrays are the same object.

    Two directions, both closed to `ROUND_TRIP_ATOL`: re-rendering the theta reproduces
    every physical array (determinism, plus the nuisance coordinates' noise law), and
    re-whitening the rendered geology coefficients returns the theta's own whitened
    coordinates — the same check `make_t3_world` makes of its truth theta, applied to a
    corpus draw against the arrays the forward actually consumed.
    """
    reproduced = render_theta(theta, context)
    arrays_match = all(
        np.allclose(reproduced.arrays[name], values, rtol=0.0, atol=ROUND_TRIP_ATOL)
        for name, values in rendered.arrays.items()
    ) and set(reproduced.arrays) == set(rendered.arrays)
    coefficients = geology_coefficients(theta, context)
    whitened = context.rotation.T @ np.linalg.solve(context.chol, coefficients - context.mean)
    geology_residual = context.n_geology - N_P1_GEOLOGY_IN_V
    worst = max(
        float(np.max(np.abs(whitened - np.asarray(expected))))
        for whitened, expected in (
            (whitened[:N_P1_GEOLOGY_IN_V], theta.v[:N_P1_GEOLOGY_IN_V]),
            (whitened[N_P1_GEOLOGY_IN_V:], theta.z_perp[:geology_residual]),
        )
    )
    report = {
        "arrays_reproduce": bool(arrays_match),
        "geology_round_trip_max_abs": worst,
        "tolerance": ROUND_TRIP_ATOL,
    }
    if not arrays_match or worst > ROUND_TRIP_ATOL:
        raise _CorpusStageError(
            "render",
            f"theta {theta_hash(theta)} does not round-trip through its renderer: {report}",
        )
    return report


def _build_one_parent(
    *,
    ctx: RunContext,
    learning: LearningConfig,
    paths: ProjectPaths,
    worker: PersistentJuliaWorker | None,
    ledger: BudgetLedger | None,
    run_factory: RunFactory,
    design_id: str,
    parent_id: str,
    split: SplitName,
    seed: int,
    context_dir: Any,
    truth_dir: Any,
) -> ParentRow:
    # 1-2: the conditional context, conditioned on this world's own static G.
    context, _pending, generator_ref = _stage(
        "context", lambda: _world_for(design_id, seed, paths, ctx)
    )
    # 3: a NEW theta from the prior — the corpus label, not the world's internal truth.
    rng = truth_latent_rng(seed)
    theta = _stage(
        "draw",
        lambda: GaussianConditionalPrior(context).sample(1, rng)[0],
    )
    # 4: render + the physical forward under the E03 thresholds. The theta→arrays
    # round-trip is part of the render stage: a label that cannot reproduce the arrays
    # it was forwarded with is a failure, not a corpus entry (plan §6.1).
    rendered = _stage("render", lambda: render_theta(theta, context))
    round_trip = _stage("render", lambda: _render_round_trip(theta, context, rendered))
    case = _stage("render", lambda: build_inverse_case(rendered, context, paths, ctx))
    truth = _stage(
        "forward",
        lambda: run_parent_forward(
            rendered, case, ctx=ctx, worker=worker, ledger=ledger, paths=paths,
            run_factory=run_factory,
        ),
    )
    # 5: prediction and the noisy history — with the noise OF THIS THETA, explicitly.
    template = _stage("history", lambda: _history_template(context, case))
    prediction = _stage("prediction", lambda: history_prediction(truth, template, paths))
    h_seed = history_seed(seed)
    observations = _stage(
        "history",
        lambda: generate_dynamic_history(
            context, prediction, seed=h_seed, noise=rendered.noise
        ),
    )
    # 6: publish the three streams separately. The canonical inference input is the
    # SAME six-key shape E02 published (context/G/U/observations/density_schema/basis),
    # validated by the same recursive allowlist; per-world identity stays outside it.
    def _publish() -> ParentRow:
        canonical = inference_payload(
            {
                "context": _prior_context_payload(context),
                "G": context.design.get("log_k_observations", []),
                "U": [segment.model_dump(mode="json") for segment in case.controls],
                "observations": observations.model_dump(mode="json"),
                "density_schema": context.density_schema.model_dump(mode="json"),
                "basis": {
                    "basis_hash": context.density_schema.basis_hash,
                    "transform_version": context.density_schema.transform_version,
                },
            }
        )
        context_payload = {
            "schema_version": CONTEXT_SCHEMA,
            "parent_id": parent_id,
            "design_id": design_id,
            "split": split,
            "cutoff_s": observations.cutoff_s,
            "well_ids": sorted({row.well_id for row in observations.history}),
            "inference_input": canonical,
        }
        context_ref = write_json_artifact(
            context_dir / f"{parent_id}.json",
            context_payload,
            paths,
            schema_version=CONTEXT_SCHEMA,
            producer_run_id=ctx.run_id,
            parent_artifact_ids=(generator_ref.artifact_id,),
            now=datetime.now(UTC),
        )
        truth_payload = {
            "schema_version": TRUTH_SCHEMA,
            "warning": "synthetic truth; evaluator-only; never an inference payload",
            "parent_id": parent_id,
            "design_id": design_id,
            "split": split,
            "truth_seed": seed,
            "history_seed": h_seed,
            "theta": theta.model_dump(mode="json"),
            "noise": rendered.noise.model_dump(mode="json"),
            "forward": truth.forward_ref.model_dump(mode="json"),
            "model_hash": truth.model_hash,
            "physical_checks": truth.checks,
            "renderer_checks": round_trip,
            "generator": generator_ref.model_dump(mode="json"),
        }
        truth_ref = write_json_artifact(
            truth_dir / f"{parent_id}.json",
            truth_payload,
            paths,
            schema_version=TRUTH_SCHEMA,
            producer_run_id=ctx.run_id,
            parent_artifact_ids=(generator_ref.artifact_id, truth.forward_ref.artifact_id),
            now=datetime.now(UTC),
        )
        view = ViewRow(
            view_id="m36-copy0",
            prefix_months=36,
            noise_copy=0,
            observation_hash=observations.observation_hash,
            weight=1.0,
        )
        return ParentRow(
            parent_id=parent_id,
            design_id=design_id,
            split=split,
            truth_seed=seed,
            history_seed=h_seed,
            theta=theta,
            noise=rendered.noise,
            schema_id=context.density_schema.schema_id,
            basis_hash=context.density_schema.basis_hash,
            observation_hash=observations.observation_hash,
            model_hash=truth.model_hash,
            context_ref=context_ref,
            truth_ref=truth_ref,
            views=(view,),
        )

    return _stage("publish", _publish)


def _register_stream_dir(
    directory: Any, paths: ProjectPaths, ctx: RunContext, *, schema: str
) -> ArtifactRef:
    """Register a stream directory through an index artifact: one sha256 per parent.

    ArtifactRefs are files, so a whole-directory ref would be unhashable. The index
    lists every file in the directory with its digest; loading re-verifies the one file
    it actually opens against this list.
    """
    files = sorted(path for path in directory.iterdir() if path.is_file())
    index = {
        "schema_version": schema,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    return write_json_artifact(
        directory / "index.json",
        index,
        paths,
        schema_version=schema,
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )


def _verify_published_streams(
    root: Any,
    parents: Sequence[ParentRow],
    *,
    labels_ref: ArtifactRef,
    context_ref: ArtifactRef,
    truth_ref: ArtifactRef,
    paths: ProjectPaths,
) -> None:
    """Re-read every published stream and prove it against its refs (plan §6.1).

    The labels file is hashed against `labels_ref` and its parent ids against the rows
    the manifest is about to carry; each stream index is hashed against its ref and
    every file it names against its own digest, with the file set on disk exactly the
    set of complete parents. Nothing is re-derived from memory: the bytes on disk are
    what a later reader gets, so they are what is verified.
    """
    labels_path = paths.resolve(labels_ref.path)
    if sha256_file(labels_path) != labels_ref.sha256:
        raise RuntimeError(
            f"published labels {labels_ref.path} fail their own ref digest: the stream "
            "changed between writing and verification"
        )
    published_ids = set(pq.read_table(labels_path).column("parent_id").to_pylist())
    expected_ids = {parent.parent_id for parent in parents}
    if published_ids != expected_ids:
        raise RuntimeError(
            f"published labels name parents {sorted(published_ids)} but the build completed "
            f"{sorted(expected_ids)}: the labels and the manifest would disagree"
        )
    for ref, stream in ((context_ref, "context"), (truth_ref, "truth")):
        directory = root / stream
        index_path = directory / "index.json"
        if sha256_file(index_path) != ref.sha256:
            raise RuntimeError(
                f"published {stream} stream index fails its own ref digest: the stream "
                "changed between writing and verification"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected_files = {f"{parent.parent_id}.json" for parent in parents}
        if set(index["files"]) != expected_files:
            raise RuntimeError(
                f"the {stream} stream index names files {sorted(index['files'])} but the "
                f"build completed parents {sorted(expected_files)}: a stream file went "
                "missing or appeared from nowhere"
            )
        for name, digest in index["files"].items():
            if sha256_file(directory / name) != digest:
                raise RuntimeError(
                    f"published {stream} file {name} fails its index digest: the stream "
                    "was modified after publication"
                )


def _write_labels(
    path: Any, rows: list[dict[str, Any]], paths: ProjectPaths, ctx: RunContext
) -> ArtifactRef:
    schema = pa.schema(
        [
            ("parent_id", pa.string()),
            ("design_id", pa.string()),
            ("split", pa.string()),
            ("truth_seed", pa.int64()),
            ("history_seed", pa.int64()),
            ("schema_id", pa.string()),
            ("basis_hash", pa.string()),
            ("s", pa.int64()),
            ("v", pa.list_(pa.float64())),
            ("z_perp", pa.list_(pa.float64())),
            ("noise_sigma", pa.float64()),
            ("noise_rho", pa.float64()),
            ("noise_log_bias", pa.float64()),
            ("observation_hash", pa.string()),
            ("model_hash", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    sink = path.parent
    sink.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")
    return register_artifact(
        path,
        paths,
        schema_version=LABELS_SCHEMA,
        producer_run_id=ctx.run_id,
        media_type="application/vnd.apache.parquet",
        now=datetime.now(UTC),
    )


def _source_state(paths: ProjectPaths) -> tuple[str, bool]:
    """The commit and dirty flag of the project root, or an explicit unknown."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=paths.root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout.strip()
        return (commit or "unknown", bool(status))
    except (OSError, subprocess.TimeoutExpired):
        return ("unknown", True)


def load_corpus_manifest(ref: ArtifactRef, paths: ProjectPaths) -> CorpusManifest:
    """Reload and verify a published manifest: the corpus is its manifest."""
    path = paths.resolve(ref.path)
    if not path.is_file():
        raise ValueError(f"corpus manifest {ref.path} not found")
    digest = sha256_file(path)
    if digest != ref.sha256:
        raise ValueError(f"corpus manifest {ref.path} failed identity check")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CorpusManifest.model_validate(payload)


def default_context_spec(cutoff_s: float, max_wells: int, max_months: int) -> ContextSpec:
    """The feature contract of the first context builder (plan §7.1)."""
    return ContextSpec(
        builder_version="e03-context-1",
        well_time_features=WELL_TIME_FEATURES,
        static_features=(
            "well_i_normalized",
            "well_j_normalized",
            "well_is_producer",
            "log_k_support_value",
            "log_k_support_sigma",
        ),
        edge_features=(
            "dx_normalized",
            "dy_normalized",
            "distance_normalized",
            "same_layer",
            "layer_delta",
        ),
        control_kinds=("bhp", "liquid_rate", "water_rate", "disabled"),
        units={
            "control_rate": "m3_sc/day",
            "bhp": "Pa",
            "time": "s",
            "length": "m",
            "watercut": "1",
        },
        cutoff_s=cutoff_s,
        max_wells=max_wells,
        max_months=max_months,
        allowed_context_keys=(
            "context",
            "G",
            "U",
            "observations",
            "well_ids",
            "cutoff_s",
        ),
    )


__all__ = [
    "CONTEXT_SCHEMA",
    "CORPUS_COMMAND",
    "CORPUS_TRUTH_TAG",
    "CORPUS_HISTORY_TAG",
    "CorpusForwardError",
    "LABELS_SCHEMA",
    "ROUND_TRIP_ATOL",
    "TRUTH_BALANCE_CUMULATIVE_MAX",
    "TRUTH_BALANCE_STEP_MEDIAN_MAX",
    "TRUTH_RATE_CONTROL_RELATIVE_MAX",
    "TRUTH_SCHEMA",
    "WELL_TIME_FEATURES",
    "ParentForwardOutcome",
    "build_learning_corpus",
    "default_context_spec",
    "history_prediction",
    "history_seed",
    "load_corpus_manifest",
    "run_parent_forward",
    "truth_latent_rng",
]
