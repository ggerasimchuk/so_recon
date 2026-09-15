"""Execution and comparison contracts for the registered T1/T2/T4 native matrix."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from so_recon.config.inference import InferenceConfig
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.geology.renderer import swap_t2_layers, with_t4_remote_state
from so_recon.inference.checkpoint import save_state
from so_recon.inference.contracts import (
    HistoryRow,
    ObservationBundle,
    PosteriorBundle,
    PriorContext,
    SMCState,
    TargetEvaluation,
    ThetaRecord,
)
from so_recon.inference.smc import infer
from so_recon.inference.target import PhysicalTarget, RunFactory, require_complete_forward
from so_recon.observation.bins import rounding_grid
from so_recon.observation.predict import predict_observations
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import read_array
from so_recon.simulator.contracts import CaseBundle, ForwardResult, OutputRequest
from so_recon.simulator.forward import (
    BASE_MAX_NONLINEAR_ITERATIONS,
    DEFAULT_MAX_TIMESTEP_DAYS,
    SolverConfig,
    simulate,
)
from so_recon.simulator.results import load_forward_result, write_forward_result
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.inverse_worlds import (
    generate_dynamic_history,
    inference_payload,
    make_inverse_world,
)
from so_recon.validation.inverse_metrics import compare_so, weighted_quantile
from so_recon.validation.observability import compare_pair

PHYSICAL_EXPERIMENT_SCHEMA = "e02-physical-experiment-1"
PHYSICAL_SMC_RUN_SCHEMA = "e02-physical-smc-run-1"
PHYSICAL_COMPARISON_SCHEMA = "e02-physical-smc-comparison-1"
PHYSICAL_STATE_MONTHS = (0, 12, 24, 36)


def report_zone_matrix(design_id: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the immutable 16x16x2 report zones used by every matrix comparison."""
    if design_id not in {"e02-t1-v1", "e02-t2-v1", "e02-t4-v1"}:
        raise ValueError(f"unsupported physical E02 design {design_id!r}")
    nx = ny = 16
    nz = 2
    masks: list[np.ndarray] = []
    names: list[str] = []

    def mask_for(*, layers: tuple[int, ...], x: range, y: range) -> np.ndarray:
        mask = np.zeros(nx * ny * nz, dtype=np.float64)
        for layer in layers:
            for j in y:
                for i in x:
                    mask[i + nx * (j + ny * layer)] = 1.0
        return mask

    for layer in range(nz):
        names.append(f"layer-{layer}")
        masks.append(mask_for(layers=(layer,), x=range(nx), y=range(ny)))
    for layer in range(nz):
        for x_label, x in (("west", range(0, 8)), ("east", range(8, 16))):
            for y_label, y in (("south", range(0, 8)), ("north", range(8, 16))):
                names.append(f"{x_label}-{y_label}-layer-{layer}")
                masks.append(mask_for(layers=(layer,), x=x, y=y))
    if design_id == "e02-t4-v1":
        remote_x = range(12, 16)
        remote_y = range(8, 16)
        names.append("remote-east")
        masks.append(mask_for(layers=(0, 1), x=remote_x, y=remote_y))
        for layer in range(nz):
            names.append(f"remote-east-layer-{layer}")
            masks.append(mask_for(layers=(layer,), x=remote_x, y=remote_y))
    return tuple(names), np.stack(masks)


def _two_start_se(values: np.ndarray) -> np.ndarray:
    if values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError("precision screen requires exactly two finite starts")
    return np.std(values, axis=0, ddof=1) / math.sqrt(2.0)


def convergence_screen(summaries: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Apply the frozen paired N32/N64 screen without adapting any threshold."""
    indexed: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in summaries:
        key = (int(cast(int, row["n_particles"])), int(cast(int, row["seed"])))
        if key in indexed:
            raise ValueError(f"duplicate physical SMC summary for N{key[0]} seed {key[1]}")
        indexed[key] = row
    expected = {(particles, seed) for particles in (32, 64) for seed in (11, 12)}
    if set(indexed) != expected:
        raise ValueError(f"physical SMC screen requires {sorted(expected)}, got {sorted(indexed)}")

    ordered = [indexed[(particles, seed)] for particles in (32, 64) for seed in (11, 12)]
    names = tuple(str(value) for value in cast(Sequence[object], ordered[0]["zone_names"]))
    if not names or len(set(names)) != len(names):
        raise ValueError("zone_names must be non-empty and unique")

    medians: dict[tuple[int, int], np.ndarray] = {}
    widths: dict[tuple[int, int], np.ndarray] = {}
    families: dict[tuple[int, int], dict[str, float]] = {}
    all_family_names: set[str] = set()
    complete = True
    for row in ordered:
        key = (int(cast(int, row["n_particles"])), int(cast(int, row["seed"])))
        if tuple(str(value) for value in cast(Sequence[object], row["zone_names"])) != names:
            raise ValueError("all physical summaries must use identical ordered report zones")
        medians[key] = np.asarray(row["zone_medians"], dtype=np.float64)
        widths[key] = np.asarray(row["zone_widths"], dtype=np.float64)
        if medians[key].shape != (len(names),) or widths[key].shape != (len(names),):
            raise ValueError("zone medians and widths must have one value per report zone")
        if not np.isfinite(medians[key]).all() or not np.isfinite(widths[key]).all():
            raise ValueError("zone summaries must be finite")
        if np.any(widths[key] < 0.0):
            raise ValueError("zone widths cannot be negative")
        probability = {
            str(name): float(cast(float | int | str, value))
            for name, value in cast(Mapping[object, object], row["family_probabilities"]).items()
        }
        if not probability or not np.isfinite(list(probability.values())).all():
            raise ValueError("family probabilities must be finite and non-empty")
        if any(value < 0.0 for value in probability.values()) or not math.isclose(
            sum(probability.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("family probabilities must be normalized")
        families[key] = probability
        all_family_names.update(probability)
        beta = cast(float | int | str, row["beta"])
        complete = complete and row.get("algorithm_status") == "COMPLETE" and float(beta) == 1.0

    paired_median_gaps = np.stack(
        [np.abs(medians[(64, seed)] - medians[(32, seed)]) for seed in (11, 12)]
    )
    paired_width_gaps = []
    for seed in (11, 12):
        baseline = widths[(32, seed)]
        refined = widths[(64, seed)]
        difference = np.abs(refined - baseline)
        relative = np.zeros_like(difference)
        np.divide(difference, baseline, out=relative, where=baseline > 0.02)
        paired_width_gaps.append(np.where(baseline > 0.02, relative, difference))
    width_gaps = np.stack(paired_width_gaps)
    width_limits = np.stack([np.where(widths[(32, seed)] > 0.02, 0.25, 0.01) for seed in (11, 12)])
    paired_family_gaps = np.asarray(
        [
            [
                abs(families[(64, seed)].get(name, 0.0) - families[(32, seed)].get(name, 0.0))
                for name in sorted(all_family_names)
            ]
            for seed in (11, 12)
        ],
        dtype=np.float64,
    )
    median_se = _two_start_se(np.stack([medians[(64, seed)] for seed in (11, 12)]))
    family_se = _two_start_se(
        np.asarray(
            [
                [families[(64, seed)].get(name, 0.0) for name in sorted(all_family_names)]
                for seed in (11, 12)
            ],
            dtype=np.float64,
        )
    )
    checks = {
        "complete_beta_one": complete,
        "zone_median_gap": bool(np.all(paired_median_gaps <= 0.03)),
        "family_mass_gap": bool(np.all(paired_family_gaps <= 0.15)),
        "zone_width_gap": bool(np.all(width_gaps <= width_limits)),
        "median_precision": bool(np.all(median_se <= 0.01)),
        "family_precision": bool(np.all(family_se <= 0.05)),
    }
    return {
        "schema_version": "e02-physical-smc-comparison-1",
        "status": "PASS" if all(checks.values()) else "POSTERIOR_NOT_CONVERGED",
        "checks": checks,
        "zone_names": list(names),
        "family_names": sorted(all_family_names),
        "paired_zone_median_gaps": paired_median_gaps.tolist(),
        "paired_zone_width_gaps": width_gaps.tolist(),
        "paired_zone_width_limits": width_limits.tolist(),
        "paired_family_mass_gaps": paired_family_gaps.tolist(),
        "n64_zone_median_mc_se": median_se.tolist(),
        "n64_family_mass_mc_se": family_se.tolist(),
        "thresholds": {
            "zone_median_gap": 0.03,
            "family_mass_gap": 0.15,
            "relative_width_gap": 0.25,
            "absolute_width_gap_if_baseline_at_most_0.02": 0.01,
            "zone_median_mc_se": 0.01,
            "family_mass_mc_se": 0.05,
        },
    }


def _state_times(case: CaseBundle) -> tuple[float, ...]:
    return tuple(float(case.report_edges_s[month]) for month in PHYSICAL_STATE_MONTHS)


def _history_template(context: PriorContext, case: CaseBundle) -> ObservationBundle:
    grid = rounding_grid(0.01)
    producers = sorted(
        well.well_id
        for well in case.wells
        if any(
            segment.well_id == well.well_id and segment.role == "producer"
            for segment in case.controls
        )
    )
    rows = tuple(
        HistoryRow(
            well_id=well_id,
            month_index=month,
            raw_value=0.0,
            bin_index=0,
            quality_group="watercut-0.01",
            observed_valid=True,
            reset=False,
        )
        for well_id in producers
        for month in range(len(case.report_edges_s) - 1)
    )
    return ObservationBundle(
        history=rows,
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(value) for value in grid.edges)},
        cutoff_s=float(case.report_edges_s[-1]),
        information_hash=context.information_hash,
        observation_hash=sha256_json(
            {
                "status": "PENDING_PHYSICAL_TRUTH",
                "model_hash": case.model_hash,
                "rows": [row.model_dump(mode="json") for row in rows],
            }
        ),
    )


def _load_truth_case(
    design_id: str, truth_ref: ArtifactRef, paths: ProjectPaths
) -> tuple[CaseBundle, ThetaRecord | None]:
    path = paths.resolve(truth_ref.path)
    if not path.is_file() or sha256_file(path) != truth_ref.sha256:
        raise ValueError(f"truth generator artifact {truth_ref.path} failed identity check")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if design_id in {"e02-t2-v1", "e02-t4-v1"}:
        return (
            CaseBundle.model_validate(payload["case"]),
            ThetaRecord.model_validate(payload["theta"]),
        )
    case_ref = payload.get("case_ref")
    if not isinstance(case_ref, dict):
        raise ValueError("T1 world manifest carries no case_ref")
    case_path = paths.resolve(str(case_ref["path"]))
    if sha256_file(case_path) != str(case_ref["sha256"]):
        raise ValueError("T1 truth case failed world-manifest identity check")
    return CaseBundle.model_validate_json(case_path.read_text(encoding="utf-8")), None


def _publish_forward(result: ForwardResult, ctx: RunContext, paths: ProjectPaths) -> ArtifactRef:
    path = ctx.run_dir / "forward_result.json"
    write_forward_result(result, path)
    ref = register_artifact(
        path,
        paths,
        schema_version=result.schema_version,
        producer_run_id=ctx.run_id,
        media_type="application/json",
        now=datetime.now(UTC),
    )
    ctx.add_output("forward_result", ref)
    return ref


def _so_states(result: ForwardResult, paths: ProjectPaths) -> np.ndarray:
    if "so" in result.states:
        values = read_array(result.states["so"], paths)
    elif "sw" in result.states and result.physics_class == "OW":
        values = 1.0 - read_array(result.states["sw"], paths)
    else:
        raise ValueError(f"forward {result.job_id} has no OW saturation state")
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (len(result.times_s), 512) or not np.isfinite(out).all():
        raise ValueError(f"forward {result.job_id} has invalid state shape {out.shape}")
    if np.any((out < -1.0e-8) | (out > 1.0 + 1.0e-8)):
        raise ValueError(f"forward {result.job_id} has saturation outside [0,1]")
    return out


def _truth_checks(result: ForwardResult, paths: ProjectPaths) -> dict[str, Any]:
    states = _so_states(result, paths)
    if result.balances_path is None:
        raise ValueError("complete physical truth has no balance artifact")
    balances = pq.read_table(paths.resolve(result.balances_path)).to_pylist()
    if {str(row["balance"]) for row in balances} != {
        "full_system_surface",
        "reservoir_connections",
    }:
        raise ValueError("physical truth must publish both independent balance systems")
    cumulative = max(float(row["cumulative_relative"]) for row in balances)
    step = max(float(row["median_step_relative"]) for row in balances)
    checks = {
        "complete": result.status == "COMPLETE",
        "finite_bounded_so": bool(
            np.isfinite(states).all() and np.all((states >= 0) & (states <= 1))
        ),
        "balance_cumulative": cumulative <= 1.0e-3,
        "balance_step_median": step <= 1.0e-4,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "balance_cumulative_relative_max": cumulative,
        "balance_step_median_relative_max": step,
        "so_min": float(states.min()),
        "so_max": float(states.max()),
    }


def _history_seed(parent_seed: int) -> int:
    child = np.random.SeedSequence(parent_seed).spawn(4)[2]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def _pair_evidence(
    *,
    design_id: str,
    theta: ThetaRecord | None,
    context: PriorContext,
    observations: ObservationBundle,
    state_times_s: tuple[float, ...],
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
    parent_run_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    if design_id == "e02-t1-v1":
        return None
    if theta is None:
        raise ValueError(f"{design_id} pair diagnostic requires the registered truth theta")
    prior = GaussianConditionalPrior(context)
    target = PhysicalTarget(
        prior,
        prior,
        context,
        observations,
        worker,
        ledger,
        run_factory,
        parent_run_ids=parent_run_ids,
        state_times_s=state_times_s,
    )
    if design_id == "e02-t2-v1":
        first, second = theta, swap_t2_layers(theta, context)
        names, matrix = report_zone_matrix(design_id)
        support = matrix[:2]
        support_names = names[:2]
    else:
        first = with_t4_remote_state(theta, context, -1.0)
        second = with_t4_remote_state(theta, context, 1.0)
        names, matrix = report_zone_matrix(design_id)
        support = matrix[-3:]
        support_names = names[-3:]
    evaluations = (target.evaluate(first), target.evaluate(second))
    if any(evaluation.forward_ref is None for evaluation in evaluations):
        raise ValueError("a physical ambiguity pair did not publish both forwards")
    refs = tuple(cast(ArtifactRef, evaluation.forward_ref) for evaluation in evaluations)
    results = tuple(
        load_forward_result(worker.paths.resolve(ref.path), worker.paths) for ref in refs
    )
    predictions = tuple(
        predict_observations(result, observations, worker.paths) for result in results
    )
    metrics = compare_pair(
        predictions[0],
        predictions[1],
        _so_states(results[0], worker.paths)[-1],
        _so_states(results[1], worker.paths)[-1],
        support,
    )
    return {
        **metrics,
        "support_names": list(support_names),
        "theta_a": first.model_dump(mode="json"),
        "theta_b": second.model_dump(mode="json"),
        "forward_refs": [ref.model_dump(mode="json") for ref in refs],
    }


def prepare_physical_experiment(
    *,
    ctx: RunContext,
    experiment_id: str,
    design_id: str,
    truth_seed: int,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
) -> ArtifactRef:
    """Run a registered truth and ambiguity control before any posterior computation."""
    paths = worker.paths
    context, _pending, generator_ref = make_inverse_world(design_id, truth_seed, paths, ctx)
    case, truth_theta = _load_truth_case(design_id, generator_ref, paths)
    times = _state_times(case)
    truth_result = require_complete_forward(
        simulate(
            case,
            OutputRequest(state_times_s=times, keep_native_restart=False, chunk_months=1),
            worker=worker,
            ctx=ctx,
            ledger=ledger,
            solver_config=SolverConfig(
                max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
                max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
            ),
        )
    )
    truth_forward_ref = _publish_forward(truth_result, ctx, paths)
    template = _history_template(context, case)
    prediction = predict_observations(truth_result, template, paths)
    history_seed = _history_seed(truth_seed)
    observations = generate_dynamic_history(context, prediction, seed=history_seed)
    allowed = inference_payload(
        {
            "context": context.model_dump(mode="json"),
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
    inference_ref = write_json_artifact(
        ctx.run_dir / "inference_input.json",
        {"schema_version": "e02-inference-input-1", **allowed},
        paths,
        schema_version="e02-inference-input-1",
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(generator_ref.artifact_id, truth_forward_ref.artifact_id),
        now=datetime.now(UTC),
    )
    ctx.add_output("inference_input", inference_ref)
    pair = _pair_evidence(
        design_id=design_id,
        theta=truth_theta,
        context=context,
        observations=observations,
        state_times_s=times,
        worker=worker,
        ledger=ledger,
        run_factory=run_factory,
        parent_run_ids=(ctx.run_id,),
    )
    truth_checks = _truth_checks(truth_result, paths)
    status = truth_checks["status"] == "PASS" and (
        pair is None or pair["status"] == "AMBIGUITY_DEMONSTRATED"
    )
    ref = write_json_artifact(
        ctx.run_dir / "physical_experiment.json",
        {
            "schema_version": PHYSICAL_EXPERIMENT_SCHEMA,
            "status": "PASS" if status else "FAIL",
            "experiment_id": experiment_id,
            "design_id": design_id,
            "truth_seed": truth_seed,
            "stream_names": ["truth_latent", "static_G", "history_noise", "schedule"],
            "history_seed": history_seed,
            "context": context.model_dump(mode="json"),
            "observations": observations.model_dump(mode="json"),
            "inference_input": inference_ref.model_dump(mode="json"),
            "truth": {
                "generator": generator_ref.model_dump(mode="json"),
                "forward": truth_forward_ref.model_dump(mode="json"),
                "checks": truth_checks,
            },
            "pair": pair,
            "state_times_s": list(times),
            "report_zones": list(report_zone_matrix(design_id)[0]),
            "budget": {
                "session": ledger.session_totals().model_dump(mode="json"),
                "cumulative": ledger.cumulative_totals().model_dump(mode="json"),
            },
        },
        paths,
        schema_version=PHYSICAL_EXPERIMENT_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(generator_ref.artifact_id, truth_forward_ref.artifact_id),
        now=datetime.now(UTC),
    )
    ctx.add_output("physical_experiment", ref)
    return ref


class _InitialCaptureTarget:
    def __init__(self, target: PhysicalTarget, count: int) -> None:
        self._target = target
        self._count = count
        self.initial: list[TargetEvaluation] = []
        self.fingerprint = target.fingerprint
        self.checkpoint_hashes = target.checkpoint_hashes

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
        evaluation = self._target.evaluate(theta)
        if len(self.initial) < self._count:
            self.initial.append(evaluation)
        return evaluation


def _evaluation_states(
    evaluations: Sequence[TargetEvaluation], paths: ProjectPaths
) -> tuple[np.ndarray, list[ForwardResult]]:
    results: list[ForwardResult] = []
    states: list[np.ndarray] = []
    for evaluation in evaluations:
        if evaluation.forward_ref is None:
            raise ValueError("physical SMC evaluation has no forward artifact")
        result = load_forward_result(paths.resolve(evaluation.forward_ref.path), paths)
        results.append(result)
        states.append(_so_states(result, paths))
    return np.stack(states), results


def _family_probabilities(state: SMCState, weights: np.ndarray) -> dict[str, float]:
    names = sorted({particle.evaluation.theta.s for particle in state.particles})
    return {
        str(name): float(
            weights[
                np.asarray(
                    [particle.evaluation.theta.s == name for particle in state.particles],
                    dtype=np.bool_,
                )
            ].sum()
        )
        for name in names
    }


def _state_summary(
    *,
    state: SMCState,
    initial: Sequence[TargetEvaluation],
    truth: ForwardResult,
    truth_case: CaseBundle,
    design_id: str,
    paths: ProjectPaths,
) -> dict[str, Any]:
    final_evaluations = [particle.evaluation for particle in state.particles]
    final_states, final_results = _evaluation_states(final_evaluations, paths)
    prior_states, _prior_results = _evaluation_states(initial, paths)
    truth_states = _so_states(truth, paths)
    weights = np.exp(np.asarray(state.log_weights, dtype=np.float64))
    weights /= weights.sum()
    prior_weights = np.full(len(initial), 1.0 / len(initial), dtype=np.float64)
    names, zone_mask = report_zone_matrix(design_id)
    volume = np.asarray(read_array(truth_case.grid.cell_volume_m3, paths), dtype=np.float64)
    porosity = np.asarray(read_array(truth_case.rock.porosity, paths), dtype=np.float64)
    truth_pv = volume * porosity
    zone_pv = zone_mask * truth_pv[None, :]
    zone_total = zone_pv.sum(axis=1)
    if np.any(zone_total <= 0.0):
        raise ValueError("every registered physical report zone must have positive truth PV")

    def zone_values(samples: np.ndarray) -> np.ndarray:
        values = np.einsum("zc,ptc->ptz", zone_pv, samples)
        return np.asarray(values / zone_total[None, None, :], dtype=np.float64)

    final_zones = zone_values(final_states)
    prior_zones = zone_values(prior_states)
    probabilities = np.asarray([0.05, 0.5, 0.95], dtype=np.float64)
    final_quantiles = np.empty((final_states.shape[1], len(names), 3), dtype=np.float64)
    prior_quantiles = np.empty_like(final_quantiles)
    state_quantiles = np.empty((final_states.shape[1], 512, 3), dtype=np.float64)
    for time_index in range(final_states.shape[1]):
        for zone in range(len(names)):
            final_quantiles[time_index, zone] = weighted_quantile(
                final_zones[:, time_index, zone], weights, probabilities
            )
            prior_quantiles[time_index, zone] = weighted_quantile(
                prior_zones[:, time_index, zone], prior_weights, probabilities
            )
        for cell in range(512):
            state_quantiles[time_index, cell] = weighted_quantile(
                final_states[:, time_index, cell], weights, probabilities
            )
    last_median = state_quantiles[-1, :, 1]
    state_errors = compare_so(last_median, truth_states[-1], truth_pv)
    last_zone_median = final_quantiles[-1, :, 1]
    distances = np.sqrt(np.mean((final_zones[:, -1, :] - last_zone_median[None, :]) ** 2, axis=1))
    representative = int(np.argmin(distances))
    return {
        "state_times_s": list(truth.times_s),
        "zone_names": list(names),
        "zone_medians": last_zone_median.tolist(),
        "zone_widths": (final_quantiles[-1, :, 2] - final_quantiles[-1, :, 0]).tolist(),
        "prior_zone_widths": (prior_quantiles[-1, :, 2] - prior_quantiles[-1, :, 0]).tolist(),
        "zone_quantiles_by_time": final_quantiles.tolist(),
        "prior_zone_quantiles_by_time": prior_quantiles.tolist(),
        "cell_quantiles_by_time": state_quantiles.tolist(),
        "truth_so_by_time": truth_states.tolist(),
        "representative_particle_index": representative,
        "representative_so_by_time": final_states[representative].tolist(),
        "state_error_month36": state_errors,
        "family_probabilities": _family_probabilities(state, weights),
        "physical_model_count": len({result.model_hash for result in final_results}),
    }


def run_physical_smc(
    *,
    ctx: RunContext,
    experiment_ref: ArtifactRef,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
    config: InferenceConfig,
) -> ArtifactRef:
    """Run one separately budgeted physical SMC start and publish state-aware evidence."""
    paths = worker.paths
    experiment_path = paths.resolve(experiment_ref.path)
    if sha256_file(experiment_path) != experiment_ref.sha256:
        raise ValueError("physical experiment artifact failed identity check")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != PHYSICAL_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported physical experiment schema")
    if experiment.get("status") != "PASS":
        raise ValueError("physical SMC requires a passing truth/control experiment")
    context = PriorContext.model_validate(experiment["context"])
    observations = ObservationBundle.model_validate(experiment["observations"])
    truth_ref = ArtifactRef.model_validate(experiment["truth"]["forward"])
    truth = load_forward_result(paths.resolve(truth_ref.path), paths)
    generator_ref = ArtifactRef.model_validate(experiment["truth"]["generator"])
    truth_case, _ = _load_truth_case(str(experiment["design_id"]), generator_ref, paths)
    prior = GaussianConditionalPrior(context)
    physical = PhysicalTarget(
        prior,
        prior,
        context,
        observations,
        worker,
        ledger,
        run_factory,
        parent_run_ids=(ctx.run_id, experiment_ref.producer_run_id),
        state_times_s=tuple(float(value) for value in experiment["state_times_s"]),
    )
    target = _InitialCaptureTarget(physical, config.n_particles)
    state = infer(
        target,
        prior,
        prior.context.density_schema,
        config,
        ctx.run_dir / "working",
        lambda: False,
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
    if state.algorithm_status != "COMPLETE" or state.beta != 1.0:
        state_summary: dict[str, Any] | None = None
    else:
        if len(target.initial) != config.n_particles:
            raise ValueError("physical prior ensemble was not captured completely")
        state_summary = _state_summary(
            state=state,
            initial=target.initial,
            truth=truth,
            truth_case=truth_case,
            design_id=str(experiment["design_id"]),
            paths=paths,
        )
    weights = np.exp(np.asarray(state.log_weights, dtype=np.float64))
    families = _family_probabilities(state, weights / weights.sum()) if len(weights) else {}
    payload = {
        "schema_version": PHYSICAL_SMC_RUN_SCHEMA,
        "experiment": experiment_ref.model_dump(mode="json"),
        "experiment_id": experiment["experiment_id"],
        "design_id": experiment["design_id"],
        "algorithm_status": state.algorithm_status,
        "beta": state.beta,
        "n_particles": config.n_particles,
        "seed": config.seed,
        "log_evidence": state.log_evidence,
        "unique_ancestors": len({particle.ancestor_id for particle in state.particles}),
        "family_probabilities": families,
        "state": state_summary,
        "beta_history": list(state.diagnostics.get("beta_history", [])),
        "ess_history": list(state.diagnostics.get("ess_history", [])),
        "resampling": list(state.diagnostics.get("resampling", [])),
        "config": config.model_dump(mode="json"),
        "target_hashes": physical.checkpoint_hashes,
        "checkpoint": checkpoint_ref.model_dump(mode="json"),
        "budget": {
            "session": ledger.session_totals().model_dump(mode="json"),
            "cumulative": ledger.cumulative_totals().model_dump(mode="json"),
        },
    }
    if state_summary is not None:
        payload.update(
            {
                "zone_names": state_summary["zone_names"],
                "zone_medians": state_summary["zone_medians"],
                "zone_widths": state_summary["zone_widths"],
            }
        )
    ref = write_json_artifact(
        ctx.run_dir / "physical_smc.json",
        payload,
        paths,
        schema_version=PHYSICAL_SMC_RUN_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(experiment_ref.artifact_id, checkpoint_ref.artifact_id),
        now=datetime.now(UTC),
    )
    ctx.add_output("physical_smc", ref)
    manifest = json.loads(paths.resolve(checkpoint_ref.path).read_text(encoding="utf-8"))
    particles_ref = ArtifactRef.model_validate(manifest["shards"]["particles"])
    physical_refs = tuple(
        sorted(
            {
                evaluation.forward_ref.artifact_id: evaluation.forward_ref
                for evaluation in (particle.evaluation for particle in state.particles)
                if evaluation.forward_ref is not None
            }.values(),
            key=lambda item: item.artifact_id,
        )
    )
    posterior = PosteriorBundle(
        state_ref=checkpoint_ref,
        particles_ref=particles_ref,
        diagnostics_ref=ref,
        ledger_ref=ledger_ref,
        algorithm_status=state.algorithm_status,
        beta=state.beta,
        convergence_status="NOT_ASSESSED",
        physical_state_refs=physical_refs,
        parent_run_ids=(experiment_ref.producer_run_id,),
    )
    posterior_ref = write_json_artifact(
        ctx.run_dir / "posterior_bundle.json",
        posterior.model_dump(mode="json"),
        paths,
        schema_version="e02-posterior-bundle-1",
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(
            checkpoint_ref.artifact_id,
            particles_ref.artifact_id,
            ref.artifact_id,
            ledger_ref.artifact_id,
        ),
        now=datetime.now(UTC),
    )
    ctx.add_output("posterior_bundle", posterior_ref)
    return ref


def publish_physical_comparison(
    *,
    ctx: RunContext,
    run_refs: Sequence[ArtifactRef],
    experiment_ref: ArtifactRef,
    paths: ProjectPaths,
) -> ArtifactRef:
    summaries: list[dict[str, Any]] = []
    for ref in run_refs:
        path = paths.resolve(ref.path)
        if sha256_file(path) != ref.sha256:
            raise ValueError(f"physical SMC artifact {ref.path} failed identity check")
        payload = json.loads(path.read_text(encoding="utf-8"))
        parent = ArtifactRef.model_validate(payload["experiment"])
        if parent.artifact_id != experiment_ref.artifact_id:
            raise ValueError("physical SMC comparison cannot mix parent experiments")
        summaries.append(payload)
    verdict = convergence_screen(summaries)
    payload = {
        **verdict,
        "experiment": experiment_ref.model_dump(mode="json"),
        "runs": [ref.model_dump(mode="json") for ref in run_refs],
    }
    ref = write_json_artifact(
        ctx.run_dir / "physical_smc_comparison.json",
        payload,
        paths,
        schema_version=PHYSICAL_COMPARISON_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(
            experiment_ref.artifact_id,
            *(item.artifact_id for item in run_refs),
        ),
        now=datetime.now(UTC),
    )
    ctx.add_output("physical_smc_comparison", ref)
    return ref


__all__ = [
    "PHYSICAL_COMPARISON_SCHEMA",
    "PHYSICAL_EXPERIMENT_SCHEMA",
    "PHYSICAL_SMC_RUN_SCHEMA",
    "convergence_screen",
    "prepare_physical_experiment",
    "publish_physical_comparison",
    "report_zone_matrix",
    "run_physical_smc",
]
