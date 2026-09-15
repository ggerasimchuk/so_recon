"""Independent fixed-grid reference integration for reduced physical inversions."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from so_recon.inference.contracts import F64, FIXED_NOISE_THETA, ObservationBundle
from so_recon.inference.target import evaluate_loglik, require_complete_forward
from so_recon.observation.predict import predict_observations
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext, RunRecord
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import load_case
from so_recon.simulator.contracts import ForwardResult, OutputRequest
from so_recon.simulator.forward import SolverConfig, simulate
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.reduced_inverse import ReducedDesign, build_reduced_case

REFERENCE_SCHEMA_VERSION = "e02-reduced-reference-3"
SUPPORTED_REFERENCE_SCHEMA_VERSIONS = (
    "e02-reduced-reference-1",
    "e02-reduced-reference-2",
    REFERENCE_SCHEMA_VERSION,
)
REFERENCE_LIMIT = 7.0
QUANTILE_METHOD = "piecewise-linear-density-cdf-1"
LEGACY_QUANTILE_METHOD = "node-mass-cdf-1"


class RunFactory(Protocol):
    def __call__(self, command: str, parent_run_ids: tuple[str, ...]) -> RunContext: ...


SimulateFunction = Callable[..., ForwardResult]


def reference_nodes(n_nodes: int) -> F64:
    """Return one of the pre-registered nested ``2^k + 1`` reference grids."""
    intervals = n_nodes - 1
    if intervals < 16 or intervals & (intervals - 1):
        raise ValueError("reference node count must be 2^k + 1 with at least 17 nodes")
    return np.linspace(-REFERENCE_LIMIT, REFERENCE_LIMIT, n_nodes, dtype=np.float64)


def reference_artifact_from_path(path: Path, paths: ProjectPaths) -> ArtifactRef:
    """Recover and verify the immutable reference named by its producer run."""
    resolved = path if path.is_absolute() else paths.resolve(str(path))
    relative = paths.relative(resolved)
    run_path = resolved.parent / "run.json"
    try:
        run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
        ref = run.outputs["reference"]
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"{relative}: cannot recover its reference ArtifactRef: {exc}") from exc
    if paths.resolve(ref.path) != resolved:
        raise ValueError(f"{relative}: producer run names {ref.path!r} as its reference output")
    _verify_artifact(ref, paths, label="parent reference")
    return ref


def _verify_artifact(ref: ArtifactRef, paths: ProjectPaths, *, label: str) -> Path:
    try:
        path = paths.resolve(ref.path)
    except ValueError as exc:
        raise ValueError(f"{label} names an invalid path {ref.path!r}: {exc}") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing at {ref.path}")
    digest = sha256_file(path)
    size = path.stat().st_size
    if digest != ref.sha256 or ref.artifact_id != ref.sha256 or size != ref.size_bytes:
        raise ValueError(
            f"{label} failed immutable identity check: digest={digest}, size={size}, "
            f"declared digest={ref.sha256}, artifact_id={ref.artifact_id}, "
            f"size={ref.size_bytes}"
        )
    return path


def _same_payload(actual: object, expected: object, *, label: str) -> None:
    if sha256_json(actual) != sha256_json(expected):
        raise ValueError(f"parent reference {label} does not match this refinement")


def _result_record(
    node: Mapping[str, Any],
    *,
    ledger: BudgetLedger,
    paths: ProjectPaths,
) -> tuple[str, str]:
    """Re-prove one reused physical node against its run, case, result and ledger."""
    run_id = str(node.get("run_id", ""))
    job_id = str(node.get("job_id", ""))
    model_hash = str(node.get("model_hash", ""))
    run_path = paths.runs / run_id / "run.json"
    try:
        run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"reused node run {run_id!r} is unreadable: {exc}") from exc
    if run.run_id != run_id or run.command != "e02-reduced-reference-node" or run.status != "PASS":
        raise ValueError(
            f"reused node run {run_id!r} is not a completed reduced-reference-node run"
        )
    try:
        case_ref = run.outputs["case"]
    except KeyError as exc:
        raise ValueError(f"reused node run {run_id!r} has no published case") from exc
    case_path = _verify_artifact(case_ref, paths, label=f"reused node {run_id} case")
    case = load_case(case_path, paths)
    if case.model_hash != model_hash:
        raise ValueError(
            f"reused node {run_id!r} declares model {model_hash}, case has {case.model_hash}"
        )
    matching = [
        entry
        for entry in (*ledger.record.inherited_entries, *ledger.record.entries)
        if entry.job_id == job_id
    ]
    if len(matching) != 1:
        raise ValueError(
            f"reused node {run_id!r}/{job_id!r} has {len(matching)} matching ledger entries"
        )
    entry = matching[0]
    if (
        entry.state != "COMPLETE"
        or entry.status != "COMPLETE"
        or entry.model_hash != model_hash
        or entry.case_sha256 != case_ref.sha256
    ):
        raise ValueError(f"reused node {run_id!r}/{job_id!r} is not ledger-complete for its case")
    result_path = paths.runs / run_id / job_id / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"reused node result {result_path} is unreadable: {exc}") from exc
    if not isinstance(result, dict) or (
        result.get("status") != "COMPLETE"
        or result.get("job_id") != job_id
        or result.get("model_hash") != model_hash
        or result.get("case_sha256") != case_ref.sha256
    ):
        raise ValueError(f"reused node result {result_path} has inconsistent physical identity")
    return case_ref.sha256, sha256_file(result_path)


def _reusable_nodes(
    parent_ref: ArtifactRef,
    nodes: F64,
    design: ReducedDesign,
    observations: ObservationBundle,
    ledger: BudgetLedger,
    paths: ProjectPaths,
) -> dict[float, dict[str, Any]]:
    path = _verify_artifact(parent_ref, paths, label="parent reference")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"parent reference {parent_ref.path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in (
        SUPPORTED_REFERENCE_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"parent reference has unsupported schema {payload.get('schema_version')!r}"
        )
    if payload.get("status") != "UNRESOLVED_REFERENCE":
        raise ValueError("only an UNRESOLVED_REFERENCE may be refined")
    if (
        payload.get("schema_version") == REFERENCE_SCHEMA_VERSION
        and payload.get("quantile_method") != QUANTILE_METHOD
    ):
        raise ValueError("parent reference uses an incompatible continuous quantile method")
    _same_payload(payload.get("design"), design.model_dump(mode="json"), label="design")
    _same_payload(
        payload.get("observations"),
        observations.model_dump(mode="json"),
        label="observations",
    )
    _same_payload(payload.get("noise"), FIXED_NOISE_THETA.model_dump(mode="json"), label="noise")
    fine = payload.get("fine")
    records = payload.get("nodes")
    if not isinstance(fine, dict) or not isinstance(records, list):
        raise ValueError("parent reference has no complete fine grid and node records")
    try:
        parent_count = int(fine["n_nodes"])
        parent_nodes = np.asarray(fine["nodes"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"parent reference grid is invalid: {exc}") from exc
    if nodes.size != 2 * (parent_count - 1) + 1:
        raise ValueError(
            f"refinement must add one nested level: parent has {parent_count}, "
            f"target has {nodes.size}"
        )
    if parent_nodes.shape != (parent_count,) or not np.array_equal(nodes[::2], parent_nodes):
        raise ValueError("parent reference nodes are not the exact nested subset of target nodes")
    if len(records) != parent_count:
        raise ValueError(
            f"parent reference has {len(records)} node records for {parent_count} nodes"
        )
    reused: dict[float, dict[str, Any]] = {}
    for source_index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"parent node {source_index} is not an object")
        z = float(raw.get("z", math.nan))
        if z != float(parent_nodes[source_index]) or z in reused:
            raise ValueError(f"parent node {source_index} has inconsistent or duplicate z={z}")
        in_support = raw.get("log_likelihood_in_support")
        value = raw.get("log_likelihood")
        if (
            in_support is not True
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"parent node {source_index} lacks a finite in-support likelihood")
        case_sha256, result_sha256 = _result_record(raw, ledger=ledger, paths=paths)
        reused[z] = {
            **raw,
            "source_index": source_index,
            "source_reference_artifact_id": parent_ref.artifact_id,
            "evaluation_source": "reused",
            "case_sha256": case_sha256,
            "result_sha256": result_sha256,
        }
    return reused


def _log_values(values: F64, *, label: str) -> F64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {array.shape}")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{label} may contain finite values or -inf")
    return array


def _continuous_quantiles(nodes: F64, log_density: F64, probabilities: F64) -> F64:
    """Invert the CDF of the nonnegative piecewise-linear density on ``nodes``."""
    if (
        probabilities.ndim != 1
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("quantile probabilities must be finite and strictly inside (0,1)")
    anchor = float(np.max(log_density))
    if not math.isfinite(anchor):
        raise ValueError("quadrature has no target support on its nodes")
    density = np.exp(log_density - anchor)
    gaps = np.diff(nodes)
    areas = 0.5 * (density[:-1] + density[1:]) * gaps
    total = float(np.sum(areas))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("piecewise-linear quadrature density has no finite mass")
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(areas)))
    cumulative[-1] = total
    quantiles = np.empty(probabilities.size, dtype=np.float64)
    for output_index, probability in enumerate(probabilities):
        target = float(probability * total)
        interval = int(np.searchsorted(cumulative, target, side="right") - 1)
        interval = min(interval, areas.size - 1)
        remaining = target - float(cumulative[interval])
        gap = float(gaps[interval])
        left = float(density[interval])
        slope = float((density[interval + 1] - density[interval]) / gap)
        discriminant = max(0.0, left * left + 2.0 * slope * remaining)
        denominator = left + math.sqrt(discriminant)
        if denominator <= 0.0:
            raise ValueError("piecewise-linear quantile falls in a zero-density interval")
        offset = 2.0 * remaining / denominator
        quantiles[output_index] = float(nodes[interval]) + min(max(offset, 0.0), gap)
    return quantiles


def quadrature_reference(
    nodes: F64,
    prior_log_density: F64,
    log_likelihood: F64,
    integration_weights: F64,
    *,
    quantile_method: str = LEGACY_QUANTILE_METHOD,
) -> dict[str, F64 | float]:
    """Normalize explicit quadrature masses and return moments, quantiles and evidence."""
    points = np.asarray(nodes, dtype=np.float64)
    prior = _log_values(prior_log_density, label="prior log density")
    likelihood = _log_values(log_likelihood, label="log likelihood")
    widths = np.asarray(integration_weights, dtype=np.float64)
    if points.ndim != 1 or points.size < 2:
        raise ValueError("quadrature needs at least two one-dimensional nodes")
    if (
        prior.shape != points.shape
        or likelihood.shape != points.shape
        or widths.shape != points.shape
    ):
        raise ValueError(
            f"quadrature shape mismatch: nodes {points.shape}, prior {prior.shape}, "
            f"likelihood {likelihood.shape}, weights {widths.shape}"
        )
    if not np.isfinite(points).all() or np.any(np.diff(points) <= 0.0):
        raise ValueError("quadrature nodes must be finite and strictly increasing")
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise ValueError("quadrature integration weights must be finite and positive")
    log_density = prior + likelihood
    logmass = log_density + np.log(widths)
    logz = float(logsumexp(logmass))
    if not math.isfinite(logz):
        raise ValueError("quadrature has no target support on its nodes")
    weights: F64 = np.exp(logmass - logz)
    mean = float(weights @ points)
    variance = float(weights @ ((points - mean) ** 2))
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    probabilities = np.array([0.05, 0.5, 0.95], dtype=np.float64)
    indices = np.searchsorted(cdf, probabilities, side="left")
    legacy_quantiles: F64 = points[indices]
    if quantile_method == LEGACY_QUANTILE_METHOD:
        quantiles = legacy_quantiles
    elif quantile_method == QUANTILE_METHOD:
        expected_widths = trapezoid_weights(points)
        if not np.array_equal(widths, expected_widths):
            raise ValueError("piecewise-linear quantiles require exact trapezoid weights")
        quantiles = _continuous_quantiles(points, log_density, probabilities)
    else:
        raise ValueError(f"unsupported reference quantile method {quantile_method!r}")
    return {
        "weights": weights,
        "mean": mean,
        "variance": variance,
        "quantiles": quantiles,
        "legacy_node_quantiles": legacy_quantiles,
        "logz": logz,
    }


def trapezoid_weights(nodes: F64) -> F64:
    """Integration widths for a nonuniform, strictly increasing trapezoid grid."""
    points = np.asarray(nodes, dtype=np.float64)
    if points.ndim != 1 or points.size < 2:
        raise ValueError("trapezoid integration needs at least two one-dimensional nodes")
    gaps = np.diff(points)
    if not np.isfinite(points).all() or np.any(gaps <= 0.0):
        raise ValueError("trapezoid nodes must be finite and strictly increasing")
    widths = np.empty_like(points)
    widths[0] = 0.5 * gaps[0]
    widths[-1] = 0.5 * gaps[-1]
    widths[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return widths


def _comparison(coarse: dict[str, F64 | float], fine: dict[str, F64 | float]) -> dict[str, float]:
    coarse_quantiles = np.asarray(coarse["quantiles"], dtype=np.float64)
    fine_quantiles = np.asarray(fine["quantiles"], dtype=np.float64)
    coarse_z = math.exp(float(coarse["logz"]))
    fine_z = math.exp(float(fine["logz"]))
    refinement_error = abs(fine_z - coarse_z)
    evidence_lower = max(0.0, fine_z - refinement_error)
    tail_mass = float(2.0 * norm.sf(REFERENCE_LIMIT))
    return {
        "mean_change": abs(float(fine["mean"]) - float(coarse["mean"])),
        "max_quantile_change": float(np.max(np.abs(fine_quantiles - coarse_quantiles))),
        "relative_evidence_change": refinement_error / fine_z,
        "evidence": fine_z,
        "evidence_refinement_error": refinement_error,
        "evidence_lower": evidence_lower,
        "omitted_prior_tail_mass": tail_mass,
        "tail_to_evidence_lower": math.inf if evidence_lower == 0.0 else tail_mass / evidence_lower,
    }


def reference_status(metrics: dict[str, float]) -> str:
    """Apply the pre-registered refinement and omitted-tail criteria."""
    passing = (
        metrics["mean_change"] < 0.02
        and metrics["max_quantile_change"] < 0.03
        and metrics["relative_evidence_change"] < 0.02
        and metrics["tail_to_evidence_lower"] < 1.0e-6
    )
    return "CONVERGED" if passing else "UNRESOLVED_REFERENCE"


def run_reduced_reference(
    design: ReducedDesign,
    observations: ObservationBundle,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
    *,
    parent_run_ids: tuple[str, ...] = (),
    simulate_fn: SimulateFunction = simulate,
    n_nodes: int = 33,
    parent_reference: ArtifactRef | None = None,
) -> ArtifactRef:
    """Run one nested physical reference and compare it with the preceding grid.

    This routine deliberately does not import SMC particles, temperatures or weight helpers.
    A failed physical node is an execution failure, while a numerically insufficient grid is
    a published ``UNRESOLVED_REFERENCE`` scientific outcome. Refinement accepts exactly one
    immutable parent level and re-proves each reused node against its run, case, native result
    and the inherited ledger before trusting the stored likelihood.
    """
    nodes = reference_nodes(n_nodes)
    if parent_reference is None and n_nodes != 33:
        raise ValueError("a reference above 33 nodes requires its immediate parent reference")
    if parent_reference is not None and ledger.record.parent_ledger_sha256 is None:
        raise ValueError("reference refinement requires a resumed ledger with a parent digest")
    reused = (
        {}
        if parent_reference is None
        else _reusable_nodes(
            parent_reference,
            nodes,
            design,
            observations,
            ledger,
            worker.paths,
        )
    )
    log_likelihood = np.empty(nodes.size, dtype=np.float64)
    node_records: list[dict[str, Any]] = []
    request = OutputRequest(
        state_times_s=design.report_edges_s,
        keep_native_restart=False,
        chunk_months=1,
    )
    solver = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)
    for index, z in enumerate(nodes):
        reused_record = reused.get(float(z))
        if reused_record is not None:
            log_likelihood[index] = float(reused_record["log_likelihood"])
            node_records.append({**reused_record, "index": index})
            continue
        ctx = run_factory("e02-reduced-reference-node", parent_run_ids)
        try:
            case = build_reduced_case(float(z), design, worker.paths, ctx)
            result = require_complete_forward(
                simulate_fn(
                    case,
                    request,
                    worker=worker,
                    ctx=ctx,
                    ledger=ledger,
                    solver_config=solver,
                )
            )
            prediction = predict_observations(result, observations, worker.paths)
            scored = evaluate_loglik(prediction, observations, FIXED_NOISE_THETA)
            log_likelihood[index] = scored.value
            node_records.append(
                {
                    "index": index,
                    "z": float(z),
                    "model_hash": result.model_hash,
                    "job_id": result.job_id,
                    "run_id": ctx.run_id,
                    "log_likelihood": scored.value if scored.value_in_support else None,
                    "log_likelihood_in_support": scored.value_in_support,
                    "evaluation_source": "native",
                    "case_sha256": result.case_sha256,
                    "result_path": result.solver_metadata.get("result_path"),
                    "result_sha256": result.solver_metadata.get("result_sha256"),
                }
            )
            ctx.finish("PASS")
        except BaseException as exc:
            ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
            raise

    prior = norm.logpdf(nodes)
    fine = quadrature_reference(
        nodes,
        prior,
        log_likelihood,
        trapezoid_weights(nodes),
        quantile_method=QUANTILE_METHOD,
    )
    coarse_nodes = nodes[::2]
    coarse = quadrature_reference(
        coarse_nodes,
        prior[::2],
        log_likelihood[::2],
        trapezoid_weights(coarse_nodes),
        quantile_method=QUANTILE_METHOD,
    )
    metrics = _comparison(coarse, fine)
    status = reference_status(metrics)
    node_parent_ids = tuple(dict.fromkeys(str(record["run_id"]) for record in node_records))
    reference_parent_ids = () if parent_reference is None else (parent_reference.producer_run_id,)
    parent = run_factory(
        "e02-reduced-reference", (*parent_run_ids, *reference_parent_ids, *node_parent_ids)
    )
    ledger_sha256 = sha256_file(ledger.path)
    session_totals = ledger.session_totals()
    cumulative_totals = ledger.cumulative_totals()
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "diagnostic_protocol": "e02-reduced-reference-diagnostic-v1",
        "quantile_method": QUANTILE_METHOD,
        "status": status,
        "design": design.model_dump(mode="json"),
        "observations": observations.model_dump(mode="json"),
        "noise": FIXED_NOISE_THETA.model_dump(mode="json"),
        "coarse": {
            "n_nodes": int(coarse_nodes.size),
            "mean": coarse["mean"],
            "variance": coarse["variance"],
            "quantiles": np.asarray(coarse["quantiles"]).tolist(),
            "legacy_node_quantiles": np.asarray(coarse["legacy_node_quantiles"]).tolist(),
            "logz": coarse["logz"],
        },
        "fine": {
            "n_nodes": int(nodes.size),
            "nodes": nodes.tolist(),
            "weights": np.asarray(fine["weights"]).tolist(),
            "mean": fine["mean"],
            "variance": fine["variance"],
            "quantiles": np.asarray(fine["quantiles"]).tolist(),
            "legacy_node_quantiles": np.asarray(fine["legacy_node_quantiles"]).tolist(),
            "logz": fine["logz"],
        },
        "refinement": metrics,
        "nodes": node_records,
        "lineage": {
            "parent_reference": (
                None if parent_reference is None else parent_reference.model_dump(mode="json")
            ),
            "parent_ledger_sha256": ledger.record.parent_ledger_sha256,
            "parent_session_id": ledger.record.parent_session_id,
            "ledger_path": worker.paths.relative(ledger.path),
            "ledger_sha256": ledger_sha256,
            "session_id": ledger.record.session_id,
            "reused_node_count": len(reused),
            "new_node_count": int(nodes.size - len(reused)),
        },
        "budget": {
            "session": session_totals.model_dump(mode="json"),
            "cumulative": cumulative_totals.model_dump(mode="json"),
        },
    }
    ref = write_json_artifact(
        parent.run_dir / "reduced_reference.json",
        payload,
        worker.paths,
        schema_version=REFERENCE_SCHEMA_VERSION,
        producer_run_id=parent.run_id,
        parent_artifact_ids=(() if parent_reference is None else (parent_reference.artifact_id,)),
        now=datetime.now(UTC),
    )
    parent.finish("PASS" if status == "CONVERGED" else "FAIL", outputs={"reference": ref})
    return ref


__all__ = [
    "LEGACY_QUANTILE_METHOD",
    "QUANTILE_METHOD",
    "REFERENCE_SCHEMA_VERSION",
    "quadrature_reference",
    "reference_artifact_from_path",
    "reference_nodes",
    "reference_status",
    "run_reduced_reference",
    "trapezoid_weights",
]
