"""Independent fixed-grid reference integration for reduced physical inversions."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from so_recon.inference.contracts import F64, FIXED_NOISE_THETA, ObservationBundle
from so_recon.inference.target import evaluate_loglik, require_complete_forward
from so_recon.observation.predict import predict_observations
from so_recon.registry.artifact import ArtifactRef, write_json_artifact
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.contracts import ForwardResult, OutputRequest
from so_recon.simulator.forward import SolverConfig, simulate
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.reduced_inverse import ReducedDesign, build_reduced_case

REFERENCE_SCHEMA_VERSION = "e02-reduced-reference-1"
REFERENCE_LIMIT = 7.0


class RunFactory(Protocol):
    def __call__(self, command: str, parent_run_ids: tuple[str, ...]) -> RunContext: ...


SimulateFunction = Callable[..., ForwardResult]


def _log_values(values: F64, *, label: str) -> F64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {array.shape}")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{label} may contain finite values or -inf")
    return array


def quadrature_reference(
    nodes: F64,
    prior_log_density: F64,
    log_likelihood: F64,
    integration_weights: F64,
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
    logmass = prior + likelihood + np.log(widths)
    logz = float(logsumexp(logmass))
    if not math.isfinite(logz):
        raise ValueError("quadrature has no target support on its nodes")
    weights: F64 = np.exp(logmass - logz)
    mean = float(weights @ points)
    variance = float(weights @ ((points - mean) ** 2))
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    indices = np.searchsorted(cdf, np.array([0.05, 0.5, 0.95]), side="left")
    quantiles: F64 = points[indices]
    return {
        "weights": weights,
        "mean": mean,
        "variance": variance,
        "quantiles": quantiles,
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
) -> ArtifactRef:
    """Run the 33-node physical reference and independently compare its 17-node subset.

    This routine deliberately does not import SMC particles, temperatures or weight helpers.
    A failed physical node is an execution failure, while a numerically insufficient grid is
    a published ``UNRESOLVED_REFERENCE`` scientific outcome.
    """
    nodes = np.linspace(-REFERENCE_LIMIT, REFERENCE_LIMIT, 33, dtype=np.float64)
    log_likelihood = np.empty(nodes.size, dtype=np.float64)
    node_records: list[dict[str, Any]] = []
    request = OutputRequest(
        state_times_s=design.report_edges_s,
        keep_native_restart=False,
        chunk_months=1,
    )
    solver = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)
    for index, z in enumerate(nodes):
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
                }
            )
            ctx.finish("PASS")
        except BaseException as exc:
            ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
            raise

    prior = norm.logpdf(nodes)
    fine = quadrature_reference(nodes, prior, log_likelihood, trapezoid_weights(nodes))
    coarse_nodes = nodes[::2]
    coarse = quadrature_reference(
        coarse_nodes,
        prior[::2],
        log_likelihood[::2],
        trapezoid_weights(coarse_nodes),
    )
    metrics = _comparison(coarse, fine)
    status = reference_status(metrics)
    parent = run_factory(
        "e02-reduced-reference", tuple(record["run_id"] for record in node_records)
    )
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "status": status,
        "design": design.model_dump(mode="json"),
        "observations": observations.model_dump(mode="json"),
        "noise": FIXED_NOISE_THETA.model_dump(mode="json"),
        "coarse": {
            "n_nodes": 17,
            "mean": coarse["mean"],
            "variance": coarse["variance"],
            "quantiles": np.asarray(coarse["quantiles"]).tolist(),
            "logz": coarse["logz"],
        },
        "fine": {
            "n_nodes": 33,
            "nodes": nodes.tolist(),
            "weights": np.asarray(fine["weights"]).tolist(),
            "mean": fine["mean"],
            "variance": fine["variance"],
            "quantiles": np.asarray(fine["quantiles"]).tolist(),
            "logz": fine["logz"],
        },
        "refinement": metrics,
        "nodes": node_records,
    }
    ref = write_json_artifact(
        parent.run_dir / "reduced_reference.json",
        payload,
        worker.paths,
        schema_version=REFERENCE_SCHEMA_VERSION,
        producer_run_id=parent.run_id,
        now=datetime.now(UTC),
    )
    parent.finish("PASS" if status == "CONVERGED" else "FAIL", outputs={"reference": ref})
    return ref


__all__ = [
    "REFERENCE_SCHEMA_VERSION",
    "quadrature_reference",
    "reference_status",
    "run_reduced_reference",
    "trapezoid_weights",
]
