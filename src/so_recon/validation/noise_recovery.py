"""Noise-only sigma/rho recovery on one immutable physical prediction bank."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import t

from so_recon.inference.contracts import (
    NU_FIXED,
    RHO_MAX,
    SIGMA_MIN,
    SIGMA_SPAN,
    HistoryRow,
    LogRow,
    ModelObservations,
    NoiseTheta,
    ObservationSupportMismatch,
)
from so_recon.observation.bins import G_MAX, BinGrid, rounding_grid, to_g_space
from so_recon.observation.history import PreviousObservation, history_loglik
from so_recon.observation.logs import draw_logs, logs_loglik
from so_recon.observation.noise import draw_history
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_json_artifact
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext

NOISE_RECOVERY_SCHEMA = "e02-noise-recovery-1"


def _log_diff_exp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b - a
    if np.any(delta >= 0.0):
        raise FloatingPointError("vectorized Student-t interval lost positive mass")
    return np.asarray(a + np.log(-np.expm1(delta)), dtype=np.float64)


def _log_interval(
    lo: float,
    hi: float,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    z_lo = (lo - mu) / sigma
    z_hi = (hi - mu) / sigma
    positive = z_lo >= 0.0
    out = np.empty(np.broadcast_shapes(z_lo.shape, z_hi.shape), dtype=np.float64)
    if np.any(positive):
        out[positive] = _log_diff_exp(
            np.asarray(t.logsf(z_lo[positive], NU_FIXED)),
            np.asarray(t.logsf(z_hi[positive], NU_FIXED)),
        )
    negative = ~positive
    if np.any(negative):
        out[negative] = _log_diff_exp(
            np.asarray(t.logcdf(z_hi[negative], NU_FIXED)),
            np.asarray(t.logcdf(z_lo[negative], NU_FIXED)),
        )
    return out


def selected_history_loglik_grid(
    rows: Sequence[HistoryRow],
    prediction: Mapping[tuple[str, int], float | None],
    sigmas: np.ndarray,
    rhos: np.ndarray,
    grid: BinGrid,
) -> np.ndarray:
    """Score only the observed bins over a Cartesian sigma/rho grid."""
    sigma_values = np.asarray(sigmas, dtype=np.float64)
    rho_values = np.asarray(rhos, dtype=np.float64)
    if (
        sigma_values.ndim != 1
        or rho_values.ndim != 1
        or not np.isfinite(sigma_values).all()
        or not np.isfinite(rho_values).all()
        or np.any(sigma_values <= 0.0)
        or np.any((rho_values < 0.0) | (rho_values >= 1.0))
    ):
        raise ValueError("sigma/rho grids must be finite 1D arrays in their physical support")
    sigma_matrix = np.broadcast_to(sigma_values[:, None], (sigma_values.size, rho_values.size))
    rho_matrix = np.broadcast_to(rho_values[None, :], sigma_matrix.shape)
    result = np.zeros_like(sigma_matrix)
    g_edges = to_g_space(grid.edges)
    previous_by_well: dict[str, PreviousObservation] = {}
    last_month: dict[str, int] = {}
    for row in rows:
        previous_month = last_month.get(row.well_id)
        if previous_month is not None and row.month_index <= previous_month:
            raise ValueError("history rows must be strictly increasing within each well")
        last_month[row.well_id] = row.month_index
        if row.reset:
            previous_by_well.pop(row.well_id, None)
        if not row.observed_valid:
            continue
        if row.quality_group not in {"watercut-0.01", "metered"}:
            raise ValueError(f"one supplied grid cannot score group {row.quality_group!r}")
        if row.bin_index is None or row.bin_index >= grid.n_bins:
            raise ValueError(f"observed row {row.key} has invalid bin {row.bin_index!r}")
        predicted = prediction.get(row.key)
        if predicted is None:
            raise ValueError(f"observed row {row.key} has no physical prediction")
        predicted_g = float(to_g_space(np.asarray(float(predicted), dtype=np.float64)))
        previous = previous_by_well.get(row.well_id)
        if previous is None:
            mu = np.full_like(rho_matrix, predicted_g)
        else:
            old_month, old_center, old_prediction = previous
            old_g = to_g_space(np.asarray([old_center, old_prediction], dtype=np.float64))
            residual = float(old_g[0] - old_g[1])
            mu = predicted_g + rho_matrix ** (row.month_index - old_month) * residual
        scale = sigma_matrix * row.sigma_multiplier
        selected = _log_interval(
            float(g_edges[row.bin_index]),
            float(g_edges[row.bin_index + 1]),
            mu,
            scale,
        )
        normalizer = _log_interval(0.0, G_MAX, mu, scale)
        result += selected - normalizer
        previous_by_well[row.well_id] = (
            row.month_index,
            float(grid.centers[row.bin_index]),
            float(predicted),
        )
    return result


def _normal_quadrature(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes < 9 or n_nodes % 2 == 0:
        raise ValueError("Gauss-Hermite noise grids use an odd node count of at least 9")
    roots, weights = np.polynomial.hermite.hermgauss(n_nodes)
    return np.sqrt(2.0) * roots, weights / math.sqrt(math.pi)


def _smooth_quantiles(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mass = weights / weights.sum()
    midpoint_cdf = np.cumsum(mass) - 0.5 * mass
    result = np.interp(
        np.asarray([0.05, 0.95]),
        midpoint_cdf,
        values,
        left=float(values[0]),
        right=float(values[-1]),
    )
    return [float(value) for value in result]


def noise_posterior(
    rows: Sequence[HistoryRow],
    prediction: Mapping[tuple[str, int], float | None],
    grid: BinGrid,
    *,
    n_nodes: int,
) -> dict[str, Any]:
    """Integrate the transformed standard-normal nuisance prior deterministically."""
    latent, prior_weights = _normal_quadrature(n_nodes)
    sigma = SIGMA_MIN + SIGMA_SPAN * ndtr(latent)
    rho = RHO_MAX * ndtr(latent)
    log_likelihood = selected_history_loglik_grid(rows, prediction, sigma, rho, grid)
    log_mass = log_likelihood + np.log(prior_weights)[:, None] + np.log(prior_weights)[None, :]
    log_evidence = float(logsumexp(log_mass))
    weights = np.exp(log_mass - log_evidence)
    sigma_weights = weights.sum(axis=1)
    rho_weights = weights.sum(axis=0)
    sigma_mean = float(sigma_weights @ sigma)
    rho_mean = float(rho_weights @ rho)
    sigma_centered = sigma[:, None] - sigma_mean
    rho_centered = rho[None, :] - rho_mean
    covariance = float(np.sum(weights * sigma_centered * rho_centered))
    sigma_variance = float(sigma_weights @ ((sigma - sigma_mean) ** 2))
    rho_variance = float(rho_weights @ ((rho - rho_mean) ** 2))
    denominator = math.sqrt(sigma_variance * rho_variance)
    return {
        "n_nodes_per_dimension": n_nodes,
        "sigma_mean": sigma_mean,
        "rho_mean": rho_mean,
        "sigma_quantiles": _smooth_quantiles(sigma, sigma_weights),
        "rho_quantiles": _smooth_quantiles(rho, rho_weights),
        "sigma_rho_correlation": 0.0 if denominator == 0.0 else covariance / denominator,
        "log_evidence": log_evidence,
    }


def compare_noise_grids(coarse: Mapping[str, Any], fine: Mapping[str, Any]) -> dict[str, Any]:
    sigma_quantile_change = float(
        np.max(
            np.abs(
                np.asarray(fine["sigma_quantiles"], dtype=np.float64)
                - np.asarray(coarse["sigma_quantiles"], dtype=np.float64)
            )
        )
    )
    rho_quantile_change = float(
        np.max(
            np.abs(
                np.asarray(fine["rho_quantiles"], dtype=np.float64)
                - np.asarray(coarse["rho_quantiles"], dtype=np.float64)
            )
        )
    )
    fine_evidence = math.exp(float(fine["log_evidence"]))
    evidence_change = abs(fine_evidence - math.exp(float(coarse["log_evidence"]))) / fine_evidence
    checks = {
        "sigma_mean": abs(float(fine["sigma_mean"]) - float(coarse["sigma_mean"])) < 0.0015,
        "rho_mean": abs(float(fine["rho_mean"]) - float(coarse["rho_mean"])) < 0.02,
        "sigma_quantiles": sigma_quantile_change < 0.004,
        "rho_quantiles": rho_quantile_change < 0.04,
        "evidence": evidence_change < 0.02,
    }
    return {
        "status": "CONVERGED" if all(checks.values()) else "UNRESOLVED_REFERENCE",
        "checks": checks,
        "sigma_mean_change": abs(float(fine["sigma_mean"]) - float(coarse["sigma_mean"])),
        "rho_mean_change": abs(float(fine["rho_mean"]) - float(coarse["rho_mean"])),
        "sigma_quantile_change": sigma_quantile_change,
        "rho_quantile_change": rho_quantile_change,
        "relative_evidence_change": evidence_change,
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson inputs require 0 <= successes <= trials and trials > 0")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(center - half), float(center + half)


def calibrate_noise_recovery(
    template: Sequence[HistoryRow],
    prediction: Mapping[tuple[str, int], float | None],
    grid: BinGrid,
    *,
    seeds: Sequence[int] = tuple(range(8000, 8200)),
    coarse_nodes: int = 33,
    fine_nodes: int = 65,
) -> dict[str, object]:
    """Run prior-predictive repeated recovery with no physical model evaluations."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("noise recovery needs at least one explicit seed")
    if seed_values != tuple(range(seed_values[0], seed_values[0] + len(seed_values))):
        raise ValueError("noise seeds must form one explicit contiguous sequence")
    records: list[dict[str, Any]] = []
    sigma_covered = 0
    rho_covered = 0
    low_bins = 0
    high_bins = 0
    reports = 0
    for seed in seed_values:
        rng = np.random.default_rng(seed)
        latent_sigma, latent_rho = rng.standard_normal(2)
        truth = NoiseTheta(
            sigma=SIGMA_MIN + SIGMA_SPAN * float(ndtr(latent_sigma)),
            rho=RHO_MAX * float(ndtr(latent_rho)),
            nu=NU_FIXED,
            log_bias=0.0,
        )
        observed = draw_history(template, prediction, truth, {template[0].quality_group: grid}, rng)
        coarse = noise_posterior(observed, prediction, grid, n_nodes=coarse_nodes)
        fine = noise_posterior(observed, prediction, grid, n_nodes=fine_nodes)
        refinement = compare_noise_grids(coarse, fine)
        sigma_interval = tuple(float(value) for value in fine["sigma_quantiles"])
        rho_interval = tuple(float(value) for value in fine["rho_quantiles"])
        sigma_hit = sigma_interval[0] <= truth.sigma <= sigma_interval[1]
        rho_hit = rho_interval[0] <= truth.rho <= rho_interval[1]
        sigma_covered += int(sigma_hit)
        rho_covered += int(rho_hit)
        bins = [int(row.bin_index) for row in observed if row.bin_index is not None]
        low_bins += bins.count(0)
        high_bins += bins.count(grid.n_bins - 1)
        reports += len(bins)
        records.append(
            {
                "seed": seed,
                "truth_latent": [float(latent_sigma), float(latent_rho)],
                "truth": truth.model_dump(mode="json"),
                "observed_bins": bins,
                "coarse": coarse,
                "fine": fine,
                "refinement": refinement,
                "sigma_covered_90": sigma_hit,
                "rho_covered_90": rho_hit,
            }
        )
    sigma_wilson = wilson_interval(sigma_covered, len(records))
    rho_wilson = wilson_interval(rho_covered, len(records))
    converged = all(record["refinement"]["status"] == "CONVERGED" for record in records)
    coverage_checks = {
        "sigma": sigma_wilson[0] <= 0.9 <= sigma_wilson[1],
        "rho": rho_wilson[0] <= 0.9 <= rho_wilson[1],
    }
    correlations = np.asarray(
        [record["fine"]["sigma_rho_correlation"] for record in records],
        dtype=np.float64,
    )
    truth_values = np.asarray(
        [[record["truth"]["sigma"], record["truth"]["rho"]] for record in records],
        dtype=np.float64,
    )
    truth_correlation = np.asarray(
        np.corrcoef(truth_values[:, 0], truth_values[:, 1]), dtype=np.float64
    )
    return {
        "schema_version": NOISE_RECOVERY_SCHEMA,
        "status": "PASS" if converged and all(coverage_checks.values()) else "FAIL",
        "repetitions": len(records),
        "seeds": [seed_values[0], seed_values[-1]],
        "quadrature": {
            "kind": "gauss-hermite-standard-normal-2d",
            "coarse_nodes_per_dimension": coarse_nodes,
            "fine_nodes_per_dimension": fine_nodes,
            "new_physical_forwards": 0,
        },
        "all_refinements_converged": converged,
        "coverage": {
            "nominal": 0.9,
            "sigma": sigma_covered / len(records),
            "sigma_wilson_95": sigma_wilson,
            "rho": rho_covered / len(records),
            "rho_wilson_95": rho_wilson,
            "checks": coverage_checks,
        },
        "boundary_bins": {
            "reports": reports,
            "zero_count": low_bins,
            "one_count": high_bins,
            "zero_frequency": low_bins / reports,
            "one_frequency": high_bins / reports,
            "zero_wilson_95": wilson_interval(low_bins, reports),
            "one_wilson_95": wilson_interval(high_bins, reports),
        },
        "sigma_rho_correlation": {
            "posterior_mean": float(np.mean(correlations)),
            "posterior_min": float(np.min(correlations)),
            "posterior_max": float(np.max(correlations)),
            "truth_sample": float(truth_correlation[0, 1]),
        },
        "records": records,
    }


def noise_contract_checks(
    template: Sequence[HistoryRow],
    prediction: Mapping[tuple[str, int], float | None],
    grid: BinGrid,
    truth_result_path: Path,
) -> dict[str, Any]:
    """Exercise endpoints, missing/reset/dry masks and saved-state log semantics."""
    endpoint_rows = (
        HistoryRow(
            well_id="P1",
            month_index=0,
            raw_value=0.0,
            bin_index=0,
            quality_group="watercut-0.01",
            observed_valid=True,
            reset=False,
        ),
        HistoryRow(
            well_id="P1",
            month_index=1,
            raw_value=1.0,
            bin_index=grid.n_bins - 1,
            quality_group="watercut-0.01",
            observed_valid=True,
            reset=False,
        ),
    )
    endpoint_likelihood = history_loglik(
        endpoint_rows,
        {("P1", 0): 0.0, ("P1", 1): 1.0},
        NoiseTheta(sigma=0.03, rho=0.4, nu=NU_FIXED, log_bias=0.0),
        {"watercut-0.01": grid},
    )

    dry_prediction = dict(prediction)
    dry_prediction[("P1", 3)] = None
    drawn = draw_history(
        template,
        dry_prediction,
        NoiseTheta(sigma=0.03, rho=0.4, nu=NU_FIXED, log_bias=0.0),
        {"watercut-0.01": grid},
        np.random.default_rng(20260915),
    )
    month3 = next(row for row in drawn if row.month_index == 3)
    observed_dry_refused = False
    forced_observed = tuple(
        row.model_copy(update={"observed_valid": True, "raw_value": 0.0, "bin_index": 0})
        if row.month_index == 3
        else row
        for row in template
    )
    try:
        history_loglik(
            forced_observed,
            dry_prediction,
            NoiseTheta(sigma=0.03, rho=0.4, nu=NU_FIXED, log_bias=0.0),
            {"watercut-0.01": grid},
        )
    except ObservationSupportMismatch:
        observed_dry_refused = True

    result = json.loads(truth_result_path.read_text(encoding="utf-8"))
    states = result.get("extraction", {}).get("states", {})
    times = np.asarray(states.get("times_s", []), dtype=np.float64)
    so = np.asarray(states.get("so", []), dtype=np.float64)
    if result.get("status") != "COMPLETE" or times.shape != (13,) or so.shape != (13, 16):
        raise ValueError("truth result must contain all 13 saved report states for 16 cells")
    candidate_times = (float(times[10]), float(times[11]))
    support = {
        (observation_id, time): float(np.mean(so[index, cells]))
        for observation_id, cells in (("L-west", slice(0, 8)), ("L-east", slice(8, 16)))
        for index, time in ((10, candidate_times[0]), (11, candidate_times[1]))
    }
    model = ModelObservations(fw={}, so_support=support, model_hash=str(result["model_hash"]))
    log_rows = tuple(
        LogRow(
            observation_id=observation_id,
            support_cell_ids=(
                tuple(range(0, 8)) if observation_id == "L-west" else tuple(range(8, 16))
            ),
            support_weights=(0.125,) * 8,
            bin_index=0,
            date_times_s=candidate_times,
            date_weights=(0.4, 0.6),
            date_group="D-reduced",
            bias_group="logs",
            valid=True,
            use="likelihood",
        )
        for observation_id in ("L-west", "L-east")
    )
    log_grid = rounding_grid(0.01)
    log_noise = NoiseTheta(sigma=0.03, rho=0.4, nu=NU_FIXED, log_bias=0.01)
    drawn_logs = draw_logs(
        log_rows, model, log_noise, {"logs": log_grid}, np.random.default_rng(20260915)
    )
    log_likelihood = logs_loglik(drawn_logs, model, log_noise, {"logs": log_grid})
    shifted = logs_loglik(
        drawn_logs,
        model,
        log_noise.model_copy(update={"log_bias": -0.01}),
        {"logs": log_grid},
    )
    return {
        "endpoint_bins": {
            "values": [0.0, 1.0],
            "finite_joint_log_likelihood": math.isfinite(endpoint_likelihood.value),
        },
        "history_masks": {
            "month3_missing_preserved": (
                not month3.observed_valid and month3.raw_value is None and month3.bin_index is None
            ),
            "reset_month7_present": any(row.month_index == 7 and row.reset for row in template),
            "observed_dry_prediction_refused": observed_dry_refused,
        },
        "saved_state_log": {
            "truth_result_sha256": sha256_file(truth_result_path),
            "report_state_count": int(times.size),
            "candidate_times_s": list(candidate_times),
            "joint_date_factor_count": log_likelihood.n_used,
            "drawn_bins": [row.bin_index for row in drawn_logs],
            "finite_log_likelihood": math.isfinite(log_likelihood.value),
            "shared_bias_changes_likelihood": shifted.value != log_likelihood.value,
        },
    }


def publish_noise_recovery(
    *,
    ctx: RunContext,
    template: Sequence[HistoryRow],
    prediction: Mapping[tuple[str, int], float | None],
    grid: BinGrid,
    truth_result_path: Path,
    paths: ProjectPaths,
    parent_refs: Sequence[ArtifactRef],
) -> ArtifactRef:
    """Run and publish the fixed 200-replicate noise-only acceptance suite."""
    calibration = calibrate_noise_recovery(template, prediction, grid)
    contracts = noise_contract_checks(template, prediction, grid, truth_result_path)
    contract_pass = bool(
        contracts["endpoint_bins"]["finite_joint_log_likelihood"]
        and all(contracts["history_masks"].values())
        and contracts["saved_state_log"]["report_state_count"] == 13
        and contracts["saved_state_log"]["joint_date_factor_count"] == 1
        and contracts["saved_state_log"]["finite_log_likelihood"]
        and contracts["saved_state_log"]["shared_bias_changes_likelihood"]
    )
    payload = {
        **calibration,
        "status": "PASS" if calibration["status"] == "PASS" and contract_pass else "FAIL",
        "prediction_bank": {
            "model_hash": sha256_json(
                {f"{well}:{month}": value for (well, month), value in sorted(prediction.items())}
            ),
            "values": {
                f"{well}:{month}": value for (well, month), value in sorted(prediction.items())
            },
            "new_physical_forwards": 0,
        },
        "contract_checks": contracts,
        "contract_pass": contract_pass,
        "parents": [ref.model_dump(mode="json") for ref in parent_refs],
    }
    return write_json_artifact(
        ctx.run_dir / "noise_recovery.json",
        payload,
        paths,
        schema_version=NOISE_RECOVERY_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=tuple(ref.artifact_id for ref in parent_refs),
        now=datetime.now(UTC),
    )


__all__ = [
    "NOISE_RECOVERY_SCHEMA",
    "calibrate_noise_recovery",
    "compare_noise_grids",
    "noise_posterior",
    "noise_contract_checks",
    "publish_noise_recovery",
    "selected_history_loglik_grid",
    "wilson_interval",
]
