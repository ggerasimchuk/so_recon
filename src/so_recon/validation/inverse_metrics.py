"""Particle-weighted metrics on one fixed, conservative report support."""

from __future__ import annotations

import numpy as np

from so_recon.inference.contracts import F64


def _normalised_weights(weights: F64, n: int) -> F64:
    mass = np.asarray(weights, dtype=np.float64)
    if mass.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {mass.shape}")
    if not np.isfinite(mass).all() or np.any(mass < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(mass.sum())
    if total <= 0.0:
        raise ValueError("weights must carry positive total mass")
    return mass / total


def weighted_quantile(values: F64, weights: F64, probabilities: F64) -> F64:
    """Inverse empirical CDF with stable sorting and particle weights."""
    samples = np.asarray(values, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("values must be a finite one-dimensional array")
    if probs.ndim != 1 or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be a one-dimensional array in [0, 1]")
    mass = _normalised_weights(weights, samples.size)
    order = np.argsort(samples, kind="stable")
    ordered = samples[order]
    cdf = np.cumsum(mass[order])
    cdf[-1] = 1.0
    return ordered[np.searchsorted(cdf, probs, side="left")]


def weighted_crps(values: F64, weights: F64, truth: float) -> float:
    """Exact weighted empirical CRPS."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all() or not np.isfinite(truth):
        raise ValueError("CRPS inputs must be finite and one-dimensional")
    mass = _normalised_weights(weights, samples.size)
    first = float(mass @ np.abs(samples - truth))
    pairwise = np.abs(samples[:, None] - samples[None, :])
    return first - 0.5 * float(mass @ pairwise @ mass)


def inventory_quantiles(
    cell_inventory_by_particle: F64,
    particle_weights: F64,
    probabilities: F64,
) -> F64:
    """Quantiles of each particle's summed inventory, never sums of cell quantiles."""
    inventory = np.asarray(cell_inventory_by_particle, dtype=np.float64)
    if inventory.ndim != 2 or not np.isfinite(inventory).all():
        raise ValueError("cell inventories must be a finite (particle, cell) matrix")
    return weighted_quantile(inventory.sum(axis=1), particle_weights, probabilities)


def compare_so(estimate: F64, truth: F64, truth_pv: F64) -> dict[str, float]:
    """PV-weighted errors on the truth's immutable valid support."""
    estimated = np.asarray(estimate, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    pv = np.asarray(truth_pv, dtype=np.float64)
    if estimated.shape != actual.shape or pv.shape != actual.shape or actual.ndim != 1:
        raise ValueError(
            f"estimate, truth and truth_pv must share a 1D shape, got "
            f"{estimated.shape}, {actual.shape}, {pv.shape}"
        )
    if np.any(np.isfinite(pv) & (pv < 0.0)):
        raise ValueError("truth pore volume cannot be negative")
    valid = np.isfinite(actual) & np.isfinite(pv) & (pv > 0.0)
    if not valid.any():
        raise ValueError("no comparable valid truth support")
    if not np.isfinite(estimated[valid]).all():
        raise ValueError("missing estimate on valid truth support")
    weights = pv[valid] / pv[valid].sum()
    error = estimated[valid] - actual[valid]
    return {
        "mae": float(weights @ np.abs(error)),
        "rmse": float(np.sqrt(weights @ (error**2))),
    }


__all__ = [
    "compare_so",
    "inventory_quantiles",
    "weighted_crps",
    "weighted_quantile",
]
