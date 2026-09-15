"""Numerically stable weight, CESS and tempering algebra for E02 SMC."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from so_recon.inference.contracts import F64

NORMALIZATION_ATOL = 1e-12
MIN_BETA_ADVANCE = 1e-12
BETA_TOLERANCE = 1e-10
MAX_BETA_BISECTIONS = 80


class NoTargetSupport(RuntimeError):
    """Every particle has zero mass under the next target."""


class CessSupportDiscontinuity(RuntimeError):
    """CESS jumps below policy immediately to the right of the current beta."""


def _log_vector(values: F64, *, label: str) -> F64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector, got {array.shape}")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{label} may contain finite log values or -inf, got {array!r}")
    return array


def _normalized(logw: F64) -> F64:
    values = _log_vector(logw, label="log weights")
    total = float(logsumexp(values))
    if not math.isfinite(total):
        raise NoTargetSupport("all weights are zero")
    if not math.isclose(total, 0.0, rel_tol=0.0, abs_tol=NORMALIZATION_ATOL):
        raise ValueError(f"log weights must be normalized, logsumexp is {total!r}")
    return values


def normalize_log_weights(logw: F64) -> tuple[F64, float]:
    """Normalize extended-real log weights and return the removed log normalizer."""
    values = _log_vector(logw, label="log weights")
    normalizer = float(logsumexp(values))
    if not math.isfinite(normalizer):
        raise NoTargetSupport("all weights are zero")
    normalized: F64 = values - normalizer
    # One compensated pass removes the rounding residue introduced when a very large
    # normalizer is subtracted from nearby values (for example 1000 and 999).
    residue = float(logsumexp(normalized))
    normalized = normalized - residue
    normalizer += residue
    return normalized, normalizer


def ess(logw: F64) -> float:
    """Effective sample size of normalized weights, including exact zero masses."""
    values = _normalized(logw)
    return float(math.exp(-float(logsumexp(2.0 * values))))


def cess(logw: F64, ell: F64, delta: float) -> float:
    """Conditional ESS for an incremental tempering exponent ``delta``."""
    weights = _normalized(logw)
    increments = _log_vector(ell, label="log target increments")
    if increments.shape != weights.shape:
        raise ValueError(
            f"log target increments shape {increments.shape} does not match weights {weights.shape}"
        )
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError(f"delta must be finite and non-negative, got {delta!r}")
    n = weights.size
    if delta == 0.0:
        return float(n)

    active = np.isfinite(weights)
    supported = active & np.isfinite(increments)
    if not np.any(supported):
        raise NoTargetSupport("all incremental target weights are zero")

    # Subtracting the largest finite ell is an exact common shift in log-space.  It keeps
    # both logsumexp calls small and makes CESS invariant even to a +10000 likelihood shift.
    anchor = float(np.max(increments[supported]))
    x = delta * (increments[active] - anchor)
    lw = weights[active]
    a = float(logsumexp(lw + x))
    b = float(logsumexp(lw + 2.0 * x))
    if not math.isfinite(a) or not math.isfinite(b):
        raise NoTargetSupport("all incremental target weights are zero")
    return float(n * math.exp(2.0 * a - b))


def next_beta(beta: float, logw: F64, ell: F64, target_fraction: float) -> float:
    """Choose the next exponent by CESS bisection, or return exact one."""
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0,1], got {beta!r}")
    if not math.isfinite(target_fraction) or not 0.0 < target_fraction <= 1.0:
        raise ValueError(f"target_fraction must be in (0,1], got {target_fraction!r}")
    weights = _normalized(logw)
    increments = _log_vector(ell, label="log target increments")
    if increments.shape != weights.shape:
        raise ValueError("log target increments and weights must have the same shape")
    if beta == 1.0:
        return 1.0
    maximum = 1.0 - beta
    target = target_fraction * weights.size
    if cess(weights, increments, maximum) >= target:
        return 1.0
    probe = min(MIN_BETA_ADVANCE, maximum)
    if cess(weights, increments, probe) < target:
        raise CessSupportDiscontinuity(
            "CESS_SUPPORT_DISCONTINUITY: no policy-compliant positive beta advance"
        )

    low = probe
    high = maximum
    for _ in range(MAX_BETA_BISECTIONS):
        if high - low <= BETA_TOLERANCE:
            break
        midpoint = 0.5 * (low + high)
        if cess(weights, increments, midpoint) >= target:
            low = midpoint
        else:
            high = midpoint
    advance = 0.5 * (low + high)
    if advance < MIN_BETA_ADVANCE:
        raise CessSupportDiscontinuity("CESS_SUPPORT_DISCONTINUITY: beta advance is below 1e-12")
    return float(beta + advance)


def bridge_log_target(beta: float, log_p0: float, log_l: float, log_r: float) -> float:
    """Log of ``r^(1-beta) (p0 L)^beta`` with exact endpoint semantics."""
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0,1], got {beta!r}")
    for label, value in (("log_p0", log_p0), ("log_l", log_l), ("log_r", log_r)):
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"{label} must be finite or -inf, got {value!r}")
    if beta == 0.0:
        return float(log_r)
    posterior = log_p0 + log_l
    if beta == 1.0:
        return float(posterior)
    if posterior == -math.inf or log_r == -math.inf:
        return -math.inf
    return float((1.0 - beta) * log_r + beta * posterior)


__all__ = [
    "CessSupportDiscontinuity",
    "NoTargetSupport",
    "bridge_log_target",
    "cess",
    "ess",
    "next_beta",
    "normalize_log_weights",
]
