"""Resampling schemes whose ancestry is reproducible from the stored RNG state."""

from __future__ import annotations

import numpy as np

from so_recon.inference.contracts import F64, I64
from so_recon.inference.weights import _normalized


def systematic_resample(logw: F64, rng: np.random.Generator) -> I64:
    """Draw exactly ``N`` parent indices by one-offset systematic resampling."""
    weights = _normalized(logw)
    probabilities = np.exp(weights)
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    n = weights.size
    positions = rng.random() / n + np.arange(n, dtype=np.float64) / n
    indices: I64 = np.searchsorted(cdf, positions, side="right").astype(np.int64)
    return indices


__all__ = ["systematic_resample"]
