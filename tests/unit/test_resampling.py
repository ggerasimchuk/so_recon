"""Systematic resampling preserves ancestry under normalized nonuniform weights."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so_recon.inference.resampling import systematic_resample


def test_seeded_systematic_resampling_has_exact_indices() -> None:
    rng = np.random.default_rng(91)
    actual = systematic_resample(np.log([0.05, 0.15, 0.3, 0.5]), rng)
    np.testing.assert_array_equal(actual, np.array([1, 2, 3, 3], dtype=np.int64))
    assert actual.dtype == np.int64


def test_expected_offspring_match_nonuniform_weights() -> None:
    rng = np.random.default_rng(912)
    probabilities = np.array([0.05, 0.15, 0.3, 0.5])
    totals = np.zeros(4, dtype=np.int64)
    repetitions = 10_000
    for _ in range(repetitions):
        totals += np.bincount(systematic_resample(np.log(probabilities), rng), minlength=4)
    frequencies = totals / (repetitions * probabilities.size)
    np.testing.assert_allclose(frequencies, probabilities, atol=0.003)


def test_zero_weight_parent_gets_no_offspring() -> None:
    indices = systematic_resample(
        np.array([-math.inf, math.log(0.4), math.log(0.6)]), np.random.default_rng(4)
    )
    assert 0 not in indices


@pytest.mark.parametrize(
    "bad",
    [np.log([0.4, 0.4]), np.array([0.0, math.nan]), np.array([0.0, math.inf])],
)
def test_resampling_refuses_unnormalized_or_nonfinite_weights(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        systematic_resample(bad, np.random.default_rng(1))
