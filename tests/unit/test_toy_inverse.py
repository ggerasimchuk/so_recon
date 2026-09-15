"""Independent analytic oracles for the Gaussian and disconnected toy targets."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from so_recon.validation.toy_inverse import (
    BimodalPrior,
    bimodal_reference,
    gaussian_reference,
    prior_predictive_calibration,
)


def test_reference_is_analytic() -> None:
    mean, variance, logz = gaussian_reference(0.0, 1.0, 2.0, 1.0)
    assert mean == 1.0
    assert variance == 0.5
    assert np.isclose(logz, -0.5 * np.log(4 * np.pi) - 1.0)


def test_gaussian_reference_refuses_invalid_measures() -> None:
    with pytest.raises(ValueError, match="variances"):
        gaussian_reference(0.0, 0.0, 2.0, 1.0)


def test_bimodal_reference_uses_normalized_family_evidence() -> None:
    reference = bimodal_reference()
    assert reference["family_probabilities"] == pytest.approx((0.7, 0.3))
    assert sum(reference["family_probabilities"]) == pytest.approx(1.0)
    assert reference["conditional_means"] == pytest.approx(
        (-2.8235294117647056, 2.8235294117647056)
    )
    assert reference["conditional_variance"] == pytest.approx(4.0 / 17.0)

    numerical = 0.0
    for probability, mean in zip(
        BimodalPrior.FAMILY_PROBABILITIES,
        BimodalPrior.FAMILY_MEANS,
        strict=True,
    ):
        value, _ = quad(
            lambda x, probability=probability, mean=mean: (
                probability
                * math.exp(-0.5 * ((x - mean) / 0.5) ** 2)
                / (0.5 * math.sqrt(2.0 * math.pi))
                * math.exp(-0.5 * (x / 2.0) ** 2)
                / (2.0 * math.sqrt(2.0 * math.pi))
            ),
            -math.inf,
            math.inf,
        )
        numerical += value
    assert reference["evidence"] == pytest.approx(numerical, rel=1e-11)


def test_repeated_prior_predictive_coverage_includes_the_declared_ninety_percent() -> None:
    report = prior_predictive_calibration(20260915, repetitions=200, n_particles=32)
    smc_low, smc_high = report["smc_coverage_wilson_95"]
    exact_low, exact_high = report["exact_coverage_wilson_95"]
    assert smc_low <= 0.9 <= smc_high, report
    assert exact_low <= 0.9 <= exact_high, report
    assert len(report["smc_ranks"]) == 200
    assert len(report["exact_ranks"]) == 200
    assert "ancestry" in report["smc_draw_disclosure"]
    assert "independent" in report["exact_draw_disclosure"]
