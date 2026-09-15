"""Deterministic sigma/rho quadrature uses the same recursive rounded likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from so_recon.inference.contracts import HistoryRow, NoiseTheta
from so_recon.observation.bins import rounding_grid
from so_recon.observation.history import history_loglik
from so_recon.validation.noise_recovery import (
    noise_posterior,
    selected_history_loglik_grid,
)


def _rows() -> tuple[HistoryRow, ...]:
    return (
        HistoryRow(
            well_id="P1",
            month_index=0,
            raw_value=0.2,
            bin_index=2,
            quality_group="metered",
            observed_valid=True,
            reset=False,
        ),
        HistoryRow(
            well_id="P1",
            month_index=1,
            raw_value=None,
            bin_index=None,
            quality_group="metered",
            observed_valid=False,
            reset=True,
        ),
        HistoryRow(
            well_id="P1",
            month_index=2,
            raw_value=0.7,
            bin_index=7,
            quality_group="metered",
            observed_valid=True,
            reset=False,
        ),
    )


def test_vectorized_selected_bin_likelihood_matches_the_public_scorer() -> None:
    grid = rounding_grid(0.1)
    rows = _rows()
    prediction = {("P1", 0): 0.25, ("P1", 2): 0.65}
    sigmas = np.array([0.03, 0.06])
    rhos = np.array([0.2, 0.7])

    actual = selected_history_loglik_grid(rows, prediction, sigmas, rhos, grid)

    for i, sigma in enumerate(sigmas):
        for j, rho in enumerate(rhos):
            expected = history_loglik(
                rows,
                prediction,
                NoiseTheta(sigma=float(sigma), rho=float(rho), nu=5.0, log_bias=0.0),
                {"metered": grid},
            )
            assert actual[i, j] == pytest.approx(expected.value, abs=1e-12)


def test_noise_posterior_reports_physical_marginals_and_refinement_inputs() -> None:
    result = noise_posterior(
        _rows(),
        {("P1", 0): 0.25, ("P1", 2): 0.65},
        rounding_grid(0.1),
        n_nodes=17,
    )

    assert 0.015 < result["sigma_mean"] < 0.08
    assert 0.0 < result["rho_mean"] < 0.8
    assert len(result["sigma_quantiles"]) == 2
    assert len(result["rho_quantiles"]) == 2
    assert -1.0 <= result["sigma_rho_correlation"] <= 1.0
    assert np.isfinite(result["log_evidence"])
