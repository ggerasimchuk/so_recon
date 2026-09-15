"""Calendar-aware history likelihood tests for E02 Task 4."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so_recon.inference.contracts import (
    HistoryRow,
    NoiseTheta,
    ObservationSupportMismatch,
)
from so_recon.observation.bins import rounding_grid
from so_recon.observation.history import conditional_mean, history_loglik

OBSERVATION_HASH = "d" * 64


def row(
    month: int,
    *,
    well: str = "P1",
    observed: bool = True,
    bin_index: int = 1,
    reset: bool = False,
    multiplier: float = 1.0,
) -> HistoryRow:
    return HistoryRow(
        well_id=well,
        month_index=month,
        raw_value=0.5 if observed else None,
        bin_index=bin_index if observed else None,
        quality_group="metered",
        observed_valid=observed,
        reset=reset,
        sigma_multiplier=multiplier,
    )


def test_gap_uses_calendar_distance_and_observed_bin_center() -> None:
    def g(x: float) -> float:
        return float(np.arcsin(np.sqrt(x)))

    actual = conditional_mean(0.4, (2, 0.8, 0.2), 5, 0.5, False)
    assert np.isclose(actual, g(0.4) + 0.5**3 * (g(0.8) - g(0.2)))
    assert conditional_mean(0.4, (2, 0.8, 0.2), 5, 0.5, True) == g(0.4)


def test_unobserved_months_do_not_become_the_previous_residual() -> None:
    rows = (row(2, bin_index=2), row(3, observed=False), row(4, observed=False), row(5))
    predictions = {("P1", 2): 0.2, ("P1", 5): 0.4}
    noise = NoiseTheta(sigma=0.08, rho=0.5, nu=5.0, log_bias=0.0)
    grid = rounding_grid(0.5)

    result = history_loglik(
        rows, predictions, noise, {"metered": grid}, observation_hash=OBSERVATION_HASH
    )

    expected_mu = conditional_mean(0.4, (2, 1.0, 0.2), 5, 0.5, False)
    from so_recon.observation.bins import bin_log_probs

    expected_last = bin_log_probs(grid, mu=expected_mu, sigma=0.08, nu=5.0)[1]
    assert result.terms[-1] == pytest.approx(float(expected_last))


def test_reset_in_an_unobserved_month_clears_memory() -> None:
    rows = (row(1, bin_index=2), row(2, observed=False, reset=True), row(5))
    predictions = {("P1", 1): 0.2, ("P1", 5): 0.4}
    noise = NoiseTheta(sigma=0.08, rho=0.7, nu=5.0, log_bias=0.0)
    grid = rounding_grid(0.5)

    result = history_loglik(
        rows, predictions, noise, {"metered": grid}, observation_hash=OBSERVATION_HASH
    )

    from so_recon.observation.bins import bin_log_probs, to_g_space

    expected_last = bin_log_probs(grid, mu=float(to_g_space(np.array(0.4))), sigma=0.08, nu=5.0)[1]
    assert result.terms[-1] == pytest.approx(float(expected_last))


def test_wells_keep_independent_recursive_buffers() -> None:
    rows = (row(1, well="P1", bin_index=2), row(1, well="P2", bin_index=0), row(2, well="P1"))
    predictions = {("P1", 1): 0.2, ("P2", 1): 0.8, ("P1", 2): 0.4}
    noise = NoiseTheta(sigma=0.1, rho=0.5, nu=5.0, log_bias=0.0)
    grid = rounding_grid(0.5)

    result = history_loglik(
        rows, predictions, noise, {"metered": grid}, observation_hash=OBSERVATION_HASH
    )

    from so_recon.observation.bins import bin_log_probs

    expected_mu = conditional_mean(0.4, (1, 1.0, 0.2), 2, 0.5, False)
    assert result.terms[2] == pytest.approx(
        float(bin_log_probs(grid, mu=expected_mu, sigma=0.1, nu=5.0)[1])
    )


@pytest.mark.parametrize("prediction", [None, math.nan, -0.01, 1.01])
def test_an_observed_row_without_a_valid_physical_prediction_is_not_masked(
    prediction: float | None,
) -> None:
    with pytest.raises(ObservationSupportMismatch):
        history_loglik(
            (row(0),),
            {("P1", 0): prediction},
            NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0),
            {"metered": rounding_grid(0.5)},
            observation_hash=OBSERVATION_HASH,
        )


def test_no_valid_observations_is_the_unit_likelihood() -> None:
    result = history_loglik(
        (row(0, observed=False),),
        {},
        NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0),
        {"metered": rounding_grid(0.5)},
        observation_hash=OBSERVATION_HASH,
    )
    assert result.value == 0.0
    assert result.terms == ()
    assert result.n_used == 0


def test_quality_multiplier_changes_only_that_rows_scale() -> None:
    grid = rounding_grid(0.5)
    noise = NoiseTheta(sigma=0.03, rho=0.0, nu=5.0, log_bias=0.0)
    ordinary = history_loglik(
        (row(0),), {("P1", 0): 0.1}, noise, {"metered": grid}, observation_hash=OBSERVATION_HASH
    )
    suspect = history_loglik(
        (row(0, multiplier=3.0),),
        {("P1", 0): 0.1},
        noise,
        {"metered": grid},
        observation_hash=OBSERVATION_HASH,
    )
    assert ordinary.value != suspect.value


def test_rows_for_one_well_must_have_increasing_calendar_months() -> None:
    with pytest.raises(ValueError, match="increasing"):
        history_loglik(
            (row(2), row(1)),
            {("P1", 1): 0.5, ("P1", 2): 0.5},
            NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0),
            {"metered": rounding_grid(0.5)},
            observation_hash=OBSERVATION_HASH,
        )
