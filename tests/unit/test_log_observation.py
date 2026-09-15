"""Unknown log dates are one joint latent choice per date group."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from so_recon.inference.contracts import (
    LogRow,
    ModelObservations,
    NoiseTheta,
    ObservationSupportMismatch,
)
from so_recon.observation.bins import bin_log_probs, rounding_grid, to_g_space
from so_recon.observation.logs import draw_logs, log_date_mixture, logs_loglik

OBSERVATION_HASH = "d" * 64
MODEL_HASH = "e" * 64


def row(
    observation_id: str = "L1",
    *,
    bin_index: int = 1,
    dates: tuple[float, ...] = (0.0, 1.0),
    weights: tuple[float, ...] = (0.25, 0.75),
    date_group: str = "D1",
    valid: bool = True,
    use: str = "likelihood",
) -> LogRow:
    return LogRow(
        observation_id=observation_id,
        support_cell_ids=(0,),
        support_weights=(1.0,),
        bin_index=bin_index if valid else None,
        date_times_s=dates,
        date_weights=weights,
        date_group=date_group,
        bias_group="logs",
        valid=valid,
        use=use,
    )


def prediction(values: dict[tuple[str, float], float]) -> ModelObservations:
    return ModelObservations(fw={}, so_support=values, model_hash=MODEL_HASH)


def test_dates_are_marginalized() -> None:
    result = log_date_mixture(np.log([0.1, 0.8]), np.array([0.25, 0.75]))
    assert np.isclose(result, np.log(0.625))
    assert not np.isclose(result, np.log(0.8))


def test_rows_with_one_unknown_date_are_marginalized_jointly() -> None:
    grid = rounding_grid(0.5)
    rows = (row("L1", bin_index=0), row("L2", bin_index=2))
    model = prediction(
        {
            ("L1", 0.0): 0.1,
            ("L1", 1.0): 0.8,
            ("L2", 0.0): 0.2,
            ("L2", 1.0): 0.9,
        }
    )
    noise = NoiseTheta(sigma=0.04, rho=0.4, nu=5.0, log_bias=0.0)

    actual = logs_loglik(rows, model, noise, {"logs": grid}, observation_hash=OBSERVATION_HASH)

    joint_by_date = []
    for time in (0.0, 1.0):
        joint_by_date.append(
            sum(
                float(
                    bin_log_probs(
                        grid,
                        mu=float(
                            to_g_space(np.asarray(model.so_support[(item.observation_id, time)]))
                        ),
                        sigma=0.03,
                        nu=5.0,
                    )[item.bin_index]
                )
                for item in rows
                if item.bin_index is not None
            )
        )
    expected = logsumexp(np.log([0.25, 0.75]) + joint_by_date)
    independent_mixtures = sum(
        logsumexp(
            np.log([0.25, 0.75])
            + [
                float(
                    bin_log_probs(
                        grid,
                        mu=float(
                            to_g_space(np.asarray(model.so_support[(item.observation_id, time)]))
                        ),
                        sigma=0.03,
                        nu=5.0,
                    )[item.bin_index]
                )
                for time in (0.0, 1.0)
            ]
        )
        for item in rows
        if item.bin_index is not None
    )
    assert actual.value == pytest.approx(float(expected))
    assert actual.value != pytest.approx(float(independent_mixtures))
    assert actual.n_used == 1  # one coupled likelihood factor, the shared date group


def test_different_date_groups_are_independent_factors() -> None:
    grid = rounding_grid(0.5)
    rows = (row("L1", date_group="D1"), row("L2", date_group="D2"))
    model = prediction(
        {
            ("L1", 0.0): 0.1,
            ("L1", 1.0): 0.7,
            ("L2", 0.0): 0.2,
            ("L2", 1.0): 0.8,
        }
    )
    result = logs_loglik(
        rows,
        model,
        NoiseTheta(sigma=0.1, rho=0.3, nu=5.0, log_bias=0.0),
        {"logs": grid},
        observation_hash=OBSERVATION_HASH,
    )
    assert result.n_used == 2
    assert result.value == pytest.approx(sum(result.terms))


@pytest.mark.parametrize("use", ["condition_prior", "heldout_diagnostic"])
def test_a_row_already_used_elsewhere_is_not_counted_again(use: str) -> None:
    result = logs_loglik(
        (row(use=use),),
        prediction({("L1", 0.0): 0.2, ("L1", 1.0): 0.3}),
        NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0),
        {"logs": rounding_grid(0.5)},
        observation_hash=OBSERVATION_HASH,
    )
    assert result.value == 0.0
    assert result.n_used == 0


def test_one_bias_is_shared_by_every_row_in_the_bundle() -> None:
    grid = rounding_grid(0.5)
    rows = (row("L1"), row("L2"))
    model = prediction(
        {
            ("L1", 0.0): 0.2,
            ("L1", 1.0): 0.4,
            ("L2", 0.0): 0.3,
            ("L2", 1.0): 0.5,
        }
    )
    centered = logs_loglik(
        rows,
        model,
        NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0),
        {"logs": grid},
        observation_hash=OBSERVATION_HASH,
    )
    shifted = logs_loglik(
        rows,
        model,
        NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.2),
        {"logs": grid},
        observation_hash=OBSERVATION_HASH,
    )
    assert centered.value != shifted.value


def test_a_requested_date_without_a_saved_state_is_refused() -> None:
    with pytest.raises(ObservationSupportMismatch, match="date"):
        logs_loglik(
            (row(),),
            prediction({("L1", 0.0): 0.2}),
            NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0),
            {"logs": rounding_grid(0.5)},
            observation_hash=OBSERVATION_HASH,
        )


def test_rows_in_one_date_group_must_name_the_same_latent_date_law() -> None:
    inconsistent = row("L2", dates=(0.0, 2.0), weights=(0.25, 0.75))
    with pytest.raises(ValueError, match="same candidate"):
        logs_loglik(
            (row("L1"), inconsistent),
            prediction(
                {
                    ("L1", 0.0): 0.2,
                    ("L1", 1.0): 0.3,
                    ("L2", 0.0): 0.2,
                    ("L2", 2.0): 0.3,
                }
            ),
            NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0),
            {"logs": rounding_grid(0.5)},
            observation_hash=OBSERVATION_HASH,
        )


def test_generator_uses_one_date_draw_for_a_shared_group() -> None:
    grid = rounding_grid(0.5)
    rows = (row("L1"), row("L2"))
    model = prediction(
        {
            ("L1", 0.0): 0.001,
            ("L1", 1.0): 0.999,
            ("L2", 0.0): 0.001,
            ("L2", 1.0): 0.999,
        }
    )
    noise = NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0)
    rng = np.random.default_rng(8102)
    for _ in range(200):
        drawn = draw_logs(rows, model, noise, {"logs": grid}, rng)
        assert (drawn[0].bin_index, drawn[1].bin_index) in {(0, 0), (2, 2)}


def test_average_posterior_date_mass_recovers_the_prior_date_mass() -> None:
    """Law of total expectation on 1,000 cheap draws from a two-date mixture."""
    grid = rounding_grid(0.5)
    item = row(weights=(0.3, 0.7))
    model = prediction({("L1", 0.0): 0.1, ("L1", 1.0): 0.8})
    noise = NoiseTheta(sigma=0.1, rho=0.2, nu=5.0, log_bias=0.0)
    per_date = np.stack(
        [
            np.exp(
                bin_log_probs(
                    grid,
                    mu=float(to_g_space(np.asarray(model.so_support[("L1", time)]))),
                    sigma=0.03,
                    nu=5.0,
                )
            )
            for time in (0.0, 1.0)
        ]
    )
    posterior_mass = []
    rng = np.random.default_rng(7304)
    for _ in range(1_000):
        drawn = draw_logs((item,), model, noise, {"logs": grid}, rng)[0]
        assert drawn.bin_index is not None
        mass = np.asarray([0.3, 0.7]) * per_date[:, drawn.bin_index]
        posterior_mass.append(mass / mass.sum())
    assert np.mean(posterior_mass, axis=0) == pytest.approx([0.3, 0.7], abs=0.05)


@pytest.mark.parametrize(
    "log_terms,weights",
    [
        (np.array([0.0]), np.array([0.5, 0.5])),
        (np.array([0.0, 1.0]), np.array([-0.1, 1.1])),
        (np.array([0.0, 1.0]), np.array([0.2, 0.2])),
        (np.array([math.nan, 0.0]), np.array([0.5, 0.5])),
    ],
)
def test_invalid_date_mixture_is_refused(log_terms: np.ndarray, weights: np.ndarray) -> None:
    with pytest.raises(ValueError):
        log_date_mixture(log_terms, weights)
