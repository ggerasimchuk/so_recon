"""The generator and scorer must be the same recursive discrete law."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from so_recon.inference.contracts import HistoryRow, NoiseTheta
from so_recon.observation.bins import bin_log_probs, rounding_grid
from so_recon.observation.history import conditional_mean, history_loglik
from so_recon.observation.noise import draw_history

OBSERVATION_HASH = "d" * 64


def template(month: int, *, observed: bool = True, reset: bool = False) -> HistoryRow:
    return HistoryRow(
        well_id="P1",
        month_index=month,
        raw_value=0.5 if observed else None,
        bin_index=1 if observed else None,
        quality_group="metered",
        observed_valid=observed,
        reset=reset,
    )


def sequence_log_probability(
    sequence: tuple[int, ...], predictions: tuple[float, ...], noise: NoiseTheta
) -> float:
    grid = rounding_grid(0.5)
    previous: tuple[int, float, float] | None = None
    terms: list[float] = []
    for month, (chosen, predicted) in enumerate(zip(sequence, predictions, strict=True)):
        mu = conditional_mean(predicted, previous, month, noise.rho, False)
        terms.append(float(bin_log_probs(grid, mu=mu, sigma=noise.sigma, nu=noise.nu)[chosen]))
        previous = (month, float(grid.centers[chosen]), predicted)
    return float(sum(terms))


def test_all_three_month_sequences_form_one_joint_probability_law() -> None:
    noise = NoiseTheta(sigma=0.2, rho=0.6, nu=5.0, log_bias=0.0)
    predictions = (0.2, 0.4, 0.7)
    log_probabilities = np.array(
        [
            sequence_log_probability(sequence, predictions, noise)
            for sequence in itertools.product(range(3), repeat=3)
        ]
    )
    assert abs(logsumexp(log_probabilities)) < 1e-10


def test_scorer_matches_the_joint_probability_used_by_the_generator() -> None:
    grid = rounding_grid(0.5)
    noise = NoiseTheta(sigma=0.2, rho=0.6, nu=5.0, log_bias=0.0)
    predictions = {("P1", 0): 0.2, ("P1", 1): 0.4, ("P1", 2): 0.7}
    generated = draw_history(
        (template(0), template(1), template(2)),
        predictions,
        noise,
        {"metered": grid},
        np.random.default_rng(20260915),
    )
    scored = history_loglik(
        generated, predictions, noise, {"metered": grid}, observation_hash=OBSERVATION_HASH
    )
    sequence = tuple(int(row.bin_index) for row in generated if row.bin_index is not None)
    assert scored.value == pytest.approx(sequence_log_probability(sequence, (0.2, 0.4, 0.7), noise))


def test_twenty_thousand_draws_match_the_enumerated_law() -> None:
    grid = rounding_grid(0.5)
    noise = NoiseTheta(sigma=0.2, rho=0.6, nu=5.0, log_bias=0.0)
    predictions = {("P1", 0): 0.2, ("P1", 1): 0.4, ("P1", 2): 0.7}
    rows = (template(0), template(1), template(2))
    sequences = list(itertools.product(range(3), repeat=3))
    expected = np.exp(
        np.array(
            [sequence_log_probability(sequence, (0.2, 0.4, 0.7), noise) for sequence in sequences]
        )
    )
    counts = np.zeros(len(sequences), dtype=np.int64)
    lookup = {sequence: index for index, sequence in enumerate(sequences)}
    rng = np.random.default_rng(4102)
    for _ in range(20_000):
        drawn = draw_history(rows, predictions, noise, {"metered": grid}, rng)
        sequence = tuple(int(row.bin_index) for row in drawn if row.bin_index is not None)
        counts[lookup[sequence]] += 1

    # Deterministic multinomial intervals: six binomial standard errors plus one count.
    tolerance = 6.0 * np.sqrt(expected * (1.0 - expected) / 20_000) + 1.0 / 20_000
    assert np.all(np.abs(counts / 20_000 - expected) <= tolerance)


def test_rho_zero_is_the_product_of_independent_bins() -> None:
    grid = rounding_grid(0.5)
    noise = NoiseTheta(sigma=0.2, rho=0.0, nu=5.0, log_bias=0.0)
    predictions = (0.2, 0.4, 0.7)
    sequence = (2, 0, 1)
    joint = sequence_log_probability(sequence, predictions, noise)
    independent = sum(
        float(
            bin_log_probs(
                grid,
                mu=conditional_mean(predicted, None, month, 0.0, False),
                sigma=noise.sigma,
                nu=noise.nu,
            )[chosen]
        )
        for month, (chosen, predicted) in enumerate(zip(sequence, predictions, strict=True))
    )
    assert joint == pytest.approx(independent)


def test_missing_rows_stay_missing_and_a_reset_still_takes_effect() -> None:
    rows = (template(0), template(1, observed=False, reset=True), template(2))
    predictions = {("P1", 0): 0.2, ("P1", 2): 0.7}
    drawn = draw_history(
        rows,
        predictions,
        NoiseTheta(sigma=0.2, rho=0.8, nu=5.0, log_bias=0.0),
        {"metered": rounding_grid(0.5)},
        np.random.default_rng(44),
    )
    assert drawn[1].raw_value is None
    assert drawn[1].bin_index is None
    assert not drawn[1].observed_valid


def test_a_fixed_rng_seed_reproduces_the_published_history() -> None:
    rows = (template(0), template(1), template(2))
    predictions = {("P1", 0): 0.2, ("P1", 1): 0.4, ("P1", 2): 0.7}
    noise = NoiseTheta(sigma=0.2, rho=0.6, nu=5.0, log_bias=0.0)
    args = (rows, predictions, noise, {"metered": rounding_grid(0.5)})
    left = draw_history(*args, np.random.default_rng(90210))
    right = draw_history(*args, np.random.default_rng(90210))
    assert left == right
