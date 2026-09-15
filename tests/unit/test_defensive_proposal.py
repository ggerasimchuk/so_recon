"""A defensive proposal keeps every prior mode in support and stays normalized."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so_recon.inference.contracts import ThetaRecord
from so_recon.inference.proposals import DefensiveMixture, mixture_log_prob
from so_recon.registry.hashing import sha256_json


class DiscreteDensity:
    def __init__(self, probabilities: tuple[float, ...], *, name: str) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.fingerprint = sha256_json({"name": name, "p": probabilities})

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        choices = rng.choice(len(self.probabilities), size=n, p=self.probabilities)
        return tuple(theta(int(choice)) for choice in choices)

    def log_prob(self, value: ThetaRecord) -> float:
        probability = float(self.probabilities[value.s])
        return -math.inf if probability == 0.0 else math.log(probability)


def theta(state: int) -> ThetaRecord:
    return ThetaRecord(
        schema_id="discrete",
        s=state,
        v=(0.0,),
        z_perp=(),
        basis_hash="a" * 64,
    )


def test_mixture_log_prob_has_exact_endpoint_and_refuses_no_defence() -> None:
    assert mixture_log_prob(math.log(0.1), math.log(0.4), 0.25) == pytest.approx(math.log(0.175))
    assert mixture_log_prob(-math.inf, math.log(0.4), 1.0) == pytest.approx(math.log(0.4))
    with pytest.raises(ValueError, match="epsilon"):
        mixture_log_prob(0.0, 0.0, 0.0)


def test_a_proposal_that_misses_modes_is_repaired_and_integrates_to_one() -> None:
    prior = DiscreteDensity((0.2, 0.3, 0.5), name="prior")
    missed_mode_q = DiscreteDensity((1.0, 0.0, 0.0), name="q")
    mixture = DefensiveMixture(prior, missed_mode_q, epsilon=0.2)
    actual = np.exp([mixture.log_prob(theta(i)) for i in range(3)])
    np.testing.assert_allclose(actual, [0.84, 0.06, 0.10], atol=1e-15)
    assert actual.sum() == pytest.approx(1.0)
    assert np.all(actual > 0.0)


def test_defensive_draw_frequencies_match_the_mixture_density() -> None:
    prior = DiscreteDensity((0.2, 0.3, 0.5), name="prior")
    q = DiscreteDensity((1.0, 0.0, 0.0), name="q")
    draws = DefensiveMixture(prior, q, epsilon=0.2).sample(30_000, np.random.default_rng(93))
    frequency = np.bincount([draw.s for draw in draws], minlength=3) / len(draws)
    np.testing.assert_allclose(frequency, [0.84, 0.06, 0.10], atol=0.01)


def test_epsilon_one_delegates_exactly_to_the_prior() -> None:
    prior = DiscreteDensity((0.2, 0.3, 0.5), name="prior")
    q = DiscreteDensity((1.0, 0.0, 0.0), name="q")
    defensive = DefensiveMixture(prior, q, epsilon=1.0)
    assert defensive.log_prob(theta(2)) == prior.log_prob(theta(2))
    assert defensive.sample(20, np.random.default_rng(7)) == prior.sample(
        20, np.random.default_rng(7)
    )


def test_invalid_sample_size_is_refused_before_component_draws() -> None:
    prior = DiscreteDensity((0.2, 0.3, 0.5), name="prior")
    q = DiscreteDensity((1.0, 0.0, 0.0), name="q")
    with pytest.raises(ValueError, match="at least one"):
        DefensiveMixture(prior, q, epsilon=0.2).sample(0, np.random.default_rng(1))
