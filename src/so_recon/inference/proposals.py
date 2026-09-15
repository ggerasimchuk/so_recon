"""Normalized defensive mixtures for a proposal that cannot lose prior modes."""

from __future__ import annotations

import math

import numpy as np

from so_recon.geology.density import Density
from so_recon.inference.contracts import ThetaRecord
from so_recon.registry.hashing import sha256_json


def mixture_log_prob(log_q: float, log_p0: float, epsilon: float) -> float:
    """Log density of ``(1-epsilon) q + epsilon p0`` without linear underflow."""
    if not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError(f"epsilon must be in (0,1], got {epsilon!r}")
    for label, value in (("log_q", log_q), ("log_p0", log_p0)):
        if math.isnan(value) or value == math.inf:
            raise ValueError(f"{label} must be finite or -inf, got {value!r}")
    if epsilon == 1.0:
        return float(log_p0)
    return float(np.logaddexp(math.log1p(-epsilon) + log_q, math.log(epsilon) + log_p0))


class DefensiveMixture:
    """A proposal with at least ``epsilon`` times the prior mass everywhere."""

    def __init__(self, prior: Density, proposal: Density, epsilon: float) -> None:
        # Validate through the public numerical primitive so constructor and scorer cannot
        # disagree on whether the requested defensive law exists.
        mixture_log_prob(0.0, 0.0, epsilon)
        self.prior = prior
        self.proposal = proposal
        self.epsilon = epsilon
        self._fingerprint = sha256_json(
            {
                "kind": "defensive-mixture-1",
                "prior": prior.fingerprint,
                "proposal": proposal.fingerprint,
                "epsilon": epsilon,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def log_prob(self, theta: ThetaRecord) -> float:
        return mixture_log_prob(
            self.proposal.log_prob(theta), self.prior.log_prob(theta), self.epsilon
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got n={n}")
        if self.epsilon == 1.0:
            return self.prior.sample(n, rng)
        from_prior = rng.random(n) < self.epsilon
        n_prior = int(from_prior.sum())
        n_proposal = n - n_prior
        prior_draws = iter(self.prior.sample(n_prior, rng) if n_prior else ())
        proposal_draws = iter(self.proposal.sample(n_proposal, rng) if n_proposal else ())
        return tuple(
            next(prior_draws) if choose_prior else next(proposal_draws)
            for choose_prior in from_prior
        )


__all__ = ["DefensiveMixture", "mixture_log_prob"]
