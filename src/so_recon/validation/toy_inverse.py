"""Cheap analytic targets used to verify the inference engine independently of Julia."""

from __future__ import annotations

import math

from scipy.stats import norm

from so_recon.geology.density import Density
from so_recon.inference.contracts import TargetEvaluation, ThetaRecord
from so_recon.registry.hashing import sha256_json


class GaussianToyTarget:
    """One Gaussian prior and one Gaussian observation with an arbitrary proposal r."""

    def __init__(
        self,
        prior_mean: float,
        prior_variance: float,
        y: float,
        observation_variance: float,
        proposal: Density,
    ) -> None:
        if prior_variance <= 0.0 or observation_variance <= 0.0:
            raise ValueError("Gaussian variances must be positive")
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.y = y
        self.observation_variance = observation_variance
        self.proposal = proposal
        self.fingerprint = sha256_json(
            {
                "kind": "gaussian-toy-target-1",
                "prior_mean": prior_mean,
                "prior_variance": prior_variance,
                "y": y,
                "observation_variance": observation_variance,
                "proposal": proposal.fingerprint,
            }
        )

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
        if len(theta.v) != 1 or theta.z_perp:
            raise ValueError("GaussianToyTarget expects one v coordinate and no residual block")
        x = theta.v[0]
        log_p0 = float(norm.logpdf(x, self.prior_mean, math.sqrt(self.prior_variance)))
        log_l = float(norm.logpdf(self.y, x, math.sqrt(self.observation_variance)))
        return TargetEvaluation(
            theta=theta,
            log_p0=log_p0,
            log_p0_in_support=True,
            log_l=log_l,
            log_l_in_support=True,
            log_r=self.proposal.log_prob(theta),
            log_r_in_support=True,
            forward_ref=None,
            cache_key=sha256_json(
                {"target": self.fingerprint, "theta": theta.model_dump(mode="json")}
            ),
        )


__all__ = ["GaussianToyTarget"]
