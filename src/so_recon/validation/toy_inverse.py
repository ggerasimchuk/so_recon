"""Cheap analytic targets used to verify the inference engine independently of Julia."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import norm

from so_recon.config.inference import InferenceConfig
from so_recon.geology.density import Density
from so_recon.inference.contracts import DensitySchema, SMCState, TargetEvaluation, ThetaRecord
from so_recon.inference.proposals import DefensiveMixture
from so_recon.inference.resampling import systematic_resample
from so_recon.inference.smc import infer
from so_recon.inference.weights import ess, next_beta, normalize_log_weights
from so_recon.registry.hashing import sha256_json

TOY_BASIS_HASH = "7" * 64


def gaussian_reference(m0: float, v0: float, y: float, vy: float) -> tuple[float, float, float]:
    """Analytic posterior mean, variance and log evidence for Gaussian conjugacy."""
    if v0 <= 0.0 or vy <= 0.0:
        raise ValueError("Gaussian variances must be positive")
    variance = 1.0 / (1.0 / v0 + 1.0 / vy)
    mean = variance * (m0 / v0 + y / vy)
    return (
        float(mean),
        float(variance),
        float(norm.logpdf(y, m0, math.sqrt(v0 + vy))),
    )


class GaussianThetaDensity:
    """A normalized one-coordinate Gaussian density on one toy family."""

    def __init__(self, schema: DensitySchema, mean: float, variance: float) -> None:
        if variance <= 0.0 or schema.n_v != 1 or schema.n_residual != 0:
            raise ValueError("GaussianThetaDensity needs variance>0 and a (1,0) schema")
        if len(schema.families) != 1:
            raise ValueError("GaussianThetaDensity has exactly one family")
        self.schema = schema
        self.mean = mean
        self.variance = variance
        self.fingerprint = sha256_json(
            {
                "kind": "gaussian-theta-density-1",
                "schema": schema.model_dump(mode="json"),
                "mean": mean,
                "variance": variance,
            }
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got {n}")
        values = rng.normal(self.mean, math.sqrt(self.variance), size=n)
        family = self.schema.families[0]
        return tuple(
            ThetaRecord(
                schema_id=self.schema.schema_id,
                s=family,
                v=(float(value),),
                z_perp=(),
                basis_hash=self.schema.basis_hash,
            )
            for value in values
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        self.schema.validate_theta(theta)
        return float(norm.logpdf(theta.v[0], self.mean, math.sqrt(self.variance)))


class BimodalPrior:
    """The normalized two-family prior from the disconnected-target acceptance case."""

    FAMILY_PROBABILITIES = (0.7, 0.3)
    FAMILY_MEANS = (-3.0, 3.0)
    CONDITIONAL_VARIANCE = 0.25

    def __init__(self, schema: DensitySchema) -> None:
        if schema.n_v != 1 or schema.n_residual != 0 or schema.families != (0, 1):
            raise ValueError("BimodalPrior requires the two-family (1,0) toy schema")
        self.schema = schema
        self.fingerprint = sha256_json(
            {"kind": "bimodal-prior-1", "schema": schema.model_dump(mode="json")}
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got {n}")
        families = rng.choice(2, size=n, p=self.FAMILY_PROBABILITIES)
        values = rng.normal(
            np.asarray(self.FAMILY_MEANS)[families],
            math.sqrt(self.CONDITIONAL_VARIANCE),
        )
        return tuple(
            ThetaRecord(
                schema_id=self.schema.schema_id,
                s=int(family),
                v=(float(value),),
                z_perp=(),
                basis_hash=self.schema.basis_hash,
            )
            for family, value in zip(families, values, strict=True)
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        self.schema.validate_theta(theta)
        return float(
            math.log(self.FAMILY_PROBABILITIES[theta.s])
            + norm.logpdf(
                theta.v[0],
                self.FAMILY_MEANS[theta.s],
                math.sqrt(self.CONDITIONAL_VARIANCE),
            )
        )


class MissedModeProposal:
    """A normalized q that has the correct continuous law in s=0 and zero mass in s=1."""

    def __init__(self, prior: BimodalPrior) -> None:
        self.prior = prior
        self.fingerprint = sha256_json(
            {"kind": "missed-mode-proposal-1", "prior": prior.fingerprint}
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got {n}")
        values = rng.normal(
            self.prior.FAMILY_MEANS[0],
            math.sqrt(self.prior.CONDITIONAL_VARIANCE),
            size=n,
        )
        return tuple(
            ThetaRecord(
                schema_id=self.prior.schema.schema_id,
                s=0,
                v=(float(value),),
                z_perp=(),
                basis_hash=self.prior.schema.basis_hash,
            )
            for value in values
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        self.prior.schema.validate_theta(theta)
        if theta.s == 1:
            return -math.inf
        return float(
            norm.logpdf(
                theta.v[0],
                self.prior.FAMILY_MEANS[0],
                math.sqrt(self.prior.CONDITIONAL_VARIANCE),
            )
        )


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


class BimodalToyTarget:
    """Disconnected normalized prior with a Gaussian observation shared by both families."""

    def __init__(
        self,
        prior: BimodalPrior,
        proposal: Density,
        *,
        y: float = 0.0,
        observation_variance: float = 4.0,
    ) -> None:
        if observation_variance <= 0.0:
            raise ValueError("observation variance must be positive")
        self.prior = prior
        self.proposal = proposal
        self.y = y
        self.observation_variance = observation_variance
        self.fingerprint = sha256_json(
            {
                "kind": "bimodal-toy-target-1",
                "prior": prior.fingerprint,
                "proposal": proposal.fingerprint,
                "y": y,
                "observation_variance": observation_variance,
            }
        )

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
        log_p0 = self.prior.log_prob(theta)
        log_l = float(norm.logpdf(self.y, theta.v[0], math.sqrt(self.observation_variance)))
        log_r = self.proposal.log_prob(theta)
        return TargetEvaluation(
            theta=theta,
            log_p0=log_p0,
            log_p0_in_support=True,
            log_l=log_l,
            log_l_in_support=True,
            log_r=log_r,
            log_r_in_support=log_r != -math.inf,
            forward_ref=None,
            cache_key=sha256_json(
                {"target": self.fingerprint, "theta": theta.model_dump(mode="json")}
            ),
        )


def bimodal_reference(*, y: float = 0.0, observation_variance: float = 4.0) -> dict[str, object]:
    """Analytic family masses, conditional moments and evidence for BimodalToyTarget."""
    prior_probabilities = np.asarray(BimodalPrior.FAMILY_PROBABILITIES)
    means = np.asarray(BimodalPrior.FAMILY_MEANS)
    prior_variance = BimodalPrior.CONDITIONAL_VARIANCE
    evidence_by_family = norm.pdf(y, means, math.sqrt(prior_variance + observation_variance))
    masses = prior_probabilities * evidence_by_family
    evidence = float(masses.sum())
    family_probabilities = masses / evidence
    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / observation_variance)
    posterior_means = posterior_variance * (means / prior_variance + y / observation_variance)
    return {
        "family_probabilities": tuple(float(value) for value in family_probabilities),
        "conditional_means": tuple(float(value) for value in posterior_means),
        "conditional_variance": float(posterior_variance),
        "evidence": evidence,
    }


def _weighted_summary(state: SMCState) -> dict[str, float]:
    weights = np.exp(np.asarray(state.log_weights, dtype=np.float64))
    values = np.asarray(
        [particle.evaluation.theta.v[0] for particle in state.particles],
        dtype=np.float64,
    )
    mean = float(weights @ values)
    variance = float(weights @ ((values - mean) ** 2))
    family_one = float(
        weights
        @ np.asarray(
            [particle.evaluation.theta.s == 1 for particle in state.particles],
            dtype=np.float64,
        )
    )
    return {
        "mean": mean,
        "variance": variance,
        "evidence": math.exp(state.log_evidence),
        "family_one_probability": family_one,
        "beta": state.beta,
        "unique_ancestors": len({particle.ancestor_id for particle in state.particles}),
    }


def toy_suite(seed: int, n_particles: int) -> dict[str, object]:
    """Run one Gaussian and one missed-mode acceptance case with fixed settings."""
    gaussian_schema = DensitySchema(
        schema_id="gaussian-toy-1",
        n_v=1,
        n_residual=0,
        families=(0,),
        basis_hash=TOY_BASIS_HASH,
        transform_version="identity-1",
    )
    gaussian_q = GaussianThetaDensity(gaussian_schema, -0.5, 2.0)
    gaussian_target = GaussianToyTarget(0.0, 1.0, 2.0, 1.0, gaussian_q)
    config = InferenceConfig(
        n_particles=n_particles,
        cess_fraction=0.8,
        resample_fraction=0.5,
        max_beta_steps=24,
        moves_per_level=2,
        seed=seed,
        rw_scale=0.8,
        pcn_scale=0.2,
    )
    bimodal_schema = gaussian_schema.model_copy(
        update={"schema_id": "bimodal-toy-1", "families": (0, 1)}
    )
    bimodal_prior = BimodalPrior(bimodal_schema)
    missed = MissedModeProposal(bimodal_prior)
    defensive = DefensiveMixture(bimodal_prior, missed, epsilon=0.2)
    bimodal_target = BimodalToyTarget(bimodal_prior, defensive)
    with TemporaryDirectory(prefix="so-recon-toy-") as directory:
        root = Path(directory)
        gaussian_state = infer(
            gaussian_target,
            gaussian_q,
            gaussian_schema,
            config,
            root / "gaussian",
            lambda: False,
        )
        bimodal_state = infer(
            bimodal_target,
            defensive,
            bimodal_schema,
            config.model_copy(update={"seed": seed + 1}),
            root / "bimodal",
            lambda: False,
        )
    return {
        "seed": seed,
        "n_particles": n_particles,
        "gaussian": _weighted_summary(gaussian_state),
        "bimodal": _weighted_summary(bimodal_state),
        "algorithm_status": {
            "gaussian": gaussian_state.algorithm_status,
            "bimodal": bimodal_state.algorithm_status,
        },
    }


def _randomized_rank(truth: float, draws: np.ndarray, rng: np.random.Generator) -> int:
    below = int(np.sum(draws < truth))
    tied = int(np.sum(draws == truth))
    return below + (int(rng.integers(0, tied + 1)) if tied else 0)


def _wilson(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson inputs require 0 <= successes <= trials and trials > 0")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(center - half), float(center + half)


def _randomized_central_interval(
    draws: np.ndarray,
    rng: np.random.Generator,
    *,
    nominal: float = 0.9,
) -> tuple[float, float]:
    """Randomize adjacent central order-statistic intervals to exact nominal coverage.

    For ``n`` independent posterior draws, the truth rank is uniform on ``0..n`` under
    prior-predictive repetition.  A central interval dropping ``k`` order statistics from
    each tail therefore has exact coverage ``(n-1-2k)/(n+1)``.  Usually no integer ``k``
    equals 0.9, so this selects between the adjacent two before seeing the truth.
    """
    ordered = np.sort(np.asarray(draws, dtype=np.float64))
    n = ordered.size
    if n < 2 or not 0.0 < nominal < 1.0:
        raise ValueError("a randomized central interval needs n>=2 and nominal in (0,1)")
    coverages = np.asarray([(n - 1 - 2 * k) / (n + 1) for k in range(n // 2)], dtype=np.float64)
    above = np.flatnonzero(coverages >= nominal)
    below = np.flatnonzero(coverages <= nominal)
    if not above.size:
        k = 0
    elif not below.size:
        k = int(above[-1])
    else:
        k_high = int(above[-1])
        k_low = int(below[0])
        high = float(coverages[k_high])
        low = float(coverages[k_low])
        if high == low:
            k = k_high
        else:
            choose_high = (nominal - low) / (high - low)
            k = k_high if rng.random() < choose_high else k_low
    return float(ordered[k]), float(ordered[-1 - k])


def _tempered_gaussian_draws(y: float, n_particles: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap vectorized SMC oracle using the same declared beta/ESS/RW rules as the engine."""
    particles = rng.normal(size=n_particles)
    log_weights = np.full(n_particles, -math.log(n_particles))
    log_l = np.asarray(norm.logpdf(y, particles, 1.0), dtype=np.float64)
    beta = 0.0
    while beta < 1.0:
        new_beta = next_beta(beta, log_weights, log_l, 0.8)
        log_weights, _ = normalize_log_weights(log_weights + (new_beta - beta) * log_l)
        beta = new_beta
        if ess(log_weights) < 0.5 * n_particles:
            indices = systematic_resample(log_weights, rng)
            particles = particles[indices]
            log_l = log_l[indices]
            log_weights = np.full(n_particles, -math.log(n_particles))

        # Two fixed, non-adaptive RW sweeps, matching the E02 default.  This is deliberately
        # implemented separately
        # from the phase machine: the repeated calibration is an independent numerical
        # acceptance route for the density, while Task 9 tests exact engine state/resume.
        for _ in range(2):
            for index in range(n_particles):
                proposed = float(particles[index] + 0.8 * rng.normal())
                proposed_log_l = float(norm.logpdf(y, proposed, 1.0))
                log_alpha = min(
                    0.0,
                    float(norm.logpdf(proposed))
                    + beta * proposed_log_l
                    - float(norm.logpdf(particles[index]))
                    - beta * float(log_l[index]),
                )
                if math.log(float(rng.random())) < log_alpha:
                    particles[index] = proposed
                    log_l[index] = proposed_log_l
    return particles[systematic_resample(log_weights, rng)]


def prior_predictive_calibration(
    seed: int,
    *,
    repetitions: int = 200,
    n_particles: int = 32,
) -> dict[str, object]:
    """Repeated Gaussian simulation-based calibration plus an independent exact control.

    The SMC side uses fixed CESS=0.8N, ESS=0.5N and two non-adaptive RW sweeps per level.
    Its 32 final systematic descendants share an ensemble and ancestry, so their rank
    histogram is reported as a diagnostic, not mislabelled as an independent-draw proof.
    The exact conjugate draws are the independent reference control.
    """
    if repetitions < 1 or n_particles < 2:
        raise ValueError("calibration needs repetitions>0 and at least two particles")
    rng = np.random.default_rng(seed)
    smc_ranks: list[int] = []
    exact_ranks: list[int] = []
    smc_covered = 0
    exact_covered = 0
    for _ in range(repetitions):
        truth = float(rng.normal())
        y = float(rng.normal(truth, 1.0))
        smc_draws = _tempered_gaussian_draws(y, n_particles, rng)

        mean, variance, _ = gaussian_reference(0.0, 1.0, y, 1.0)
        exact_draws = rng.normal(mean, math.sqrt(variance), size=n_particles)
        smc_ranks.append(_randomized_rank(truth, smc_draws, rng))
        exact_ranks.append(_randomized_rank(truth, exact_draws, rng))
        smc_interval = _randomized_central_interval(smc_draws, rng)
        exact_interval = _randomized_central_interval(exact_draws, rng)
        smc_covered += int(smc_interval[0] <= truth <= smc_interval[1])
        exact_covered += int(exact_interval[0] <= truth <= exact_interval[1])
    return {
        "repetitions": repetitions,
        "posterior_draws_per_repetition": n_particles,
        "smc_ranks": smc_ranks,
        "exact_ranks": exact_ranks,
        "smc_coverage": smc_covered / repetitions,
        "exact_coverage": exact_covered / repetitions,
        "smc_coverage_wilson_95": _wilson(smc_covered, repetitions),
        "exact_coverage_wilson_95": _wilson(exact_covered, repetitions),
        "smc_draw_disclosure": "systematic descendants share ensemble and ancestry",
        "exact_draw_disclosure": "independent analytic Gaussian posterior draws",
    }


__all__ = [
    "BimodalPrior",
    "BimodalToyTarget",
    "GaussianThetaDensity",
    "GaussianToyTarget",
    "MissedModeProposal",
    "bimodal_reference",
    "gaussian_reference",
    "prior_predictive_calibration",
    "toy_suite",
]
