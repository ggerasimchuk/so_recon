"""Fixed-seed numeric acceptance for posterior density, evidence and missed modes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from so_recon.inference.weights import normalize_log_weights
from so_recon.validation.toy_inverse import (
    bimodal_reference,
    gaussian_reference,
    toy_suite,
)


@pytest.fixture(scope="module")
def acceptance_runs() -> dict[int, list[dict[str, object]]]:
    # N=64 is the frozen acceptance ensemble.  The first five seeds are paired with N=32
    # for the resolution report; neither list is edited after seeing individual failures.
    seeds = list(range(20))
    return {
        64: [toy_suite(seed, 64) for seed in seeds],
        32: [toy_suite(seed, 32) for seed in seeds[:5]],
    }


def _summary(run: dict[str, object], target: str) -> dict[str, float]:
    value = run[target]
    assert isinstance(value, dict)
    return value  # type: ignore[return-value]


def test_twenty_seed_gaussian_posterior_and_evidence_acceptance(
    acceptance_runs: dict[int, list[dict[str, object]]],
) -> None:
    runs = acceptance_runs[64]
    summaries = [_summary(run, "gaussian") for run in runs]
    mean = float(np.mean([row["mean"] for row in summaries]))
    variance = float(np.mean([row["variance"] + (row["mean"] - mean) ** 2 for row in summaries]))
    evidence = np.asarray([row["evidence"] for row in summaries])
    reference_mean, reference_variance, reference_logz = gaussian_reference(0.0, 1.0, 2.0, 1.0)
    reference_z = math.exp(reference_logz)
    report = {
        "mean": mean,
        "reference_mean": reference_mean,
        "variance": variance,
        "reference_variance": reference_variance,
        "mean_evidence": float(evidence.mean()),
        "reference_evidence": reference_z,
        "evidence_mc_se": float(evidence.std(ddof=1) / math.sqrt(len(evidence))),
        "per_run_failures": [
            run["seed"]
            for run in runs
            if run["algorithm_status"] != {"gaussian": "COMPLETE", "bimodal": "COMPLETE"}
        ],
    }
    assert abs(mean - reference_mean) <= 0.08 * math.sqrt(reference_variance), report
    assert abs(variance / reference_variance - 1.0) <= 0.12, report
    assert abs(evidence.mean() / reference_z - 1.0) <= 0.10, report
    assert not report["per_run_failures"], report


def test_defensive_proposal_recovers_the_family_that_q_never_draws(
    acceptance_runs: dict[int, list[dict[str, object]]],
) -> None:
    runs = acceptance_runs[64]
    family = np.asarray([_summary(run, "bimodal")["family_one_probability"] for run in runs])
    expected = bimodal_reference()["family_probabilities"][1]
    report = {
        "mean_family_one": float(family.mean()),
        "reference": expected,
        "mc_se": float(family.std(ddof=1) / math.sqrt(len(family))),
        "zero_family_runs": [runs[i]["seed"] for i in np.flatnonzero(family == 0.0)],
    }
    assert abs(family.mean() - expected) <= 0.06, report
    assert np.count_nonzero(family > 0.0) >= 18, report


def test_n32_n64_report_uses_paired_unchanged_seeds(
    acceptance_runs: dict[int, list[dict[str, object]]],
) -> None:
    low = acceptance_runs[32]
    high = acceptance_runs[64][:5]
    assert [run["seed"] for run in low] == [run["seed"] for run in high]
    reference_mean, _, _ = gaussian_reference(0.0, 1.0, 2.0, 1.0)
    paired_report = [
        {
            "seed": low_run["seed"],
            "n32_abs_mean_error": abs(_summary(low_run, "gaussian")["mean"] - reference_mean),
            "n64_abs_mean_error": abs(_summary(high_run, "gaussian")["mean"] - reference_mean),
        }
        for low_run, high_run in zip(low, high, strict=True)
    ]
    assert all(np.isfinite(list(row.values())).all() for row in paired_report)


def test_prior_proposal_correction_is_not_likelihood_only_weighting() -> None:
    rng = np.random.default_rng(221)
    particles = rng.normal(3.0, 1.0, size=100_000)
    log_p0 = norm.logpdf(particles, 0.0, 1.0)
    log_l = norm.logpdf(2.0, particles, 1.0)
    log_r = norm.logpdf(particles, 3.0, 1.0)
    correct, _ = normalize_log_weights(log_p0 + log_l - log_r)
    flawed, _ = normalize_log_weights(log_l)
    correct_mean = float(np.exp(correct) @ particles)
    flawed_mean = float(np.exp(flawed) @ particles)
    assert correct_mean == pytest.approx(1.0, abs=0.03)
    assert flawed_mean > 2.3


def test_empty_data_preserves_the_prior_even_when_r_differs() -> None:
    rng = np.random.default_rng(222)
    from_prior = rng.normal(size=80_000)
    uniform = np.full(from_prior.size, -math.log(from_prior.size))
    assert np.exp(uniform) @ from_prior == pytest.approx(0.0, abs=0.02)
    assert np.exp(uniform) @ (from_prior**2) == pytest.approx(1.0, abs=0.03)

    from_q = rng.normal(2.0, math.sqrt(2.0), size=160_000)
    logw = norm.logpdf(from_q) - norm.logpdf(from_q, 2.0, math.sqrt(2.0))
    corrected = logw - logsumexp(logw)
    weights = np.exp(corrected)
    mean = float(weights @ from_q)
    variance = float(weights @ ((from_q - mean) ** 2))
    assert mean == pytest.approx(0.0, abs=0.04)
    assert variance == pytest.approx(1.0, abs=0.06)
