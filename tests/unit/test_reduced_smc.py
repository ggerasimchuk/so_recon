"""Acceptance arithmetic for the four-run reduced native SMC comparison."""

from __future__ import annotations

from so_recon.validation.reduced_smc import compare_reduced_smc_summaries


def _summary(n: int, seed: int, mean: float, quantiles: tuple[float, ...]) -> dict[str, object]:
    return {
        "n_particles": n,
        "seed": seed,
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
        "mean": mean,
        "quantiles": list(quantiles),
        "log_evidence": -7.75,
        "unique_ancestors": n // 2,
    }


def test_reduced_smc_requires_both_seeds_at_both_particle_counts() -> None:
    reference = {"mean": 0.70, "quantiles": [0.48, 0.68, 0.98], "logz": -7.75}
    rows = [
        _summary(32, 11, 0.69, (0.47, 0.67, 0.97)),
        _summary(32, 12, 0.71, (0.49, 0.69, 0.99)),
        _summary(64, 11, 0.695, (0.475, 0.675, 0.975)),
        _summary(64, 12, 0.705, (0.485, 0.685, 0.985)),
    ]

    result = compare_reduced_smc_summaries(rows, reference)

    assert result["status"] == "PASS"
    assert result["checks"] == {"N32": True, "N64": True}
    assert result["by_particles"]["32"]["mean_tolerance"] == 0.05
    assert result["by_particles"]["64"]["quantile_tolerances"] == [0.08] * 3


def test_beta_one_is_not_enough_when_reference_statistics_fail() -> None:
    reference = {"mean": 0.70, "quantiles": [0.48, 0.68, 0.98], "logz": -7.75}
    rows = [
        _summary(n, seed, 1.2, (0.9, 1.1, 1.4))
        for n in (32, 64)
        for seed in (11, 12)
    ]

    result = compare_reduced_smc_summaries(rows, reference)

    assert result["status"] == "FAIL"
    assert result["checks"] == {"N32": False, "N64": False}


def test_reduced_smc_refuses_missing_or_incomplete_run_identity() -> None:
    reference = {"mean": 0.70, "quantiles": [0.48, 0.68, 0.98], "logz": -7.75}
    rows = [
        _summary(32, 11, 0.70, (0.48, 0.68, 0.98)),
        _summary(32, 12, 0.70, (0.48, 0.68, 0.98)),
        _summary(64, 11, 0.70, (0.48, 0.68, 0.98)),
    ]

    try:
        compare_reduced_smc_summaries(rows, reference)
    except ValueError as exc:
        assert "exactly seeds 11/12" in str(exc)
    else:
        raise AssertionError("a missing N64 seed must not be silently accepted")
