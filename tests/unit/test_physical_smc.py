"""Pure contracts for the registered T1/T2/T4 native inference matrix."""

from __future__ import annotations

import numpy as np

from so_recon.validation.physical_smc import (
    convergence_screen,
    report_zone_matrix,
)


def _summary(
    particles: int,
    seed: int,
    medians: list[float],
    widths: list[float],
    family_one: float,
) -> dict[str, object]:
    return {
        "n_particles": particles,
        "seed": seed,
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
        "zone_names": ["layer-0", "layer-1"],
        "zone_medians": medians,
        "zone_widths": widths,
        "family_probabilities": {"0": 1.0 - family_one, "1": family_one},
    }


def test_report_zones_include_layers_quadrants_and_t4_remote_support() -> None:
    common_names, common = report_zone_matrix("e02-t1-v1")
    assert common.shape == (10, 512)
    assert common_names[:2] == ("layer-0", "layer-1")
    np.testing.assert_array_equal(common[:2].sum(axis=1), [256.0, 256.0])

    remote_names, remote = report_zone_matrix("e02-t4-v1")
    assert remote.shape == (13, 512)
    assert remote_names[-3:] == ("remote-east", "remote-east-layer-0", "remote-east-layer-1")
    np.testing.assert_array_equal(remote[-3:].sum(axis=1), [64.0, 32.0, 32.0])


def test_convergence_screen_applies_paired_gaps_and_precision_limits() -> None:
    rows = [
        _summary(32, 11, [0.60, 0.50], [0.20, 0.10], 0.40),
        _summary(32, 12, [0.61, 0.51], [0.21, 0.11], 0.42),
        _summary(64, 11, [0.61, 0.51], [0.18, 0.09], 0.41),
        _summary(64, 12, [0.62, 0.52], [0.19, 0.10], 0.43),
    ]
    result = convergence_screen(rows)
    assert result["status"] == "PASS"
    assert result["checks"] == {
        "complete_beta_one": True,
        "zone_median_gap": True,
        "family_mass_gap": True,
        "zone_width_gap": True,
        "median_precision": True,
        "family_precision": True,
    }


def test_convergence_screen_does_not_widen_threshold_for_noisy_starts() -> None:
    rows = [
        _summary(32, 11, [0.40, 0.50], [0.20, 0.10], 0.20),
        _summary(32, 12, [0.60, 0.51], [0.21, 0.11], 0.60),
        _summary(64, 11, [0.41, 0.51], [0.18, 0.09], 0.21),
        _summary(64, 12, [0.61, 0.52], [0.19, 0.10], 0.61),
    ]
    result = convergence_screen(rows)
    assert result["status"] == "POSTERIOR_NOT_CONVERGED"
    assert result["checks"]["median_precision"] is False
    assert result["checks"]["family_precision"] is False


def test_near_zero_width_uses_absolute_not_relative_gap() -> None:
    rows = [
        _summary(32, 11, [0.6, 0.5], [0.01, 0.0], 0.4),
        _summary(32, 12, [0.6, 0.5], [0.01, 0.0], 0.4),
        _summary(64, 11, [0.6, 0.5], [0.019, 0.009], 0.4),
        _summary(64, 12, [0.6, 0.5], [0.019, 0.009], 0.4),
    ]
    assert convergence_screen(rows)["checks"]["zone_width_gap"] is True
