"""Weighted inverse metrics use particle mass and fixed truth support."""

from __future__ import annotations

import numpy as np
import pytest

from so_recon.validation.inverse_metrics import (
    compare_so,
    inventory_quantiles,
    weighted_crps,
    weighted_quantile,
)


def test_weighted_quantile_is_inverse_empirical_cdf() -> None:
    out = weighted_quantile(
        np.array([10.0, 0.0, 10.0]),
        np.array([0.1, 0.8, 0.1]),
        np.array([0.5, 0.9, 0.95]),
    )
    np.testing.assert_allclose(out, [0.0, 10.0, 10.0])


def test_quantile_of_zone_sum_is_not_sum_of_cell_quantiles() -> None:
    inventories = np.array([[0.0, 10.0], [10.0, 0.0]])
    summary = inventory_quantiles(inventories, np.array([0.5, 0.5]), np.array([0.95]))
    assert summary[0] == 10.0
    cellwise_wrong = sum(
        weighted_quantile(inventories[:, cell], np.array([0.5, 0.5]), np.array([0.95]))[0]
        for cell in range(2)
    )
    assert cellwise_wrong == 20.0


def test_weighted_crps_matches_direct_definition() -> None:
    values = np.array([0.0, 2.0])
    weights = np.array([0.75, 0.25])
    assert weighted_crps(values, weights, 1.0) == pytest.approx(0.625)


def test_so_comparison_uses_fixed_truth_pore_volume() -> None:
    metrics = compare_so(
        np.array([0.4, 0.7, 0.2]),
        np.array([0.2, 0.8, np.nan]),
        np.array([1.0, 3.0, 100.0]),
    )
    assert metrics["mae"] == pytest.approx(0.125)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.0175))


def test_missing_estimate_on_valid_truth_support_is_invalid() -> None:
    with pytest.raises(ValueError, match="missing estimate"):
        compare_so(np.array([0.4, np.nan]), np.array([0.2, 0.8]), np.ones(2))
