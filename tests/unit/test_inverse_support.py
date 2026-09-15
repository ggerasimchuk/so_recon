"""Conservative report support keeps geometry validity distinct from empty pore volume."""

from __future__ import annotations

import numpy as np

from so_recon.validation.support import aggregate_state


def test_zone_so_uses_pore_volume() -> None:
    zone = aggregate_state(
        np.array([[1.0, 3.0], [0.0, 0.0]]),
        np.array([0.2, 0.6]),
        np.ones(2),
        np.full(2, 0.2),
    )
    assert np.isclose(zone.so[0], 0.5)
    assert np.isnan(zone.so[1])
    assert zone.n_rem[1] == 0.0
    assert zone.n_above_sor[1] == 0.0
    np.testing.assert_array_equal(zone.valid_geometry, [True, True])


def test_invalid_geometry_is_not_relabelled_as_zero_inventory() -> None:
    zone = aggregate_state(
        np.array([[1.0, np.nan], [1.0, -0.1]]),
        np.array([0.2, 0.6]),
        np.ones(2),
        np.full(2, 0.2),
    )
    np.testing.assert_array_equal(zone.valid_geometry, [False, False])
    assert np.isnan(zone.pv).all()
    assert np.isnan(zone.so).all()
    assert np.isnan(zone.n_rem).all()
    assert np.isnan(zone.n_above_sor).all()


def test_inventory_uses_bo_and_only_counts_oil_above_sor() -> None:
    zone = aggregate_state(
        np.array([[2.0, 2.0]]),
        np.array([0.1, 0.5]),
        np.array([1.0, 2.0]),
        np.array([0.2, 0.2]),
    )
    assert zone.n_rem[0] == 0.7
    assert zone.n_above_sor[0] == 0.3
