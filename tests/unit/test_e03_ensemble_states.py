"""E03 Task 08 — the correct state evaluator (plan §0.2 items 4-5,7, §5.5, §10.2, §9).

The rows this suite pins:

* the §9 evaluator-PV hand example — particles and truth with DIFFERENT phi, so a
  truth-weighted aggregation passed off as physical zonal So produces a different,
  wrong number;
* the posterior MEAN is the primary estimator (the E02 cellwise-median primary is the
  named anti-pattern); median/quantiles stay additional products;
* OW inventory is quantiles of each particle's SUM, never sums of cellwise quantiles
  (the anticorrelated example where the two disagree);
* zero particle PV in a zone: So undefined with an explicit mask, inventory 0, weight
  fraction recorded; a missing estimate where truth PV is POSITIVE is a FAILURE of the
  zone, not an exclusion;
* overlapping zones (the E02 layers-plus-quadrants matrix) never enter one PV-weighted
  primary score; layer aggregates are separately labelled diagnostics;
* malformed weights (negative, non-finite, zero mass, wrong length) are refused;
* the representative particle is an actual particle chosen by the DECLARED rule;
* s probabilities cover every admissible hypothesis;
* map geometry: the eight physical quadrants project onto the grid, and the figure
  axes are the physical extents in metres.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from so_recon.synthetic.loop_designs import t5_support_zone_names
from so_recon.validation.ensemble_states import (
    DEFAULT_QUANTILE_PROBABILITIES,
    REPRESENTATIVE_RULE,
    ZoneSupport,
    aggregate_particle_zones,
    ensemble_state_products,
    layer_diagnostic_support,
    phase_history_fit,
    primary_quadrant_support,
    products_payload,
    score_ensemble_states,
    zone_value_map,
)

# --------------------------------------------------------------------------------------
# fixtures: a two-cell world with one zone, hand-computed everywhere
# --------------------------------------------------------------------------------------


def _support(
    rows: list[list[float]], names: list[str], kind: str = "primary_quadrants"
) -> ZoneSupport:
    return ZoneSupport(names=tuple(names), matrix=np.asarray(rows, dtype=np.float64), kind=kind)


ONE_ZONE = _support([[1.0, 1.0]], ["both-cells"])
TWO_CELL_ZONES = _support([[1.0, 0.0], [0.0, 1.0]], ["left", "right"])

#: The §9 Evaluator PV row: two particles whose own phi differ from each other and from
#: the truth. Every expected number below is hand-computed from the §5.5 formulas.
SO_PARTICLES = np.array([[0.2, 0.6], [0.4, 0.2]])
PV_PARTICLES = np.array([[1.0, 3.0], [3.0, 1.0]])
BO_PARTICLES = np.ones((2, 2))
WEIGHTS = np.array([0.5, 0.5])
SO_TRUTH = np.array([0.5, 0.5])
PV_TRUTH = np.array([1.0, 1.0])


def _nine_row_zones():
    return aggregate_particle_zones(
        so=SO_PARTICLES, pv=PV_PARTICLES, bo=BO_PARTICLES, support=ONE_ZONE
    )


# --------------------------------------------------------------------------------------
# §9: each particle's own PV; truth scored with truth's own PV
# --------------------------------------------------------------------------------------


def test_per_particle_pv_aggregation_matches_the_hand_example() -> None:
    zones = _nine_row_zones()
    # So_0 = (1*0.2 + 3*0.6) / 4 = 0.5 ; So_1 = (3*0.4 + 1*0.2) / 4 = 0.35
    np.testing.assert_allclose(zones.so[:, 0], [0.5, 0.35])
    np.testing.assert_array_equal(zones.so_defined, np.ones((2, 1), dtype=bool))
    # N_i,z = sum_c A_zc PV_i,c So_i,c / Bo_i,c = [2.0, 1.4] with Bo = 1
    np.testing.assert_allclose(zones.inventory[:, 0], [2.0, 1.4])
    np.testing.assert_allclose(zones.zone_pv[:, 0], [4.0, 4.0])


def test_primary_score_uses_each_own_pv_and_catches_truth_weighting() -> None:
    zones = _nine_row_zones()
    products = ensemble_state_products(zones, WEIGHTS, support=ONE_ZONE)
    # So_hat = (0.5 + 0.35) / 2 = 0.425
    assert products.posterior_mean_so[0] == pytest.approx(0.425)
    scores = score_ensemble_states(
        zones,
        WEIGHTS,
        truth_so=SO_TRUTH,
        truth_pv=PV_TRUTH,
        support=ONE_ZONE,
    )
    # So_truth = (1*0.5 + 1*0.5)/2 = 0.5, MAE = |0.425 - 0.5| = 0.075
    assert scores.mae == pytest.approx(0.075)
    assert scores.rmse == pytest.approx(0.075)
    # The E02 anti-pattern (truth PV weighting every particle) would print
    # So_hat = 0.35 and MAE = 0.15: this row is what the test exists to catch.
    assert scores.mae != pytest.approx(0.15)
    assert scores.truth_zone_so[0] == pytest.approx(0.5)
    # CRPS = E|X-truth| - 0.5 E|X-X'| = 0.075 - 0.5*0.075 = 0.0375 (mass-weighted)
    assert scores.crps == pytest.approx(0.0375)


def test_coverage_and_width_are_truth_pv_weighted_over_valid_zones() -> None:
    only = _support([[1.0]], ["only-cell"])
    zones = aggregate_particle_zones(
        so=np.array([[0.35], [0.5]]),
        pv=np.ones((2, 1)),
        bo=np.ones((2, 1)),
        support=only,
    )
    inside = score_ensemble_states(
        zones,
        np.array([0.5, 0.5]),
        truth_so=np.array([0.45]),
        truth_pv=np.array([2.0]),
        support=only,
    )
    # q05 = 0.35 <= 0.45 <= q95 = 0.5 -> covered; width = 0.15
    assert inside.coverage == pytest.approx(1.0)
    assert inside.mean_width == pytest.approx(0.15)
    assert inside.mae == pytest.approx(0.025)
    outside = score_ensemble_states(
        zones,
        np.array([0.5, 0.5]),
        truth_so=np.array([0.6]),
        truth_pv=np.array([2.0]),
        support=only,
    )
    assert outside.coverage == pytest.approx(0.0)


def test_primary_estimator_is_the_posterior_mean_not_the_median() -> None:
    zones = _nine_row_zones()
    products = ensemble_state_products(zones, WEIGHTS, support=ONE_ZONE)
    assert products.estimator == "posterior_mean"
    # median (q50) stays an additional product and differs from the mean here
    assert products.so_quantiles[0, 1] == pytest.approx(0.35)
    assert products.posterior_mean_so[0] == pytest.approx(0.425)
    scores = score_ensemble_states(
        zones, WEIGHTS, truth_so=SO_TRUTH, truth_pv=PV_TRUTH, support=ONE_ZONE
    )
    # a median-primary evaluator would score 0.15
    assert scores.mae == pytest.approx(0.075)


# --------------------------------------------------------------------------------------
# §5.5 inventory: quantiles of per-particle sums
# --------------------------------------------------------------------------------------


def test_inventory_quantiles_are_quantiles_of_particle_sums() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.1, 0.0], [0.0, 0.1]]),
        pv=np.full((2, 2), 100.0),
        bo=np.ones((2, 2)),
        support=ONE_ZONE,
    )
    products = ensemble_state_products(zones, np.array([0.5, 0.5]), support=ONE_ZONE)
    # Each particle's zone sum is 10.0, so every quantile of the sums is 10.0.
    np.testing.assert_allclose(products.inventory_quantiles[0], [10.0, 10.0, 10.0])
    # Summing the per-cell q95 (10.0 + 10.0 = 20.0) is the forbidden alternative.
    assert products.inventory_quantiles[0, 2] != pytest.approx(20.0)


def test_phase_history_fit_is_the_weighted_crps_mean_over_time() -> None:
    predicted = np.array([[0.0, 0.5], [1.0, 1.5]])
    observed = np.array([0.0, 1.5])
    equal = phase_history_fit(predicted, np.array([0.5, 0.5]), observed)
    # each time: first = 0.5, E|X-X'| (mass-weighted) = 0.5 -> crps = 0.5 - 0.25
    assert equal == pytest.approx(0.25)
    tilted = phase_history_fit(predicted, np.array([0.75, 0.25]), observed)
    # time 0: first = .25, pairwise .375 -> 0.0625 ; time 1: first = .75 -> 0.5625
    assert tilted == pytest.approx((0.0625 + 0.5625) / 2.0)


# --------------------------------------------------------------------------------------
# zero PV, masks, and the missing-estimate failure
# --------------------------------------------------------------------------------------


def test_zero_pv_zone_has_undefined_so_zero_inventory_and_a_mask() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]),
        pv=np.array([[1.0, 0.0]]),
        bo=np.ones((1, 2)),
        support=TWO_CELL_ZONES,
    )
    assert zones.so[0, 0] == pytest.approx(0.3)
    assert np.isnan(zones.so[0, 1])
    assert zones.inventory[0, 1] == 0.0
    assert not zones.so_defined[0, 1]
    products = ensemble_state_products(zones, np.array([1.0]), support=TWO_CELL_ZONES)
    assert products.zone_weight_so_defined[0] == pytest.approx(1.0)
    assert products.zone_weight_so_defined[1] == pytest.approx(0.0)
    assert np.isnan(products.posterior_mean_so[1])


def test_mean_renormalises_over_defined_particles_and_records_the_fraction() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.9], [0.6, 0.8]]),
        pv=np.array([[1.0, 0.0], [1.0, 1.0]]),
        bo=np.ones((2, 2)),
        support=TWO_CELL_ZONES,
    )
    products = ensemble_state_products(zones, np.array([0.25, 0.75]), support=TWO_CELL_ZONES)
    # zone 0: both particles defined -> 0.25*0.3 + 0.75*0.6 = 0.525
    assert products.posterior_mean_so[0] == pytest.approx(0.525)
    assert products.zone_weight_so_defined[0] == pytest.approx(1.0)
    # zone 1: only particle 1 has PV -> the mean is that particle's value, renormalised
    assert products.posterior_mean_so[1] == pytest.approx(0.8)
    assert products.zone_weight_so_defined[1] == pytest.approx(0.75)


def test_missing_estimate_on_positive_truth_pv_is_a_failure_not_an_exclusion() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]),
        pv=np.array([[1.0, 0.0]]),
        bo=np.ones((1, 2)),
        support=TWO_CELL_ZONES,
    )
    with pytest.raises(ValueError, match="missing estimate"):
        score_ensemble_states(
            zones,
            np.array([1.0]),
            truth_so=np.array([0.3, 0.8]),
            truth_pv=np.array([1.0, 1.0]),
            support=TWO_CELL_ZONES,
        )


def test_truth_missing_so_on_positive_truth_pv_is_refused() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]),
        pv=np.array([[1.0, 1.0]]),
        bo=np.ones((1, 2)),
        support=TWO_CELL_ZONES,
    )
    with pytest.raises(ValueError, match="truth"):
        score_ensemble_states(
            zones,
            np.array([1.0]),
            truth_so=np.array([0.3, np.nan]),
            truth_pv=np.array([1.0, 1.0]),
            support=TWO_CELL_ZONES,
        )


def test_particle_so_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="so"):
        aggregate_particle_zones(
            so=np.array([[1.4, 0.6]]),
            pv=np.ones((1, 2)),
            bo=np.ones((1, 2)),
            support=ONE_ZONE,
        )


def test_zero_truth_pv_zone_leaves_the_score_and_its_weights_untouched() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]),
        pv=np.array([[1.0, 1.0]]),
        bo=np.ones((1, 2)),
        support=TWO_CELL_ZONES,
    )
    scores = score_ensemble_states(
        zones,
        np.array([1.0]),
        truth_so=np.array([0.3, 0.6]),
        truth_pv=np.array([1.0, 0.0]),
        support=TWO_CELL_ZONES,
    )
    # zone 1 carries zero truth PV: it is out of the score, not silently repaired
    assert scores.mae == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# §0.2 item 7: one primary score never mixes overlapping supports
# --------------------------------------------------------------------------------------


def test_overlapping_primary_support_is_refused() -> None:
    from so_recon.validation.physical_smc import report_zone_matrix

    names, matrix = report_zone_matrix("e02-t1-v1")
    overlapping = ZoneSupport(names=tuple(names), matrix=matrix, kind="primary_quadrants")
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]), pv=np.ones((1, 2)), bo=np.ones((1, 2)), support=ONE_ZONE
    )
    with pytest.raises(ValueError, match="overlap"):
        score_ensemble_states(
            zones,
            np.array([1.0]),
            truth_so=SO_TRUTH,
            truth_pv=PV_TRUTH,
            support=overlapping,
        )


def test_primary_support_is_the_eight_disjoint_physical_quadrants() -> None:
    support = primary_quadrant_support()
    assert support.names == t5_support_zone_names()
    assert support.matrix.shape == (8, 512)
    assert support.kind == "primary_quadrants"
    np.testing.assert_array_equal(support.matrix.sum(axis=0), np.ones(512))


def test_layer_aggregates_are_separately_labelled_diagnostics() -> None:
    layers = layer_diagnostic_support()
    assert layers.names == ("layer-0", "layer-1")
    assert layers.kind == "diagnostic_layer_aggregate"
    assert layers.matrix.shape == (2, 512)
    # a layer aggregate is NOT a primary support
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]), pv=np.ones((1, 2)), bo=np.ones((1, 2)), support=ONE_ZONE
    )
    with pytest.raises(ValueError, match="primary"):
        score_ensemble_states(
            zones,
            np.array([1.0]),
            truth_so=SO_TRUTH,
            truth_pv=PV_TRUTH,
            support=layers,
        )


# --------------------------------------------------------------------------------------
# malformed inputs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights",
    [np.array([-0.5, 1.5]), np.array([np.nan, 1.0]), np.array([np.inf, 1.0]), np.array([0.0, 0.0])],
)
def test_malformed_weights_are_refused(weights: np.ndarray) -> None:
    zones = _nine_row_zones()
    with pytest.raises(ValueError, match="weights"):
        ensemble_state_products(zones, weights, support=ONE_ZONE)


def test_wrong_length_weights_are_refused() -> None:
    zones = _nine_row_zones()
    with pytest.raises(ValueError, match="weights"):
        ensemble_state_products(zones, np.array([1.0]), support=ONE_ZONE)


# --------------------------------------------------------------------------------------
# the representative particle and s probabilities
# --------------------------------------------------------------------------------------


def test_representative_particle_is_an_actual_particle_by_the_declared_rule() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.2, 0.6], [0.5, 0.2]]),
        pv=np.ones((2, 2)),
        bo=np.ones((2, 2)),
        support=TWO_CELL_ZONES,
    )
    weights = np.array([0.25, 0.75])
    products = ensemble_state_products(zones, weights, support=TWO_CELL_ZONES)
    # mean = [0.425, 0.3]; equal zone PV -> equal zone weights
    # d_0 = sqrt(.5*.225^2 + .5*.3^2) = sqrt(0.0703125); d_1 = sqrt(0.0078125)
    assert products.representative_particle_index == 1
    assert products.representative_rule == REPRESENTATIVE_RULE
    np.testing.assert_allclose(products.representative_zone_so, [0.5, 0.2])


def test_s_probabilities_cover_every_admissible_hypothesis() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.2, 0.6], [0.4, 0.2], [0.3, 0.3]]),
        pv=np.ones((3, 2)),
        bo=np.ones((3, 2)),
        support=ONE_ZONE,
    )
    weights = np.array([0.2, 0.2, 0.6])
    labels = (1, 1, 0)
    products = ensemble_state_products(
        zones, weights, support=ONE_ZONE, s_labels=labels, admissible_s=(0, 1)
    )
    assert products.s_probabilities == {"0": 0.6, "1": 0.4}
    wider = ensemble_state_products(
        zones, weights, support=ONE_ZONE, s_labels=labels, admissible_s=(0, 1, 2)
    )
    assert wider.s_probabilities == {"0": 0.6, "1": 0.4, "2": 0.0}
    with pytest.raises(ValueError, match="admissible"):
        ensemble_state_products(
            zones, weights, support=ONE_ZONE, s_labels=(3, 1, 0), admissible_s=(0, 1)
        )


# --------------------------------------------------------------------------------------
# maps: geometry and axes
# --------------------------------------------------------------------------------------


def _grid_products() -> tuple:
    support = primary_quadrant_support()
    n_cells = support.matrix.shape[1]
    rng = np.random.default_rng(41)
    so = np.clip(0.3 + 0.1 * rng.standard_normal((3, n_cells)), 0.05, 0.95)
    zones = aggregate_particle_zones(
        so=so, pv=np.ones((3, n_cells)), bo=np.ones((3, n_cells)), support=support
    )
    products = ensemble_state_products(
        zones, np.full(3, 1.0 / 3.0), support=support, s_labels=(0, 1, 0), admissible_s=(0, 1)
    )
    return support, products


def test_zone_value_map_projects_quadrants_onto_the_grid() -> None:
    support, _products = _grid_products()
    values = np.arange(8, dtype=np.float64)
    grid = zone_value_map(support, values, shape=(16, 16, 2))
    assert grid.shape == (2, 16, 16)
    # t5 order per layer: west-south, east-south, west-north, east-north
    assert grid[0, 2, 3] == values[0]
    assert grid[0, 2, 11] == values[1]
    assert grid[0, 9, 3] == values[2]
    assert grid[0, 9, 11] == values[3]
    assert grid[1, 2, 3] == values[4]
    assert grid[1, 9, 11] == values[7]


def test_zone_value_map_marks_ambiguous_cells_as_missing() -> None:
    support = _support([[1.0, 1.0], [0.0, 1.0]], ["both", "right"])
    grid = zone_value_map(support, np.array([1.0, 2.0]), shape=(1, 1, 2))
    assert grid.shape == (2, 1, 1)
    assert grid[0, 0, 0] == pytest.approx(1.0)  # cell 0 lies in "both" alone
    assert np.isnan(grid[1, 0, 0])  # cell 1 lies in two zones: ambiguous


def test_state_figure_uses_physical_axes_and_declared_geometry(tmp_path: Path) -> None:
    from so_recon.validation.learned_comparison import (
        ensemble_state_figure,
        render_ensemble_state_maps,
    )

    support, products = _grid_products()
    figure = ensemble_state_figure(
        products, support, shape=(16, 16, 2), extent_m=(100.0, 100.0, 20.0), layer=0
    )
    image_axes = [axis for axis in figure.axes if axis.get_images()]
    assert len(image_axes) == 4  # mean, median, width, representative
    for axis in image_axes:
        assert axis.get_xlim() == pytest.approx((0.0, 100.0))
        assert axis.get_ylim() == pytest.approx((0.0, 100.0))
    with_truth = ensemble_state_figure(
        products,
        support,
        shape=(16, 16, 2),
        extent_m=(100.0, 100.0, 20.0),
        layer=0,
        truth_zone_so=np.full(8, 0.5),
    )
    assert len([axis for axis in with_truth.axes if axis.get_images()]) == 6
    paths = render_ensemble_state_maps(
        products, support, shape=(16, 16, 2), extent_m=(100.0, 100.0, 20.0), output_dir=tmp_path
    )
    assert len(paths) == 2
    for path in paths:
        assert path.stat().st_size > 10_000
        image = mpimg.imread(path)
        assert image.shape[0] > 300 and image.shape[1] > 300


# --------------------------------------------------------------------------------------
# the operational payload is truth-free and JSON-safe
# --------------------------------------------------------------------------------------


def test_products_payload_is_json_safe_and_carries_no_truth_field() -> None:
    zones = aggregate_particle_zones(
        so=np.array([[0.3, 0.6]]),
        pv=np.array([[1.0, 0.0]]),
        bo=np.ones((1, 2)),
        support=TWO_CELL_ZONES,
    )
    products = ensemble_state_products(zones, np.array([1.0]), support=TWO_CELL_ZONES)
    payload = products_payload(products)
    import json

    encoded = json.dumps(payload, allow_nan=False)
    assert "truth" not in encoded
    assert payload["zone_weight_so_defined"] == [1.0, 0.0]
    assert payload["posterior_mean_so"][1] is None  # NaN travelled as null
    assert payload["quantile_probabilities"] == list(DEFAULT_QUANTILE_PROBABILITIES)
