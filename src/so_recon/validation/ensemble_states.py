"""E03 ensemble state aggregation and metrics on the declared report support.

Plan §0.2 items 4-5 name the two E02 defects this module exists to not repeat:

* `_state_summary` aggregated EVERY particle with the TRUTH pore volume. Here each
  particle is aggregated with its OWN PV (`So_i,z = sum_c A_zc PV_i,c So_i,c /
  sum_c A_zc PV_i,c`, plan §5.5) and the truth is aggregated with truth's own PV;
  truth PV returns only as the WEIGHT of the error score
  (`MAE = sum_z PV_truth,z |So_hat_z - So_truth,z| / sum_z PV_truth,z`).
* the primary error was a cellwise median. Here the posterior MEAN
  (`So_hat_z = sum_i W_i So_i,z`) is the frozen primary estimator; median/quantiles
  are additional products.

The primary support is the eight disjoint per-layer quadrants in physical coordinates
(the T5 declaration, reused — there is no second zone system). Overlapping zones
(the E02 layers-plus-quadrants matrix) never enter one PV-weighted primary score
(§0.2 item 7); layer aggregates exist only as separately labelled diagnostics.

OW inventory is reported in surface units as `N_remaining_i,z = sum_c A_zc PV_i,c
So_i,c / Bo_i,c` — a per-particle SUM first, then weighted quantiles of the sum
distribution; cellwise quantiles are never summed. Inventory is NOT recoverable
reserves. Zero particle PV in a zone: So is undefined (NaN, explicit mask),
inventory is 0, and the defined weight fraction is recorded. A missing estimate on
a zone where the TRUTH PV is positive is a FAILURE of the zone, never an exclusion.

Truth enters this module in `score_ensemble_states` and `phase_history_fit` only:
every other function builds operational products without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from so_recon.inference.contracts import F64
from so_recon.synthetic.loop_designs import t5_quadrant_masks
from so_recon.validation.inverse_metrics import (
    _normalised_weights,
    compare_so,
    weighted_crps,
    weighted_quantile,
)

#: Frozen by plan §10.2: Q05/Q50/Q95 travel with every ensemble product.
DEFAULT_QUANTILE_PROBABILITIES = (0.05, 0.5, 0.95)

#: The declared representative rule (plan §10.2): among ACTUAL particles, the one
#: whose zonal So vector is nearest the posterior mean in the PV-weighted RMS sense,
#: zone weights being the ensemble-mean zonal PV. An averaged state is never drawn.
REPRESENTATIVE_RULE = "nearest_to_posterior_mean_by_mean_zone_pv"

PRIMARY_SUPPORT_KIND = "primary_quadrants"
DIAGNOSTIC_LAYER_KIND = "diagnostic_layer_aggregate"

#: The E03 inference grid (16x16x2, 100x100x20 m): the grid the primary quadrants are
#: declared on in physical coordinates, so coarse inference and the 48x48x2 fine truth
#: share one definition.
INFERENCE_GRID_SHAPE = (16, 16, 2)
INFERENCE_GRID_EXTENT_M = (100.0, 100.0, 20.0)


@dataclass(frozen=True)
class ZoneSupport:
    """One declared report support: names, cell membership and its role."""

    names: tuple[str, ...]
    matrix: F64
    kind: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                f"zone matrix must be a non-empty (zone, cell) array, got {matrix.shape}"
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError("zone matrix must be finite and non-negative")
        if len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError("zone names must be non-empty and unique")
        if len(self.names) != matrix.shape[0]:
            raise ValueError(f"{len(self.names)} names for {matrix.shape[0]} zone rows")
        object.__setattr__(self, "matrix", matrix)


def check_support_disjoint(support: ZoneSupport) -> None:
    """Refuse overlapping zones: no cell may sit in two zones of one support (§0.2 item 7)."""
    coverage = support.matrix.sum(axis=0)
    if np.any(coverage > 1.0 + 1.0e-12):
        overlapping = np.flatnonzero(coverage > 1.0 + 1.0e-12)
        raise ValueError(
            f"support of kind {support.kind!r} overlaps on {overlapping.size} cells "
            f"(first at cell {int(overlapping[0])}): overlapping zones cannot enter one "
            "PV-weighted primary score"
        )


def primary_quadrant_support(
    shape: tuple[int, int, int] = INFERENCE_GRID_SHAPE,
    extent_m: tuple[float, float, float] = INFERENCE_GRID_EXTENT_M,
) -> ZoneSupport:
    """The eight disjoint per-layer quadrants in physical coordinates (plan §10.2).

    Reuses the T5 declaration (`t5_quadrant_masks`) verbatim: the same definition
    covers the 16x16x2 inference grid and the 48x48x2 fine truth, and the masks are
    disjoint by construction. There is no second zone system.
    """
    names, matrix = t5_quadrant_masks(shape, extent_m)
    support = ZoneSupport(names=tuple(names), matrix=matrix, kind=PRIMARY_SUPPORT_KIND)
    check_support_disjoint(support)
    return support


def layer_diagnostic_support(
    shape: tuple[int, int, int] = INFERENCE_GRID_SHAPE,
    extent_m: tuple[float, float, float] = INFERENCE_GRID_EXTENT_M,
) -> ZoneSupport:
    """Whole-layer aggregates, built as unions of the primary quadrants.

    A separately labelled DIAGNOSTIC (plan §10.2): layer rows are never re-counted in
    the primary score, and this support is refused wherever a primary is required.
    """
    quadrants = primary_quadrant_support(shape, extent_m)
    per_layer = shape[2]
    matrix = np.stack(
        [quadrants.matrix[layer * 4 : (layer + 1) * 4].sum(axis=0) for layer in range(per_layer)]
    )
    names = tuple(f"layer-{layer}" for layer in range(per_layer))
    return ZoneSupport(names=names, matrix=matrix, kind=DIAGNOSTIC_LAYER_KIND)


@dataclass(frozen=True)
class ParticleZones:
    """Per-particle zonal aggregates; every array is (particle, zone)."""

    so: F64
    inventory: F64
    zone_pv: F64
    so_defined: np.ndarray


def _particle_cells(values: F64, n_particles: int, n_cells: int, label: str) -> F64:
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (n_particles, n_cells):
        raise ValueError(f"{label} must have shape ({n_particles}, {n_cells}), got {out.shape}")
    return out


def aggregate_particle_zones(
    *,
    so: F64,
    pv: F64,
    bo: F64,
    support: ZoneSupport,
) -> ParticleZones:
    """Aggregate each particle onto the support with ITS OWN pore volume (plan §5.5).

    `So_i,z = sum_c A_zc PV_i,c So_i,c / sum_c A_zc PV_i,c` per particle; a zone with
    zero particle PV leaves So undefined (NaN, explicit mask) and inventory exactly 0.
    """
    n_zones, n_cells = support.matrix.shape
    if so.ndim != 2:
        raise ValueError(f"so must be a (particle, cell) matrix, got {so.shape}")
    n_particles = so.shape[0]
    oil = _particle_cells(so, n_particles, n_cells, label="so")
    pore = _particle_cells(pv, n_particles, n_cells, label="pv")
    formation_volume = _particle_cells(bo, n_particles, n_cells, label="bo")
    if not np.isfinite(oil).all() or np.any((oil < -1.0e-12) | (oil > 1.0 + 1.0e-12)):
        raise ValueError("so must be finite and inside [0, 1] on every cell")
    if not np.isfinite(pore).all() or np.any(pore < 0.0):
        raise ValueError("pv must be finite and non-negative on every cell")
    if not np.isfinite(formation_volume).all():
        raise ValueError("bo must be finite on every cell")
    active = pore > 0.0
    if np.any(~(formation_volume > 0.0) & active):
        raise ValueError("bo must be positive on every cell with positive pv")

    matrix = support.matrix
    zone_pv = pore @ matrix.T
    numerator = (pore * oil) @ matrix.T
    defined = zone_pv > 0.0
    zone_so = np.full_like(zone_pv, np.nan)
    np.divide(numerator, zone_pv, out=zone_so, where=defined)
    # PV=0 contributes exactly zero to the inventory regardless of Bo on that cell.
    cell_inventory = np.zeros_like(oil)
    np.divide(pore * oil, formation_volume, out=cell_inventory, where=active)
    inventory = cell_inventory @ matrix.T
    inventory[~defined] = 0.0
    return ParticleZones(so=zone_so, inventory=inventory, zone_pv=zone_pv, so_defined=defined)


@dataclass(frozen=True)
class EnsembleStateProducts:
    """Truth-free operational products on one support at one report time."""

    zone_names: tuple[str, ...]
    support_kind: str
    estimator: str
    posterior_mean_so: F64
    so_quantiles: F64
    inventory_quantiles: F64
    zone_weight_so_defined: F64
    mean_zone_pv: F64
    representative_particle_index: int
    representative_rule: str
    representative_zone_so: F64
    s_probabilities: dict[str, float]
    quantile_probabilities: tuple[float, ...]
    n_particles: int


def _zone_weighted_quantiles(
    zones: ParticleZones,
    weights: F64,
    probabilities: tuple[float, ...],
) -> tuple[F64, F64, F64, F64]:
    """Per-zone mean/quantiles over defined particles, and quantiles of the sums."""
    n_zones = zones.so.shape[1]
    probs = np.asarray(probabilities, dtype=np.float64)
    mean = np.full(n_zones, np.nan)
    defined_weight = np.zeros(n_zones)
    quantiles = np.full((n_zones, probs.size), np.nan)
    inventory = np.full((n_zones, probs.size), np.nan)
    for zone in range(n_zones):
        defined = zones.so_defined[:, zone]
        mass = weights[defined]
        total = float(mass.sum())
        defined_weight[zone] = total
        if total <= 0.0:
            continue
        values = zones.so[defined, zone]
        mean[zone] = float(mass @ values / total)
        quantiles[zone] = weighted_quantile(values, mass, probs)
        # §5.5: quantiles of each particle's summed inventory, never sums of cell
        # quantiles — inventory is defined for every particle (0 where PV is 0).
        inventory[zone] = weighted_quantile(zones.inventory[:, zone], weights, probs)
    return mean, quantiles, inventory, defined_weight


def representative_particle(zones: ParticleZones, weights: F64, posterior_mean: F64) -> int:
    """The DECLARED rule: the actual particle nearest the posterior mean.

    Distance is the RMS gap over zones weighted by the ensemble-mean zonal PV (an
    operational quantity — no truth enters the rule), restricted to zones where the
    mean exists. Ties break to the lowest particle index.
    """
    zone_weight = zones.zone_pv.T @ weights
    usable = np.isfinite(posterior_mean) & (zone_weight > 0.0)
    if not usable.any():
        raise ValueError("no zone carries a defined posterior mean: no representative")
    share = zone_weight[usable] / zone_weight[usable].sum()
    gaps = zones.so[:, usable] - posterior_mean[usable]
    with np.errstate(invalid="ignore"):
        distance = np.sqrt(np.asarray(share) @ (gaps**2).T)
    # A particle undefined in some usable zone has no distance; it is not eligible,
    # and if no particle is defined everywhere the mean is, the rule has no winner.
    finite = np.isfinite(distance)
    if not finite.any():
        raise ValueError("no particle is defined on every zone with a defined mean")
    return int(np.argmin(np.where(finite, distance, np.inf)))


def ensemble_state_products(
    zones: ParticleZones,
    weights: F64,
    *,
    support: ZoneSupport,
    s_labels: Sequence[int] | None = None,
    admissible_s: Sequence[int] | None = None,
    probabilities: tuple[float, ...] = DEFAULT_QUANTILE_PROBABILITIES,
) -> EnsembleStateProducts:
    """Posterior-mean primary products; truth is not an input of this function."""
    n_particles = zones.so.shape[0]
    if zones.so.shape[1] != support.matrix.shape[0]:
        raise ValueError(
            f"{zones.so.shape[1]} aggregated zones for a support of {support.matrix.shape[0]}"
        )
    mass = _normalised_weights(weights, n_particles)
    mean, quantiles, inventory, defined_weight = _zone_weighted_quantiles(
        zones, mass, probabilities
    )
    index = representative_particle(zones, mass, mean)
    return EnsembleStateProducts(
        zone_names=support.names,
        support_kind=support.kind,
        estimator="posterior_mean",
        posterior_mean_so=mean,
        so_quantiles=quantiles,
        inventory_quantiles=inventory,
        zone_weight_so_defined=defined_weight,
        mean_zone_pv=zones.zone_pv.T @ mass,
        representative_particle_index=index,
        representative_rule=REPRESENTATIVE_RULE,
        representative_zone_so=zones.so[index],
        s_probabilities=_s_probabilities(s_labels, admissible_s, mass),
        quantile_probabilities=tuple(float(p) for p in probabilities),
        n_particles=n_particles,
    )


def _s_probabilities(
    s_labels: Sequence[int] | None,
    admissible_s: Sequence[int] | None,
    weights: F64,
) -> dict[str, float]:
    """Particle-weight probability of every admissible s (plan §10.2)."""
    if s_labels is None:
        return {}
    labels = [int(label) for label in s_labels]
    if len(labels) != weights.size:
        raise ValueError(
            f"{len(labels)} s labels for {weights.size} particles: one label per particle"
        )
    admissible = (
        sorted(int(value) for value in admissible_s) if admissible_s else sorted(set(labels))
    )
    outside = sorted(set(labels) - set(admissible))
    if outside:
        raise ValueError(f"s labels {outside} are not among the admissible hypotheses {admissible}")
    return {str(value): float(weights[np.asarray(labels) == value].sum()) for value in admissible}


@dataclass(frozen=True)
class EnsembleStateScores:
    """Truth-conditional scores of one ensemble on one support (plan §10.2)."""

    mae: float
    rmse: float
    crps: float
    coverage: float
    mean_width: float
    zone_error: F64
    truth_zone_so: F64
    truth_zone_pv: F64
    zone_crps: F64
    zone_covered: np.ndarray
    quantile_probabilities: tuple[float, ...]


def score_ensemble_states(
    zones: ParticleZones,
    weights: F64,
    *,
    truth_so: F64,
    truth_pv: F64,
    support: ZoneSupport,
    probabilities: tuple[float, ...] = DEFAULT_QUANTILE_PROBABILITIES,
) -> EnsembleStateScores:
    """Score the posterior mean against truth, PV-weighted by the TRUTH zones.

    This is the only function in the module that sees truth. A zone with positive
    truth PV and no defined estimate fails the whole call (§5.5) instead of being
    dropped from the score.
    """
    if support.kind != PRIMARY_SUPPORT_KIND:
        raise ValueError(
            f"primary scores require the {PRIMARY_SUPPORT_KIND!r} support, got "
            f"{support.kind!r}: layer aggregates and remote zones are diagnostics"
        )
    check_support_disjoint(support)
    n_particles = zones.so.shape[0]
    mass = _normalised_weights(weights, n_particles)

    actual = np.asarray(truth_so, dtype=np.float64)
    pore = np.asarray(truth_pv, dtype=np.float64)
    n_cells = support.matrix.shape[1]
    if actual.shape != (n_cells,) or pore.shape != (n_cells,):
        raise ValueError(
            f"truth_so and truth_pv must have shape ({n_cells},), got {actual.shape}, {pore.shape}"
        )
    if not np.isfinite(pore).all() or np.any(pore < 0.0):
        raise ValueError("truth_pv must be finite and non-negative")
    if np.any(~np.isfinite(actual) & (pore > 0.0)):
        raise ValueError("truth so is missing on a cell with positive truth PV")

    matrix = support.matrix
    truth_zone_pv = matrix @ pore
    truth_numerator = matrix @ (pore * actual)
    valid = truth_zone_pv > 0.0
    truth_zone_so = np.full(matrix.shape[0], np.nan)
    np.divide(truth_numerator, truth_zone_pv, out=truth_zone_so, where=valid)

    mean, quantiles, _inventory, _defined = _zone_weighted_quantiles(zones, mass, probabilities)
    missing = [support.names[zone] for zone in np.flatnonzero(valid & ~np.isfinite(mean))]
    if missing:
        raise ValueError(
            f"missing estimate on positive truth PV in zone(s) {missing}: a zone the "
            "truth has pore volume in is a FAILURE, not an exclusion (plan §5.5)"
        )
    errors = compare_so(mean[valid], truth_zone_so[valid], truth_zone_pv[valid])

    zone_pv_weight = truth_zone_pv[valid] / truth_zone_pv[valid].sum()
    zone_crps = np.full(matrix.shape[0], np.nan)
    zone_covered = np.zeros(matrix.shape[0], dtype=bool)
    widths = quantiles[:, -1] - quantiles[:, 0]
    for zone in np.flatnonzero(valid):
        defined = zones.so_defined[:, zone]
        zone_crps[zone] = weighted_crps(
            zones.so[defined, zone], mass[defined], float(truth_zone_so[zone])
        )
        zone_covered[zone] = bool(quantiles[zone, 0] <= truth_zone_so[zone] <= quantiles[zone, -1])
    zone_error = mean - truth_zone_so
    return EnsembleStateScores(
        mae=errors["mae"],
        rmse=errors["rmse"],
        crps=float(zone_pv_weight @ zone_crps[valid]),
        coverage=float(zone_pv_weight @ zone_covered[valid].astype(np.float64)),
        mean_width=float(zone_pv_weight @ widths[valid]),
        zone_error=zone_error,
        truth_zone_so=truth_zone_so,
        truth_zone_pv=truth_zone_pv,
        zone_crps=zone_crps,
        zone_covered=zone_covered,
        quantile_probabilities=tuple(float(p) for p in probabilities),
    )


def phase_history_fit(predicted: F64, weights: F64, observed: F64) -> float:
    """Weighted CRPS of the phase history, averaged over the observed time bins."""
    values = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(observed, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("predicted history must be a finite (particle, time) matrix")
    if actual.shape != (values.shape[1],) or not np.isfinite(actual).all():
        raise ValueError("observed history must be finite with one value per time bin")
    mass = _normalised_weights(weights, values.shape[0])
    return float(
        np.mean(
            [
                weighted_crps(values[:, index], mass, float(actual[index]))
                for index in range(values.shape[1])
            ]
        )
    )


def zone_value_map(support: ZoneSupport, zone_values: F64, shape: tuple[int, int, int]) -> F64:
    """Project one value per zone onto the (nz, ny, nx) grid.

    A cell takes its zone's value when exactly one zone holds it; cells covered by
    none or by several zones (diagnostic supports) are NaN rather than guessed.
    """
    values = np.asarray(zone_values, dtype=np.float64)
    if values.shape != (support.matrix.shape[0],):
        raise ValueError(
            f"zone_values must have one entry per zone ({support.matrix.shape[0]}), "
            f"got {values.shape}"
        )
    nx, ny, nz = shape
    if nx * ny * nz != support.matrix.shape[1]:
        raise ValueError(
            f"shape {shape} has {nx * ny * nz} cells for a support of {support.matrix.shape[1]}"
        )
    coverage = support.matrix.sum(axis=0)
    cell_values = np.full(support.matrix.shape[1], np.nan)
    unique = coverage == 1.0
    cell_values[unique] = values[np.argmax(support.matrix[:, unique], axis=0)]
    return cell_values.reshape((nx, ny, nz), order="F").transpose(2, 1, 0)


def _json_scalar(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def products_payload(products: EnsembleStateProducts) -> dict[str, Any]:
    """A JSON-safe payload of the operational products (NaN travels as null)."""
    return {
        "estimator": products.estimator,
        "posterior_mean_so": [_json_scalar(value) for value in products.posterior_mean_so],
        "so_quantiles": [[_json_scalar(value) for value in row] for row in products.so_quantiles],
        "inventory_quantiles": [
            [_json_scalar(value) for value in row] for row in products.inventory_quantiles
        ],
        "zone_weight_so_defined": [
            _json_scalar(value) for value in products.zone_weight_so_defined
        ],
        "mean_zone_pv": [_json_scalar(value) for value in products.mean_zone_pv],
        "representative_particle_index": products.representative_particle_index,
        "representative_rule": products.representative_rule,
        "representative_zone_so": [
            _json_scalar(value) for value in products.representative_zone_so
        ],
        "s_probabilities": dict(products.s_probabilities),
        "quantile_probabilities": list(products.quantile_probabilities),
        "n_particles": products.n_particles,
    }


__all__ = [
    "DEFAULT_QUANTILE_PROBABILITIES",
    "DIAGNOSTIC_LAYER_KIND",
    "INFERENCE_GRID_EXTENT_M",
    "INFERENCE_GRID_SHAPE",
    "PRIMARY_SUPPORT_KIND",
    "REPRESENTATIVE_RULE",
    "EnsembleStateProducts",
    "EnsembleStateScores",
    "ParticleZones",
    "ZoneSupport",
    "aggregate_particle_zones",
    "check_support_disjoint",
    "ensemble_state_products",
    "layer_diagnostic_support",
    "phase_history_fit",
    "primary_quadrant_support",
    "products_payload",
    "representative_particle",
    "score_ensemble_states",
    "zone_value_map",
]
