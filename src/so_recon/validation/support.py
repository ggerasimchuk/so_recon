"""Conservative aggregation of physical states onto fixed report zones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so_recon.inference.contracts import F64


@dataclass(frozen=True)
class ZoneState:
    """One state on report support; every array is indexed by zone."""

    pv: F64
    so: F64
    n_rem: F64
    n_above_sor: F64
    valid_geometry: np.ndarray


def _cell_vector(values: F64, n_cells: int, *, label: str) -> F64:
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (n_cells,):
        raise ValueError(f"{label} must have shape ({n_cells},), got {out.shape}")
    return out


def aggregate_state(
    intersection_pv: F64,
    so: F64,
    bo: F64,
    sorw: F64,
) -> ZoneState:
    """Aggregate by intersection pore volume without repairing malformed geometry.

    A valid empty zone has zero inventory and missing saturation.  A malformed zone has
    missing values for every extensive and intensive output, so it cannot masquerade as a
    genuinely empty region.
    """
    weights = np.asarray(intersection_pv, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"intersection_pv must be (zone, cell), got {weights.shape}")
    n_zones, n_cells = weights.shape
    oil = _cell_vector(so, n_cells, label="so")
    formation_volume = _cell_vector(bo, n_cells, label="bo")
    residual = _cell_vector(sorw, n_cells, label="sorw")
    cell_physics_valid = (
        np.isfinite(oil)
        & np.isfinite(formation_volume)
        & np.isfinite(residual)
        & (oil >= 0.0)
        & (oil <= 1.0)
        & (formation_volume > 0.0)
        & (residual >= 0.0)
        & (residual <= 1.0)
    )
    valid = np.isfinite(weights).all(axis=1) & (weights >= 0.0).all(axis=1)
    if not cell_physics_valid.all():
        involved = weights[:, ~cell_physics_valid]
        valid &= np.isfinite(involved).all(axis=1) & (involved == 0.0).all(axis=1)

    pv = np.full(n_zones, np.nan, dtype=np.float64)
    mean_so = np.full(n_zones, np.nan, dtype=np.float64)
    n_rem = np.full(n_zones, np.nan, dtype=np.float64)
    n_above = np.full(n_zones, np.nan, dtype=np.float64)
    for zone in np.flatnonzero(valid):
        zone_weights = weights[zone]
        total = float(zone_weights.sum())
        pv[zone] = total
        if total == 0.0:
            n_rem[zone] = 0.0
            n_above[zone] = 0.0
            continue
        mean_so[zone] = float(zone_weights @ oil / total)
        n_rem[zone] = float(zone_weights @ (oil / formation_volume))
        n_above[zone] = float(zone_weights @ (np.maximum(oil - residual, 0.0) / formation_volume))
    return ZoneState(
        pv=pv,
        so=mean_so,
        n_rem=n_rem,
        n_above_sor=n_above,
        valid_geometry=valid,
    )


__all__ = ["ZoneState", "aggregate_state"]
