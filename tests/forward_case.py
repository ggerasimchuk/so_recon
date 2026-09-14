"""A small, complete and valid E01 forward case, built through the public contracts.

Shared by the unit tests and by the Julia exchange test, which needs a real `CaseBundle`
on disk for the Julia decoder to read. Not a test module: pytest collects `test_*.py` only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.simulator.case_io import cartesian_neighbors, compute_model_hash, write_array
from so_recon.simulator.contracts import (
    CELL_AXES,
    CELL_DIM_AXES,
    CONTROL_RATE_UNIT,
    CONTROL_RATE_UNIT_KEY,
    DIM_CELL_AXES,
    FACE_AXES,
    MILLIDARCY_M2,
    SECONDS_PER_DAY,
    ArrayRef,
    BoundarySpec,
    CaseBundle,
    ControlSegment,
    FluidSpec,
    GridSpec,
    InitialStateSpec,
    ObservationSpec,
    RockSpec,
    WellSpec,
)

SHAPE = (2, 2, 2)
N_CELLS = 8

# The anti-transpose fixture: two times, three cells, every value distinct and no symmetry
# to hide behind, so a transpose or a reshape changes which cell holds which number.
ASYMMETRIC = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float64)

# A case has to exist before its own content can be hashed, so it is built with a legal
# stand-in digest and the real hash is written in afterwards.
PLACEHOLDER_HASH = "e" * 64


#: Depth of the top face of the fixture box. The grid is 20 m of reservoir in two 10 m
#: layers below it, so the cell centres are at 1005 m and 1015 m, and the wells' 1000 m
#: `reference_depth_m` is that top face — a datum 5 m above the shallowest perforation,
#: which is what the Julia adapter origins the mesh at.
DATUM_M = 1000.0


def cell_centers(shape: tuple[int, int, int]) -> NDArray[np.float64]:
    nx, ny, nz = shape
    out = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = i + nx * (j + ny * k)
                # z is depth, positive down: deeper layers get larger z.
                out[cell] = (50.0 * i + 25.0, 50.0 * j + 25.0, DATUM_M + 10.0 * k + 5.0)
    return out


def write_case_arrays(
    paths: ProjectPaths, *, subdir: str = "arrays", **overrides: NDArray[Any]
) -> dict[str, ArrayRef]:
    """Publish the arrays a valid 2x2x2 case needs; `overrides` replaces single values."""
    data: dict[str, tuple[NDArray[Any], str, tuple[str, ...]]] = {
        "cell_centers_m": (cell_centers(SHAPE), "m", CELL_DIM_AXES),
        "cell_volume_m3": (np.full(N_CELLS, 50.0 * 50.0 * 10.0), "m3", CELL_AXES),
        "neighbors": (cartesian_neighbors(SHAPE), "1", FACE_AXES),
        "porosity": (np.full(N_CELLS, 0.25), "1", CELL_AXES),
        "permeability_m2": (np.full((3, N_CELLS), 100.0 * MILLIDARCY_M2), "m2", DIM_CELL_AXES),
        "pressure_pa": (np.full(N_CELLS, 2.5e7), "Pa", CELL_AXES),
        "sw": (np.full(N_CELLS, 0.3), "1", CELL_AXES),
    }
    refs: dict[str, ArrayRef] = {}
    for name, (values, unit, axes) in data.items():
        refs[name] = write_array(
            paths.artifacts / subdir / f"{name}.h5",
            "values",
            overrides.get(name, values),
            unit=unit,
            axis_order=axes,
            paths=paths,
        )
    return refs


def grid_spec(refs: dict[str, ArrayRef]) -> GridSpec:
    return GridSpec(
        shape=SHAPE,
        extent_m=(100.0, 100.0, 20.0),
        cell_centers_m=refs["cell_centers_m"],
        cell_volume_m3=refs["cell_volume_m3"],
        neighbors=refs["neighbors"],
    )


def control_segments() -> tuple[ControlSegment, ...]:
    return (
        ControlSegment(
            start_s=0.0,
            end_s=30.0 * SECONDS_PER_DAY,
            well_id="INJ1",
            role="injector",
            target="water_rate",
            # m3_sc/day: the human-facing unit the exchange carries. Julia divides by
            # SECONDS_PER_DAY at model construction to reach the native m3_sc/s.
            value=50.0,
            bhp_limit_pa=4.0e7,
            connection_open=(True, True),
        ),
        ControlSegment(
            start_s=0.0,
            end_s=30.0 * SECONDS_PER_DAY,
            well_id="PRO1",
            role="producer",
            target="liquid_rate",
            value=40.0,  # m3_sc/day, total standard liquid (SPEC 9.1)
            bhp_limit_pa=1.0e7,
            connection_open=(True, True),
        ),
    )


def build_case(refs: dict[str, ArrayRef], **overrides: Any) -> CaseBundle:
    """A valid case over `refs`, with its real model hash, unless `overrides` says otherwise."""
    fields: dict[str, Any] = {
        "case_id": "case-e01-0001",
        "world_id": "world-e01-0001",
        "sector_id": None,
        "start_date": "2020-01-01",
        "cutoff": "2020-01-31",
        "report_edges_s": (0.0, 15.0 * SECONDS_PER_DAY, 30.0 * SECONDS_PER_DAY),
        "grid": grid_spec(refs),
        "rock": RockSpec(porosity=refs["porosity"], permeability_m2=refs["permeability_m2"]),
        "fluids": FluidSpec(),
        # Crossflow is declared true because the native wellbore couples its connections and
        # JutulDarcy 0.3.11 has no switch that turns that off; the adapter refuses a
        # multi-connection well that declares otherwise (`check_crossflow` in model.jl)
        # instead of silently simulating the opposite semantics.
        "wells": (
            WellSpec(well_id="INJ1", cells=(0, 4), reference_depth_m=DATUM_M, allow_crossflow=True),
            WellSpec(well_id="PRO1", cells=(3, 7), reference_depth_m=DATUM_M, allow_crossflow=True),
        ),
        "controls": control_segments(),
        "initial": InitialStateSpec(
            kind="explicit",
            pressure_pa=refs["pressure_pa"],
            sw=refs["sw"],
            meaning="synthetic_initial",
        ),
        "boundary": BoundarySpec(kind="closed", cells=()),
        "observations": ObservationSpec(dynamic_channels=(), pressure_available=False),
        "renderer_version": "e01.0",
        "units": {
            "pressure": "Pa",
            "permeability": "m2",
            "time": "s",
            CONTROL_RATE_UNIT_KEY: CONTROL_RATE_UNIT,
        },
        "seeds": {"fixture": 20260913},
        "source_hashes": {"generator": "a" * 64, "config": "b" * 64},
        "model_hash": PLACEHOLDER_HASH,
    }
    fields.update(overrides)
    case = CaseBundle(**fields)
    if "model_hash" not in overrides:
        case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    return case
