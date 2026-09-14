"""Cases for E01.8: one long enough to restart inside, and two that share nothing.

Not a test module — pytest collects `test_*.py` only. Two builders live here:

* `six_month_case` — half a year over a 4x1x2 box, with a completion event at the start of
  month 3 and a role switch at the start of month 4. The checkpoint of the restart test is
  taken at the end of month 3, so the mask event is strictly inside the prefix and the role
  switch is exactly at the boundary: a continuation that quietly reused the prefix's
  controls, or that lost the switch, produces different numbers rather than the same ones.
* `isolation_cases` — A and B, differing in permeability, fluids, initial pressure and
  saturation, and completion masks. They are what an A -> B -> A isolation check needs: if
  anything of B survived in the process, the second A cannot reproduce the first.
"""

from __future__ import annotations

from datetime import date
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
from so_recon.simulator.schedule import month_edges_s

#: The box every case here is built on: four cells along x, two layers, 10 m cubes. Small
#: enough to run many times inside a P0 session and asymmetric enough that a transposed or
#: reused array changes the answer.
SHAPE = (4, 1, 2)
N_CELLS = 8
CELL_EDGE_M = 10.0
DATUM_M = 1000.0

#: The injector is the i=0 column and the producer the i=3 column, one cell in each layer.
INJECTOR_CELLS = (0, 4)
PRODUCER_CELLS = (3, 7)

START = date(2020, 1, 1)
RESTART_MONTHS = 6

#: Which month boundary each event of the six-month case lands on.
MASK_EVENT_MONTH = 2  # the start of month 3
ROLE_SWITCH_MONTH = 3  # the start of month 4, and the checkpoint

RATE_M3_DAY = 2.0
PRODUCER_BHP_LIMIT_PA = 5.0e6
INJECTOR_BHP_LIMIT_PA = 4.0e7
INITIAL_PRESSURE_PA = 2.0e7

_PLACEHOLDER_HASH = "e" * 64


def cell_centers() -> NDArray[np.float64]:
    nx, ny, nz = SHAPE
    out = np.zeros((N_CELLS, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                out[i + nx * (j + ny * k)] = (
                    CELL_EDGE_M * i + 0.5 * CELL_EDGE_M,
                    CELL_EDGE_M * j + 0.5 * CELL_EDGE_M,
                    # z is depth, positive down.
                    DATUM_M + CELL_EDGE_M * k + 0.5 * CELL_EDGE_M,
                )
    return out


def write_arrays(
    paths: ProjectPaths,
    subdir: str,
    *,
    permeability_md: float,
    pressure_pa: float,
    sw: float,
) -> dict[str, ArrayRef]:
    data: dict[str, tuple[NDArray[Any], str, tuple[str, ...]]] = {
        "cell_centers_m": (cell_centers(), "m", CELL_DIM_AXES),
        "cell_volume_m3": (np.full(N_CELLS, CELL_EDGE_M**3), "m3", CELL_AXES),
        "neighbors": (cartesian_neighbors(SHAPE), "1", FACE_AXES),
        "porosity": (np.full(N_CELLS, 0.2), "1", CELL_AXES),
        "permeability_m2": (
            np.full((3, N_CELLS), permeability_md * MILLIDARCY_M2),
            "m2",
            DIM_CELL_AXES,
        ),
        "pressure_pa": (np.full(N_CELLS, pressure_pa), "Pa", CELL_AXES),
        "sw": (np.full(N_CELLS, sw), "1", CELL_AXES),
    }
    return {
        name: write_array(
            paths.artifacts / subdir / f"{name}.h5",
            "values",
            values,
            unit=unit,
            axis_order=axes,
            paths=paths,
        )
        for name, (values, unit, axes) in data.items()
    }


def _case(
    refs: dict[str, ArrayRef],
    *,
    case_id: str,
    months: int,
    controls: tuple[ControlSegment, ...],
    fluids: FluidSpec,
) -> CaseBundle:
    edges = month_edges_s(START, months)
    cutoff = date(
        START.year + (START.month - 1 + months) // 12, (START.month - 1 + months) % 12 + 1, 1
    )
    case = CaseBundle(
        case_id=case_id,
        world_id=f"world-{case_id}",
        start_date=START.isoformat(),
        cutoff=cutoff.isoformat(),
        report_edges_s=edges,
        grid=GridSpec(
            shape=SHAPE,
            extent_m=(CELL_EDGE_M * SHAPE[0], CELL_EDGE_M * SHAPE[1], CELL_EDGE_M * SHAPE[2]),
            cell_centers_m=refs["cell_centers_m"],
            cell_volume_m3=refs["cell_volume_m3"],
            neighbors=refs["neighbors"],
        ),
        rock=RockSpec(porosity=refs["porosity"], permeability_m2=refs["permeability_m2"]),
        fluids=fluids,
        wells=(
            WellSpec(
                well_id="INJ1",
                cells=INJECTOR_CELLS,
                reference_depth_m=DATUM_M,
                allow_crossflow=True,
            ),
            WellSpec(
                well_id="PRO1",
                cells=PRODUCER_CELLS,
                reference_depth_m=DATUM_M,
                allow_crossflow=True,
            ),
        ),
        controls=controls,
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=refs["pressure_pa"],
            sw=refs["sw"],
            meaning="synthetic_initial",
        ),
        boundary=BoundarySpec(kind="closed", cells=()),
        observations=ObservationSpec(dynamic_channels=(), pressure_available=False),
        renderer_version="e01.8",
        units={
            "pressure": "Pa",
            "permeability": "m2",
            "time": "s",
            CONTROL_RATE_UNIT_KEY: CONTROL_RATE_UNIT,
        },
        seeds={"fixture": 20260913},
        source_hashes={"generator": "d" * 64},
        model_hash=_PLACEHOLDER_HASH,
    )
    return case.model_copy(update={"model_hash": compute_model_hash(case)})


def six_month_controls(edges: tuple[float, ...]) -> tuple[ControlSegment, ...]:
    """Two wells over six months, with a mask event and a role switch on month edges."""
    both, upper_only = (True, True), (True, False)
    mask_at = edges[MASK_EVENT_MONTH]
    switch_at = edges[ROLE_SWITCH_MONTH]
    end = edges[-1]

    def injecting(a: float, b: float) -> ControlSegment:
        return ControlSegment(
            start_s=a,
            end_s=b,
            well_id="INJ1",
            role="injector",
            target="water_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=INJECTOR_BHP_LIMIT_PA,
            connection_open=both,
        )

    def producing(a: float, b: float, mask: tuple[bool, ...]) -> ControlSegment:
        return ControlSegment(
            start_s=a,
            end_s=b,
            well_id="PRO1",
            role="producer",
            target="liquid_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=PRODUCER_BHP_LIMIT_PA,
            connection_open=mask,
        )

    return (
        injecting(edges[0], mask_at),
        injecting(mask_at, switch_at),
        # The role switch, exactly at the checkpoint: the two wells swap. The injector
        # becomes a producer on a total standard liquid rate and the producer becomes a
        # water injector, each on the limit the other had. It is a real change of role in
        # the contract's own vocabulary, and it is deliberately as well conditioned as the
        # first three months — this case exists to test a restart, and a case that tested
        # the solver's patience instead would confuse the two.
        ControlSegment(
            start_s=switch_at,
            end_s=end,
            well_id="INJ1",
            role="producer",
            target="liquid_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=PRODUCER_BHP_LIMIT_PA,
            connection_open=both,
        ),
        producing(edges[0], mask_at, both),
        # The completion event: the producer's lower perforation closes at the start of
        # month 3, strictly inside the part the checkpoint covers.
        producing(mask_at, switch_at, upper_only),
        ControlSegment(
            start_s=switch_at,
            end_s=end,
            well_id="PRO1",
            role="injector",
            target="water_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=INJECTOR_BHP_LIMIT_PA,
            connection_open=both,
        ),
    )


def six_month_case(paths: ProjectPaths, *, subdir: str = "restart-arrays") -> CaseBundle:
    refs = write_arrays(
        paths, subdir, permeability_md=100.0, pressure_pa=INITIAL_PRESSURE_PA, sw=0.3
    )
    edges = month_edges_s(START, RESTART_MONTHS)
    return _case(
        refs,
        case_id="case-e01-restart-6m",
        months=RESTART_MONTHS,
        controls=six_month_controls(edges),
        fluids=FluidSpec(),
    )


def future_policy(edges: tuple[float, ...]) -> tuple[ControlSegment, ...]:
    """The part of the six-month schedule that starts at the checkpoint, unchanged."""
    switch_at = edges[ROLE_SWITCH_MONTH]
    return tuple(segment for segment in six_month_controls(edges) if segment.start_s >= switch_at)


def _one_month_controls(
    edges: tuple[float, ...], *, mask: tuple[bool, ...]
) -> tuple[ControlSegment, ...]:
    return (
        ControlSegment(
            start_s=edges[0],
            end_s=edges[-1],
            well_id="INJ1",
            role="injector",
            target="water_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=INJECTOR_BHP_LIMIT_PA,
            connection_open=(True, True),
        ),
        ControlSegment(
            start_s=edges[0],
            end_s=edges[-1],
            well_id="PRO1",
            role="producer",
            target="liquid_rate",
            value=RATE_M3_DAY,
            bhp_limit_pa=PRODUCER_BHP_LIMIT_PA,
            connection_open=mask,
        ),
    )


def isolation_cases(paths: ProjectPaths) -> tuple[CaseBundle, CaseBundle]:
    """A and B: different rock, different fluids, different initial state, different masks.

    Every one of those differences is one a process could leak: a cached transmissibility, a
    viscosity parameter, an initial state left in a simulator, a perforation mask applied to
    the wrong model. If any of them survived a job, the second A could not reproduce the
    first.
    """
    edges = month_edges_s(START, 1)
    a_refs = write_arrays(paths, "iso-a", permeability_md=100.0, pressure_pa=2.0e7, sw=0.3)
    b_refs = write_arrays(paths, "iso-b", permeability_md=450.0, pressure_pa=2.6e7, sw=0.55)
    case_a = _case(
        a_refs,
        case_id="case-e01-isolation-a",
        months=1,
        controls=_one_month_controls(edges, mask=(True, True)),
        fluids=FluidSpec(),
    )
    case_b = _case(
        b_refs,
        case_id="case-e01-isolation-b",
        months=1,
        controls=_one_month_controls(edges, mask=(True, False)),
        # A different educational fluid: heavier, more viscous oil and a different water
        # compressibility. Still the same OW system, so the same adapter builds it.
        fluids=FluidSpec(
            density_sc_kg_m3=(1010.0, 870.0),
            viscosity_pa_s=(0.0012, 0.0065),
            compressibility_pa_inv=(5.0e-10, 1.4e-9),
        ),
    )
    return case_a, case_b
