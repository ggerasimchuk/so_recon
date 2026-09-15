"""Registered 1D physical inversion whose latent changes oil mobility, not rate-scale."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import model_validator
from scipy.special import ndtr

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import (
    cartesian_neighbors,
    compute_model_hash,
    validate_case,
    write_arrays,
)
from so_recon.simulator.contracts import (
    CELL_AXES,
    CELL_DIM_AXES,
    DIM_CELL_AXES,
    FACE_AXES,
    MILLIDARCY_M2,
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
from so_recon.simulator.schedule import month_edges, month_edges_s
from so_recon.synthetic.p1 import OBSERVATION_SCHEMA, P1Design, oil_connected_hydrostatic_pa

REDUCED_RENDERER_VERSION = "e02-reduced-corey-1"


class ReducedDesign(StrictModel):
    """The fixed 16-cell, 12-calendar-month displacement design."""

    design_id: Literal["e02-reduced-corey-v1"] = "e02-reduced-corey-v1"
    shape: tuple[int, int, int] = (16, 1, 1)
    extent_m: tuple[float, float, float] = (160.0, 10.0, 10.0)
    porosity: float = 0.25
    permeability_md: tuple[float, float, float] = (100.0, 100.0, 5.0)
    initial_sw: float = 0.2
    start_date: str = "2000-01-01"
    n_months: int = 12
    rate_m3_sc_day: float = 2.0
    injector_bhp_limit_pa: float = 30.0e6
    producer_bhp_limit_pa: float = 5.0e6

    @model_validator(mode="after")
    def _fixed_design_is_physical(self) -> ReducedDesign:
        if self.shape != (16, 1, 1) or self.extent_m != (160.0, 10.0, 10.0):
            raise ValueError(
                f"reduced design v1 fixes shape/extent, got {self.shape}/{self.extent_m}"
            )
        if date.fromisoformat(self.start_date).day != 1 or self.n_months != 12:
            raise ValueError("reduced design v1 is twelve calendar months from a month edge")
        if not 0.0 < self.porosity < 1.0 or self.initial_sw != 0.2:
            raise ValueError("reduced porosity must be fractional and v1 starts at Sw=0.2")
        if any(value <= 0.0 for value in self.permeability_md):
            raise ValueError("reduced permeability must be positive in all directions")
        if (
            self.rate_m3_sc_day <= 0.0
            or self.injector_bhp_limit_pa <= 0.0
            or self.producer_bhp_limit_pa <= 0.0
        ):
            raise ValueError("reduced controls and BHP limits must be positive")
        return self

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def report_edges_s(self) -> tuple[float, ...]:
        return month_edges_s(self.start, self.n_months)

    @property
    def cutoff(self) -> str:
        return month_edges(self.start, self.n_months)[-1].isoformat()

    @property
    def n_cells(self) -> int:
        return math.prod(self.shape)


def oil_corey_exponent(z: float) -> float:
    """Declared latent transform ``n_o = 1.5 + 2 Phi(z)``."""
    if not math.isfinite(z):
        raise ValueError(f"reduced latent z must be finite, got {z!r}")
    return 1.5 + 2.0 * float(ndtr(z))


def _centers(design: ReducedDesign) -> np.ndarray:
    nx, ny, nz = design.shape
    dx, dy, dz = (
        extent / count for extent, count in zip(design.extent_m, design.shape, strict=True)
    )
    return np.asarray(
        [
            ((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz)
            for k in range(nz)
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.float64,
    )


def _empty_observations() -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([], schema=OBSERVATION_SCHEMA), sink, compression="snappy")
    payload: bytes = sink.getvalue().to_pybytes()
    return payload


def build_reduced_case(
    z: float,
    design: ReducedDesign,
    paths: ProjectPaths,
    ctx: RunContext,
) -> CaseBundle:
    """Publish one reduced physical case; no truth or observation value enters F."""
    exponent = oil_corey_exponent(z)
    identity = sha256_json(
        {"renderer": REDUCED_RENDERER_VERSION, "design": design.model_dump(), "z": z}
    )
    root = paths.artifacts / "reduced_cases" / identity
    centers = _centers(design)
    cell_size = [
        extent / count for extent, count in zip(design.extent_m, design.shape, strict=True)
    ]
    cell_volume = math.prod(cell_size)
    grid = write_arrays(
        root / "grid.h5",
        {
            "cell_centers_m": (centers, "m", CELL_DIM_AXES),
            "cell_volume_m3": (
                np.full(design.n_cells, cell_volume, dtype=np.float64),
                "m3",
                CELL_AXES,
            ),
            "neighbors": (cartesian_neighbors(design.shape), "1", FACE_AXES),
        },
        paths=paths,
    )
    permeability = (
        np.repeat(
            np.asarray(design.permeability_md, dtype=np.float64)[:, None],
            design.n_cells,
            axis=1,
        )
        * MILLIDARCY_M2
    )
    rock = write_arrays(
        root / "rock.h5",
        {
            "porosity": (
                np.full(design.n_cells, design.porosity, dtype=np.float64),
                "1",
                CELL_AXES,
            ),
            "permeability_m2": (permeability, "m2", DIM_CELL_AXES),
        },
        paths=paths,
    )
    fluids = FluidSpec(corey_exponents=(2.0, exponent), residual_saturations=(0.2, 0.2))
    initial = write_arrays(
        root / "initial.h5",
        {
            "pressure_pa": (
                oil_connected_hydrostatic_pa(centers[:, 2], design=P1Design(), fluids=fluids),
                "Pa",
                CELL_AXES,
            ),
            "sw": (
                np.full(design.n_cells, design.initial_sw, dtype=np.float64),
                "1",
                CELL_AXES,
            ),
        },
        paths=paths,
    )
    observation_ref = write_artifact(
        root / "observations.parquet",
        _empty_observations(),
        paths,
        schema_version="e02-reduced-observations-1",
        producer_run_id=ctx.run_id,
        media_type="application/vnd.apache.parquet",
        now=datetime.now(UTC),
    )
    ctx.add_output(f"reduced_case.{identity}.observations", observation_ref)
    horizon = design.report_edges_s[-1]
    wells = (
        WellSpec(
            well_id="I1",
            cells=(0,),
            reference_depth_m=0.0,
            model="simple",
            allow_crossflow=False,
        ),
        WellSpec(
            well_id="P1",
            cells=(design.n_cells - 1,),
            reference_depth_m=0.0,
            model="simple",
            allow_crossflow=False,
        ),
    )
    controls = (
        ControlSegment(
            start_s=0.0,
            end_s=horizon,
            well_id="I1",
            role="injector",
            target="water_rate",
            value=design.rate_m3_sc_day,
            bhp_limit_pa=design.injector_bhp_limit_pa,
            connection_open=(True,),
        ),
        ControlSegment(
            start_s=0.0,
            end_s=horizon,
            well_id="P1",
            role="producer",
            target="liquid_rate",
            value=design.rate_m3_sc_day,
            bhp_limit_pa=design.producer_bhp_limit_pa,
            connection_open=(True,),
        ),
    )
    case = CaseBundle(
        case_id=f"reduced-{identity[:16]}",
        world_id=f"reduced-{identity[:16]}",
        start_date=design.start_date,
        cutoff=design.cutoff,
        report_edges_s=design.report_edges_s,
        grid=GridSpec(
            shape=design.shape,
            extent_m=design.extent_m,
            cell_centers_m=grid["cell_centers_m"],
            cell_volume_m3=grid["cell_volume_m3"],
            neighbors=grid["neighbors"],
        ),
        rock=RockSpec(porosity=rock["porosity"], permeability_m2=rock["permeability_m2"]),
        fluids=fluids,
        wells=wells,
        controls=controls,
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=initial["pressure_pa"],
            sw=initial["sw"],
            meaning="synthetic_initial",
        ),
        boundary=BoundarySpec(kind="closed", cells=()),
        observations=ObservationSpec(
            table_path=observation_ref.path,
            sha256=observation_ref.sha256,
            dynamic_channels=("fw", "controls"),
            pressure_available=False,
        ),
        renderer_version=REDUCED_RENDERER_VERSION,
        units={
            "pressure": "Pa",
            "permeability": "m2",
            "time": "s",
            "length": "m",
            "control_rate": "m3_sc/day",
        },
        seeds={"design": 0},
        source_hashes={
            "design": sha256_json(design.model_dump(mode="json")),
            "latent": sha256_json({"z": z}),
        },
        model_hash="e" * 64,
    )
    case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    validation = validate_case(case, paths)
    if not validation.valid:
        raise ValueError("invalid reduced case: " + "; ".join(validation.errors))
    return case


__all__ = [
    "REDUCED_RENDERER_VERSION",
    "ReducedDesign",
    "build_reduced_case",
    "oil_corey_exponent",
]
