"""Physical T2/T4 designs used by the E02 observability experiments.

The designs live outside the E01 P1 generator so their deliberate symmetry and remote
initial-state uncertainty cannot silently change the five frozen E01 parent worlds.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from scipy.special import ndtr

from so_recon.config.schema import StrictModel
from so_recon.simulator.contracts import (
    MILLIDARCY_M2,
    ControlSegment,
    FluidSpec,
    WellSpec,
)
from so_recon.simulator.schedule import month_edges, month_edges_s
from so_recon.synthetic.p1 import (
    BASE_RATE_M3_SC_DAY,
    COMPLETION_EVENT_WELL,
    COMPLETION_REOPEN_EDGE,
    COMPLETION_SHUT_EDGE,
    DATUM_DEPTH_M,
    INITIAL_SW,
    INJECTOR_MAX_BHP_PA,
    MAX_PERMEABILITY_MD,
    MIN_PERMEABILITY_MD,
    MODES,
    N_LAYERS,
    N_MODES,
    PRODUCER_MIN_BHP_PA,
    RATE_MODULATION,
    WELL_RADIUS_M,
    P1Design,
    oil_connected_hydrostatic_pa,
)

INVERSE_DESIGN_KEY = "inverse_design"
INVERSE_RENDERER_VERSION = "e02-ambiguous-physical-renderer-1"


class InverseSupport(NamedTuple):
    observation_id: str
    cell_ids: tuple[int, ...]


class InversePhysicalDesign(StrictModel):
    """A frozen physical definition for either layer symmetry (T2) or remote ambiguity (T4)."""

    design_id: Literal["e02-t2-v1", "e02-t4-v1"]
    shape: tuple[int, int, int] = (16, 16, 2)
    extent_m: tuple[float, float, float] = (100.0, 100.0, 20.0)
    n_months: int = 36
    start_date: str = "2000-01-01"

    @model_validator(mode="after")
    def _fixed_geometry(self) -> InversePhysicalDesign:
        if self.shape != (16, 16, 2) or self.extent_m != (100.0, 100.0, 20.0):
            raise ValueError("E02 T2/T4 v1 fix a 16x16x2, 100x100x20 m box")
        if self.n_months != 36 or date.fromisoformat(self.start_date).day != 1:
            raise ValueError("E02 T2/T4 v1 fix 36 calendar months from a month edge")
        return self

    @property
    def n_cells(self) -> int:
        return math.prod(self.shape)

    @property
    def cells_per_layer(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def cell_size_m(self) -> tuple[float, float, float]:
        return tuple(
            extent / count for extent, count in zip(self.extent_m, self.shape, strict=True)
        )  # type: ignore[return-value]

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def cutoff(self) -> str:
        return month_edges(self.start, self.n_months)[-1].isoformat()

    @property
    def report_edges_s(self) -> tuple[float, ...]:
        return month_edges_s(self.start, self.n_months)

    @property
    def n_geology(self) -> int:
        return 12 if self.design_id == "e02-t2-v1" else 13

    @property
    def n_state_residual(self) -> int:
        return 0 if self.design_id == "e02-t2-v1" else 1

    @property
    def schema_id(self) -> str:
        return "e02-t2-symmetric-12" if self.design_id == "e02-t2-v1" else "e02-t4-17d"

    @property
    def transform_version(self) -> str:
        return f"{self.design_id}-conditional-1"

    @property
    def layer_base_permeability_md(self) -> tuple[float, float]:
        return (80.0, 80.0) if self.design_id == "e02-t2-v1" else (120.0, 40.0)

    @property
    def kz_over_kx(self) -> float:
        return 1.0e-4 if self.design_id == "e02-t2-v1" else 0.05

    @property
    def well_columns(self) -> tuple[tuple[str, tuple[int, int], str], ...]:
        producer_x = 13 if self.design_id == "e02-t2-v1" else 6
        return (
            ("I1", (2, 2), "injector"),
            ("I2", (2, 13), "injector"),
            ("P1", (producer_x, 2), "producer"),
            ("P2", (producer_x, 13), "producer"),
        )

    def payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "n_geology": self.n_geology,
            "n_state_residual": self.n_state_residual,
            "schema_id": self.schema_id,
            "transform_version": self.transform_version,
            "layer_base_permeability_md": list(self.layer_base_permeability_md),
            "kz_over_kx": self.kz_over_kx,
            "well_columns": [
                {"well_id": name, "column": list(column), "role": role}
                for name, column, role in self.well_columns
            ],
            "remote_zone": {"x_index_min": 12, "y_index_min": 8}
            if self.design_id == "e02-t4-v1"
            else None,
            "initial_state_meaning": (
                "synthetic_nonvirgin_initial_state"
                if self.design_id == "e02-t4-v1"
                else "synthetic_initial"
            ),
        }


def inverse_design(design_id: str) -> InversePhysicalDesign:
    return InversePhysicalDesign.model_validate({"design_id": design_id})


def _cell_centers(design: InversePhysicalDesign) -> NDArray[np.float64]:
    nx, ny, nz = design.shape
    dx, dy, dz = design.cell_size_m
    return np.asarray(
        [
            ((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz)
            for k in range(nz)
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.float64,
    )


def remote_mask(design: InversePhysicalDesign) -> NDArray[np.float64]:
    """T4's declared west-to-east transition, repeated identically in both layers."""
    nx, ny, nz = design.shape
    one_layer = np.empty(nx * ny, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            if i <= 9:
                value = 0.0
            elif i == 10:
                value = 1.0 / 3.0
            elif i == 11:
                value = 2.0 / 3.0
            else:
                value = 1.0
            one_layer[i + nx * j] = value
    return np.tile(one_layer, nz)


def permeability_multiplier(design: InversePhysicalDesign) -> NDArray[np.float64]:
    nx, ny, nz = design.shape
    one_layer = np.empty(nx * ny, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            value = 1.0 if i <= 9 else (0.5 if i == 10 else 0.1 if i == 11 else 0.01)
            one_layer[i + nx * j] = value
    return np.tile(one_layer, nz)


def render_inverse_coefficients(
    coefficients: NDArray[np.float64],
    design: InversePhysicalDesign,
    *,
    state_coordinate: float = 0.0,
) -> dict[str, NDArray[np.float64]]:
    """Render the T2/T4 geology and, for T4, its independent non-virgin initial state."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (design.n_geology,):
        raise ValueError(
            f"{design.design_id} expects {design.n_geology} coefficients, got {values.shape}"
        )
    if not np.isfinite(values).all() or not math.isfinite(state_coordinate):
        raise ValueError("inverse coefficients and state coordinate must be finite")

    nx, ny, _ = design.shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = remote_mask(design)[: design.cells_per_layer].reshape((nx, ny), order="F")
    multiplier = permeability_multiplier(design)
    fields: list[NDArray[np.float64]] = []
    k_layers: list[NDArray[np.float64]] = []
    phi_layers: list[NDArray[np.float64]] = []
    for layer in range(N_LAYERS):
        layer_coefficients = values[layer * N_MODES : (layer + 1) * N_MODES]
        field = sum(
            coefficient
            * np.cos(np.pi * mode_x * xx)
            * np.cos(np.pi * mode_y * yy)
            / (1 + mode_x + mode_y)
            for coefficient, (mode_x, mode_y) in zip(layer_coefficients, MODES, strict=True)
        )
        if design.design_id == "e02-t4-v1":
            field = field + values[12] * mask * np.cos(np.pi * yy)
        fields.append(field.ravel(order="F"))
        k_layers.append(
            np.exp(math.log(design.layer_base_permeability_md[layer]) + field).ravel(order="F")
        )
        phi_layers.append((0.12 + 0.18 / (1.0 + np.exp(-0.7 * field))).ravel(order="F"))

    k_md = np.concatenate(k_layers) * multiplier
    phi = np.concatenate(phi_layers)
    if not np.isfinite(k_md).all() or not np.isfinite(phi).all():
        raise ValueError("inverse renderer produced non-finite rock")
    if float(k_md.min()) < MIN_PERMEABILITY_MD or float(k_md.max()) > MAX_PERMEABILITY_MD:
        raise ValueError(
            f"inverse permeability spans [{float(k_md.min()):g}, {float(k_md.max()):g}] "
            "mD outside the declared renderer range"
        )
    if not ((phi > 0.0) & (phi < 1.0)).all():
        raise ValueError("inverse porosity lies outside (0,1)")

    centers = _cell_centers(design)
    sw = np.full(design.n_cells, INITIAL_SW, dtype=np.float64)
    if design.design_id == "e02-t4-v1":
        sw = sw + 0.25 * float(ndtr(state_coordinate)) * remote_mask(design)
    if not ((sw >= 0.2) & (sw <= 0.45)).all():
        raise ValueError("T4 initial Sw lies outside its declared [0.2,0.45] support")
    dx, dy, dz = design.cell_size_m
    kx = k_md * MILLIDARCY_M2
    arrays = {
        "cell_centers_m": centers,
        "cell_volume_m3": np.full(design.n_cells, dx * dy * dz, dtype=np.float64),
        "porosity": phi,
        "permeability_m2": np.vstack([kx, kx, kx * design.kz_over_kx]),
        "log_permeability_m2": np.log(kx),
        "generator_field": np.concatenate(fields),
        "pressure_pa": oil_connected_hydrostatic_pa(
            centers[:, 2], design=P1Design(), fluids=FluidSpec()
        ),
        "sw": sw,
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("inverse renderer produced non-finite arrays")
    return arrays


def inverse_well_specs(design: InversePhysicalDesign) -> tuple[WellSpec, ...]:
    nx, ny, nz = design.shape
    return tuple(
        WellSpec(
            well_id=name,
            cells=tuple(i + nx * (j + ny * layer) for layer in range(nz)),
            radius_m=WELL_RADIUS_M,
            reference_depth_m=DATUM_DEPTH_M,
            model="multisegment",
            allow_crossflow=True,
        )
        for name, (i, j), _role in design.well_columns
    )


def inverse_control_segments(design: InversePhysicalDesign) -> tuple[ControlSegment, ...]:
    segments: list[ControlSegment] = []
    for well_id, _column, role in design.well_columns:
        for first, last, factor in RATE_MODULATION:
            cuts = [first, last]
            if design.design_id == "e02-t4-v1" and well_id == COMPLETION_EVENT_WELL:
                cuts += [
                    edge
                    for edge in (COMPLETION_SHUT_EDGE, COMPLETION_REOPEN_EDGE)
                    if first < edge < last
                ]
            bounds = sorted(set(cuts))
            for start, end in zip(bounds, bounds[1:], strict=False):
                lower_shut = (
                    design.design_id == "e02-t4-v1"
                    and well_id == COMPLETION_EVENT_WELL
                    and COMPLETION_SHUT_EDGE <= start < COMPLETION_REOPEN_EDGE
                )
                segments.append(
                    ControlSegment(
                        start_s=design.report_edges_s[start],
                        end_s=design.report_edges_s[end],
                        well_id=well_id,
                        role=role,
                        target="water_rate" if role == "injector" else "liquid_rate",
                        value=BASE_RATE_M3_SC_DAY * factor,
                        bhp_limit_pa=(
                            INJECTOR_MAX_BHP_PA if role == "injector" else PRODUCER_MIN_BHP_PA
                        ),
                        connection_open=(True, not lower_shut),
                    )
                )
    return tuple(segments)


def inverse_supports(design: InversePhysicalDesign) -> tuple[InverseSupport, ...]:
    nx, ny, nz = design.shape
    supports: list[InverseSupport] = []
    for well_id, (i, j), _role in design.well_columns:
        cells = tuple(i + nx * (j + ny * layer) for layer in range(nz))
        if design.design_id == "e02-t2-v1":
            supports.append(InverseSupport(f"{well_id}-layer-average-log_permeability_m2", cells))
        else:
            supports.extend(
                InverseSupport(f"{well_id}-L{layer}-log_permeability_m2", (cell,))
                for layer, cell in enumerate(cells)
            )
    return tuple(supports)


__all__ = [
    "INVERSE_DESIGN_KEY",
    "INVERSE_RENDERER_VERSION",
    "InversePhysicalDesign",
    "InverseSupport",
    "inverse_control_segments",
    "inverse_design",
    "inverse_supports",
    "inverse_well_specs",
    "permeability_multiplier",
    "remote_mask",
    "render_inverse_coefficients",
]
