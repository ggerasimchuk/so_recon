"""Versioned typed records for a forward case, its job and its result (plan 3.2).

These records are the contract every later E01 task builds on, so they carry the axis,
unit and identity conventions of plan 3.1 in their types rather than in prose:

* `cell_id = i + nx*(j + ny*k)`, zero-based everywhere in Python and in the exchange
  files; Julia converts to 1-based only while building its model.
* z is depth, positive down; the horizontal frame is a local Cartesian one, so `crs` is
  null by construction. Gravity is 9.80665 m/s², and zero gravity is legal only for a
  fixture that marks itself analytical.
* Pressure is Pa, permeability m², time seconds. Human-facing rates are m3_sc/day and
  native ones m3_sc/s; a unit outside `KNOWN_UNITS` is an error, never a guess.
* Two-phase tuples are ordered `(water, oil)`.

Where the table of plan 3.2 spells a value out (`z_positive='down'`, `chunk_months: int=1`)
that value is the default here. A nullable field defaults to `None`; a field that is
neither nullable nor given a default in the table is required, because a physical switch
such as `allow_crossflow` must be stated rather than inherited.

`StrictModel` forbids unknown fields and freezes the instance. Frozen is not immutability
of the science: the arrays themselves live in HDF5 files and are pinned by the SHA-256 in
`ArrayRef`, which is what actually makes a case reproducible.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_validator, model_validator

from so_recon.config.schema import StrictModel
from so_recon.paths import validate_relative_path

# Annotated as Literal so the record defaults below stay the literal type they declare.
CASE_SCHEMA_VERSION: Literal["case-1"] = "case-1"
JOB_SCHEMA_VERSION: Literal["job-1"] = "job-1"
FORWARD_SCHEMA_VERSION: Literal["forward-1"] = "forward-1"

SECONDS_PER_DAY = 86400.0
MILLIDARCY_M2 = 9.869233e-16
STANDARD_GRAVITY_M_S2 = 9.80665

#: SPEC 3.3: the original attempt plus at most one registered numerical retry.
MAX_ATTEMPTS = 2

#: Saturations and phase splits must sum to one; a larger drift is an error, never
#: something to renormalise away.
SATURATION_SUM_TOLERANCE = 1e-12

#: Every case must declare, as data, the unit its control rates are written in. The Julia
#: side divides by SECONDS_PER_DAY at model construction, so a case that silently carried
#: native m3_sc/s would run every rate control 86400 times off. Requiring the key — and
#: requiring it to say m3_sc/day — makes that disagreement a validation error rather than a
#: plausible-looking result.
CONTROL_RATE_UNIT_KEY = "control_rate"
CONTROL_RATE_UNIT = "m3_sc/day"

#: Axis names. They are recorded in the HDF5 files and compared on read, so a transposed
#: array cannot be mistaken for the one that was written.
CELL_AXES = ("cell",)
CELL_DIM_AXES = ("cell", "dim")
DIM_CELL_AXES = ("dim", "cell")
FACE_AXES = ("face", "side")
TIME_CELL_AXES = ("time", "cell")

#: Every unit the E01 contract speaks. An unknown unit is rejected rather than carried
#: along, because a silently mislabelled array is indistinguishable from a correct one
#: until the physics is already wrong.
KNOWN_UNITS = frozenset(
    {
        "1",
        "m",
        "m2",
        "m3",
        "s",
        "K",
        "Pa",
        "Pa.s",
        "1/Pa",
        "kg/m3",
        "m/s2",
        "m3_sc/s",
        "m3_sc/day",
    }
)

_ZERO_DIGEST = "0" * 64


def _reject_placeholder_digest(value: str) -> str:
    """An absent hash is `None` with a reason, never a zero string (plan 3.2)."""
    if value == _ZERO_DIGEST:
        raise ValueError(
            "digest is the all-zero placeholder; an absent hash must be null with a reason"
        )
    return value


Sha256 = Annotated[
    str, Field(pattern=r"^[0-9a-f]{64}$"), AfterValidator(_reject_placeholder_digest)
]

#: In 4.0 every input path is project-relative and resolved through `ProjectPaths`.
RelativePath = Annotated[str, AfterValidator(validate_relative_path)]

ForwardStatus = Literal[
    "COMPLETE",
    "INVALID_INPUT",
    "PHYSICALLY_INVALID",
    "CONTROL_INFEASIBLE",
    "NUMERICAL_FAILURE",
    "RESOURCE_FAILURE",
    "TIMEOUT",
    "PROTOCOL_FAILURE",
    "INCOMPLETE_BUDGET",
]


class ArrayRef(StrictModel):
    """A numeric array in an HDF5 dataset, described well enough to be checked on read."""

    path: RelativePath
    dataset: str = Field(min_length=1)
    sha256: Sha256
    shape: tuple[int, ...]
    dtype: Literal["float64", "int64", "bool"]
    unit: str
    axis_order: tuple[str, ...]

    @field_validator("shape")
    @classmethod
    def _positive_dimensions(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("shape must have at least one dimension")
        if any(n < 1 for n in v):
            raise ValueError(f"shape dimensions must be positive, got {v}")
        return v

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, v: str) -> str:
        if v not in KNOWN_UNITS:
            raise ValueError(f"unit {v!r} is not one of the known units {sorted(KNOWN_UNITS)}")
        return v

    @model_validator(mode="after")
    def _axis_order_describes_the_shape(self) -> ArrayRef:
        if len(self.axis_order) != len(self.shape):
            raise ValueError(
                f"axis_order {self.axis_order} does not describe shape {self.shape}: "
                f"{len(self.axis_order)} names for {len(self.shape)} dimensions"
            )
        if len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError(f"axis_order names must be unique, got {self.axis_order}")
        return self


def _require_array(
    label: str,
    ref: ArrayRef,
    *,
    unit: str,
    dtype: str,
    axis_order: tuple[str, ...],
) -> None:
    """Check the declared metadata of an array a record depends on."""
    if ref.unit != unit:
        raise ValueError(f"{label}: unit must be {unit!r}, got {ref.unit!r}")
    if ref.dtype != dtype:
        raise ValueError(f"{label}: dtype must be {dtype!r}, got {ref.dtype!r}")
    if ref.axis_order != axis_order:
        raise ValueError(f"{label}: axis_order must be {axis_order}, got {ref.axis_order}")


class GridSpec(StrictModel):
    """A Cartesian grid. `neighbors` is the face list; its values are checked on disk."""

    shape: tuple[int, int, int]
    extent_m: tuple[float, float, float]
    cell_centers_m: ArrayRef
    cell_volume_m3: ArrayRef
    neighbors: ArrayRef
    z_positive: Literal["down"] = "down"
    crs: None = None

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def n_faces(self) -> int:
        nx, ny, nz = self.shape
        return (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)

    @model_validator(mode="after")
    def _consistent(self) -> GridSpec:
        if any(n < 1 for n in self.shape):
            raise ValueError(f"grid shape must be positive in every direction, got {self.shape}")
        if any(e <= 0.0 for e in self.extent_m):
            raise ValueError(f"grid extent_m must be positive, got {self.extent_m}")
        _require_array(
            "cell_centers_m",
            self.cell_centers_m,
            unit="m",
            dtype="float64",
            axis_order=CELL_DIM_AXES,
        )
        if self.cell_centers_m.shape != (self.n_cells, 3):
            raise ValueError(
                f"cell_centers_m: expected shape {(self.n_cells, 3)}, "
                f"got {self.cell_centers_m.shape}"
            )
        _require_array(
            "cell_volume_m3", self.cell_volume_m3, unit="m3", dtype="float64", axis_order=CELL_AXES
        )
        if self.cell_volume_m3.shape != (self.n_cells,):
            raise ValueError(
                f"cell_volume_m3: expected shape {(self.n_cells,)}, got {self.cell_volume_m3.shape}"
            )
        _require_array("neighbors", self.neighbors, unit="1", dtype="int64", axis_order=FACE_AXES)
        if self.neighbors.shape[1] != 2:
            raise ValueError(
                f"neighbors: every face has exactly two sides, got shape {self.neighbors.shape}"
            )
        return self


class RockSpec(StrictModel):
    porosity: ArrayRef
    permeability_m2: ArrayRef
    rock_compressibility_pa_inv: float = 0.0

    @field_validator("rock_compressibility_pa_inv")
    @classmethod
    def _pore_volume_is_constant(cls, v: float) -> float:
        if v != 0.0:
            raise ValueError(
                "E01 fixes rock_compressibility_pa_inv at 0.0: the pore volume is constant "
                f"in this educational OW model (plan 3.1), got {v}"
            )
        return v

    @model_validator(mode="after")
    def _arrays_are_described_correctly(self) -> RockSpec:
        _require_array("porosity", self.porosity, unit="1", dtype="float64", axis_order=CELL_AXES)
        _require_array(
            "permeability_m2",
            self.permeability_m2,
            unit="m2",
            dtype="float64",
            axis_order=DIM_CELL_AXES,
        )
        if self.permeability_m2.shape[0] != 3:
            raise ValueError(
                "permeability_m2: must carry three directions on its first axis, "
                f"got shape {self.permeability_m2.shape}"
            )
        return self


class FluidSpec(StrictModel):
    """The educational two-phase oil-water model of plan 3.1. Tuples are (water, oil)."""

    kind: Literal["OW"] = "OW"
    density_sc_kg_m3: tuple[float, float] = (1000.0, 800.0)
    viscosity_pa_s: tuple[float, float] = (0.001, 0.003)
    compressibility_pa_inv: tuple[float, float] = (4e-10, 1e-9)
    p_sc_pa: float = Field(default=101325.0, gt=0.0)
    t_sc_k: float = Field(default=288.15, gt=0.0)
    corey_exponents: tuple[float, float] = (2.0, 2.0)
    residual_saturations: tuple[float, float] = (0.2, 0.2)
    kr_endpoints: tuple[float, float] = (1.0, 1.0)
    pc_model: Literal["zero"] = "zero"
    educational: bool = True
    analytical_limit: bool = False

    @property
    def mobile_saturation_range(self) -> tuple[float, float]:
        """[Swc, 1 - Sorw]: the interval a water saturation has to stay inside."""
        swc, sorw = self.residual_saturations
        return swc, 1.0 - sorw

    @field_validator("density_sc_kg_m3", "viscosity_pa_s", "corey_exponents", "kr_endpoints")
    @classmethod
    def _positive_pair(cls, v: tuple[float, float]) -> tuple[float, float]:
        if any(x <= 0.0 for x in v):
            raise ValueError(f"must be positive for both phases, got {v}")
        return v

    @field_validator("compressibility_pa_inv")
    @classmethod
    def _non_negative_pair(cls, v: tuple[float, float]) -> tuple[float, float]:
        if any(x < 0.0 for x in v):
            raise ValueError(f"must be non-negative for both phases, got {v}")
        return v

    @field_validator("kr_endpoints")
    @classmethod
    def _endpoints_are_relative(cls, v: tuple[float, float]) -> tuple[float, float]:
        if any(x > 1.0 for x in v):
            raise ValueError(f"relative permeability endpoints must not exceed 1, got {v}")
        return v

    @field_validator("residual_saturations")
    @classmethod
    def _leaves_a_mobile_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        swc, sorw = v
        if not (0.0 <= swc < 1.0 and 0.0 <= sorw < 1.0):
            raise ValueError(f"residual_saturations must lie in [0,1), got {v}")
        if swc + sorw >= 1.0:
            raise ValueError(
                f"residual_saturations {v} leave no mobile range [Swc, 1-Sorw]; "
                "invalid input is refused, not clipped"
            )
        return v


class WellSpec(StrictModel):
    well_id: str = Field(min_length=1)
    cells: tuple[int, ...]
    radius_m: float = Field(default=0.1, gt=0.0)
    reference_depth_m: float
    model: Literal["simple", "multisegment"] = "multisegment"
    allow_crossflow: bool

    @field_validator("cells")
    @classmethod
    def _unique_connections(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("a well needs at least one connection cell")
        if any(c < 0 for c in v):
            raise ValueError(f"cell ids are zero-based and non-negative, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate connection cells in one well: {v}")
        return v


class ControlSegment(StrictModel):
    """One well's control over one time interval. Times are seconds from the case start.

    `value` is a HUMAN-facing quantity, and which one depends on `target`:

    * `liquid_rate` — total standard liquid rate in **m3_sc/day**
    * `water_rate` — standard water rate in **m3_sc/day**
    * `bhp` — bottom-hole pressure in **Pa**
    * `disabled` — unitless and ignored; it must be 0.0

    Rates cross the JSON boundary in m3_sc/day and Julia divides by `SECONDS_PER_DAY` to
    reach the native m3_sc/s when it builds the model — in the same place, and for the same
    reason, that it converts zero-based cell ids to one-based ones (plan 3.1: "человеческие
    control rates — m3_sc/day, native — m3_sc/s"). Getting this wrong is silent and costs a
    factor of 86400, so the convention is not left to prose: every `CaseBundle` has to
    declare it in `units[CONTROL_RATE_UNIT_KEY]`, and that declaration is validated.
    """

    start_s: float = Field(ge=0.0)
    end_s: float
    well_id: str = Field(min_length=1)
    role: Literal["producer", "injector", "shut"]
    target: Literal["liquid_rate", "water_rate", "bhp", "disabled"]
    value: float
    bhp_limit_pa: float | None
    connection_open: tuple[bool, ...]

    @field_validator("connection_open")
    @classmethod
    def _non_empty_mask(cls, v: tuple[bool, ...]) -> tuple[bool, ...]:
        if not v:
            raise ValueError("connection_open must name every connection of the well")
        return v

    @model_validator(mode="after")
    def _control_is_physically_meaningful(self) -> ControlSegment:
        if not self.end_s > self.start_s:
            raise ValueError(
                f"end_s must be greater than start_s, got {self.start_s} .. {self.end_s}"
            )
        # SPEC 9.1: a producer is given a TOTAL standard liquid rate, never a per-phase
        # rate; an injector is given a standard water rate. Splitting the phases at the
        # control would prescribe the very thing the forward model is supposed to predict.
        allowed: dict[str, frozenset[str]] = {
            "producer": frozenset({"liquid_rate", "bhp"}),
            "injector": frozenset({"water_rate", "bhp"}),
            "shut": frozenset({"disabled"}),
        }
        if self.target not in allowed[self.role]:
            if self.role == "shut":
                raise ValueError(f"a shut well must carry target='disabled', got {self.target!r}")
            if self.role == "producer":
                raise ValueError(
                    "a producer is controlled on a total standard liquid rate or on bhp, "
                    f"not on the phase rate {self.target!r} (SPEC 9.1)"
                )
            raise ValueError(
                "an injector is controlled on a standard water rate or on bhp, "
                f"not on {self.target!r} (SPEC 9.1)"
            )
        if self.target == "disabled":
            if self.value != 0.0:
                raise ValueError(f"a disabled control carries value 0.0, got {self.value}")
        elif not self.value > 0.0:
            raise ValueError(
                f"value must be positive for target {self.target!r}: production is positive "
                f"in public tables, got {self.value}"
            )
        if self.target in ("bhp", "disabled"):
            if self.bhp_limit_pa is not None:
                raise ValueError(
                    f"bhp_limit_pa is meaningless for target {self.target!r}; leave it null"
                )
        elif self.bhp_limit_pa is not None and not self.bhp_limit_pa > 0.0:
            raise ValueError(f"bhp_limit_pa must be positive when given, got {self.bhp_limit_pa}")
        return self


class InitialStateSpec(StrictModel):
    kind: Literal["equilibrium", "explicit", "native_restart"]
    pressure_pa: ArrayRef | None = None
    sw: ArrayRef | None = None
    restart: RestartRef | None = None
    meaning: Literal["synthetic_initial", "developed_state"]

    @model_validator(mode="after")
    def _kind_determines_the_payload(self) -> InitialStateSpec:
        if self.kind == "explicit":
            if self.pressure_pa is None or self.sw is None:
                raise ValueError("an explicit initial state needs both pressure_pa and sw")
            if self.restart is not None:
                raise ValueError("an explicit initial state carries no restart reference")
            _require_array(
                "pressure_pa", self.pressure_pa, unit="Pa", dtype="float64", axis_order=CELL_AXES
            )
            _require_array("sw", self.sw, unit="1", dtype="float64", axis_order=CELL_AXES)
        elif self.kind == "equilibrium":
            if self.pressure_pa is not None or self.sw is not None or self.restart is not None:
                raise ValueError(
                    "an equilibrium initial state is computed, so it carries no arrays "
                    "and no restart reference"
                )
        elif self.restart is None:
            raise ValueError("a native_restart initial state needs a restart reference")
        elif self.pressure_pa is not None or self.sw is not None:
            raise ValueError(
                "a native_restart initial state takes its fields from the restart, "
                "not from explicit arrays"
            )
        return self


class BoundarySpec(StrictModel):
    kind: Literal["closed", "pressure_water"]
    cells: tuple[int, ...]
    pressure_pa: float | None = None
    trans_flow: float | None = None
    fractional_flow: tuple[float, float] = (1.0, 0.0)

    @field_validator("fractional_flow")
    @classmethod
    def _is_a_split(cls, v: tuple[float, float]) -> tuple[float, float]:
        if any(not 0.0 <= x <= 1.0 for x in v):
            raise ValueError(f"fractional_flow entries must lie in [0,1], got {v}")
        if abs(sum(v) - 1.0) > SATURATION_SUM_TOLERANCE:
            raise ValueError(f"fractional_flow must sum to 1, got {v}")
        return v

    @model_validator(mode="after")
    def _kind_determines_the_payload(self) -> BoundarySpec:
        if self.kind == "closed":
            if self.cells or self.pressure_pa is not None or self.trans_flow is not None:
                raise ValueError(
                    "a closed boundary names no cells and carries no pressure or transmissibility"
                )
            return self
        if not self.cells:
            raise ValueError("a pressure_water boundary needs at least one cell")
        if any(c < 0 for c in self.cells):
            raise ValueError(f"boundary cell ids are zero-based and non-negative: {self.cells}")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"duplicate boundary cells: {self.cells}")
        if self.pressure_pa is None or not self.pressure_pa > 0.0:
            raise ValueError(
                f"a pressure_water boundary needs a positive pressure_pa, got {self.pressure_pa}"
            )
        if self.trans_flow is None or not self.trans_flow > 0.0:
            raise ValueError(
                f"a pressure_water boundary needs a positive trans_flow, got {self.trans_flow}"
            )
        return self


class ObservationSpec(StrictModel):
    table_path: RelativePath | None = None
    sha256: Sha256 | None = None
    dynamic_channels: tuple[str, ...]
    pressure_available: bool
    truth_access: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def _table_and_digest_travel_together(self) -> ObservationSpec:
        if (self.table_path is None) != (self.sha256 is None):
            raise ValueError(
                "table_path and sha256 are both present or both absent: an observations "
                f"table without a digest cannot be checked (got {self.table_path!r})"
            )
        # An empty observations table is allowed: E01 is forward-synthetic and a case may
        # legitimately carry nothing to condition on.
        if self.table_path is None and self.dynamic_channels:
            raise ValueError(
                f"dynamic_channels {self.dynamic_channels} name an observations table, "
                "but table_path is null"
            )
        if self.table_path is not None and not self.dynamic_channels:
            raise ValueError("an observations table must declare at least one dynamic channel")
        return self


def _parse_iso_date(label: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


class CaseBundle(StrictModel):
    """Everything the forward operator F needs, with nothing that F does not read."""

    schema_version: Literal["case-1"] = CASE_SCHEMA_VERSION
    spec_version: Literal["4.0"] = "4.0"
    case_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    sector_id: str | None = None
    information_mode: Literal["synthetic_forward"] = "synthetic_forward"
    start_date: str
    cutoff: str
    report_edges_s: tuple[float, ...]
    grid: GridSpec
    rock: RockSpec
    fluids: FluidSpec
    wells: tuple[WellSpec, ...]
    controls: tuple[ControlSegment, ...]
    initial: InitialStateSpec
    boundary: BoundarySpec
    observations: ObservationSpec
    gravity_m_s2: float = STANDARD_GRAVITY_M_S2
    renderer_version: str = Field(min_length=1)
    units: dict[str, str]
    seeds: dict[str, int]
    # For a synthetic case these name the generator, its configuration and the arrays;
    # raw CSV digests are not invented for something no CSV produced (plan 2.5).
    source_hashes: dict[str, Sha256]
    model_hash: Sha256

    @field_validator("report_edges_s")
    @classmethod
    def _edges_start_at_zero_and_increase(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) < 2:
            raise ValueError(f"report_edges_s must describe at least one report interval: {v}")
        if v[0] != 0.0:
            raise ValueError(f"report_edges_s must start at 0 seconds, got {v[0]}")
        if any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(f"report_edges_s must be strictly increasing, got {v}")
        return v

    @field_validator("units")
    @classmethod
    def _units_are_known(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = sorted({unit for unit in v.values() if unit not in KNOWN_UNITS})
        if unknown:
            raise ValueError(f"unknown units {unknown}; known units are {sorted(KNOWN_UNITS)}")
        # The control-rate convention travels with the case, not in anyone's memory: Julia
        # converts `ControlSegment.value` to native m3_sc/s by dividing by SECONDS_PER_DAY,
        # and a case that meant m3_sc/s all along would be wrong by that factor in silence.
        declared = v.get(CONTROL_RATE_UNIT_KEY)
        if declared is None:
            raise ValueError(
                f"units must declare {CONTROL_RATE_UNIT_KEY!r}: a case has to say which unit "
                f"its ControlSegment rates are written in (expected {CONTROL_RATE_UNIT!r})"
            )
        if declared != CONTROL_RATE_UNIT:
            raise ValueError(
                f"units[{CONTROL_RATE_UNIT_KEY!r}] must be {CONTROL_RATE_UNIT!r}, got "
                f"{declared!r}; control rates cross the exchange as human day rates and "
                "Julia converts to native m3_sc/s when it builds the model"
            )
        return v

    @model_validator(mode="after")
    def _cross_record_consistency(self) -> CaseBundle:
        self._check_dates()
        self._check_gravity()
        self._check_cell_counts()
        self._check_wells()
        self._check_controls()
        return self

    def _check_dates(self) -> None:
        start = _parse_iso_date("start_date", self.start_date)
        cutoff = _parse_iso_date("cutoff", self.cutoff)
        if cutoff < start:
            raise ValueError(f"cutoff {self.cutoff} precedes start_date {self.start_date}")

    def _check_gravity(self) -> None:
        if self.gravity_m_s2 == STANDARD_GRAVITY_M_S2:
            return
        if self.gravity_m_s2 == 0.0 and self.fluids.analytical_limit:
            return
        raise ValueError(
            f"gravity_m_s2 must be {STANDARD_GRAVITY_M_S2}; zero gravity is allowed only for a "
            f"fixture that sets fluids.analytical_limit (got {self.gravity_m_s2})"
        )

    def _check_cell_counts(self) -> None:
        n = self.grid.n_cells
        if self.rock.porosity.shape != (n,):
            raise ValueError(
                f"rock.porosity: grid has {n} cells, array has shape {self.rock.porosity.shape}"
            )
        if self.rock.permeability_m2.shape != (3, n):
            raise ValueError(
                f"rock.permeability_m2: grid has {n} cells, array has shape "
                f"{self.rock.permeability_m2.shape}"
            )
        for label, ref in (
            ("initial.pressure_pa", self.initial.pressure_pa),
            ("initial.sw", self.initial.sw),
        ):
            if ref is not None and ref.shape != (n,):
                raise ValueError(f"{label}: grid has {n} cells, array has shape {ref.shape}")
        outside = sorted(c for c in self.boundary.cells if not 0 <= c < n)
        if outside:
            raise ValueError(f"boundary.cells {outside} lie outside the grid of {n} cells")

    def _check_wells(self) -> None:
        n = self.grid.n_cells
        seen_ids: set[str] = set()
        claimed: dict[int, str] = {}
        for well in self.wells:
            if well.well_id in seen_ids:
                raise ValueError(f"duplicate well id {well.well_id!r}")
            seen_ids.add(well.well_id)
            for cell in well.cells:
                if not 0 <= cell < n:
                    raise ValueError(
                        f"well {well.well_id!r}: cell {cell} lies outside the grid of {n} cells"
                    )
                if cell in claimed:
                    raise ValueError(
                        f"connection cell {cell} is claimed by two wells: "
                        f"{claimed[cell]!r} and {well.well_id!r}"
                    )
                claimed[cell] = well.well_id

    def _check_controls(self) -> None:
        by_id = {w.well_id: w for w in self.wells}
        horizon = self.report_edges_s[-1]
        spans: dict[str, list[tuple[float, float]]] = {}
        for segment in self.controls:
            well = by_id.get(segment.well_id)
            if well is None:
                raise ValueError(f"control segment names an unknown well {segment.well_id!r}")
            if len(segment.connection_open) != len(well.cells):
                raise ValueError(
                    f"well {well.well_id!r}: connection_open has "
                    f"{len(segment.connection_open)} entries for {len(well.cells)} connections"
                )
            if segment.end_s > horizon:
                raise ValueError(
                    f"well {well.well_id!r}: control segment ends at {segment.end_s} s, past the "
                    f"last report edge at {horizon} s"
                )
            for start, end in spans.setdefault(segment.well_id, []):
                if segment.start_s < end and start < segment.end_s:
                    raise ValueError(
                        f"well {segment.well_id!r}: control segments overlap, "
                        f"[{start}, {end}) and [{segment.start_s}, {segment.end_s})"
                    )
            spans[segment.well_id].append((segment.start_s, segment.end_s))


class RestartRef(StrictModel):
    manifest_path: RelativePath
    sha256: Sha256
    completed_report_step: int = Field(ge=0)
    completed_time_s: float = Field(ge=0.0)
    model_hash: Sha256
    schedule_prefix_hash: Sha256
    environment_lock_hash: Sha256
    native_format: Literal["Jutul-native"] = "Jutul-native"


class OutputRequest(StrictModel):
    state_times_s: tuple[float, ...]
    keep_native_restart: bool
    chunk_months: int = Field(default=1, ge=1)
    diagnostic_substeps: bool = False

    @field_validator("state_times_s")
    @classmethod
    def _times_increase(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("state_times_s must request at least one state")
        if v[0] < 0.0:
            raise ValueError(f"state_times_s must be non-negative, got {v[0]}")
        if any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(f"state_times_s must be strictly increasing, got {v}")
        return v


class JobDescriptor(StrictModel):
    schema_version: Literal["job-1"] = JOB_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    case_path: RelativePath
    case_sha256: Sha256
    model_hash: Sha256
    solver_config_path: RelativePath
    solver_config_sha256: Sha256
    output_request: OutputRequest
    seed: int = Field(ge=0)
    result_dir: RelativePath
    # SPEC 3.3: the original attempt plus at most one registered numerical retry.
    attempt: int = Field(ge=1, le=MAX_ATTEMPTS)
    resume_from: RestartRef | None = None


class CostRecord(StrictModel):
    """Measured cost. None of this enters a model hash: it describes the machine, not F."""

    wall_s: float = Field(ge=0.0)
    cpu_s: float = Field(ge=0.0)
    peak_rss_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    accepted_steps: int = Field(ge=0)
    cut_steps: int = Field(ge=0)
    nonlinear_iterations: int = Field(ge=0)
    retry_count: int = Field(ge=0, le=MAX_ATTEMPTS - 1)
    measurement_method: str = Field(min_length=1)


class ForwardResult(StrictModel):
    schema_version: Literal["forward-1"] = FORWARD_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    case_sha256: Sha256
    model_hash: Sha256
    physics_class: str = Field(min_length=1)
    status: ForwardStatus
    reason: str | None = None
    completed_time_s: float = Field(ge=0.0)
    times_s: tuple[float, ...]
    states: dict[str, ArrayRef]
    monthly_path: RelativePath | None = None
    connections_path: RelativePath | None = None
    balances_path: RelativePath | None = None
    restart: RestartRef | None = None
    solver_metadata: dict[str, str]
    cost: CostRecord
    parent_attempt_ids: tuple[str, ...]

    @field_validator("times_s")
    @classmethod
    def _times_increase(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if v and v[0] < 0.0:
            raise ValueError(f"times_s must be non-negative, got {v[0]}")
        if any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(f"times_s must be strictly increasing, got {v}")
        return v

    @model_validator(mode="after")
    def _completeness_matches_the_status(self) -> ForwardResult:
        missing = [
            name
            for name, value in (
                ("monthly_path", self.monthly_path),
                ("connections_path", self.connections_path),
                ("balances_path", self.balances_path),
            )
            if value is None
        ]
        if self.status != "COMPLETE":
            # A failed or partial result may drop its outputs, but never silently: the
            # absence has to carry a reason (SPEC 18.4, plan 3.2).
            if not self.reason:
                raise ValueError(
                    f"an unsuccessful result ({self.status}) must give a reason for what is "
                    "missing; a null path with no reason is indistinguishable from a bug"
                )
            return self
        if self.reason is not None:
            raise ValueError(f"a COMPLETE result carries no reason, got {self.reason!r}")
        if missing:
            raise ValueError(f"a COMPLETE result must carry every result path; missing {missing}")
        if not self.states:
            raise ValueError("a COMPLETE result must carry the requested states")
        if not self.times_s:
            raise ValueError("a COMPLETE result must carry the time axis it covered")
        if self.completed_time_s != self.times_s[-1]:
            raise ValueError(
                f"a COMPLETE result ends at its last state time {self.times_s[-1]}, "
                f"got completed_time_s {self.completed_time_s}"
            )
        for name, ref in sorted(self.states.items()):
            # HDF5 states are (n_times, n_cells); the recorded axis_order is what proves it.
            if ref.axis_order != TIME_CELL_AXES:
                raise ValueError(
                    f"states[{name!r}]: axis_order must be {TIME_CELL_AXES}, got {ref.axis_order}"
                )
            if ref.shape[0] != len(self.times_s):
                raise ValueError(
                    f"states[{name!r}]: first axis has {ref.shape[0]} entries but times_s has "
                    f"{len(self.times_s)}"
                )
        return self


# InitialStateSpec and CaseBundle are declared in the order of the plan's table, which puts
# RestartRef after them; their forward reference to it is resolved here.
InitialStateSpec.model_rebuild()
CaseBundle.model_rebuild()
