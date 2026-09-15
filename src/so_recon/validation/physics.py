"""E01.9 — the physics evaluator: fixed tolerances, an analytic reference, and a verdict.

Three responsibilities, kept apart on purpose.

**The tolerances are data, not code.** `load_tolerances` reads `configs/e01_tolerances.yml`
and refuses anything outside the versioned key set (Task 9.1 plus the v2 absolute phase
limit): a missing key is a
refusal rather than a default, and an unknown key is a refusal rather than a value nobody
reads. A threshold that can be defaulted is a threshold nobody agreed to, and a scoring run
whose config silently gained a key is a run whose gate moved.

**The Buckley-Leverett reference is analytic.** `bl_saturation` is the exact weak solution
in the limit the fixture declares — Corey `n=2`, equal viscosity, `Swc=Sor=0`,
incompressible, horizontal, `Pc=0` — and there is no second numerical solver anywhere in
this project. `bl_cell_average` is the layer on top of it that makes a comparison legal at
all: a cell average is not a point value, and comparing a quasi-pointwise reference against
a cell-averaged answer puts the whole error of the shock's location into the verdict.

**The verdict reads PUBLISHED artifacts.** `evaluate_physics` opens the files a forward
really published — `states.h5`, `balances.parquet`, `connections.parquet` — and scores them.
Plan §12.9 has the stage validator read the results rather than re-run the suite, so a
metric that only exists inside a test is invisible to the gate that has to cite it. What the
evaluator brings of its own is the FIXTURE REGISTRY below: a verification case's geometry,
its injected volume and its horizon are part of the case's definition, not of its answer, so
they are declared here and the integration test proves the published case matches them.

Denominators, once. A rate or a volume is divided by `max(abs(reference), 1e-6)` in its own
unit and a pressure by `max(abs(reference), 1 Pa)`, exactly as `configs/e01_tolerances.yml`
states; a zero rate is checked absolutely and never as a relative error against zero.
Version 2 compares near-zero phase volumes against its explicit absolute tolerance,
with crossover absolute_limit / relative_limit. Raw phase-relative errors remain reported.

On the duplication with `validation.balance`: that module has carried
`CUMULATIVE_RELATIVE_TOLERANCE`, `MEDIAN_STEP_RELATIVE_TOLERANCE` and `BALANCE_FLOOR_M3_SC`
since Task 7, and its own tests assert against them. The config is the single source of
truth for what the EVALUATOR scores — nothing below ever reads those constants — while they
remain the defaults of the low-level helper and of `BalanceMetrics.within_spec_tolerance`.
`tests/unit/test_physics_metrics.py` asserts the two agree, so the duplication is pinned
rather than left to drift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml
from numpy.typing import NDArray
from pydantic import model_validator

from so_recon.config.schema import StrictModel
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.contracts import SECONDS_PER_DAY, STANDARD_GRAVITY_M_S2

#: Where the fixed tolerance block lives, relative to the repository root.
DEFAULT_TOLERANCES_RELPATH = "configs/e01_tolerances.yml"

#: The schema the tolerance block declares. A config from another schema is refused rather
#: than read field by field: the names would still resolve and would mean something else.
TOLERANCE_SCHEMA_VERSION = "e01-tolerances-2"

#: Task 9.1 tolerances plus the explicit v2 absolute phase-volume tolerance.
#: `load_tolerances` requires
#: EXACTLY these: see the module docstring for why neither direction is forgiving.
REQUIRED_TOLERANCE_KEYS: frozenset[str] = frozenset(
    {
        "balance_cumulative_relative_max",
        "balance_step_median_relative_max",
        "balance_cumulative_target",
        "balance_absolute_floor_m3_sc",
        "saturation_sum_abs_max",
        "saturation_bound_slack",
        "closed_connection_mass_kg_s_max",
        "closed_state_saturation_drift_max",
        "closed_pressure_relative_drift_max",
        "hydrostatic_gradient_relative_max",
        "hydrostatic_saturation_drift_max",
        "hydrostatic_pressure_relative_drift_max",
        "restart_saturation_abs_max",
        "restart_pressure_relative_max",
        "restart_volume_relative_max",
        "restart_inventory_relative_max",
        "rate_control_relative_max",
        "bl_pv_l1_max_at_128",
        "bl_refinement_ratio_max",
        "five_spot_symmetry_abs_max",
        "refinement_so_pv_mae_max",
        "refinement_inventory_relative_max",
        "refinement_monthly_volume_relative_max",
        "refinement_monthly_volume_absolute_max_m3_sc",
    }
)

#: Task 13: where the BLACK-OIL capability's own tolerance block lives, and what it declares.
#:
#: It is a separate file from the oil-water block because the capability is a separate gate
#: (plan 13.5). Adding keys to `configs/e01_tolerances.yml` would have changed that file's
#: digest, and every P0 and P1 suite report already published names that digest as the
#: agreement it was scored against — so the oil-water evidence would have been invalidated by
#: a capability that has nothing to do with it.
DEFAULT_BLACKOIL_TOLERANCES_RELPATH = "configs/e01_blackoil_tolerances.yml"
BLACKOIL_TOLERANCE_SCHEMA_VERSION = "e01-blackoil-tolerances-1"

REQUIRED_BLACKOIL_TOLERANCE_KEYS: frozenset[str] = frozenset(
    {
        "blackoil_saturation_sum_abs_max",
        "blackoil_gas_balance_relative_max",
        "blackoil_gas_inventory_closure_relative_max",
        "blackoil_free_gas_shortfall_max",
        "blackoil_bubble_point_shortfall_max",
        "blackoil_closed_saturation_drift_max",
        "blackoil_closed_gas_inventory_relative_max",
        "blackoil_restart_saturation_abs_max",
        "blackoil_restart_pressure_relative_max",
        "blackoil_restart_rs_relative_max",
        "blackoil_restart_inventory_relative_max",
        "blackoil_restart_surface_volume_relative_max",
    }
)


def load_blackoil_tolerances(path: Path) -> dict[str, float]:
    """Read the black-oil capability's fixed tolerance block, or refuse it.

    The same exact-key-set rule as `load_tolerances`, against a different key set and a
    different schema version. Nothing here falls back on the oil-water block: a black-oil
    threshold that silently borrowed a liquid one would be a gas number measured against a
    liquid agreement.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a tolerance config is a mapping, got {type(payload).__name__}")
    declared = payload.get("schema_version")
    if declared != BLACKOIL_TOLERANCE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: declares schema_version {declared!r}; this build reads "
            f"{BLACKOIL_TOLERANCE_SCHEMA_VERSION!r}"
        )
    values = {key: value for key, value in payload.items() if key != "schema_version"}
    missing = sorted(REQUIRED_BLACKOIL_TOLERANCE_KEYS - set(values))
    unknown = sorted(set(values) - REQUIRED_BLACKOIL_TOLERANCE_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path}: the black-oil tolerance block is not the one this build scores against. "
            f"Missing {missing}, unknown {unknown}; a missing threshold is never defaulted and "
            "an unknown one is never ignored"
        )
    out: dict[str, float] = {}
    for key, value in sorted(values.items()):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: {key} must be a number, got {value!r}")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{path}: {key} must be finite and positive, got {value!r}")
        out[key] = float(value)
    return out


#: The plan's relative denominators, stated once (Task 9.1).
VOLUME_DENOMINATOR_FLOOR = 1e-6
PRESSURE_DENOMINATOR_FLOOR_PA = 1.0

#: An L1 below this is the arithmetic's own zero rather than a discretisation error, and
#: dividing by it would manufacture a refinement ratio out of round-off. It is the plan's
#: own volume floor, applied to a pore-volume-weighted saturation error.
BL_L1_FLOOR = VOLUME_DENOMINATOR_FLOOR


def load_tolerances(path: Path) -> dict[str, float]:
    """Read the fixed tolerance block, or refuse it.

    Returns the numeric tolerances only: `schema_version` is a property of the file, not a
    threshold, so it is checked and dropped rather than handed on as if it were a number.
    """
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a tolerance config is a mapping, got {type(payload).__name__}")
    declared = payload.get("schema_version")
    if declared != TOLERANCE_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: declares schema_version {declared!r}; this build reads "
            f"{TOLERANCE_SCHEMA_VERSION!r}"
        )
    values = {key: value for key, value in payload.items() if key != "schema_version"}
    missing = sorted(REQUIRED_TOLERANCE_KEYS - set(values))
    unknown = sorted(set(values) - REQUIRED_TOLERANCE_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path}: the tolerance block is not the one this build scores against. "
            f"Missing {missing}, unknown {unknown}; a missing threshold is never defaulted "
            "and an unknown one is never ignored"
        )
    try:
        return require_tolerances(values)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def require_tolerances(tolerances: Mapping[str, float]) -> dict[str, float]:
    """The same exact-key-set rule, applied to a mapping a caller assembled by hand."""
    missing = sorted(REQUIRED_TOLERANCE_KEYS - set(tolerances))
    unknown = sorted(set(tolerances) - REQUIRED_TOLERANCE_KEYS)
    if missing or unknown:
        raise ValueError(
            f"the tolerance mapping is not the fixed block: missing {missing}, unknown "
            f"{unknown}; load it with load_tolerances({DEFAULT_TOLERANCES_RELPATH!r})"
        )
    out: dict[str, float] = {}
    for key, value in sorted(tolerances.items()):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key} must be a number, got {value!r}")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive, got {value!r}")
        out[key] = float(value)
    return out


# --------------------------------------------------------------------------------------
# 9.4 the Buckley-Leverett analytic reference
# --------------------------------------------------------------------------------------

#: The Welge tangent of the quadratic Corey pair at equal viscosity: `f(S)/S = f'(S)` has
#: the single root `S = 1/sqrt(2)`, and that is the shock saturation.
BL_SHOCK_SATURATION = 1 / math.sqrt(2)


def bl_saturation(x: NDArray[np.float64] | Sequence[float], t_pvi: float) -> NDArray[np.float64]:
    """Water saturation of the Buckley-Leverett solution at `x` (pore volumes) and `t_pvi`.

    The limit is the one the `:bl` fixture declares and nothing else: Corey exponents
    `(2, 2)`, equal phase viscosity, `Swc = Sor = 0`, incompressible, horizontal, `Pc = 0`.
    In that limit `f_w(S) = S^2 / (S^2 + (1-S)^2)`, the Welge tangent gives the shock at
    `S* = 1/sqrt(2)`, and behind the shock the rarefaction is the inverse of `f_w'`.

    `x` and `t_pvi` are both in pore volumes: `x` is the cumulative pore volume fraction
    from the injection face and `t_pvi` the injected pore volumes, so the solution is a pure
    function of `x/t` and carries no permeability, area or rate of its own.
    """
    x = np.asarray(x, dtype=np.float64)
    if t_pvi < 0 or (x < 0).any():
        raise ValueError("negative BL coordinate/time")
    if t_pvi == 0:
        return np.where(x == 0, 1.0, 0.0)
    shock_s = 1 / np.sqrt(2)
    grid = np.linspace(shock_s, 1.0, 20001)
    denominator = grid**2 + (1 - grid) ** 2
    derivative = 2 * grid * (1 - grid) / denominator**2
    xi = x / t_pvi
    behind = np.interp(xi, derivative[::-1], grid[::-1])
    return np.where(xi <= derivative[0], behind, 0.0)


def bl_front_position(t_pvi: float) -> float:
    """Where the shock is at `t_pvi`, in pore volumes: `x = t * f'(S*)`."""
    if t_pvi < 0:
        raise ValueError("negative BL coordinate/time")
    s = BL_SHOCK_SATURATION
    derivative = 2 * s * (1 - s) / (s**2 + (1 - s) ** 2) ** 2
    return float(t_pvi * derivative)


def bl_cell_average(
    edges_pv: NDArray[np.float64] | Sequence[float],
    t_pvi: float,
    *,
    subpoints: int = 64,
) -> NDArray[np.float64]:
    """The reference AVERAGED over each cell, by midpoint quadrature on `subpoints` samples.

    A finite-volume answer is a cell average, so comparing it against a quasi-pointwise
    reference charges the verdict for the shock's position inside whichever cell holds it —
    an error of up to the whole saturation jump, which on 128 cells is larger than the L1
    the result is gated on. The quadrature is a midpoint rule precisely because the solution
    is discontinuous: no sample ever lands on the jump, and the error in the one cell that
    contains it is bounded by half a subinterval of that cell.

    `edges_pv` is the cumulative pore volume fraction at every cell boundary, so the
    averages it returns are pore-volume averages even on a non-uniform grid.
    """
    edges = np.asarray(edges_pv, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"edges_pv describes the cell boundaries of a 1D column, got {edges.shape}"
        )
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges_pv must be strictly increasing")
    if subpoints < 1:
        raise ValueError(f"subpoints must be positive, got {subpoints}")
    left = edges[:-1, None]
    width = np.diff(edges)[:, None]
    offsets = (np.arange(subpoints, dtype=np.float64) + 0.5) / subpoints
    samples = bl_saturation((left + width * offsets[None, :]).ravel(), t_pvi)
    return np.asarray(samples.reshape(-1, subpoints).mean(axis=1), dtype=np.float64)


# --------------------------------------------------------------------------------------
# 10.3 the support aggregation
# --------------------------------------------------------------------------------------


def aggregate_so(
    so: NDArray[np.float64],
    pv: NDArray[np.float64],
    zone_id: NDArray[np.int64],
    n_zones: int,
) -> NDArray[np.float64]:
    """Pore-volume-weighted mean oil saturation of each of `n_zones` support zones.

    A support average is a PORE VOLUME average and never a cell-count one: two cells of the
    same zone at 0.2 and 0.8 with pore volumes 1 and 3 average to 0.65, not to 0.5, and a
    refinement comparison built on the unweighted mean would be comparing two different
    quantities as if they were one.

    A zone that collected NO pore volume comes back `nan`, never `0.0`. Zero is a saturation
    somebody will plot, cite and average further; the absence of a measurement is not a dry
    zone, and `np.divide(..., where=den > 0)` is what keeps the two apart.

    This is the KERNEL. Shape, finiteness and index-range validation happen in the caller
    (`compare_refinement` does them for the published states it reads) so that this function
    has exactly one behaviour and no opinion about where its inputs came from. Inventory is
    NOT aggregated here and must never be reconstructed as `mean So * mean B`: a mean of a
    ratio is not the ratio of the means, and `_component_inventory_m3_sc` sums it cell by
    cell instead.
    """
    num = np.bincount(zone_id, weights=pv * so, minlength=n_zones)
    den = np.bincount(zone_id, weights=pv, minlength=n_zones)
    mean: NDArray[np.float64] = np.divide(num, den, out=np.full(n_zones, np.nan), where=den > 0)
    return mean


# --------------------------------------------------------------------------------------
# 9.2 the record one check produces
# --------------------------------------------------------------------------------------

CheckStatus = Literal["PASS", "FAIL", "NOT_RUN"]


class PhysicsCheck(StrictModel):
    """What one verification fixture was scored on, and what came out.

    `metrics` is what was measured and `thresholds` is what it was measured against. A
    threshold with no metric beside it is an UNRUN metric, which is why a check that could
    not run still carries its thresholds: the report of plan 9.8 has to name the metrics
    that were never produced, and omitting them would make an absent check look like a
    passing one.

    `input_hashes` is the SHA-256 of every artifact that was read, keyed by the role it was
    read as, so the verdict names the bytes it is about and not merely the case it claims.
    """

    name: str
    status: CheckStatus
    metrics: dict[str, float]
    thresholds: dict[str, float]
    input_hashes: dict[str, str]
    evidence_paths: tuple[str, ...]
    reason: str | None

    @model_validator(mode="after")
    def _a_verdict_states_its_grounds(self) -> PhysicsCheck:
        if not self.name:
            raise ValueError("a physics check is named")
        if self.status == "PASS" and self.reason is not None:
            raise ValueError(f"{self.name}: a passing check carries no reason ({self.reason!r})")
        if self.status != "PASS" and not self.reason:
            raise ValueError(f"{self.name}: a {self.status} check has to say why")
        if self.status == "NOT_RUN" and self.metrics:
            raise ValueError(
                f"{self.name}: a check that did not run reports no metrics, got "
                f"{sorted(self.metrics)}"
            )
        if not all(np.isfinite(value) for value in self.metrics.values()):
            raise ValueError(f"{self.name}: a metric is nonfinite: {self.metrics}")
        return self

    @property
    def unrun_metrics(self) -> tuple[str, ...]:
        """The thresholds this check was to be scored against but produced no metric for."""
        gates = _FIXTURES[self.name].gates if self.name in _FIXTURES else ()
        return tuple(gate.metric for gate in gates if gate.metric not in self.metrics)


# --------------------------------------------------------------------------------------
# the fixture registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One scored comparison: `metric` against `threshold`, in the stated direction."""

    metric: str
    threshold: str
    direction: Literal["at_most", "greater_than"]

    def failed(self, value: float, limit: float) -> bool:
        return value > limit if self.direction == "at_most" else not value > limit


@dataclass(frozen=True)
class Fixture:
    """A registered verification case: what it is, what it publishes, how it is scored.

    The constants are the CASE's, not the answer's — a fixture's geometry, its injected
    volume and its horizon are decided when the fixture is written, so they belong here and
    the integration test proves the published case carries them.
    """

    family: Literal["closed", "hydrostatic", "segregation", "bl", "five_spot", "refinement"]
    roles: tuple[str, ...]
    gates: tuple[Gate, ...]
    #: Thresholds that are not tolerances: a sign, a direction, a structural zero.
    structural_thresholds: dict[str, float]
    #: Depth of each cell centre, m, for the vertical fixtures. Empty otherwise.
    cell_center_depth_m: tuple[float, ...] = ()
    #: `(nx, ny, nz)` of the case, for the areal fixtures whose metric is a reflection.
    grid_shape: tuple[int, int, int] = ()  # type: ignore[assignment]
    #: Pore volumes of water injected over `horizon_s`, for the BL fixture.
    injected_pore_volumes: float = 0.0
    horizon_s: float = 0.0


_BALANCE_GATES: tuple[Gate, ...] = (
    Gate("balance_cumulative_relative", "balance_cumulative_relative_max", "at_most"),
    Gate("balance_step_median_relative", "balance_step_median_relative_max", "at_most"),
)
_SATURATION_GATES: tuple[Gate, ...] = (
    Gate("saturation_sum_abs", "saturation_sum_abs_max", "at_most"),
    Gate("saturation_bound_violation", "saturation_bound_slack", "at_most"),
)
_CONNECTION_GATE = Gate("connection_mass_kg_s", "closed_connection_mass_kg_s_max", "at_most")

_CLOSED_GATES: tuple[Gate, ...] = (
    *_BALANCE_GATES,
    *_SATURATION_GATES,
    _CONNECTION_GATE,
    Gate("state_saturation_drift", "closed_state_saturation_drift_max", "at_most"),
    Gate("pressure_relative_drift", "closed_pressure_relative_drift_max", "at_most"),
)

_HYDROSTATIC_GATES: tuple[Gate, ...] = (
    *_BALANCE_GATES,
    *_SATURATION_GATES,
    _CONNECTION_GATE,
    Gate("hydrostatic_face_residual_relative", "hydrostatic_gradient_relative_max", "at_most"),
    Gate("state_saturation_drift", "hydrostatic_saturation_drift_max", "at_most"),
    Gate("pressure_relative_drift", "hydrostatic_pressure_relative_drift_max", "at_most"),
)

#: The segregation case is a DIRECTION claim, and the plan fixes no tolerance for a
#: direction: heavy water above light oil has to move down, by any amount at all. The zero
#: below is therefore a structural requirement and not a threshold anybody may tune.
#: The drift gates are deliberately ABSENT: this column is supposed to move, and gating it on
#: not moving would be gating it on the opposite of what it demonstrates. The balance gates
#: are present for the same reason they are everywhere else — a closed column that segregates
#: still has to conserve both components while it does it.
_SEGREGATION_GATES: tuple[Gate, ...] = (
    *_BALANCE_GATES,
    *_SATURATION_GATES,
    _CONNECTION_GATE,
    Gate("water_mean_depth_increase_m", "water_mean_depth_increase_min_m", "greater_than"),
)

#: The five-spot is a SYMMETRY claim about a field that has to vary. The symmetry gate alone
#: would be satisfied by a uniform saturation, which is symmetric about every axis and says
#: nothing at all, so the structural companion requires the answer to have a range; the zero
#: below is a structural requirement and not a threshold anybody may tune. The connection gate
#: is deliberately ABSENT — every connection of this pattern is open and carrying flow.
_FIVE_SPOT_GATES: tuple[Gate, ...] = (
    *_BALANCE_GATES,
    *_SATURATION_GATES,
    Gate("five_spot_symmetry_abs", "five_spot_symmetry_abs_max", "at_most"),
    Gate("five_spot_so_range", "five_spot_so_range_min", "greater_than"),
)

#: What a refinement pair is scored on. `support_pore_volume_relative` is structural: two
#: grids that tile the same physical zones carry the same pore volume in each of them to
#: round-off, and a mapping that is not pore-volume-conservative is a mapping whose saturation
#: comparison means nothing — so it is checked FIRST and it is not a tolerance to widen.
_REFINEMENT_GATES: tuple[Gate, ...] = (
    Gate("support_pore_volume_relative", "support_pore_volume_relative_max", "at_most"),
    Gate("so_pv_mae", "refinement_so_pv_mae_max", "at_most"),
    Gate("inventory_relative", "refinement_inventory_relative_max", "at_most"),
    Gate("monthly_volume_relative", "refinement_monthly_volume_relative_max", "at_most"),
    Gate(
        "monthly_volume_near_zero_absolute_m3_sc",
        "refinement_monthly_volume_absolute_max_m3_sc",
        "at_most",
    ),
)

_BL_GATES: tuple[Gate, ...] = (
    *_BALANCE_GATES,
    *_SATURATION_GATES,
    Gate("bl_pv_l1_at_128", "bl_pv_l1_max_at_128", "at_most"),
    Gate("bl_refinement_ratio_64_to_128", "bl_refinement_ratio_max", "at_most"),
)

#: The E01 verification column: two 50 m cells whose top face is at the 1000 m fixture
#: datum, so the centres are at 1025 m and 1075 m. z is depth, positive down, and ABSOLUTE.
_COLUMN_DEPTHS_M = (1025.0, 1075.0)

_CLOSED_ROLES = ("states", "balances", "connections")

#: The 10.7 five-spot: 16 x 16 cells in one layer, over 400 x 400 x 10 m.
FIVE_SPOT_SHAPE: tuple[int, int, int] = (16, 16, 1)

#: How exactly two nested Cartesian grids have to agree on the pore volume of a shared zone.
#: Not a physical tolerance: the zones are unions of whole cells on both grids and the rock is
#: the same continuous field, so the two sums differ only by the order they were added in.
SUPPORT_PV_EXACTNESS = 1e-9

_FIXTURES: dict[str, Fixture] = {
    "closed_cell_pvt": Fixture(
        family="closed", roles=_CLOSED_ROLES, gates=_CLOSED_GATES, structural_thresholds={}
    ),
    "closed_box_pvt": Fixture(
        family="closed", roles=_CLOSED_ROLES, gates=_CLOSED_GATES, structural_thresholds={}
    ),
    "hydrostatic": Fixture(
        family="hydrostatic",
        roles=_CLOSED_ROLES,
        gates=_HYDROSTATIC_GATES,
        structural_thresholds={},
        cell_center_depth_m=_COLUMN_DEPTHS_M,
    ),
    "segregation": Fixture(
        family="segregation",
        roles=_CLOSED_ROLES,
        gates=_SEGREGATION_GATES,
        structural_thresholds={"water_mean_depth_increase_min_m": 0.0},
        cell_center_depth_m=_COLUMN_DEPTHS_M,
    ),
    "five_spot": Fixture(
        family="five_spot",
        roles=_CLOSED_ROLES,
        gates=_FIVE_SPOT_GATES,
        structural_thresholds={"five_spot_so_range_min": 0.0},
        grid_shape=FIVE_SPOT_SHAPE,
    ),
    "five_spot_refinement": Fixture(
        family="refinement",
        roles=("coarse_states", "coarse_monthly", "fine_states", "fine_monthly"),
        gates=_REFINEMENT_GATES,
        structural_thresholds={"support_pore_volume_relative_max": SUPPORT_PV_EXACTNESS},
    ),
    "bl": Fixture(
        family="bl",
        roles=("states_32", "states_64", "states_128", "balances_128"),
        gates=_BL_GATES,
        structural_thresholds={},
        injected_pore_volumes=0.2,
        horizon_s=SECONDS_PER_DAY,
    ),
}

#: Reference density of water at standard conditions (plan §3.1). The published `bw` is
#: `rho_w_sc / rho_w(p)`, so this is what turns it back into the density the flux used.
RHO_W_SC_KG_M3 = 1000.0


def registered_fixtures() -> tuple[str, ...]:
    return tuple(sorted(_FIXTURES))


# --------------------------------------------------------------------------------------
# reading published artifacts
# --------------------------------------------------------------------------------------

_STATE_DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "pressure_pa": ("Pa", ("time", "cell")),
    "sw": ("1", ("time", "cell")),
    "so": ("1", ("time", "cell")),
    "pore_volume_m3": ("m3", ("time", "cell")),
    "bw": ("1", ("time", "cell")),
    "bo": ("1", ("time", "cell")),
    "time_s": ("s", ("time",)),
}


class ArtifactUnreadable(RuntimeError):
    """A published artifact this check needs is absent, or is not what it claims to be."""


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_states(path: Path) -> dict[str, NDArray[np.float64]]:
    """Read a published `states.h5`, proving each dataset's unit and axis order first."""
    if not path.is_file():
        raise ArtifactUnreadable(f"{path}: no such published states file")
    fields: dict[str, NDArray[np.float64]] = {}
    try:
        with h5py.File(path, "r") as handle:
            for name, (unit, axes) in _STATE_DATASETS.items():
                if name not in handle:
                    raise ArtifactUnreadable(f"{path}: dataset {name!r} is missing")
                dataset = handle[name]
                stored_unit = _decode(dataset.attrs.get("unit", ""))
                stored_axes = tuple(_decode(a) for a in dataset.attrs.get("axis_order", ()))
                if stored_unit != unit or stored_axes != axes:
                    raise ArtifactUnreadable(
                        f"{path}: {name} records unit {stored_unit!r} and axis_order "
                        f"{stored_axes}, expected {unit!r} and {axes}"
                    )
                fields[name] = np.asarray(dataset[()], dtype=np.float64)
    except OSError as exc:
        raise ArtifactUnreadable(f"{path}: cannot be read as HDF5: {exc}") from exc
    times = fields["time_s"]
    for name in ("pressure_pa", "sw", "so", "pore_volume_m3", "bw", "bo"):
        if fields[name].shape[0] != times.size:
            raise ArtifactUnreadable(
                f"{path}: {name} has {fields[name].shape[0]} rows for {times.size} times"
            )
    if not all(np.isfinite(values).all() for values in fields.values()):
        raise ArtifactUnreadable(f"{path}: the published states hold nonfinite values")
    return fields


def _read_table(path: Path, columns: Sequence[str]) -> dict[str, list[Any]]:
    if not path.is_file():
        raise ArtifactUnreadable(f"{path}: no such published table")
    try:
        table = pq.read_table(path)
    except (OSError, ValueError) as exc:
        raise ArtifactUnreadable(f"{path}: cannot be read as Parquet: {exc}") from exc
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise ArtifactUnreadable(f"{path}: the table has no column {missing}")
    if table.num_rows == 0:
        raise ArtifactUnreadable(f"{path}: the published table is empty")
    return {name: table.column(name).to_pylist() for name in columns}


# --------------------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------------------


def _balance_metrics(path: Path, target: float) -> dict[str, float]:
    """The worst of BOTH published balances, over both components.

    Task 7 publishes two labelled statements — the whole model against the surface flux and
    the reservoir against the connection flux — so a reader that assumed one row per
    component would score half the table. The `balance` column is carried into the metric
    names' company below by taking the maximum over every row: a residual in either
    statement is a residual.
    """
    columns = (
        "balance",
        "component",
        "cumulative_relative",
        "median_step_relative",
        "max_step_relative",
        "absolute_residual_m3_sc",
    )
    rows = _read_table(path, columns)
    labels = sorted(set(rows["balance"]))
    if len(labels) < 2:
        raise ArtifactUnreadable(
            f"{path}: only balance {labels} is published; a forward publishes both the "
            "surface-source and the connection-flux statement"
        )
    cumulative = max(float(v) for v in rows["cumulative_relative"])
    return {
        "balance_cumulative_relative": cumulative,
        "balance_step_median_relative": max(float(v) for v in rows["median_step_relative"]),
        "balance_max_step_relative": max(float(v) for v in rows["max_step_relative"]),
        "balance_absolute_residual_m3_sc": max(float(v) for v in rows["absolute_residual_m3_sc"]),
        # SPEC 8.3's stricter aim, reported beside SPEC 23.1's gate and never as it.
        "balance_cumulative_target_met": 1.0 if cumulative <= target else 0.0,
    }


def _saturation_metrics(states: Mapping[str, NDArray[np.float64]]) -> dict[str, float]:
    sw, so = states["sw"], states["so"]
    violation = max(
        0.0,
        float(-sw.min()),
        float(sw.max() - 1.0),
        float(-so.min()),
        float(so.max() - 1.0),
    )
    return {
        "saturation_sum_abs": float(np.abs(sw + so - 1.0).max()),
        "saturation_bound_violation": violation,
    }


def _drift_metrics(states: Mapping[str, NDArray[np.float64]]) -> dict[str, float]:
    """How far the published state moved from the one the case started at.

    The reference is the state at the FIRST published time, which for every fixture here is
    the case's own initial state, so the drift is against what was put in rather than
    against the previous step.
    """
    sw, so, pressure = states["sw"], states["so"], states["pressure_pa"]
    saturation_drift = max(float(np.abs(sw - sw[0]).max()), float(np.abs(so - so[0]).max()))
    denominator = np.maximum(np.abs(pressure[0]), PRESSURE_DENOMINATOR_FLOOR_PA)
    return {
        "state_saturation_drift": saturation_drift,
        "pressure_relative_drift": float((np.abs(pressure - pressure[0]) / denominator).max()),
        "n_published_times": float(states["time_s"].size),
    }


def _connection_metrics(path: Path) -> dict[str, float]:
    """The largest mean connection mass rate any perforation carried, kg/s.

    `connections.parquet` integrates the native cross term over each report interval, so the
    published number is a mass in kg; the tolerance is a rate, and dividing by the interval
    the mass was integrated over is what turns one into the other.
    """
    rows = _read_table(path, ("total_mass_kg", "start_s", "end_s"))
    worst = 0.0
    for mass, start, end in zip(rows["total_mass_kg"], rows["start_s"], rows["end_s"], strict=True):
        duration = float(end) - float(start)
        if duration <= 0.0:
            raise ArtifactUnreadable(
                f"{path}: a connection row spans [{start}, {end}) s, which is not an interval"
            )
        worst = max(worst, abs(float(mass)) / duration)
    return {"connection_mass_kg_s": worst}


def _hydrostatic_metrics(
    states: Mapping[str, NDArray[np.float64]], depths: tuple[float, ...]
) -> dict[str, float]:
    """The DISCRETE hydrostatic face residual of the column before its first timestep.

    This is the native two-point statement and not a continuous gradient. JutulDarcy forms
    the phase potential difference across a face as `-T*(p[r] - p[l] + gdz*rho_face)` with
    `gdz = -g*(z[r] - z[l])` (`Jutul.compute_face_gdz`) and `rho_face` the ARITHMETIC average
    of the two cell densities (`Jutul.face_average`), so equilibrium is exactly

        p[r] - p[l] = g * (z[r] - z[l]) * (rho[l] + rho[r]) / 2

    and the residual below is what that flux would be driven by. The density is the model's
    own: `bw` is published as `rho_w_sc / rho_w(p)`, so `rho_w_sc / bw` is the density the
    flux used rather than an analytic formula evaluated a second time.
    """
    pressure = states["pressure_pa"][0]
    if pressure.size != len(depths):
        raise ArtifactUnreadable(
            f"the published column has {pressure.size} cells; this fixture is a column of "
            f"{len(depths)} cells at {depths} m"
        )
    z = np.asarray(depths, dtype=np.float64)
    rho = RHO_W_SC_KG_M3 / states["bw"][0]
    delta_z = np.diff(z)
    head = STANDARD_GRAVITY_M_S2 * delta_z * 0.5 * (rho[:-1] + rho[1:])
    residual = np.diff(pressure) - head
    denominator = np.maximum(np.abs(head), PRESSURE_DENOMINATOR_FLOOR_PA)
    gradient = np.diff(pressure) / delta_z
    return {
        "hydrostatic_face_residual_relative": float((np.abs(residual) / denominator).max()),
        "hydrostatic_face_residual_pa": float(np.abs(residual).max()),
        # Positive dp/dz is what agrees with z-down; the residual gate already implies it,
        # because a wrong-signed gradient misses the head by twice the head.
        "pressure_depth_gradient_pa_m": float(gradient.min()),
    }


def _segregation_metrics(
    states: Mapping[str, NDArray[np.float64]], depths: tuple[float, ...]
) -> dict[str, float]:
    """Where the water is, at the start and at the end, weighted by the water that is there.

    The water is weighted by its STANDARD volume, `Sw * PV / Bw`, and not by the reservoir
    volume `Sw * PV`. The two give the same centre of mass to within the compressibility, but
    only the standard one is conserved: a closed column whose pressure redistributes as the
    heavy phase falls changes `sum(Sw * PV)` by `c_w * dp` — measured at 1.6e-5 relative over
    this fixture's thirty days — and `water_volume_relative_change` is reported precisely so
    that a leak can be told apart from the fluid being compressible.

    The mean depth of the water is then `sum(V_w * z) / sum(V_w)`. Its mirror image, the
    centre of HEIGHT above the deepest cell centre, is reported beside it because the plan
    names both; the two are the same statement with opposite sign, and saying so is cheaper
    than leaving a reader to wonder whether they are two independent checks.
    """
    sw, pore, times = states["sw"], states["pore_volume_m3"], states["time_s"]
    if sw.shape[1] != len(depths):
        raise ArtifactUnreadable(
            f"the published column has {sw.shape[1]} cells; this fixture is a column of "
            f"{len(depths)} cells at {depths} m"
        )
    if times.size < 2:
        raise ArtifactUnreadable("a segregation fixture publishes at least two states")
    z = np.asarray(depths, dtype=np.float64)
    bw = states["bw"]
    if not (bw > 0.0).all():
        raise ArtifactUnreadable("the published bw is not positive; it cannot weigh a volume")
    water = sw * pore / bw
    totals = water.sum(axis=1)
    if not (totals > 0.0).all():
        raise ArtifactUnreadable("the published column holds no water to segregate")
    mean_depth = (water * z[None, :]).sum(axis=1) / totals
    base = float(z.max())
    return {
        "water_mean_depth_start_m": float(mean_depth[0]),
        "water_mean_depth_end_m": float(mean_depth[-1]),
        "water_mean_depth_increase_m": float(mean_depth[-1] - mean_depth[0]),
        "water_center_of_height_change_m": float((base - mean_depth[-1]) - (base - mean_depth[0])),
        "water_volume_relative_change": float(
            abs(totals[-1] - totals[0]) / max(abs(totals[0]), VOLUME_DENOMINATOR_FLOOR)
        ),
    }


def _bl_grid_error(
    states: Mapping[str, NDArray[np.float64]], fixture: Fixture, subpoints: int = 64
) -> tuple[float, float, dict[str, float]]:
    """The pore-volume L1 of one BL grid against the cell-averaged analytic reference.

    The coordinate is the cumulative pore volume fraction taken from the PUBLISHED pore
    volumes, so a non-uniform column would still be compared in the coordinate the analytic
    solution is written in. `t_pvi` comes from the fixture's own injection — the injected
    volume is part of the case, not of the answer.
    """
    sw = states["sw"][-1]
    pore = states["pore_volume_m3"][-1]
    time_s = float(states["time_s"][-1])
    total = float(pore.sum())
    if not total > 0.0:
        raise ArtifactUnreadable("the published BL column has no pore volume")
    edges = np.concatenate([[0.0], np.cumsum(pore) / total])
    t_pvi = fixture.injected_pore_volumes * time_s / fixture.horizon_s
    reference = bl_cell_average(edges, t_pvi, subpoints=subpoints)
    finer = bl_cell_average(edges, t_pvi, subpoints=2 * subpoints)
    weights = pore / total
    l1 = float(np.sum(np.abs(sw - reference) * weights))
    quadrature = float(np.sum(np.abs(reference - finer) * weights))
    extra = {
        "t_pvi": t_pvi,
        "numerical_volume_pv": float(np.sum(sw * weights)),
        "reference_volume_pv": float(np.sum(reference * weights)),
        # A first-order upwind answer is monotone in x; a positive number here is over- or
        # undershoot, which the L1 alone would average away.
        "profile_monotonicity_violation": float(max(0.0, float(np.diff(sw).max()))),
    }
    return l1, quadrature, extra


def _bl_metrics(
    grids: Mapping[int, Mapping[str, NDArray[np.float64]]], fixture: Fixture
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    errors: dict[int, float] = {}
    for n_cells in sorted(grids):
        states = grids[n_cells]
        if states["sw"].shape[1] != n_cells:
            raise ArtifactUnreadable(
                f"the states published as the {n_cells}-cell BL grid describe "
                f"{states['sw'].shape[1]} cells"
            )
        l1, quadrature, extra = _bl_grid_error(states, fixture)
        errors[n_cells] = l1
        metrics[f"bl_pv_l1_at_{n_cells}"] = l1
        metrics[f"bl_quadrature_delta_64_vs_128_at_{n_cells}"] = quadrature
        metrics[f"bl_numerical_volume_pv_at_{n_cells}"] = extra["numerical_volume_pv"]
        metrics[f"bl_profile_monotonicity_violation_at_{n_cells}"] = extra[
            "profile_monotonicity_violation"
        ]
    for coarse, fine in ((32, 64), (64, 128)):
        metrics[f"bl_refinement_ratio_{coarse}_to_{fine}"] = errors[fine] / max(
            errors[coarse], BL_L1_FLOOR
        )
    t_pvi = (
        fixture.injected_pore_volumes * float(grids[max(grids)]["time_s"][-1]) / fixture.horizon_s
    )
    metrics["bl_t_pvi"] = t_pvi
    metrics["bl_front_position_pv"] = bl_front_position(t_pvi)
    metrics["bl_reference_volume_error_pv"] = abs(
        _bl_grid_error(grids[max(grids)], fixture)[2]["reference_volume_pv"] - t_pvi
    )
    return metrics


# --------------------------------------------------------------------------------------
# 10.7 the five-spot: a reflection, measured
# --------------------------------------------------------------------------------------


def mirror_symmetry_abs(
    field: NDArray[np.float64], shape: tuple[int, int, int]
) -> tuple[float, float]:
    """The largest `|f - reflect(f)|` of an areal field about each of the two grid axes.

    `shape` is `(nx, ny, nz)` with a single layer, and cells are numbered `i + nx*j` exactly
    as the exchange declares. The reflections are `i -> nx-1-i` and `j -> ny-1-j`, which are
    the mirrors the five-spot's wells are placed about: the injectors sit at `2` and `13` on
    a 16-cell axis and the producer's four cells at `7` and `8`, so a converged answer has to
    be symmetric about both and nothing in the fixture forces it to be.
    """
    nx, ny, nz = shape
    if nz != 1:
        raise ArtifactUnreadable(f"an areal reflection needs a single layer, got shape {shape}")
    if field.size != nx * ny:
        raise ArtifactUnreadable(
            f"the published field has {field.size} cells; this fixture is {nx}x{ny}"
        )
    grid = np.asarray(field, dtype=np.float64).reshape(ny, nx)
    return (
        float(np.abs(grid - grid[:, ::-1]).max()),
        float(np.abs(grid - grid[::-1, :]).max()),
    )


def _five_spot_metrics(
    states: Mapping[str, NDArray[np.float64]], fixture: Fixture
) -> dict[str, float]:
    """The symmetry of the LAST published oil saturation, and the range it varies over."""
    final = states["so"][-1]
    about_x, about_y = mirror_symmetry_abs(final, fixture.grid_shape)
    return {
        "five_spot_symmetry_x_abs": about_x,
        "five_spot_symmetry_y_abs": about_y,
        "five_spot_symmetry_abs": max(about_x, about_y),
        # Non-vacuity: a uniform field is symmetric about every axis and demonstrates nothing.
        "five_spot_so_range": float(final.max() - final.min()),
    }


# --------------------------------------------------------------------------------------
# 10.8 the refinement comparison
# --------------------------------------------------------------------------------------


def cartesian_zone_ids(nx: int, ny: int, zones_per_side: int) -> NDArray[np.int64]:
    """The support zone each cell of an `nx x ny` areal grid falls in, zero-based.

    `zones_per_side` must divide both axes exactly, which is what makes the support a
    partition of WHOLE cells: every cell lies entirely inside one zone, so its intersection
    pore volume with that zone is its own pore volume and two nested grids correspond exactly
    without any geometry library. A support that cut cells would need real intersection areas
    and is refused here rather than approximated.
    """
    if nx <= 0 or ny <= 0 or zones_per_side <= 0:
        raise ValueError(f"a support of {zones_per_side} zones over {nx}x{ny} is not a grid")
    if nx % zones_per_side or ny % zones_per_side:
        raise ValueError(
            f"a {zones_per_side}-zone support does not tile an {nx}x{ny} grid into whole "
            "cells; the zones of a refinement comparison are unions of cells on BOTH grids"
        )
    fx, fy = nx // zones_per_side, ny // zones_per_side
    i = np.arange(nx) // fx
    j = np.arange(ny) // fy
    return np.asarray(
        (i[None, :] + zones_per_side * j[:, None]).ravel().astype(np.int64), dtype=np.int64
    )


@dataclass(frozen=True)
class CommonSupport:
    """The fixed physical zones two grids are compared on, and which cell falls in each.

    This is the `common_support` of plan 10.8: the zones are physical, they are the same on
    both grids, and because the grids are nested Cartesian ones every cell lies wholly inside
    one zone — so the intersection pore volume of a cell with its zone is the cell's own pore
    volume and `compare_refinement` can take it from the published states. The check that the
    two grids really do carry the same pore volume in each zone is the FIRST thing scored, so
    a mapping that is not pore-volume-conservative is caught before any saturation is compared.
    """

    name: str
    n_zones: int
    coarse_zone_id: NDArray[np.int64]
    fine_zone_id: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.n_zones <= 0:
            raise ValueError(f"{self.name}: a support has at least one zone, got {self.n_zones}")
        for label, ids in (("coarse", self.coarse_zone_id), ("fine", self.fine_zone_id)):
            array = np.asarray(ids)
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{self.name}: the {label} zone map is one id per cell")
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{self.name}: zone ids are integers, got {array.dtype}")
            if array.min() < 0 or array.max() >= self.n_zones:
                raise ValueError(
                    f"{self.name}: a {label} zone id lies outside [0, {self.n_zones}); the "
                    "caller validates the index range before the aggregation, not after"
                )
            if np.unique(array).size != self.n_zones:
                raise ValueError(
                    f"{self.name}: the {label} grid leaves some of the {self.n_zones} zones "
                    "empty; an empty zone is an unmeasured zone and is never a zero"
                )


#: What the refinement comparison reads from each published result.
_REFINEMENT_ROLES = ("states", "monthly")

#: The monthly phase integrals of one result, per month, in the order they are compared.
_MONTHLY_VOLUME_COLUMNS = ("oil_prod_m3_sc", "water_prod_m3_sc", "water_inj_m3_sc")


def _component_inventory_m3_sc(
    states: Mapping[str, NDArray[np.float64]], index: int
) -> dict[str, float]:
    """Standard-volume inventory of each component at one published time.

    Summed CELL BY CELL: `sum(S * PV / B)`, never `mean(S) * mean(B) * PV`. The mean of a
    ratio is not the ratio of the means, and a refinement comparison built on the latter would
    compare two different quantities and call the difference discretisation error.
    """
    pore = states["pore_volume_m3"][index]
    return {
        "water": float(np.sum(states["sw"][index] * pore / states["bw"][index])),
        "oil": float(np.sum(states["so"][index] * pore / states["bo"][index])),
    }


def _monthly_volumes(path: Path) -> dict[str, NDArray[np.float64]]:
    """Every well's monthly phase integrals, summed to the field, one row per month."""
    rows = _read_table(path, ("month_index", *_MONTHLY_VOLUME_COLUMNS))
    months = rows["month_index"]
    if any(isinstance(month, bool) or not isinstance(month, int) or month < 0 for month in months):
        raise ArtifactUnreadable(f"{path}: month_index must contain nonnegative integers")
    unique_months = set(months)
    # Check contiguity before allocating: a corrupt huge index must not request a
    # huge array, and a missing month must not silently become a zero-volume month.
    if unique_months != set(range(len(unique_months))):
        raise ArtifactUnreadable(f"{path}: month_index must cover consecutive months starting at 0")
    n_months = len(unique_months)
    out: dict[str, NDArray[np.float64]] = {}
    for column in _MONTHLY_VOLUME_COLUMNS:
        totals = np.zeros(n_months, dtype=np.float64)
        for month, value in zip(months, rows[column], strict=True):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ArtifactUnreadable(
                    f"{path}: {column} must contain finite nonnegative volumes"
                )
            total = float(totals[month]) + float(value)
            if not math.isfinite(total):
                raise ArtifactUnreadable(f"{path}: {column} monthly sum is nonfinite")
            totals[month] = total
        out[column] = totals

    return out


def _refinement_metrics(
    coarse: Mapping[str, Path],
    fine: Mapping[str, Path],
    support: CommonSupport,
    tolerances: Mapping[str, float],
) -> dict[str, float]:
    grids: dict[str, dict[str, Any]] = {}
    for label, outputs, zone_id in (
        ("coarse", coarse, support.coarse_zone_id),
        ("fine", fine, support.fine_zone_id),
    ):
        states = _read_states(outputs["states"])
        so = np.asarray(states["so"][-1], dtype=np.float64)
        pore = np.asarray(states["pore_volume_m3"][-1], dtype=np.float64)
        if so.size != zone_id.size:
            raise ArtifactUnreadable(
                f"{support.name}: the {label} result describes {so.size} cells and its zone "
                f"map covers {zone_id.size}"
            )
        if not np.isfinite(so).all() or not np.isfinite(pore).all():
            raise ArtifactUnreadable(f"{support.name}: the {label} result is not finite")
        if (pore < 0.0).any():
            raise ArtifactUnreadable(f"{support.name}: the {label} result has a negative volume")
        grids[label] = {
            "zone_so": aggregate_so(so, pore, zone_id, support.n_zones),
            "zone_pv": np.bincount(zone_id, weights=pore, minlength=support.n_zones),
            "inventory": _component_inventory_m3_sc(states, -1),
            "monthly": _monthly_volumes(outputs["monthly"]),
            "n_cells": float(so.size),
        }

    coarse_pv = grids["coarse"]["zone_pv"]
    fine_pv = grids["fine"]["zone_pv"]
    pv_denominator = np.maximum(np.abs(coarse_pv), VOLUME_DENOMINATOR_FLOOR)
    zone_so_coarse = grids["coarse"]["zone_so"]
    zone_so_fine = grids["fine"]["zone_so"]
    if not np.isfinite(zone_so_coarse).all() or not np.isfinite(zone_so_fine).all():
        raise ArtifactUnreadable(
            f"{support.name}: a support zone collected no pore volume, so its mean saturation "
            "is unmeasured; an unmeasured zone is never compared as if it were a dry one"
        )
    weights = coarse_pv / max(float(coarse_pv.sum()), VOLUME_DENOMINATOR_FLOOR)
    zone_difference = np.abs(zone_so_coarse - zone_so_fine)

    inventory_relative = max(
        abs(grids["fine"]["inventory"][name] - grids["coarse"]["inventory"][name])
        / max(abs(grids["coarse"]["inventory"][name]), VOLUME_DENOMINATOR_FLOOR)
        for name in ("water", "oil")
    )

    coarse_monthly = grids["coarse"]["monthly"]
    fine_monthly = grids["fine"]["monthly"]
    n_months = len(coarse_monthly[_MONTHLY_VOLUME_COLUMNS[0]])
    if n_months != len(fine_monthly[_MONTHLY_VOLUME_COLUMNS[0]]):
        raise ArtifactUnreadable(
            f"{support.name}: the two results report {n_months} and "
            f"{len(fine_monthly[_MONTHLY_VOLUME_COLUMNS[0]])} months; a refinement compares "
            "the same horizon"
        )
    # v2: 2% of each phase's OWN reference volume, with an explicit 1 ml
    # absolute tolerance below their crossover. The raw relative number is still
    # reported. Neither oil throughput nor injection can dilute a water error.
    absolute_limit = tolerances["refinement_monthly_volume_absolute_max_m3_sc"]
    relative_limit = tolerances["refinement_monthly_volume_relative_max"]
    crossover = absolute_limit / relative_limit
    monthly_relative = 0.0
    monthly_self_relative = 0.0
    monthly_near_zero_absolute = 0.0
    monthly_absolute = 0.0
    for column in _MONTHLY_VOLUME_COLUMNS:
        difference = np.abs(fine_monthly[column] - coarse_monthly[column])
        own = np.abs(coarse_monthly[column])
        near_zero = own <= crossover
        monthly_absolute = max(monthly_absolute, float(difference.max()))
        monthly_self_relative = max(
            monthly_self_relative,
            float((difference / np.maximum(own, VOLUME_DENOMINATOR_FLOOR)).max()),
        )
        if near_zero.any():
            monthly_near_zero_absolute = max(
                monthly_near_zero_absolute, float(difference[near_zero].max())
            )
        if (~near_zero).any():
            monthly_relative = max(
                monthly_relative, float((difference[~near_zero] / own[~near_zero]).max())
            )

    return {
        "support_pore_volume_relative": float((np.abs(fine_pv - coarse_pv) / pv_denominator).max()),
        "so_pv_mae": float(np.sum(zone_difference * weights)),
        "so_zone_max_abs": float(zone_difference.max()),
        "so_zone_range_coarse": float(zone_so_coarse.max() - zone_so_coarse.min()),
        "inventory_relative": inventory_relative,
        "monthly_volume_relative": monthly_relative,
        "monthly_volume_self_relative": monthly_self_relative,
        "monthly_volume_near_zero_absolute_m3_sc": monthly_near_zero_absolute,
        "monthly_volume_absolute_crossover_m3_sc": crossover,
        "monthly_volume_absolute_m3_sc": monthly_absolute,
        "n_zones": float(support.n_zones),
        "n_months": float(n_months),
        "coarse_cells": grids["coarse"]["n_cells"],
        "fine_cells": grids["fine"]["n_cells"],
    }


def compare_refinement(
    coarse_result: Mapping[str, Path],
    fine_result: Mapping[str, Path],
    common_support: CommonSupport,
    tolerances: Mapping[str, float],
) -> PhysicsCheck:
    """Score one refinement pair: the same continuous problem on two grids.

    `coarse_result` and `fine_result` map `states` and `monthly` to the files each forward
    really published. What comes back is a DISCRETIZATION SENSITIVITY verdict and nothing
    more: it says how much of the answer moved when the mesh and the timestep were refined,
    and it does not say the fine answer is the truth. Nothing here calls the fine grid a
    posterior, and the fixture that produced it is a forward.
    """
    fixture = _FIXTURES.get(common_support.name)
    if fixture is None or fixture.family != "refinement":
        raise ValueError(
            f"{common_support.name!r} is not a registered refinement fixture; this build "
            f"compares {[n for n, f in sorted(_FIXTURES.items()) if f.family == 'refinement']}"
        )
    checked = require_tolerances(tolerances)
    thresholds = _thresholds_for(fixture, checked)
    present: dict[str, Path] = {}
    for prefix, outputs in (("coarse", coarse_result), ("fine", fine_result)):
        for role in _REFINEMENT_ROLES:
            if role in outputs:
                present[f"{prefix}_{role}"] = Path(outputs[role])
    hashes = {role: sha256_file(path) for role, path in sorted(present.items()) if path.is_file()}
    evidence = tuple(str(path) for _, path in sorted(present.items()))

    missing = [role for role in fixture.roles if role not in present]
    if missing:
        return PhysicsCheck(
            name=common_support.name,
            status="NOT_RUN",
            metrics={},
            thresholds=thresholds,
            input_hashes=hashes,
            evidence_paths=evidence,
            reason=(
                f"{common_support.name}: no artifact was given for {missing}; the comparison "
                f"did not run and its metrics {sorted({g.metric for g in fixture.gates})} are "
                "unmeasured"
            ),
        )
    try:
        metrics = _refinement_metrics(coarse_result, fine_result, common_support, checked)
    except ArtifactUnreadable as exc:
        return PhysicsCheck(
            name=common_support.name,
            status="NOT_RUN",
            metrics={},
            thresholds=thresholds,
            input_hashes=hashes,
            evidence_paths=evidence,
            reason=f"{common_support.name}: the published artifacts could not be compared: {exc}",
        )
    failures = _failed_gates(fixture, metrics, thresholds)
    return PhysicsCheck(
        name=common_support.name,
        status="FAIL" if failures else "PASS",
        metrics=metrics,
        thresholds=thresholds,
        input_hashes=hashes,
        evidence_paths=evidence,
        reason=f"{common_support.name}: " + "; ".join(failures) if failures else None,
    )


# --------------------------------------------------------------------------------------
# 9.2 the evaluator
# --------------------------------------------------------------------------------------


def _thresholds_for(fixture: Fixture, tolerances: Mapping[str, float]) -> dict[str, float]:
    out = dict(fixture.structural_thresholds)
    for gate in fixture.gates:
        if gate.threshold in tolerances:
            out[gate.threshold] = float(tolerances[gate.threshold])
    # Reported beside the gate, never as it: SPEC 8.3's stricter cumulative aim.
    if any(gate.threshold.startswith("balance_") for gate in fixture.gates):
        out["balance_cumulative_target"] = float(tolerances["balance_cumulative_target"])
        out["balance_absolute_floor_m3_sc"] = float(tolerances["balance_absolute_floor_m3_sc"])
    return out


def _measure(
    fixture: Fixture, outputs: Mapping[str, Path], tolerances: Mapping[str, float]
) -> dict[str, float]:
    target = tolerances["balance_cumulative_target"]
    if fixture.family == "bl":
        grids = {n_cells: _read_states(outputs[f"states_{n_cells}"]) for n_cells in (32, 64, 128)}
        metrics = _bl_metrics(grids, fixture)
        metrics.update(_saturation_metrics(grids[128]))
        metrics.update(_balance_metrics(outputs["balances_128"], target))
        return metrics

    states = _read_states(outputs["states"])
    metrics = {
        **_saturation_metrics(states),
        **_drift_metrics(states),
        **_balance_metrics(outputs["balances"], target),
        **_connection_metrics(outputs["connections"]),
    }
    if fixture.family == "hydrostatic":
        metrics.update(_hydrostatic_metrics(states, fixture.cell_center_depth_m))
    elif fixture.family == "segregation":
        metrics.update(_segregation_metrics(states, fixture.cell_center_depth_m))
    elif fixture.family == "five_spot":
        metrics.update(_five_spot_metrics(states, fixture))
    return metrics


def common_metrics(
    states_path: Path, balances_path: Path, tolerances: Mapping[str, float]
) -> dict[str, float]:
    """The saturation and balance numbers EVERY published forward owes, whatever its family.

    A registered fixture is scored by `evaluate_physics`, which knows what else that fixture
    claims. A result that is not one of the seven registered fixtures — an operational
    sector, a P1 parent world — still publishes states and both balance statements, and plan
    12.9 has the stage report cite those from the published files rather than from a
    docstring. This is that subset, scored against the same frozen tolerance block and
    against nothing else.
    """
    checked = require_tolerances(tolerances)
    return {
        **_saturation_metrics(_read_states(states_path)),
        **_balance_metrics(balances_path, checked["balance_cumulative_target"]),
    }


def _failed_gates(
    fixture: Fixture, metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> list[str]:
    """Every gate of this fixture that its metrics did not clear, said in full."""
    failures: list[str] = []
    for gate in fixture.gates:
        if gate.metric not in metrics:
            failures.append(f"{gate.metric} was not measured")
            continue
        limit = thresholds[gate.threshold]
        value = metrics[gate.metric]
        if gate.failed(value, limit):
            comparison = "exceeds" if gate.direction == "at_most" else "does not exceed"
            failures.append(f"{gate.metric}={value:.6g} {comparison} {gate.threshold}={limit:g}")
    return failures


def evaluate_physics(
    case_name: str, outputs: Mapping[str, Path], tolerances: Mapping[str, float]
) -> PhysicsCheck:
    """Score one registered verification fixture against the artifacts it published.

    `outputs` maps the fixture's declared roles — `states`, `balances`, `connections`, or
    the per-grid roles of the refinement fixtures — to the files a forward really wrote.
    `tolerances` is the fixed block, exactly; an incomplete mapping is refused rather than
    completed from a default.

    Returns a verdict in every case. A fixture whose artifacts are missing or unreadable
    comes back `NOT_RUN` with the reason and with the thresholds it WOULD have been scored
    against, because plan 9.8 requires the report to name the metrics that never ran; only
    an unregistered fixture name or an unusable tolerance block raises, because neither is
    a property of the run being scored.
    """
    fixture = _FIXTURES.get(case_name)
    if fixture is None:
        raise ValueError(
            f"{case_name!r} is not a registered verification fixture; this build scores "
            f"{list(registered_fixtures())}"
        )
    if fixture.family == "refinement":
        raise ValueError(
            f"{case_name!r} is a refinement PAIR and is scored by "
            "compare_refinement(coarse, fine, support, tolerances), which needs two published "
            "results and the zones they share; evaluate_physics reads one result and cannot "
            "compare anything"
        )
    checked = require_tolerances(tolerances)
    thresholds = _thresholds_for(fixture, checked)
    present = {role: Path(outputs[role]) for role in fixture.roles if role in outputs}
    hashes = {role: sha256_file(path) for role, path in sorted(present.items()) if path.is_file()}
    evidence = tuple(str(path) for _, path in sorted(present.items()))

    missing = [role for role in fixture.roles if role not in outputs]
    if missing:
        return PhysicsCheck(
            name=case_name,
            status="NOT_RUN",
            metrics={},
            thresholds=thresholds,
            input_hashes=hashes,
            evidence_paths=evidence,
            reason=(
                f"{case_name}: no artifact was given for {missing}; the check did not run "
                f"and its metrics {sorted({gate.metric for gate in fixture.gates})} are unmeasured"
            ),
        )
    try:
        metrics = _measure(fixture, present, checked)
    except ArtifactUnreadable as exc:
        return PhysicsCheck(
            name=case_name,
            status="NOT_RUN",
            metrics={},
            thresholds=thresholds,
            input_hashes=hashes,
            evidence_paths=evidence,
            reason=f"{case_name}: the published artifacts could not be scored: {exc}",
        )

    failures = _failed_gates(fixture, metrics, thresholds)
    return PhysicsCheck(
        name=case_name,
        status="FAIL" if failures else "PASS",
        metrics=metrics,
        thresholds=thresholds,
        input_hashes=hashes,
        evidence_paths=evidence,
        reason=f"{case_name}: " + "; ".join(failures) if failures else None,
    )
