"""E01.7 — accepted-substep integrals, the published outputs, and reading them back.

Three jobs, and the order matters because it is the order a forward becomes believable in.

**Integrate, never sample.** `integrate_monthly` sums `rate * dt` over the accepted
substeps of each month. SPEC §9.3 is explicit — «Snapshot в конце месяца не заменяет это
интегрирование» — and the cost of getting it wrong is not subtle: a well on 1 m³/day for
ten days and 3 m³/day for twenty produced 70 m³, and the last rate times the month says 90.
The substeps come from the solver's own accepted mini-steps, so the sum is the same
backward-Euler quadrature the solver used and not a re-interpolation of it.

**Split by the sign the substep actually had.** Production and injection are separate
columns and every public rate is non-negative (plan 3.1). Summing a signed water rate would
let a month of injection cancel a month of production and leave a well that looks as if it
did nothing; a well that changed role inside a month keeps both groups. The water cut is
`Vw/(Vo+Vw)` over PRODUCED volumes only, and only above a volume floor: below it the ratio
is noise wearing a physical name, so the row is kept and `fw` is null with `fw_valid=false`.

**Publish, then hash, then claim.** `write_forward_outputs` writes the states file and the
Parquet tables, hashes every one of them, and only then is a `ForwardResult` built;
`publish_forward_result` writes that record LAST. A process lost half way therefore leaves
files nobody has claimed rather than a manifest pointing at outputs that do not exist.
`load_forward_result` reads the record back and re-proves what the record alone can prove:
every digest, every shape and axis order, finiteness and physical range, and that the time
axis the record claims is the one the states file actually holds. Whether the axis is the
one an `OutputRequest` ASKED for is a different question — it needs the request as well as
the result — and belongs to the stage that holds both.

The extraction the numbers come from is produced by `julia/adapter/outputs.jl` and crosses
as JSON. Nothing here re-derives a rate, a flux or a saturation: this module integrates,
splits, writes and verifies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactImmutabilityError
from so_recon.registry.atomic import fsync_dir, stage_path, write_json_atomic
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.case_io import read_array, write_arrays
from so_recon.simulator.contracts import (
    CELL_AXES,
    TIME_CELL_AXES,
    ArrayRef,
    BlackOilFluidSpec,
    BlackOilOutputs,
    CaseBundle,
    CostRecord,
    ForwardResult,
    ForwardStatus,
    JobDescriptor,
    RestartRef,
)
from so_recon.simulator.schedule import FLOW_TOLERANCE_M3, Schedule, compile_schedule
from so_recon.validation.balance import BalanceMetrics, component_balance

#: The shape of the extraction `julia/adapter/outputs.jl` writes. Bumped whenever that
#: payload changes, and checked on the way in: a worker speaking an older shape is a
#: refusal rather than a result assembled out of fields that mean something else now.
EXTRACT_SCHEMA_VERSION = "forward-extract-2"

#: Files published beside a result, and the keys their digests are recorded under in
#: `ForwardResult.solver_metadata`. `ForwardResult` has three result paths of its own
#: (monthly, connections, balances); the states file and the optional substep diagnostics
#: are named here because the record has no field for them.
STATES_FILENAME = "states.h5"
MONTHLY_FILENAME = "monthly.parquet"
CONNECTIONS_FILENAME = "connections.parquet"
BALANCES_FILENAME = "balances.parquet"
STEPS_FILENAME = "accepted_steps.parquet"
RESULT_FILENAME = "forward_result.json"

#: Water and oil, in the phase order the whole of E01 uses (plan 3.1: tuples are
#: `(water, oil)`), which is also the native `OW_PHASES` order the adapter builds.
COMPONENTS = ("water", "oil")

#: Task 13: the components a black-oil extraction balances, in the same phase order. It is a
#: SUPERSET of `COMPONENTS` and never a replacement for it — an oil-water result publishes two
#: component rows and a black-oil one three, and neither learns the other's shape.
BO_COMPONENTS = ("water", "oil", "gas")

#: Where a black-oil result's surface gas volumes go: its own table, beside `monthly.parquet`
#: rather than inside it. A standard cubic metre of gas is not a standard cubic metre of
#: liquid, so it is never summed into `liquid_prod_m3_sc` and never enters the water cut; and
#: the oil-water monthly schema — which every published E01 result already uses — does not
#: grow a column it would have to fill with zeros.
GAS_MONTHLY_FILENAME = "gas_monthly.parquet"

#: A substep boundary may miss a month edge by this much and still be the edge. The
#: schedule's own edges are day counts times 86400, so anything above float64 round-off at
#: reservoir time scales is a real crossing; a microsecond is far below any control change.
BOUNDARY_TOLERANCE_S = 1e-6

#: The columns of the accepted-substep table the aggregator consumes (plan 7.1). Public
#: rates only, non-negative, production and injection apart.
STEP_COLUMNS = (
    "well_id",
    "start_s",
    "end_s",
    "oil_prod_m3_s",
    "water_prod_m3_s",
    "water_inj_m3_s",
)

#: The columns of the per-substep connection table (plan 7.4).
CONNECTION_COLUMNS = (
    "well_id",
    "connection_id",
    "cell_id",
    "start_s",
    "end_s",
    "water_mass_kg_s",
    "oil_mass_kg_s",
    "total_mass_kg_s",
    "connection_open",
    "actual_target",
    "bhp_pa",
)

#: Physical ranges a published state has to lie in. Nothing is clipped into them.
_MAX_SATURATION_DRIFT = 1e-8


class ForwardResultIntegrityError(RuntimeError):
    """A published forward result does not prove to be what its record says it is."""


class ExtractionError(ValueError):
    """The native extraction cannot be turned into outputs, and says which field is wrong."""


# --------------------------------------------------------------------------------------
# the monthly aggregator (plan 7.5)
# --------------------------------------------------------------------------------------


def _validated_month_edges(month_edges_s: Sequence[float]) -> tuple[float, ...]:
    edges = tuple(float(e) for e in month_edges_s)
    if len(edges) < 2:
        raise ValueError(f"month_edges_s must describe at least one month, got {edges}")
    if any(not math.isfinite(e) for e in edges):
        raise ValueError(f"month_edges_s must be finite, got {edges}")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"month_edges_s must be strictly increasing, got {edges}")
    return edges


def _month_of(start: float, end: float, edges: tuple[float, ...]) -> int:
    """Which month `[start, end)` lies inside, or a refusal if it straddles two.

    A substep that crosses a month boundary cannot be attributed to either month without
    splitting it, and splitting it here would invent a rate history inside the substep that
    the solver never computed. The extractor is what must not produce one.
    """
    for index, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
        if start >= low - BOUNDARY_TOLERANCE_S and end <= high + BOUNDARY_TOLERANCE_S:
            return index
    for edge in edges[1:-1]:
        if start < edge - BOUNDARY_TOLERANCE_S < end:
            raise ValueError(
                f"substep [{start}, {end}) crosses the month boundary at {edge} s; one "
                "substep never spans two months, and splitting it here would invent a rate "
                "history inside it"
            )
    raise ValueError(
        f"substep [{start}, {end}) lies outside the reported horizon [{edges[0]}, {edges[-1]})"
    )


def integrate_monthly(steps: pa.Table, month_edges_s: Sequence[float]) -> pa.Table:
    """Integrate accepted substeps into one row per well per month.

    `steps` carries `STEP_COLUMNS`: a well, the substep's half-open interval in seconds from
    the case start, and the three public rates in m³_sc/s. Every rate is non-negative and
    production and injection are already apart, which is what stops an injected cubic metre
    from cancelling a produced one.

    Every well the table names gets a row in every month, including the months in which it
    was shut or simply had nothing to report: an absent row is indistinguishable from a
    month nobody computed, and «resource failure и numerical failure не дают физический
    нулевой likelihood без анализа» (SPEC 18.4) starts with being able to tell the two
    apart. Such a row carries zero volumes and `fw = null, fw_valid = false`.
    """
    edges = _validated_month_edges(month_edges_s)
    missing = [name for name in STEP_COLUMNS if name not in steps.column_names]
    if missing:
        raise ValueError(f"the accepted-substep table is missing columns {missing}")

    rows = steps.select(list(STEP_COLUMNS)).to_pylist()
    wells = sorted({str(row["well_id"]) for row in rows})
    n_months = len(edges) - 1
    volumes: dict[tuple[str, int], dict[str, float]] = {
        (well, month): {"oil_prod_m3_sc": 0.0, "water_prod_m3_sc": 0.0, "water_inj_m3_sc": 0.0}
        for well in wells
        for month in range(n_months)
    }
    open_s: dict[tuple[str, int], float] = dict.fromkeys(volumes, 0.0)
    spans: dict[str, list[tuple[float, float]]] = {well: [] for well in wells}

    for row in rows:
        well = str(row["well_id"])
        start, end = float(row["start_s"]), float(row["end_s"])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"well {well!r}: substep bounds must be finite, got [{start}, {end})")
        if not end > start:
            raise ValueError(
                f"well {well!r}: a substep has a positive duration, got [{start}, {end})"
            )
        for previous_start, previous_end in spans[well]:
            if start < previous_end - BOUNDARY_TOLERANCE_S and previous_start < end:
                raise ValueError(
                    f"well {well!r}: substeps overlap, [{previous_start}, {previous_end}) and "
                    f"[{start}, {end}); an accepted substep is counted exactly once"
                )
        spans[well].append((start, end))
        month = _month_of(start, end, edges)
        duration = end - start
        bucket = volumes[(well, month)]
        flowed = False
        for column, name in (
            ("oil_prod_m3_s", "oil_prod_m3_sc"),
            ("water_prod_m3_s", "water_prod_m3_sc"),
            ("water_inj_m3_s", "water_inj_m3_sc"),
        ):
            rate = float(row[column])
            if not math.isfinite(rate):
                raise ValueError(
                    f"well {well!r} substep [{start}, {end}): {column} is {rate}; a nonfinite "
                    "rate is refused, never integrated"
                )
            if rate < 0.0:
                raise ValueError(
                    f"well {well!r} substep [{start}, {end}): {column} is {rate}. Public rates "
                    "are non-negative and production and injection are separate columns; a "
                    "negative entry means the sign split was not made (plan 3.1)"
                )
            bucket[name] += rate * duration
            flowed = flowed or rate > 0.0
        if flowed:
            open_s[(well, month)] += duration

    out: list[dict[str, Any]] = []
    for well in wells:
        for month in range(n_months):
            bucket = volumes[(well, month)]
            produced = bucket["oil_prod_m3_sc"] + bucket["water_prod_m3_sc"]
            valid = produced > FLOW_TOLERANCE_M3
            out.append(
                {
                    "well_id": well,
                    "month_index": month,
                    "start_s": edges[month],
                    "end_s": edges[month + 1],
                    "oil_prod_m3_sc": bucket["oil_prod_m3_sc"],
                    "water_prod_m3_sc": bucket["water_prod_m3_sc"],
                    "water_inj_m3_sc": bucket["water_inj_m3_sc"],
                    "liquid_prod_m3_sc": produced,
                    "flowing_s": open_s[(well, month)],
                    # Produced water over produced liquid. The injected column is NOT in
                    # this ratio: a water cut built from a signed total would fall when a
                    # well injected, which is not what a water cut means.
                    "fw": (bucket["water_prod_m3_sc"] / produced) if valid else None,
                    "fw_valid": valid,
                }
            )
    return pa.Table.from_pylist(out, schema=MONTHLY_SCHEMA)


MONTHLY_SCHEMA = pa.schema(
    [
        ("well_id", pa.string()),
        ("month_index", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("oil_prod_m3_sc", pa.float64()),
        ("water_prod_m3_sc", pa.float64()),
        ("water_inj_m3_sc", pa.float64()),
        ("liquid_prod_m3_sc", pa.float64()),
        ("flowing_s", pa.float64()),
        ("fw", pa.float64()),
        ("fw_valid", pa.bool_()),
    ]
)

CONNECTION_MONTHLY_SCHEMA = pa.schema(
    [
        ("well_id", pa.string()),
        ("connection_id", pa.int64()),
        ("cell_id", pa.int64()),
        ("month_index", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("water_mass_kg", pa.float64()),
        ("oil_mass_kg", pa.float64()),
        ("total_mass_kg", pa.float64()),
        ("open_s", pa.float64()),
        ("bhp_min_pa", pa.float64()),
        ("bhp_mean_pa", pa.float64()),
        ("bhp_max_pa", pa.float64()),
    ]
)

BALANCE_SCHEMA = pa.schema(
    [
        # Which system was closed over and which source it was closed against. A reader who
        # cites one of these numbers has to be able to say which statement it is, so the
        # three labels are columns rather than prose in a docstring somewhere.
        ("balance", pa.string()),
        ("system", pa.string()),
        ("source_term", pa.string()),
        ("component", pa.string()),
        ("cumulative_relative", pa.float64()),
        ("median_step_relative", pa.float64()),
        ("max_step_relative", pa.float64()),
        ("absolute_residual_m3_sc", pa.float64()),
        ("max_step_absolute_m3_sc", pa.float64()),
        ("throughput_relative", pa.float64()),
        ("initial_inventory_m3_sc", pa.float64()),
        ("final_inventory_m3_sc", pa.float64()),
        ("net_source_m3_sc", pa.float64()),
        # How much of `net_source_m3_sc` crossed a face no well owns. A `pressure_water`
        # boundary is a source of the balanced system exactly as a wellhead is, so it is
        # inside the net source above; naming it here is what stops a reader attributing an
        # aquifer's water to a well. Zero for a closed case, which is every case but 10.6's.
        ("boundary_source_m3_sc", pa.float64()),
        ("floor_m3_sc", pa.float64()),
        ("n_steps", pa.int64()),
    ]
)

STEP_SCHEMA = pa.schema(
    [
        ("step_index", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("dt_s", pa.float64()),
        ("interval_index", pa.int64()),
        ("month_index", pa.int64()),
    ]
)


def connection_monthly_schema(components: Sequence[str]) -> pa.Schema:
    """The per-month connection schema for an extraction with these components.

    Identical to `CONNECTION_MONTHLY_SCHEMA` for the oil-water pair — same columns, same
    order — with `gas_mass_kg` inserted after the oil column for a black-oil one.
    """
    return pa.schema(
        [
            ("well_id", pa.string()),
            ("connection_id", pa.int64()),
            ("cell_id", pa.int64()),
            ("month_index", pa.int64()),
            ("start_s", pa.float64()),
            ("end_s", pa.float64()),
            *[(f"{c}_mass_kg", pa.float64()) for c in components],
            ("total_mass_kg", pa.float64()),
            ("open_s", pa.float64()),
            ("bhp_min_pa", pa.float64()),
            ("bhp_mean_pa", pa.float64()),
            ("bhp_max_pa", pa.float64()),
        ]
    )


def integrate_connections(connections: pa.Table, month_edges_s: Sequence[float]) -> pa.Table:
    """Integrate the per-substep connection fluxes into one row per connection per month.

    The mass columns are the native reservoir-well cross-term for the substep's own upwind
    state, in kg/s and with the native sign (positive out of the reservoir); they are NOT a
    surface rate distributed over connections by `kh`, which would attribute flow to a
    completion that was shut. `open_s` is how long the connection was actually open, and the
    bottom-hole pressure is summarised time-weighted over the same month, so a month's
    pressure is the pressure the well held rather than the last value of it.
    """
    edges = _validated_month_edges(month_edges_s)
    # The components the step table really carries, read off its columns rather than assumed:
    # a black-oil connection table has a gas column and an oil-water one does not, and
    # integrating the columns that are there is what stops a third component being dropped.
    components = tuple(c for c in BO_COMPONENTS if f"{c}_mass_kg_s" in connections.column_names)
    columns = [
        *[c for c in CONNECTION_COLUMNS if not c.endswith("_mass_kg_s")],
        *[f"{c}_mass_kg_s" for c in components],
        "total_mass_kg_s",
    ]
    missing = [name for name in columns if name not in connections.column_names]
    if missing:
        raise ValueError(f"the connection table is missing columns {missing}")
    rows = connections.select(columns).to_pylist()
    zero = {f"{c}_mass_kg": 0.0 for c in components}

    keys: dict[tuple[str, int], int] = {}
    totals: dict[tuple[str, int, int], dict[str, float]] = {}
    bhp: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for row in rows:
        well = str(row["well_id"])
        connection = int(row["connection_id"])
        cell = int(row["cell_id"])
        start, end = float(row["start_s"]), float(row["end_s"])
        if not end > start:
            raise ValueError(
                f"well {well!r} connection {connection}: a substep has a positive duration, "
                f"got [{start}, {end})"
            )
        month = _month_of(start, end, edges)
        keys[(well, connection)] = cell
        bucket = totals.setdefault(
            (well, connection, month), {**zero, "total_mass_kg": 0.0, "open_s": 0.0}
        )
        duration = end - start
        for column, name in (
            *[(f"{c}_mass_kg_s", f"{c}_mass_kg") for c in components],
            ("total_mass_kg_s", "total_mass_kg"),
        ):
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(
                    f"well {well!r} connection {connection} substep [{start}, {end}): "
                    f"{column} is {value}"
                )
            bucket[name] += value * duration
        if bool(row["connection_open"]):
            bucket["open_s"] += duration
        bhp.setdefault((well, month), []).append((float(row["bhp_pa"]), duration))

    out: list[dict[str, Any]] = []
    for (well, connection), cell in sorted(keys.items()):
        for month in range(len(edges) - 1):
            bucket = totals.get(
                (well, connection, month), {**zero, "total_mass_kg": 0.0, "open_s": 0.0}
            )
            samples = bhp.get((well, month), [])
            weight = sum(duration for _, duration in samples)
            out.append(
                {
                    "well_id": well,
                    "connection_id": connection,
                    "cell_id": cell,
                    "month_index": month,
                    "start_s": edges[month],
                    "end_s": edges[month + 1],
                    **bucket,
                    "bhp_min_pa": min((v for v, _ in samples), default=None),
                    "bhp_mean_pa": (
                        sum(v * d for v, d in samples) / weight if weight > 0.0 else None
                    ),
                    "bhp_max_pa": max((v for v, _ in samples), default=None),
                }
            )
    return pa.Table.from_pylist(out, schema=connection_monthly_schema(components))


#: What each published balance closes over and against, in words a reader of the table can
#: act on without opening this module.
_BALANCE_LABELS: dict[str, tuple[str, str]] = {
    "full_system_surface": (
        "reservoir + wellbores",
        "surface component flux (q_t * mix) + boundary influx",
    ),
    "reservoir_connections": (
        "reservoir",
        "reservoir-well connection flux (geometry/state/PVT) + boundary influx",
    ),
}


def balance_table(
    balances: Mapping[str, BalanceMetrics], boundary_source_m3_sc: Mapping[str, float]
) -> pa.Table:
    """Every published balance as one row per (balance, component).

    Both statements are in the same table, each labelled with the system it closed over and
    the source term it closed against, in the order `BALANCE_SYSTEMS` declares them. Plan
    12.9 has the stage validator read the PUBLISHED results rather than re-run the suite, so
    a balance that only exists inside a test is invisible to the gate that has to cite it.

    `boundary_source_m3_sc` is the part of each net source that crossed the reservoir's outer
    boundary, per component. It is REQUIRED rather than defaulted: a table that quietly wrote
    zero there would say a case had no aquifer support when nobody had looked.
    """
    rows: list[dict[str, Any]] = []
    for name, _, _ in BALANCE_SYSTEMS:
        metrics = balances[name]
        system, source_term = _BALANCE_LABELS[name]
        rows.extend(_balance_rows(name, system, source_term, metrics, boundary_source_m3_sc))
    return pa.Table.from_pylist(rows, schema=BALANCE_SCHEMA)


def _balance_rows(
    balance: str,
    system: str,
    source_term: str,
    metrics: BalanceMetrics,
    boundary_source_m3_sc: Mapping[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "balance": balance,
            "system": system,
            "source_term": source_term,
            "component": name,
            "cumulative_relative": metrics.cumulative_relative[index],
            "median_step_relative": metrics.median_step_relative[index],
            "max_step_relative": metrics.max_step_relative[index],
            "absolute_residual_m3_sc": metrics.absolute_residual[index],
            "max_step_absolute_m3_sc": metrics.max_step_absolute[index],
            "throughput_relative": metrics.throughput_relative[index],
            "initial_inventory_m3_sc": metrics.initial_inventory_m3_sc[index],
            "final_inventory_m3_sc": metrics.final_inventory_m3_sc[index],
            "net_source_m3_sc": metrics.net_source_m3_sc[index],
            "boundary_source_m3_sc": boundary_source_m3_sc[name],
            "floor_m3_sc": metrics.floor_m3_sc,
            "n_steps": metrics.n_steps,
        }
        for index, name in enumerate(metrics.components)
    ]


# --------------------------------------------------------------------------------------
# the native extraction -> tables (plan 7.3, 7.4)
# --------------------------------------------------------------------------------------


def _floats(payload: Mapping[str, Any], key: str, *, where: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ExtractionError(
            f"{where}: {key} must be a list of numbers, got {type(value).__name__}"
        )
    out: list[float] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, int | float) or isinstance(entry, bool):
            raise ExtractionError(f"{where}: {key}[{index}] is {entry!r}, not a number")
        if not math.isfinite(float(entry)):
            raise ExtractionError(f"{where}: {key}[{index}] is {entry!r}; a nonfinite value")
        out.append(float(entry))
    return out


def _matrix(payload: Mapping[str, Any], key: str, *, where: str) -> NDArray[np.float64]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ExtractionError(f"{where}: {key} must be a non-empty list of rows")
    rows = [
        _floats({"row": row}, "row", where=f"{where}.{key}[{i}]") for i, row in enumerate(value)
    ]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ExtractionError(f"{where}: {key} rows have different lengths {sorted(widths)}")
    return np.asarray(rows, dtype=np.float64)


def accepted_step_table(payload: Mapping[str, Any], schedule: Schedule) -> pa.Table:
    """The accepted-substep diagnostics: what the solver actually stepped on.

    `interval_index` is the schedule interval the substep belongs to and `month_index` the
    month that interval lies in, so a substep can be traced back to the control it ran under
    without re-deriving either.
    """
    chunk = _chunk(payload)
    starts = chunk["start_s"]
    ends = chunk["end_s"]
    intervals = chunk["interval_index"]
    rows = [
        {
            "step_index": index,
            "start_s": start,
            "end_s": end,
            "dt_s": end - start,
            "interval_index": interval,
            "month_index": schedule.month_index[interval],
        }
        for index, (start, end, interval) in enumerate(zip(starts, ends, intervals, strict=True))
    ]
    return pa.Table.from_pylist(rows, schema=STEP_SCHEMA)


def _chunk(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The substep axis of the extraction, checked for the invariants plan 7.3 names."""
    chunk = payload.get("chunk")
    if not isinstance(chunk, dict):
        raise ExtractionError("the extraction carries no `chunk` mapping")
    starts = _floats(chunk, "start_s", where="chunk")
    ends = _floats(chunk, "end_s", where="chunk")
    dt = _floats(chunk, "dt_s", where="chunk")
    intervals = chunk.get("interval_index")
    if not isinstance(intervals, list) or not all(isinstance(i, int) for i in intervals):
        raise ExtractionError("chunk.interval_index must be a list of integers")
    if not (len(starts) == len(ends) == len(dt) == len(intervals)):
        raise ExtractionError(
            f"chunk axes disagree: {len(starts)} starts, {len(ends)} ends, {len(dt)} "
            f"durations and {len(intervals)} interval indices"
        )
    if not starts:
        raise ExtractionError(
            "chunk carries no accepted substeps; a forward that stepped nowhere is not a forward"
        )
    horizon_start = float(chunk.get("horizon_start_s", starts[0]))
    horizon_end = float(chunk.get("horizon_end_s", ends[-1]))
    previous = horizon_start
    for index, (start, end, duration) in enumerate(zip(starts, ends, dt, strict=True)):
        if not duration > 0.0:
            raise ExtractionError(
                f"chunk substep {index} has dt_s {duration}; every accepted substep has a "
                "positive duration, and a failed or cut trial step is not an accepted one"
            )
        if abs((end - start) - duration) > BOUNDARY_TOLERANCE_S:
            raise ExtractionError(
                f"chunk substep {index} spans [{start}, {end}) but reports dt_s {duration}"
            )
        if abs(start - previous) > BOUNDARY_TOLERANCE_S:
            raise ExtractionError(
                f"chunk substep {index} starts at {start} s, but the previous one ended at "
                f"{previous} s; the accepted substeps tile the chunk without gaps or overlaps"
            )
        previous = end
    if abs(previous - horizon_end) > BOUNDARY_TOLERANCE_S:
        raise ExtractionError(
            f"the accepted substeps end at {previous} s, but the chunk runs to {horizon_end} s; "
            f"sum(dt) must be the chunk duration"
        )
    return {
        "start_s": starts,
        "end_s": ends,
        "dt_s": dt,
        "interval_index": [int(i) for i in intervals],
        "horizon_start_s": horizon_start,
        "horizon_end_s": horizon_end,
    }


def well_step_table(payload: Mapping[str, Any]) -> pa.Table:
    """Public per-substep well rates, split by the sign the substep actually had.

    The extraction carries the NATIVE surface phase rates, in m³_sc/s and negative for
    production. The split happens here, once: a negative rate becomes a production column
    and a positive one an injection column, so nothing downstream ever has to know the
    native sign and nothing can accidentally add the two together.

    The educational oil-water case injects water and only water (SPEC 9.1 gives an injector
    a standard water rate), so the table has no oil-injection column. A positive surface oil
    rate is therefore refused by name instead of being dropped: it would be a well injecting
    oil, which this contract cannot represent and must not silently discard.
    """
    chunk = _chunk(payload)
    wells = payload.get("wells")
    if not isinstance(wells, dict) or not wells:
        raise ExtractionError("the extraction names no wells")
    n = len(chunk["dt_s"])
    rows: list[dict[str, Any]] = []
    for well_id in sorted(wells):
        well = wells[well_id]
        if not isinstance(well, dict):
            raise ExtractionError(f"well {well_id!r}: the extraction entry is not a mapping")
        water = _floats(well, "surface_water_m3_s", where=f"wells[{well_id!r}]")
        oil = _floats(well, "surface_oil_m3_s", where=f"wells[{well_id!r}]")
        if len(water) != n or len(oil) != n:
            raise ExtractionError(
                f"well {well_id!r}: {len(water)} water and {len(oil)} oil surface rates for "
                f"{n} accepted substeps"
            )
        for index in range(n):
            oil_rate, water_rate = oil[index], water[index]
            if oil_rate > 0.0:
                raise ExtractionError(
                    f"well {well_id!r} substep {index}: the native surface oil rate is "
                    f"{oil_rate} m3_sc/s, which is oil being injected. The E01 oil-water "
                    "contract injects water only (SPEC 9.1) and has no column for it; this is "
                    "refused rather than dropped"
                )
            rows.append(
                {
                    "well_id": well_id,
                    "start_s": chunk["start_s"][index],
                    "end_s": chunk["end_s"][index],
                    # The sign the substep actually had, and a true zero either way: a
                    # negative zero in a public volume column is a sign nobody meant.
                    "oil_prod_m3_s": -oil_rate if oil_rate < 0.0 else 0.0,
                    "water_prod_m3_s": -water_rate if water_rate < 0.0 else 0.0,
                    "water_inj_m3_s": water_rate if water_rate > 0.0 else 0.0,
                }
            )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("well_id", pa.string()),
                ("start_s", pa.float64()),
                ("end_s", pa.float64()),
                ("oil_prod_m3_s", pa.float64()),
                ("water_prod_m3_s", pa.float64()),
                ("water_inj_m3_s", pa.float64()),
            ]
        ),
    )


GAS_MONTHLY_SCHEMA = pa.schema(
    [
        ("well_id", pa.string()),
        ("month_index", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("gas_prod_m3_sc", pa.float64()),
        ("gas_inj_m3_sc", pa.float64()),
    ]
)


def gas_monthly_table(payload: Mapping[str, Any], month_edges_s: Sequence[float]) -> pa.Table:
    """Surface gas volumes per well per month, in m3_sc. Task 13, and black oil only.

    It is the same integration `integrate_monthly` performs — the native surface rate over
    each accepted substep, split by the sign the substep had, bucketed into the month the
    substep belongs to — applied to the gas rate and written to a table of its own. Gas is
    kept out of `monthly.parquet` on purpose: `liquid_prod_m3_sc` and `fw` are liquid
    quantities, and a gas volume added into either would be a number nobody could interpret.
    """
    edges = _validated_month_edges(month_edges_s)
    chunk = _chunk(payload)
    wells = payload.get("wells")
    if not isinstance(wells, dict) or not wells:
        raise ExtractionError("the extraction names no wells")
    n = len(chunk["dt_s"])
    n_months = len(edges) - 1
    volumes: dict[tuple[str, int], dict[str, float]] = {
        (well, month): {"gas_prod_m3_sc": 0.0, "gas_inj_m3_sc": 0.0}
        for well in sorted(wells)
        for month in range(n_months)
    }
    for well_id in sorted(wells):
        rates = _floats(wells[well_id], "surface_gas_m3_s", where=f"wells[{well_id!r}]")
        if len(rates) != n:
            raise ExtractionError(
                f"well {well_id!r}: {len(rates)} gas surface rates for {n} accepted substeps"
            )
        for index, rate in enumerate(rates):
            start, end = chunk["start_s"][index], chunk["end_s"][index]
            duration = end - start
            month = _month_of(start, end, edges)
            if not math.isfinite(rate):
                raise ExtractionError(
                    f"well {well_id!r} substep {index}: the surface gas rate is {rate}"
                )
            bucket = volumes[(well_id, month)]
            if rate < 0.0:
                bucket["gas_prod_m3_sc"] += -rate * duration
            else:
                bucket["gas_inj_m3_sc"] += rate * duration
    return pa.Table.from_pylist(
        [
            {
                "well_id": well,
                "month_index": month,
                "start_s": edges[month],
                "end_s": edges[month + 1],
                **volumes[(well, month)],
            }
            for well in sorted(wells)
            for month in range(n_months)
        ],
        schema=GAS_MONTHLY_SCHEMA,
    )


def connection_step_table(payload: Mapping[str, Any]) -> pa.Table:
    """The per-substep connection diagnostics exactly as `outputs.jl` extracted them."""
    chunk = _chunk(payload)
    components = _components_of(payload)
    raw = payload.get("connections")
    if not isinstance(raw, list):
        raise ExtractionError("the extraction carries no `connections` list")
    n = len(chunk["dt_s"])
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ExtractionError(f"connections[{index}] is not a mapping")
        missing = [
            name for name in ("well_id", "connection_id", "cell_id", "step") if name not in entry
        ]
        if missing:
            raise ExtractionError(f"connections[{index}] is missing {missing}")
        step = int(entry["step"])
        if not 0 <= step < n:
            raise ExtractionError(
                f"connections[{index}] names substep {step}, outside the {n} accepted substeps"
            )
        rows.append(
            {
                "well_id": str(entry["well_id"]),
                "connection_id": int(entry["connection_id"]),
                "cell_id": int(entry["cell_id"]),
                "start_s": chunk["start_s"][step],
                "end_s": chunk["end_s"][step],
                "total_mass_kg_s": float(entry["total_mass_kg_s"]),
                "connection_open": bool(entry["connection_open"]),
                "actual_target": str(entry["actual_target"]),
                "bhp_pa": float(entry["bhp_pa"]),
                **{f"{c}_mass_kg_s": float(entry[f"{c}_mass_kg_s"]) for c in components},
            }
        )
    return pa.Table.from_pylist(rows, schema=connection_step_schema(components))


def connection_step_schema(components: Sequence[str]) -> pa.Schema:
    """The per-substep connection schema of an extraction with these components.

    For the oil-water pair it is exactly the schema it has always been, column for column and
    in the same order; a black-oil extraction inserts `gas_mass_kg_s` after the oil column.
    Derived from the component list rather than written twice, so a third component cannot be
    published in the extraction and quietly dropped from the table.
    """
    return pa.schema(
        [
            ("well_id", pa.string()),
            ("connection_id", pa.int64()),
            ("cell_id", pa.int64()),
            ("start_s", pa.float64()),
            ("end_s", pa.float64()),
            *[(f"{c}_mass_kg_s", pa.float64()) for c in components],
            ("total_mass_kg_s", pa.float64()),
            ("connection_open", pa.bool_()),
            ("actual_target", pa.string()),
            ("bhp_pa", pa.float64()),
        ]
    )


#: The two balances a forward publishes, each named by the SYSTEM it closes over and the
#: SOURCE it closes against. They are two different statements about the same run and are
#: kept apart in the record: never merged, never averaged, never one reported as the other.
#:
#: * `full_system_surface` — reservoir AND wellbores, against the surface component flux
#:   `q_t·mix` the native facility cross term uses. This is the balance of everything the
#:   model holds against everything that left it through a wellhead.
#: * `reservoir_connections` — the reservoir alone, against the reservoir-well cross term
#:   evaluated from the native geometry (well indices, perforation gravity), state
#:   (pressures, saturations) and PVT (densities, mobilities). This is the one plan 7.6's
#:   «проверять независимо по геометрии/state/PVT» describes most directly.
#:
#: The two differ by what the wellbores are storing at each instant, which is a real
#: quantity and not an error; a single number covering both would hide it.
BALANCE_SYSTEMS: tuple[tuple[str, str, str], ...] = (
    ("full_system_surface", "inventory_m3_sc", "net_surface_source_m3_sc"),
    ("reservoir_connections", "reservoir_inventory_m3_sc", "net_connection_source_m3_sc"),
)

#: Which of them `ForwardResult.solver_metadata` summarises and `balance_within_spec_tolerance`
#: is about. The whole-model statement is the headline; the other is beside it in the table.
HEADLINE_BALANCE = BALANCE_SYSTEMS[0][0]

#: The reservoir-only statement, by name. It is the one a quantity summed over RESERVOIR
#: cells has to be compared against: the whole-model one also holds what the wellbores are
#: storing, which is a real quantity and not a discrepancy.
RESERVOIR_BALANCE = BALANCE_SYSTEMS[1][0]


def _components_of(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Which components this extraction balances: the oil-water pair or the black-oil trio.

    The list is read from the payload and matched against the two this build knows, in the
    order they are declared. Anything else is refused rather than truncated: a balance scored
    over two of three components would close beautifully and mean nothing.
    """
    components = payload.get("components")
    if isinstance(components, list):
        found = tuple(str(c) for c in components)
        if found in (COMPONENTS, BO_COMPONENTS):
            return found
    raise ExtractionError(
        f"the extraction must balance {list(COMPONENTS)} or {list(BO_COMPONENTS)} in that "
        f"order, got {components!r}"
    )


def extraction_balances(payload: Mapping[str, Any]) -> dict[str, BalanceMetrics]:
    """Both component balances of an extraction, keyed by `BALANCE_SYSTEMS`.

    Each inventory is a native component mass divided by the phase's reference density, and
    each source was computed from quantities that never pass through an inventory — which is
    what makes a nonzero residual mean something (plan 7.6).
    """
    components = _components_of(payload)
    return {
        name: component_balance(
            _matrix(payload, inventory_key, where="extraction"),
            _matrix(payload, source_key, where="extraction"),
            components=components,
        )
        for name, inventory_key, source_key in BALANCE_SYSTEMS
    }


def extraction_balance(payload: Mapping[str, Any]) -> BalanceMetrics:
    """The headline balance: the whole model against the surface flux that crossed it."""
    return extraction_balances(payload)[HEADLINE_BALANCE]


def extraction_boundary_source(payload: Mapping[str, Any]) -> dict[str, float]:
    """Cumulative standard volume each component gained across the outer boundary, m3_sc.

    Positive INTO the reservoir, which is the sign of every other source in this file. The
    adapter has already added this into both net sources — a boundary feeds the reservoir,
    which is inside both balanced systems — so this is the NAMED part of them and never a
    third term to add on top.
    """
    components = _components_of(payload)
    matrix = _matrix(payload, "net_boundary_source_m3_sc", where="extraction")
    if matrix.ndim != 2 or matrix.shape[1] != len(components):
        raise ExtractionError(
            f"net_boundary_source_m3_sc must give one row per substep and one column per "
            f"component {list(components)}, got shape {matrix.shape}"
        )
    return {name: float(matrix[:, index].sum()) for index, name in enumerate(components)}


# --------------------------------------------------------------------------------------
# publication (plan 7.7)
# --------------------------------------------------------------------------------------


def _write_parquet(path: Path, table: pa.Table) -> str:
    """Write one Parquet table into place atomically and return its digest.

    The same sequence, and the same immutability, as `case_io.write_arrays`: staged beside
    the destination, flushed and hashed, and only then renamed. A destination that already
    holds different bytes is refused rather than replaced — a result directory belongs to
    one job (plan 3.2), so a second set of numbers arriving at that path means two different
    results are claiming the same name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stage_path(path)
    try:
        pq.write_table(table, tmp, compression="snappy")
        digest = sha256_file(tmp)
        if path.exists():
            if sha256_file(path) != digest:
                raise ArtifactImmutabilityError(f"refusing to overwrite {path}: different content")
            tmp.unlink()
        else:
            tmp.replace(path)
            fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return digest


def _state_arrays(payload: Mapping[str, Any]) -> dict[str, NDArray[np.float64]]:
    states = payload.get("states")
    if not isinstance(states, dict):
        raise ExtractionError("the extraction carries no `states` mapping")
    times = _floats(states, "times_s", where="states")
    if not times:
        raise ExtractionError("states.times_s is empty; a result must carry its time axis")
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ExtractionError(f"states.times_s must be strictly increasing, got {times}")
    fields = {}
    optional = tuple(name for name in BO_STATE_FIELDS if name in states)
    if optional and len(optional) != len(BO_STATE_FIELDS):
        raise ExtractionError(
            f"a black-oil state publishes {list(BO_STATE_FIELDS)} together; this one carries "
            f"only {list(optional)}"
        )
    for name in ("pressure_pa", "sw", "so", "pore_volume_m3", "bw", "bo", *optional):
        values = _matrix(states, name, where="states")
        if values.shape[0] != len(times):
            raise ExtractionError(
                f"states.{name} has {values.shape[0]} rows for {len(times)} requested times; "
                f"HDF5 states are (n_times, n_cells)"
            )
        fields[name] = values
    widths = {values.shape[1] for values in fields.values()}
    if len(widths) != 1:
        raise ExtractionError(f"the state fields describe different cell counts {sorted(widths)}")
    fields["times_s"] = np.asarray(times, dtype=np.float64)
    return fields


#: Unit and axis order of every dataset the states file holds. `time_s` and `cell_id` are
#: the axes themselves and therefore not `(time, cell)` fields; they live in the same file
#: so that a state array can be read back beside the axis it is indexed by.
_STATE_UNITS: dict[str, str] = {
    "pressure_pa": "Pa",
    "sw": "1",
    "so": "1",
    "pore_volume_m3": "m3",
    "bw": "1",
    "bo": "1",
    # Task 13. Present only on a black-oil result: gas saturation, dissolved gas-oil ratio
    # (m3_sc of gas per m3_sc of oil, hence dimensionless), and the gas formation volume
    # factor. An oil-water states file is unchanged — these datasets are simply not written.
    "sg": "1",
    "rs": "1",
    "bg": "1",
}

#: The three fields above, as a group. They travel together or not at all.
BO_STATE_FIELDS: tuple[str, ...] = ("sg", "rs", "bg")


def write_forward_outputs(
    payload: Mapping[str, Any],
    schedule: Schedule,
    result_dir: Path,
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Write every output file of one forward, hash each one, and describe them.

    Nothing here claims a result. The caller gets the `ArrayRef`s, the relative paths and
    the digests, and publishes the manifest afterwards — which is why a process lost in the
    middle leaves files nobody has claimed rather than a record pointing at outputs that are
    not there (plan 7.7).
    """
    fields = _state_arrays(payload)
    components = _components_of(payload)
    times = fields.pop("times_s")
    n_cells = int(next(iter(fields.values())).shape[1])
    datasets: dict[str, tuple[NDArray[Any], str, tuple[str, ...]]] = {
        name: (values, _STATE_UNITS[name], TIME_CELL_AXES) for name, values in fields.items()
    }
    # The axes travel in the same file as the fields they index, and the model's geometry
    # and rock do NOT: those are inputs the case already publishes, and copying them once
    # per timestep would make a result heavier than the model it came from (plan 7.7).
    datasets["time_s"] = (times, "s", ("time",))
    datasets["cell_id"] = (np.arange(n_cells, dtype=np.int64), "1", CELL_AXES)

    states_path = result_dir / STATES_FILENAME
    refs = write_arrays(states_path, datasets, paths=paths)

    steps = well_step_table(payload)
    connections = connection_step_table(payload)
    month_edges = schedule.month_edges_s
    monthly = integrate_monthly(steps, month_edges)
    _reject_flow_without_uptime(monthly, schedule)
    connection_months = integrate_connections(connections, month_edges)
    balances = extraction_balances(payload)
    boundary_source = extraction_boundary_source(payload)

    tables = [
        (MONTHLY_FILENAME, monthly),
        (CONNECTIONS_FILENAME, connection_months),
        (BALANCES_FILENAME, balance_table(balances, boundary_source)),
        (STEPS_FILENAME, accepted_step_table(payload, schedule)),
    ]
    # Task 13: a black-oil result publishes its surface gas volumes in a table of their own.
    # An oil-water result writes exactly the four files it always did.
    if "gas" in components:
        tables.append((GAS_MONTHLY_FILENAME, gas_monthly_table(payload, month_edges)))
    written: dict[str, str] = {}
    for filename, table in tables:
        written[filename] = _write_parquet(result_dir / filename, table)

    return {
        "states": {name: refs[name] for name in _STATE_UNITS if name in refs},
        "components": components,
        "black_oil": (
            _black_oil_outputs(fields, written, result_dir, paths) if "gas" in components else None
        ),
        "axes": {name: refs[name] for name in ("time_s", "cell_id")},
        "times_s": tuple(float(t) for t in times),
        "states_path": paths.relative(states_path),
        "states_sha256": sha256_file(states_path),
        "files": {name: paths.relative(result_dir / name) for name in written},
        "digests": written,
        "balances": balances,
        "monthly": monthly,
        "connections": connection_months,
    }


def _black_oil_outputs(
    fields: Mapping[str, NDArray[np.float64]],
    written: Mapping[str, str],
    result_dir: Path,
    paths: ProjectPaths,
) -> dict[str, Any]:
    """The gas numbers a black-oil result carries and an oil-water one has no field for.

    Free gas is `Sg * PV / Bg` and dissolved gas is `Rs * So * PV / Bo`, both in standard
    cubic metres and both per published state time. Their sum is the component gas inventory
    the balance closes on, which is what makes the dissolved term a claim that can be checked
    rather than a definition: the balance is computed from the native `TotalMasses`, and this
    split is computed from the published saturations, ratio and formation volume factors, so
    the two agreeing is evidence.
    """
    pv = fields["pore_volume_m3"]
    free = np.sum(fields["sg"] * pv / fields["bg"], axis=1)
    dissolved = np.sum(fields["rs"] * fields["so"] * pv / fields["bo"], axis=1)
    gas_monthly = result_dir / GAS_MONTHLY_FILENAME
    produced = pq.read_table(gas_monthly).column("gas_prod_m3_sc").to_pylist()
    return {
        "free_gas_m3_sc": tuple(float(v) for v in free),
        "dissolved_gas_m3_sc": tuple(float(v) for v in dissolved),
        "surface_gas_m3_sc": float(sum(produced)),
        "gas_monthly_path": paths.relative(gas_monthly),
        "gas_monthly_sha256": written[GAS_MONTHLY_FILENAME],
    }


def _reject_flow_without_uptime(monthly: pa.Table, schedule: Schedule) -> None:
    """Refuse a monthly volume over a month in which the schedule says nothing was open.

    This is the contradiction `Schedule.reject_flow_without_uptime` exists for, applied to
    the volumes that were actually integrated: E01 is given its uptime, so a volume that
    disagrees with it means the schedule and the flow are not the same case.
    """
    n_months = schedule.n_months
    by_well: dict[str, list[float]] = {}
    for row in monthly.to_pylist():
        by_well.setdefault(str(row["well_id"]), [0.0] * n_months)
        by_well[str(row["well_id"])][int(row["month_index"])] = float(
            row["liquid_prod_m3_sc"]
        ) + float(row["water_inj_m3_sc"])
    unknown = sorted(set(by_well) - set(schedule.wells))
    if unknown:
        raise ValueError(
            f"the extraction reports flow for wells {unknown}, which the case's schedule does "
            f"not control (it controls {list(schedule.wells)}); the model and the case are not "
            "the same case"
        )
    for well_id, volumes in sorted(by_well.items()):
        schedule.reject_flow_without_uptime(well_id, tuple(volumes))


def _uncovered_horizon(payload: Mapping[str, Any], schedule: Schedule) -> str | None:
    """Refuse an extraction that does not span the case's whole reported horizon.

    `_chunk` proves the accepted substeps tile THEIR OWN chunk; it cannot know what that
    chunk was supposed to be. Nothing else here can either: `integrate_monthly` emits a row
    for every month whether or not a substep reached it, and
    `Schedule.reject_flow_without_uptime` refuses flow without uptime, never uptime without
    flow. So a payload covering only the first month of a two-month case would publish as
    `COMPLETE` with the second month reading as a physical zero — a truncated forward wearing
    the shape of a well that produced nothing, which is precisely what SPEC 18.4 forbids
    («Resource failure и numerical failure не дают физический нулевой likelihood без
    анализа»).

    The refusal is `INCOMPLETE_BUDGET` rather than a protocol failure because the payload is
    well formed and self-consistent; what is wrong with it is that it stops early, which is
    what that status is for. A caller that meant to run in chunks resumes rather than
    publishes.
    """
    chunk = _chunk(payload)
    expected_start, expected_end = schedule.edges_s[0], schedule.edges_s[-1]
    problems = []
    if abs(chunk["horizon_start_s"] - expected_start) > BOUNDARY_TOLERANCE_S:
        problems.append(
            f"it starts at {chunk['horizon_start_s']} s, the case's schedule at {expected_start} s"
        )
    if abs(chunk["horizon_end_s"] - expected_end) > BOUNDARY_TOLERANCE_S:
        problems.append(
            f"it ends at {chunk['horizon_end_s']} s, the case's schedule at {expected_end} s"
        )
    if not problems:
        return None
    return (
        "the native extraction does not cover the case's reported horizon: "
        + "; ".join(problems)
        + ". A month nobody simulated would be published as a month in which nothing flowed"
    )


def publish_forward_result(
    job: JobDescriptor,
    case: CaseBundle,
    payload: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    cost: CostRecord,
    solver_metadata: Mapping[str, str],
    parent_attempt_ids: tuple[str, ...] = (),
    restart: RestartRef | None = None,
) -> ForwardResult:
    """Turn one native extraction into a published, verifiable `ForwardResult`.

    This is the path that produces results, and every refusal on it is classified rather
    than raised at the caller (plan 3.3):

    * a schedule the case cannot compile — a gap, an overlap, a segment past the last report
      edge, an event boundary that is a near-miss for a report edge — is `INVALID_INPUT`;
    * a well that could not hold the control its case demanded, and operated on a limit
      instead, is `CONTROL_INFEASIBLE` with the well, the step, what was demanded and what
      it actually ran on;
    * an extraction whose own status is not a completed simulation is passed through with
      the reason the adapter gave it;
    * an extraction that does not cover the case's whole reported horizon is
      `INCOMPLETE_BUDGET` — see `_uncovered_horizon`.

    Only a result that survives all of that gets its outputs written, and the record is
    written last, after every file has been flushed and hashed.

    `restart` is the native checkpoint the job published, if it published one. It travels
    onto the record whatever the status is: a run stopped after a completed month is
    `INCOMPLETE_BUDGET` AND resumable, and dropping the checkpoint because the status is not
    COMPLETE would throw away the only thing that makes the stop recoverable.
    """
    try:
        schedule = compile_schedule(case.report_edges_s, case.controls)
    except ValueError as exc:
        return _classified(
            job,
            case,
            "INVALID_INPUT",
            f"the case schedule cannot be compiled: {exc}",
            cost,
            solver_metadata,
            parent_attempt_ids,
            restart,
        )

    status = str(payload.get("status", ""))
    reason = payload.get("reason")
    declared_schema = payload.get("schema_version")
    if status == "COMPLETE" and declared_schema != EXTRACT_SCHEMA_VERSION:
        # A payload in a shape this side does not speak cannot be read field by field: the
        # names would still resolve and would mean something else.
        return _classified(
            job,
            case,
            "PROTOCOL_FAILURE",
            f"the native extraction declares schema_version {declared_schema!r}; this build "
            f"reads {EXTRACT_SCHEMA_VERSION!r}",
            cost,
            solver_metadata,
            parent_attempt_ids,
            restart,
        )
    if status != "COMPLETE":
        return _classified(
            job,
            case,
            _extraction_status(status),
            str(reason) if reason else f"the native extraction reported {status!r} with no reason",
            cost,
            solver_metadata,
            parent_attempt_ids,
            restart,
        )
    infeasible = control_infeasibility(payload)
    if infeasible is not None:
        return _classified(
            job,
            case,
            "CONTROL_INFEASIBLE",
            infeasible,
            cost,
            solver_metadata,
            parent_attempt_ids,
            restart,
        )
    uncovered = _uncovered_horizon(payload, schedule)
    if uncovered is not None:
        return _classified(
            job,
            case,
            "INCOMPLETE_BUDGET",
            uncovered,
            cost,
            solver_metadata,
            parent_attempt_ids,
            restart,
        )

    result_dir = paths.resolve(job.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_forward_outputs(payload, schedule, result_dir, paths)

    metadata = dict(solver_metadata)
    metadata["states_path"] = outputs["states_path"]
    metadata["states_sha256"] = outputs["states_sha256"]
    metadata["extract_schema_version"] = EXTRACT_SCHEMA_VERSION
    for filename, digest in outputs["digests"].items():
        metadata[f"{filename}.sha256"] = digest
    # The accepted-step diagnostics have no field of their own in `ForwardResult`, so the
    # record names them here; naming them is what makes them checkable on read.
    metadata[f"{STEPS_FILENAME}.path"] = outputs["files"][STEPS_FILENAME]
    balances: dict[str, BalanceMetrics] = outputs["balances"]
    # Both statements are summarised, each under its own name, and the unprefixed keys stay
    # the whole-model one so that a reader who does not know there are two is not handed the
    # reservoir-only number by accident. `balances.parquet` carries all of it in full.
    for name, metrics in sorted(balances.items()):
        prefix = "balance" if name == HEADLINE_BALANCE else f"balance.{name}"
        metadata[f"{prefix}_cumulative_relative"] = json.dumps(
            dict(zip(metrics.components, metrics.cumulative_relative, strict=True))
        )
        metadata[f"{prefix}_median_step_relative"] = json.dumps(
            dict(zip(metrics.components, metrics.median_step_relative, strict=True))
        )
        metadata[f"{prefix}_absolute_residual_m3_sc"] = json.dumps(
            dict(zip(metrics.components, metrics.absolute_residual, strict=True))
        )
        metadata[f"{prefix}_within_spec_tolerance"] = str(metrics.within_spec_tolerance).lower()
        metadata[f"{prefix}_meets_strict_target"] = str(metrics.meets_strict_target).lower()
    metadata["balance_headline"] = HEADLINE_BALANCE
    metadata["balance_systems"] = json.dumps([name for name, _, _ in BALANCE_SYSTEMS])

    black_oil: BlackOilOutputs | None = None
    if outputs["black_oil"] is not None:
        gas = dict(outputs["black_oil"])
        metadata[f"{GAS_MONTHLY_FILENAME}.sha256"] = str(gas.pop("gas_monthly_sha256"))
        fluids = case.fluids
        if not isinstance(fluids, BlackOilFluidSpec):
            raise ExtractionError(
                "the extraction balances gas, and the case it names declares fluids.kind "
                f"{fluids.kind!r}; a black-oil result is published only for a black-oil case"
            )
        black_oil = BlackOilOutputs(
            sg=outputs["states"]["sg"],
            rs=outputs["states"]["rs"],
            bg=outputs["states"]["bg"],
            pvt_source=fluids.pvt_source,
            pvt_table_hashes=dict(fluids.pvt_table_hashes),
            **gas,
        )

    times = outputs["times_s"]
    return ForwardResult(
        job_id=job.job_id,
        case_sha256=job.case_sha256,
        model_hash=case.model_hash,
        physics_class=case.fluids.kind,
        status="COMPLETE",
        reason=None,
        completed_time_s=times[-1],
        times_s=times,
        states=outputs["states"],
        monthly_path=outputs["files"][MONTHLY_FILENAME],
        connections_path=outputs["files"][CONNECTIONS_FILENAME],
        balances_path=outputs["files"][BALANCES_FILENAME],
        restart=restart,
        black_oil=black_oil,
        solver_metadata=metadata,
        cost=cost,
        parent_attempt_ids=parent_attempt_ids,
    )


#: How a native extraction status maps onto the contract's own (plan 3.3). A status the
#: adapter is not allowed to reach — `COMPLETE` is decided here, `RESOURCE_FAILURE` and
#: `PROTOCOL_FAILURE` are the transport's to decide — is itself a protocol failure: a
#: verdict nobody defined must not become a physics verdict.
_EXTRACTION_STATUSES: dict[str, ForwardStatus] = {
    "INVALID_INPUT": "INVALID_INPUT",
    "PHYSICALLY_INVALID": "PHYSICALLY_INVALID",
    "CONTROL_INFEASIBLE": "CONTROL_INFEASIBLE",
    "NUMERICAL_FAILURE": "NUMERICAL_FAILURE",
    "TIMEOUT": "TIMEOUT",
    "INCOMPLETE_BUDGET": "INCOMPLETE_BUDGET",
}


def _extraction_status(status: str) -> ForwardStatus:
    return _EXTRACTION_STATUSES.get(status, "PROTOCOL_FAILURE")


def control_infeasibility(payload: Mapping[str, Any]) -> str | None:
    """The reason a result is `CONTROL_INFEASIBLE`, or `None` when every control held.

    The adapter builds the reason, because it is the side that knows what was requested and
    what the well operated on. What happens here is the consistency check that makes the
    absence of a reason meaningful: an evidence table with an unhonoured step and no reason,
    or a reason with no unhonoured step, is a contradiction and is reported as one rather
    than resolved in favour of whichever field was read first.
    """
    evidence = payload.get("control_evidence")
    reason = payload.get("control_infeasible_reason")
    if not isinstance(evidence, dict):
        raise ExtractionError("the extraction carries no `control_evidence` mapping")
    unhonoured: list[str] = []
    for well_id in sorted(evidence):
        steps = evidence[well_id]
        if not isinstance(steps, list):
            raise ExtractionError(f"control_evidence[{well_id!r}] is not a list of steps")
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or "honoured" not in step:
                raise ExtractionError(
                    f"control_evidence[{well_id!r}][{index}] carries no `honoured` flag"
                )
            if not bool(step["honoured"]):
                unhonoured.append(f"{well_id} step {index}")
    if unhonoured and not reason:
        raise ExtractionError(
            f"the control evidence reports unhonoured controls ({', '.join(unhonoured)}) but the "
            "extraction gives no CONTROL_INFEASIBLE reason"
        )
    if reason and not unhonoured:
        raise ExtractionError(
            f"the extraction gives a CONTROL_INFEASIBLE reason ({reason!r}) but every control "
            "in the evidence was honoured"
        )
    return str(reason) if reason else None


def _classified(
    job: JobDescriptor,
    case: CaseBundle,
    status: ForwardStatus,
    reason: str,
    cost: CostRecord,
    solver_metadata: Mapping[str, str],
    parent_attempt_ids: tuple[str, ...],
    restart: RestartRef | None = None,
) -> ForwardResult:
    """An unsuccessful result: no output paths, and a reason that says what is missing.

    It may still carry a checkpoint. A stop after a completed month publishes no outputs for
    the months nobody simulated AND leaves a valid native restart; the two are not in
    tension, and dropping the second would make the stop unrecoverable.
    """
    return ForwardResult(
        job_id=job.job_id,
        case_sha256=job.case_sha256,
        model_hash=case.model_hash,
        physics_class=case.fluids.kind,
        status=status,
        reason=reason,
        completed_time_s=0.0,
        times_s=(),
        states={},
        restart=restart,
        solver_metadata=dict(solver_metadata),
        cost=cost,
        parent_attempt_ids=parent_attempt_ids,
    )


def write_forward_result(result: ForwardResult, path: Path) -> str:
    """Write the record LAST, after every file it names has been flushed and hashed."""
    write_json_atomic(path, result.model_dump(mode="json"))
    return sha256_file(path)


# --------------------------------------------------------------------------------------
# reading a result back (plan 3.2's COMPLETE checks, the half a result can prove alone)
# --------------------------------------------------------------------------------------


def load_forward_result(path: Path, paths: ProjectPaths) -> ForwardResult:
    """Read a published result and re-prove everything the result alone can prove.

    That is: every declared digest against the bytes on disk — the output tables, the states
    file, the accepted-step diagnostics AND the native checkpoint's manifest — every state
    array's shape, axis order, unit and dtype, that every published number is finite and
    physically in range, and that the time axis the record claims is the one the states file
    holds.

    What it deliberately does NOT check is the half that needs a second record: whether the
    axis is the one an `OutputRequest` asked for, and whether a restart is present because
    that request set `keep_native_restart`. Those comparisons need the request as well as
    the result, and inventing them here from the result alone would be a check that cannot
    fail.
    """
    relative = paths.relative(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = ForwardResult.model_validate(payload)
    # Before the status gate, not after it: a run stopped after a completed month is
    # INCOMPLETE_BUDGET *and* resumable, and its checkpoint is the only part of it anybody
    # will use. A digest that is only checked on the results nobody resumes is not a gate.
    if result.restart is not None:
        _verify_restart_manifest(relative, result.restart, result.solver_metadata, paths)
    if result.status != "COMPLETE":
        return result

    checked = [
        ("monthly_path", result.monthly_path, f"{MONTHLY_FILENAME}.sha256"),
        ("connections_path", result.connections_path, f"{CONNECTIONS_FILENAME}.sha256"),
        ("balances_path", result.balances_path, f"{BALANCES_FILENAME}.sha256"),
        # Named in `solver_metadata` rather than in a field of its own, because
        # `ForwardResult` has no field for them — but named, and therefore checked.
        (
            "solver_metadata['states_path']",
            result.solver_metadata.get("states_path"),
            "states_sha256",
        ),
        (
            f"solver_metadata['{STEPS_FILENAME}']",
            result.solver_metadata.get(f"{STEPS_FILENAME}.path"),
            f"{STEPS_FILENAME}.sha256",
        ),
    ]
    if result.black_oil is not None:
        # Task 13: the gas table is named by the record's own black-oil block, so it is
        # re-proved on read like every other output rather than trusted because it exists.
        checked.append(
            (
                "black_oil.gas_monthly_path",
                result.black_oil.gas_monthly_path,
                f"{GAS_MONTHLY_FILENAME}.sha256",
            )
        )
    for label, name, digest_key in checked:
        _verify_published_file(relative, label, name, result, digest_key, paths)

    fields = {
        name: _read_state(relative, name, ref, paths) for name, ref in sorted(result.states.items())
    }
    _verify_time_axis(relative, result, np.asarray(result.times_s, dtype=np.float64), paths)
    _verify_physical_ranges(relative, fields)
    return result


def _verify_restart_manifest(
    relative: str,
    restart: RestartRef,
    solver_metadata: Mapping[str, str],
    paths: ProjectPaths,
) -> None:
    """Re-prove the checkpoint manifest's digest, and the copy of it in `solver_metadata`.

    The manifest names every native file with its own SHA-256, so a manifest whose bytes
    hash to what the record declares is a trustworthy list; `restart.jl:verify_restart`
    re-hashes each file it names at resume time, before a single native byte is read. This
    is the other end of that: it stops a record from travelling with a digest for a
    checkpoint that is no longer the checkpoint it was written for.

    `solver_metadata` carries a convenience copy of the path and the digest. `RestartRef` is
    the authority, so the copy is not required — but where it exists it must agree, because
    a record that says two different things about one checkpoint is a record nobody can act
    on.
    """
    target = paths.resolve(restart.manifest_path)
    if not target.is_file():
        raise ForwardResultIntegrityError(
            f"{relative}: the restart names {restart.manifest_path}, which does not exist; an "
            "unfinished staging directory is not a checkpoint"
        )
    digest = sha256_file(target)
    if digest != restart.sha256:
        raise ForwardResultIntegrityError(
            f"{relative}: the checkpoint manifest {restart.manifest_path} hashes to {digest}, "
            f"but the record declares {restart.sha256}"
        )
    for key, actual in (
        ("restart_manifest_path", restart.manifest_path),
        ("restart_manifest_sha256", restart.sha256),
    ):
        declared = solver_metadata.get(key)
        if declared is not None and declared != actual:
            raise ForwardResultIntegrityError(
                f"{relative}: solver_metadata[{key!r}] is {declared!r}, but the restart "
                f"reference says {actual!r}"
            )


def _verify_published_file(
    relative: str,
    label: str,
    named: str | None,
    result: ForwardResult,
    digest_key: str,
    paths: ProjectPaths,
) -> None:
    if named is None:
        raise ForwardResultIntegrityError(
            f"{relative}: a COMPLETE result must name {label}; an output the record does not "
            "name is an output nobody can check"
        )
    target = paths.resolve(named)
    if not target.is_file():
        raise ForwardResultIntegrityError(
            f"{relative}: {label} names {named}, which does not exist"
        )
    declared = result.solver_metadata.get(digest_key)
    if declared is None:
        raise ForwardResultIntegrityError(
            f"{relative}: solver_metadata carries no {digest_key} for {named}; an absent hash "
            "is null with a reason, never assumed to match"
        )
    digest = sha256_file(target)
    if digest != declared:
        raise ForwardResultIntegrityError(
            f"{relative}: {named} hashes to {digest}, but the record declares {declared}"
        )


def _read_state(
    relative: str, name: str, ref: ArrayRef, paths: ProjectPaths
) -> NDArray[np.float64]:
    try:
        values = read_array(ref, paths)
    except (ValueError, OSError) as exc:
        raise ForwardResultIntegrityError(f"{relative}: states[{name!r}]: {exc}") from exc
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ForwardResultIntegrityError(f"{relative}: states[{name!r}] holds nonfinite values")
    return array


def _verify_time_axis(
    relative: str, result: ForwardResult, times: NDArray[np.float64], paths: ProjectPaths
) -> None:
    """The record's own time axis has to be the one the states file was written against.

    The axis lives in the same file as the fields it indexes, so the reference below is the
    file the record already named and hashed, pointed at the `time_s` dataset instead of a
    state one. A record whose `times_s` drifted from the axis the arrays were written
    against would otherwise index every published field by a time nobody simulated.
    """
    axis = ArrayRef(
        path=str(result.solver_metadata["states_path"]),
        dataset="time_s",
        sha256=str(result.solver_metadata["states_sha256"]),
        shape=(len(result.times_s),),
        dtype="float64",
        unit="s",
        axis_order=("time",),
    )
    try:
        stored = np.asarray(read_array(axis, paths), dtype=np.float64)
    except (ValueError, OSError) as exc:
        raise ForwardResultIntegrityError(
            f"{relative}: the states file has no usable time axis: {exc}"
        ) from exc
    if stored.shape != times.shape or not np.array_equal(stored, times):
        raise ForwardResultIntegrityError(
            f"{relative}: times_s {result.times_s} is not the time axis stored in the states "
            f"file, which holds {stored.tolist()}"
        )


def _verify_physical_ranges(relative: str, fields: Mapping[str, NDArray[np.float64]]) -> None:
    """Refuse a published state that is not a physical one. Nothing is clipped into range."""
    if "pressure_pa" in fields and not (fields["pressure_pa"] > 0.0).all():
        raise ForwardResultIntegrityError(f"{relative}: states['pressure_pa'] is not positive")
    for name in ("sw", "so", "sg"):
        if name in fields and not ((fields[name] >= 0.0) & (fields[name] <= 1.0)).all():
            raise ForwardResultIntegrityError(f"{relative}: states[{name!r}] lies outside [0,1]")
    # Every phase the result published, summed. For a black-oil result that is three, and the
    # sum is `1 + MINIMUM_COMPOSITIONAL_SATURATION` by construction in the pinned engine
    # (JutulDarcy 0.3.11, `blackoil/variables/varswitch.jl`, constant 1e-10) — far inside the
    # drift limit below, and still a real failure an order of magnitude above it.
    phases = [name for name in ("sw", "so", "sg") if name in fields]
    if "sw" in fields and "so" in fields:
        total = sum(fields[name] for name in phases)
        drift = float(np.max(np.abs(total - 1.0)))
        if drift > _MAX_SATURATION_DRIFT:
            raise ForwardResultIntegrityError(
                f"{relative}: {' + '.join(phases)} departs from 1 by {drift:g}; saturations "
                "are refused, never renormalised"
            )
    if "rs" in fields and not (fields["rs"] >= 0.0).all():
        raise ForwardResultIntegrityError(f"{relative}: states['rs'] is negative")
    for name in ("pore_volume_m3", "bw", "bo", "bg"):
        if name in fields and not (fields[name] > 0.0).all():
            raise ForwardResultIntegrityError(f"{relative}: states[{name!r}] is not positive")
