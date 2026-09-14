"""E01.6 — the calendar, and the per-interval control schedule compiled from it.

Two things live here, and neither of them is physics.

**The calendar is explicit.** `month_edges` walks real months: January 2020 is 31 days,
February 2020 is 29 and February 2021 is 28. There is no average month anywhere in this
module. A 30-day month is wrong by up to 1.7% on every monthly rate, always in the same
direction within a year, which is exactly the kind of bias a forward operator cannot be
audited out of later — the monthly volumes would simply be somebody else's months.

**The schedule is explicit too.** `compile_schedule` takes the report edges and the case's
control segments and returns the edge list a simulator can be driven with: the sorted
union of the report edges and every event boundary, plus, for each of those intervals,
exactly one control for each well. Nothing is inherited between intervals — a role, a
rate, a bottom-hole pressure and a completion mask are all restated on every interval —
because "the well kept doing whatever it was doing" is the assumption that turns a missing
segment into a plausible-looking result instead of a refusal.

Every refusal here names the well and the interval it is about. A gap, an overlap, a
segment reaching past the last report edge and an edge that almost coincides with a report
edge are all errors; the caller reports them as `INVALID_INPUT` (plan 3.3).

**Uptime is given, never reconstructed.** The open/shut intervals are the uptime: a month
in which a well is shut for half the month has half a month of uptime, and a month in
which every connection is closed has none at all. E01 does not infer an unknown field
uptime from anything, so `reject_flow_without_uptime` exists to refuse the contradiction —
a positive produced volume over a month in which nothing was open — rather than to explain
it away.
"""

from __future__ import annotations

from datetime import date

from pydantic import model_validator

from so_recon.config.schema import StrictModel
from so_recon.simulator.contracts import SECONDS_PER_DAY, ControlSegment

#: The shortest interval that is a schedule event rather than a floating-point accident.
#: Identical edges collapse into one edge, so the only way a degenerate interval can reach
#: a simulator is an event boundary that was meant to be a report edge and missed it by a
#: fraction of a second. A second is far below any control change anyone schedules and far
#: above the round-off of a day count multiplied by 86400.
MIN_INTERVAL_S = 1.0

#: A produced or injected volume below this is not flow; it is the solver's own zero.
#: Used only to decide whether a volume reported over a month with no uptime contradicts
#: that uptime (a cubic millimetre over a month cannot be a well that was actually open).
FLOW_TOLERANCE_M3 = 1e-9


def month_edges(start: date, count: int) -> tuple[date, ...]:
    """The `count + 1` month boundaries starting at `start`, as real calendar dates.

    `start` must be the first of a month: a schedule that began on the 12th would make
    "month" mean a 12th-to-12th period that no report ever aggregates over.
    """
    if start.day != 1:
        raise ValueError(f"start must be a month start (the 1st), got {start.isoformat()}")
    if count < 1:
        raise ValueError(f"count must be a positive number of months, got {count}")
    base = start.year * 12 + start.month - 1
    return tuple(date((base + i) // 12, (base + i) % 12 + 1, 1) for i in range(count + 1))


def month_edges_s(start: date, count: int) -> tuple[float, ...]:
    """The same boundaries as seconds from `start`, which is what a case carries.

    `1 day = 86400 s` (plan 3.1). The length of each month comes from the calendar above,
    so the gaps are 31, 29, 31 ... days and never twelve equal ones.
    """
    edges = month_edges(start, count)
    return tuple((edge - edges[0]).days * SECONDS_PER_DAY for edge in edges)


class Schedule(StrictModel):
    """The compiled calendar: edges, which month each interval belongs to, and controls.

    `edges_s[i] .. edges_s[i+1]` is interval `i`; `month_index[i]` is the report interval
    (the month) it lies inside; `controls_by_interval[i]` holds exactly one control for
    each well, restated rather than inherited.
    """

    edges_s: tuple[float, ...]
    month_index: tuple[int, ...]
    controls_by_interval: tuple[tuple[ControlSegment, ...], ...]

    @model_validator(mode="after")
    def _shapes_agree(self) -> Schedule:
        n = len(self.edges_s) - 1
        if n < 1:
            raise ValueError(f"a schedule needs at least one interval, got edges {self.edges_s}")
        if len(self.month_index) != n or len(self.controls_by_interval) != n:
            raise ValueError(
                f"a schedule of {n} intervals needs {n} month indices and {n} control "
                f"groups, got {len(self.month_index)} and {len(self.controls_by_interval)}"
            )
        # Every interval names every well. This is what "nothing is inherited" means as a
        # data invariant, and it is what makes `wells` below a property of the schedule
        # rather than of whichever interval happens to be read first.
        expected = {c.well_id for c in self.controls_by_interval[0]}
        for interval, controls in enumerate(self.controls_by_interval):
            named = [c.well_id for c in controls]
            if len(named) != len(set(named)) or set(named) != expected:
                raise ValueError(
                    f"interval {interval} controls {sorted(named)}, but the schedule is "
                    f"about {sorted(expected)}; every well carries exactly one control on "
                    "every interval"
                )
        return self

    @property
    def n_months(self) -> int:
        return self.month_index[-1] + 1

    @property
    def wells(self) -> tuple[str, ...]:
        """Every well the schedule controls, in a stable order."""
        return tuple(sorted({c.well_id for c in self.controls_by_interval[0]}))

    @property
    def durations_s(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.edges_s, self.edges_s[1:], strict=False))

    @property
    def month_edges_s(self) -> tuple[float, ...]:
        """The report edges the compiled intervals were grouped into, recovered.

        The compiled edge list is the union of the report edges and every event boundary,
        so the report edges are the subset of it at which `month_index` changes — plus the
        two ends. Recovering them is what lets an aggregator be given the months a case
        reports on without the case having to travel beside its own schedule.
        """
        edges = [self.edges_s[0]]
        for interval, month in enumerate(self.month_index):
            last = interval + 1 == len(self.month_index)
            if last or self.month_index[interval + 1] != month:
                edges.append(self.edges_s[interval + 1])
        return tuple(edges)

    def control_for(self, interval: int, well_id: str) -> ControlSegment:
        """The one control this well runs on this interval."""
        for control in self.controls_by_interval[interval]:
            if control.well_id == well_id:
                return control
        raise KeyError(f"well {well_id!r} has no control on interval {interval}")

    def monthly_uptime_s(self, well_id: str) -> tuple[float, ...]:
        """How long this well actually flowed in each month.

        Flow needs a surface control that is not `shut` AND at least one open connection:
        a producer whose every completion is closed moves nothing, whatever the surface
        says. The open/shut intervals are the given uptime — nothing here estimates it.
        """
        uptime = [0.0] * self.n_months
        for interval, (month, duration) in enumerate(
            zip(self.month_index, self.durations_s, strict=True)
        ):
            control = self.control_for(interval, well_id)
            if control.role != "shut" and any(control.connection_open):
                uptime[month] += duration
        return tuple(uptime)

    def reject_flow_without_uptime(self, well_id: str, volumes_m3: tuple[float, ...]) -> None:
        """Refuse a positive monthly volume over a month in which nothing was open.

        `volumes_m3` is one total per month, in the order of the report edges. This is a
        contradiction, not a calibration problem: E01 is given its uptime, so a volume
        that disagrees with it means the schedule and the flow are not the same case.
        """
        uptime = self.monthly_uptime_s(well_id)
        if len(volumes_m3) != len(uptime):
            raise ValueError(
                f"well {well_id!r}: one volume per report interval is required, got "
                f"{len(volumes_m3)} for {len(uptime)} months"
            )
        for month, (volume, open_s) in enumerate(zip(volumes_m3, uptime, strict=True)):
            if open_s == 0.0 and abs(volume) > FLOW_TOLERANCE_M3:
                raise ValueError(
                    f"well {well_id!r}: month {month} has no uptime — every interval in it "
                    f"is shut or fully isolated — but carries a flow of {volume} m3; E01 "
                    "does not reconstruct an unknown uptime from a volume"
                )


def compile_schedule(
    report_edges_s: tuple[float, ...], segments: tuple[ControlSegment, ...]
) -> Schedule:
    """Compile the report calendar and the case's control segments into one schedule.

    The edges are the sorted union of the report edges and every segment boundary, so a
    control change inside a month becomes its own interval and the month it belongs to is
    still the month. Each interval then gets exactly one control per well: no gap, no
    overlap, and nothing carried over from the interval before.
    """
    edges = _validated_report_edges(report_edges_s)
    horizon = edges[-1]
    wells = sorted({segment.well_id for segment in segments})
    if not wells:
        raise ValueError("a schedule needs at least one control segment")

    by_well: dict[str, list[ControlSegment]] = {well: [] for well in wells}
    boundaries = set(edges)
    for segment in segments:
        if segment.end_s > horizon:
            raise ValueError(
                f"well {segment.well_id!r}: control segment ends at {segment.end_s} s, past "
                f"the last report edge at {horizon} s"
            )
        by_well[segment.well_id].append(segment)
        boundaries.add(segment.start_s)
        boundaries.add(segment.end_s)

    all_edges = tuple(sorted(boundaries))
    for a, b in zip(all_edges, all_edges[1:], strict=False):
        if b - a < MIN_INTERVAL_S:
            raise ValueError(
                f"the compiled schedule has an interval [{a}, {b}) shorter than "
                f"{MIN_INTERVAL_S} s; an event boundary that close to another edge is a "
                "near-miss, not a control change"
            )

    month_index: list[int] = []
    controls_by_interval: list[tuple[ControlSegment, ...]] = []
    month = 0
    for index, (start, end) in enumerate(zip(all_edges, all_edges[1:], strict=False)):
        while edges[month + 1] <= start:
            month += 1
        month_index.append(month)
        controls_by_interval.append(
            tuple(_covering_control(by_well[well], well, index, start, end) for well in wells)
        )

    return Schedule(
        edges_s=all_edges,
        month_index=tuple(month_index),
        controls_by_interval=tuple(controls_by_interval),
    )


def _validated_report_edges(report_edges_s: tuple[float, ...]) -> tuple[float, ...]:
    """The same rule `CaseBundle` enforces, restated where the schedule is compiled."""
    if len(report_edges_s) < 2:
        raise ValueError(
            f"report_edges_s must describe at least one report interval: {report_edges_s}"
        )
    if report_edges_s[0] != 0.0:
        raise ValueError(f"report_edges_s must start at 0 seconds, got {report_edges_s[0]}")
    if any(b <= a for a, b in zip(report_edges_s, report_edges_s[1:], strict=False)):
        raise ValueError(f"report_edges_s must be strictly increasing, got {report_edges_s}")
    return tuple(report_edges_s)


def _covering_control(
    segments: list[ControlSegment], well_id: str, interval: int, start: float, end: float
) -> ControlSegment:
    """The single segment of this well that covers `[start, end)`, or a refusal.

    The interval's own boundaries are segment boundaries by construction, so a segment
    either covers the whole interval or does not touch its interior at all.
    """
    covering = [s for s in segments if s.start_s <= start and s.end_s >= end]
    if len(covering) == 1:
        return covering[0]
    if not covering:
        raise ValueError(
            f"well {well_id!r} has no control on interval {interval} [{start}, {end}) s; "
            "every well carries exactly one control on every interval"
        )
    spans = ", ".join(f"[{s.start_s}, {s.end_s})" for s in covering)
    raise ValueError(
        f"well {well_id!r}: {len(covering)} controls overlap on interval {interval} "
        f"[{start}, {end}) s: {spans}"
    )
