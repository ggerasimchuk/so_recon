"""E01.6: the calendar, and the compiled per-interval control schedule.

A month is a real month. The whole point of this module is that February 2020 is 29 days
and February 2021 is 28, so a schedule built here cannot quietly become twelve 30-day
months — a 1.7% error on every monthly rate, in the same direction every year.

The second half of the file is `compile_schedule`, which turns the report edges and the
case's control segments into the thing a simulator can actually be driven with: one edge
list, one control per well per interval, and no inheritance between intervals. Every way
that can go wrong — a gap, an overlap, an interval nobody covers, an event boundary that
is almost but not exactly a report edge — is a refusal with the well and the interval in
the message, never a silently repaired schedule.
"""

from __future__ import annotations

from datetime import date

import pytest

from so_recon.simulator.contracts import SECONDS_PER_DAY, ControlSegment
from so_recon.simulator.schedule import (
    MIN_INTERVAL_S,
    Schedule,
    compile_schedule,
    month_edges,
    month_edges_s,
)

DAY = SECONDS_PER_DAY


def _segment(
    start_days: float,
    end_days: float,
    *,
    well_id: str = "PRO1",
    role: str = "producer",
    target: str = "liquid_rate",
    value: float = 40.0,
    bhp_limit_pa: float | None = 1.0e7,
    connection_open: tuple[bool, ...] = (True, True),
) -> ControlSegment:
    return ControlSegment(
        start_s=start_days * DAY,
        end_s=end_days * DAY,
        well_id=well_id,
        role=role,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        value=value,
        bhp_limit_pa=bhp_limit_pa,
        connection_open=connection_open,
    )


def _shut(start_days: float, end_days: float, *, well_id: str = "PRO1") -> ControlSegment:
    return _segment(
        start_days,
        end_days,
        well_id=well_id,
        role="shut",
        target="disabled",
        value=0.0,
        bhp_limit_pa=None,
        connection_open=(False, False),
    )


# ------------------------------------------------------------------------ the calendar


def test_leap_year_calendar() -> None:
    edges = month_edges(date(2020, 1, 1), 3)
    assert [(b - a).days for a, b in zip(edges, edges[1:], strict=False)] == [31, 29, 31]


def test_the_same_february_is_28_days_in_a_common_year() -> None:
    """The leap day is read from the calendar, not from a rule someone remembered."""
    edges = month_edges(date(2021, 1, 1), 3)
    assert [(b - a).days for a, b in zip(edges, edges[1:], strict=False)] == [31, 28, 31]


def test_a_year_of_months_is_never_twelve_thirty_day_months() -> None:
    edges = month_edges(date(2020, 1, 1), 12)
    lengths = [(b - a).days for a, b in zip(edges, edges[1:], strict=False)]
    assert lengths == [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert sum(lengths) == 366
    # A 30-day average month would have produced 360 days and every month the same.
    assert len(set(lengths)) > 1


def test_months_roll_over_the_year_boundary() -> None:
    edges = month_edges(date(2020, 11, 1), 3)
    assert edges == (date(2020, 11, 1), date(2020, 12, 1), date(2021, 1, 1), date(2021, 2, 1))


def test_a_calendar_must_start_on_the_first_of_a_month() -> None:
    with pytest.raises(ValueError, match="month start"):
        month_edges(date(2020, 1, 15), 3)


@pytest.mark.parametrize("count", [0, -1])
def test_a_calendar_needs_at_least_one_month(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        month_edges(date(2020, 1, 1), count)


def test_month_edges_in_seconds_start_at_zero_and_use_real_month_lengths() -> None:
    """The seconds a case carries in `report_edges_s`, with `1 day = 86400 s`."""
    assert month_edges_s(date(2020, 1, 1), 2) == (0.0, 31.0 * DAY, 60.0 * DAY)
    assert month_edges_s(date(2021, 1, 1), 2) == (0.0, 31.0 * DAY, 59.0 * DAY)


# --------------------------------------------------------------- the compiled schedule


def test_a_schedule_without_events_is_the_report_calendar() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(report, (_segment(0.0, 60.0),))
    assert schedule.edges_s == report
    assert schedule.month_index == (0, 1)
    assert [len(controls) for controls in schedule.controls_by_interval] == [1, 1]
    assert schedule.wells == ("PRO1",)


def test_an_event_inside_a_month_splits_that_month_and_keeps_its_month_index() -> None:
    """A completion or a control change mid-month adds an edge; it does not add a month."""
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 45.0),
            # Same well, same rate, one perforation closed from day 45 — an event that
            # belongs to February and must not be rounded to a month boundary.
            _segment(45.0, 60.0, connection_open=(True, False)),
        ),
    )
    assert schedule.edges_s == (0.0, 31.0 * DAY, 45.0 * DAY, 60.0 * DAY)
    assert schedule.month_index == (0, 1, 1)
    assert schedule.control_for(1, "PRO1").connection_open == (True, True)
    assert schedule.control_for(2, "PRO1").connection_open == (True, False)


def test_the_report_calendar_is_recoverable_from_the_compiled_schedule() -> None:
    """The months an aggregator reports on, read back out of the compiled intervals.

    The compiled edge list is the union of the report edges and every event boundary, so the
    report edges are the subset of it where the month changes. Recovering them is what lets
    a monthly aggregator be given the case's own months without the case travelling beside
    its schedule — and getting it wrong would silently report on the events instead.
    """
    report = month_edges_s(date(2020, 1, 1), 3)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 45.0),
            _segment(45.0, 52.0, connection_open=(True, False)),
            _segment(52.0, 91.0),
        ),
    )
    assert schedule.edges_s == (0.0, 31.0 * DAY, 45.0 * DAY, 52.0 * DAY, 60.0 * DAY, 91.0 * DAY)
    assert schedule.month_index == (0, 1, 1, 1, 2)
    # Three real months — 31, 29 and 31 days — and not the five compiled intervals.
    assert schedule.month_edges_s == report
    assert schedule.month_edges_s == (0.0, 31.0 * DAY, 60.0 * DAY, 91.0 * DAY)


def test_a_role_switch_gives_one_month_two_flows() -> None:
    """SPEC 9.1 controls both roles; a well that switches mid-month is two flows, not one."""
    report = month_edges_s(date(2020, 1, 1), 1)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 10.0, role="producer", target="liquid_rate", value=40.0),
            _segment(
                10.0,
                31.0,
                role="injector",
                target="water_rate",
                value=50.0,
                bhp_limit_pa=4.0e7,
            ),
        ),
    )
    assert schedule.edges_s == (0.0, 10.0 * DAY, 31.0 * DAY)
    # One month, two intervals: an average over the month would erase the switch.
    assert schedule.month_index == (0, 0)
    assert schedule.control_for(0, "PRO1").role == "producer"
    assert schedule.control_for(1, "PRO1").role == "injector"
    assert schedule.monthly_uptime_s("PRO1") == (31.0 * DAY,)


def test_an_interval_no_control_covers_is_a_gap_and_is_refused() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    with pytest.raises(ValueError, match="PRO1.*no control"):
        compile_schedule(report, (_segment(0.0, 31.0), _segment(45.0, 60.0)))


def test_an_uncovered_first_interval_is_refused() -> None:
    report = month_edges_s(date(2020, 1, 1), 1)
    with pytest.raises(ValueError, match="PRO1.*no control"):
        compile_schedule(report, (_segment(5.0, 31.0),))


def test_two_controls_on_one_interval_are_refused() -> None:
    report = month_edges_s(date(2020, 1, 1), 1)
    with pytest.raises(ValueError, match="PRO1.*overlap"):
        compile_schedule(report, (_segment(0.0, 31.0), _segment(10.0, 31.0)))


def test_a_control_past_the_last_report_edge_is_refused() -> None:
    report = month_edges_s(date(2020, 1, 1), 1)
    with pytest.raises(ValueError, match="past the last report edge"):
        compile_schedule(report, (_segment(0.0, 40.0),))


def test_report_edges_must_start_at_zero_and_increase() -> None:
    with pytest.raises(ValueError, match="start at 0"):
        compile_schedule((DAY, 2 * DAY), (_segment(1.0, 2.0),))
    with pytest.raises(ValueError, match="strictly increasing"):
        compile_schedule((0.0, 2 * DAY, DAY), (_segment(0.0, 1.0),))
    with pytest.raises(ValueError, match="at least one report interval"):
        compile_schedule((0.0,), ())


def test_a_zero_duration_control_cannot_be_built_at_all() -> None:
    """The degenerate segment is refused by the contract, before a schedule sees it."""
    with pytest.raises(ValueError, match="end_s must be greater than start_s"):
        _segment(31.0, 31.0)


def test_an_edge_that_almost_coincides_with_a_report_edge_is_refused() -> None:
    """A near-miss edge would compile into a degenerate interval; say so instead.

    This is the only way a zero-duration interval can survive `compile_schedule`: the
    segments themselves are non-degenerate, and identical edges are one edge. Half a
    second between two edges is not a schedule event, it is an edge that was meant to be
    a report edge and missed.
    """
    report = month_edges_s(date(2020, 1, 1), 2)
    almost = 31.0 * DAY + 0.5
    with pytest.raises(ValueError, match="shorter than"):
        compile_schedule(
            report,
            (
                _segment(0.0, almost / DAY),
                _segment(almost / DAY, 60.0),
            ),
        )
    assert MIN_INTERVAL_S > 0.5


def test_an_exactly_coinciding_event_and_report_edge_are_one_edge() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(report, (_segment(0.0, 31.0), _segment(31.0, 60.0)))
    assert schedule.edges_s == report
    assert schedule.month_index == (0, 1)


def test_every_well_is_compiled_on_every_interval() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 60.0, well_id="PRO1"),
            _segment(0.0, 20.0, well_id="INJ1", role="injector", target="water_rate", value=50.0),
            _shut(20.0, 60.0, well_id="INJ1"),
        ),
    )
    assert schedule.wells == ("INJ1", "PRO1")
    assert schedule.edges_s == (0.0, 20.0 * DAY, 31.0 * DAY, 60.0 * DAY)
    # Nothing is inherited: each interval names a control for each well, by construction.
    for controls in schedule.controls_by_interval:
        assert sorted(c.well_id for c in controls) == ["INJ1", "PRO1"]


# ------------------------------------------------------------- uptime, and a zero rate


def test_a_zero_rate_is_a_shut_well_and_never_a_rate_of_zero() -> None:
    """There is no `liquid_rate = 0`: a well that does not flow is shut (SPEC 9.1).

    A zero rate control would claim a phase split over an interval in which no phase was
    produced. The contract has no way to write it, which is what makes the uptime below
    unambiguous rather than a convention.
    """
    with pytest.raises(ValueError, match="value must be positive"):
        _segment(0.0, 31.0, value=0.0)


def test_uptime_is_the_open_part_of_the_month_and_nothing_else() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 45.0),
            _shut(45.0, 60.0),
        ),
    )
    # January is fully open; February flows for 14 of its 29 days.
    assert schedule.monthly_uptime_s("PRO1") == (31.0 * DAY, 14.0 * DAY)


def test_a_well_with_every_connection_closed_has_no_uptime_even_while_it_is_a_producer() -> None:
    """Uptime is flow, not intent: a fully isolated completion produces nothing."""
    report = month_edges_s(date(2020, 1, 1), 1)
    schedule = compile_schedule(
        report,
        (
            _segment(0.0, 10.0),
            _segment(10.0, 31.0, connection_open=(False, False)),
        ),
    )
    assert schedule.monthly_uptime_s("PRO1") == (10.0 * DAY,)


def test_a_positive_flow_over_a_month_with_no_uptime_is_refused() -> None:
    """E01 does not reconstruct an unknown field uptime; a contradiction is an error."""
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(report, (_segment(0.0, 31.0), _shut(31.0, 60.0)))
    assert schedule.monthly_uptime_s("PRO1") == (31.0 * DAY, 0.0)
    schedule.reject_flow_without_uptime("PRO1", (62.0, 0.0))
    with pytest.raises(ValueError, match="PRO1.*month 1.*no uptime"):
        schedule.reject_flow_without_uptime("PRO1", (62.0, 3.5))


def test_a_flow_series_must_have_one_value_per_month() -> None:
    report = month_edges_s(date(2020, 1, 1), 2)
    schedule = compile_schedule(report, (_segment(0.0, 60.0),))
    with pytest.raises(ValueError, match="one volume per report interval"):
        schedule.reject_flow_without_uptime("PRO1", (62.0,))


def test_uptime_and_flow_checks_name_an_unknown_well() -> None:
    report = month_edges_s(date(2020, 1, 1), 1)
    schedule = compile_schedule(report, (_segment(0.0, 31.0),))
    with pytest.raises(KeyError, match="INJ9"):
        schedule.monthly_uptime_s("INJ9")
    with pytest.raises(KeyError, match="INJ9"):
        schedule.control_for(0, "INJ9")


def test_a_schedule_is_frozen() -> None:
    report = month_edges_s(date(2020, 1, 1), 1)
    schedule = compile_schedule(report, (_segment(0.0, 31.0),))
    assert isinstance(schedule, Schedule)
    with pytest.raises(ValueError, match="frozen"):
        schedule.edges_s = (0.0,)  # type: ignore[misc]
