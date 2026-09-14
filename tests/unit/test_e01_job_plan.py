"""E01.12.4 — the planned matrix against the forwards that really run.

A forward with no row is a forward nobody accounted for. The refinement diagnostic runs
FOUR forwards (`julia/README.md`: "10.7-10.8 (4 forwards)") and the P1 plan listed three,
so `bl_64` — declared in the P0 table and deferred to P1 — had no `JobOutcome` anywhere and
the launcher wall time was divided by three instead of four.

These tests read `configs/e01_jobs.json` itself. They are about the PLAN, not about a
session: nothing here launches anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _plan() -> dict[str, Any]:
    return json.loads((REPO / "configs" / "e01_jobs.json").read_text(encoding="utf-8"))


def _fixtures(jobs: list[dict[str, Any]]) -> set[str]:
    return {str(job["native_source"]).split("#")[-1] for job in jobs}


def test_the_p1_refinement_group_lists_every_forward_the_diagnostic_runs() -> None:
    """`refinement.jl` runs bl_64, bl_128, five_spot_16 and five_spot_48. All four get a row."""
    jobs = [j for j in _plan()["suites"]["p1"]["jobs"] if j["group"] == "refinement"]
    assert _fixtures(jobs) == {"bl_64", "bl_128", "five_spot_16", "five_spot_48"}
    assert len(jobs) == 4, "one launcher forward per fixture the diagnostic really runs"


def test_the_deferred_p0_row_is_the_one_executed_in_p1() -> None:
    """`bl64_halfdt` is declared by the P0 table and RUN by the P1 diagnostic.

    The P0 row keeps `deferred_to: p1` so a P0 session reports it as deferred rather than
    missing; the P1 row carries `declared_by_plan: false` so the brief's 23/17/4 declared
    counts do not move because the cost of a forward was written down.
    """
    plan = _plan()
    p0 = next(j for j in plan["suites"]["p0"]["jobs"] if j["job_id"] == "bl64_halfdt")
    assert p0["deferred_to"] == "p1"
    assert p0["declared_by_plan"] is True
    p1 = next(j for j in plan["suites"]["p1"]["jobs"] if j["job_id"] == "bl64_halfdt")
    assert p1["deferred_to"] is None
    assert p1["declared_by_plan"] is False
    assert p1["native_source"].endswith("#bl_64")
    assert p1["accounting"] == "launcher"


def test_the_declared_counts_of_the_brief_have_not_moved() -> None:
    """23 P0 / 17 P1 / 4 BO, declared, with no retry. Nothing above may change them."""
    plan = _plan()
    for suite, expected in (("p0", 23), ("p1", 17), ("bo", 4)):
        block = plan["suites"][suite]
        assert block["declared_count_without_retry"] == expected, suite
        declared = [j for j in block["jobs"] if j["declared_by_plan"]]
        assert len(declared) == expected, (suite, sorted(j["job_id"] for j in declared))


def test_the_plan_accounts_for_every_launcher_forward_the_diagnostics_run() -> None:
    """7 analytic + 9 operational + 7 controls + 4 refinement = 27 launcher forwards."""
    plan = _plan()
    counted = plan["counting"]["test_launcher_forwards"]
    assert counted["analytic_fixtures"] + counted["operations_p0"] == 16
    assert counted["controls_diagnostic"] == 7
    assert counted["refinement_p1"] == 4
    rows = [
        job
        for suite in ("p0", "p1")
        for job in plan["suites"][suite]["jobs"]
        if job["accounting"] == "launcher" and job.get("deferred_to") is None
    ]
    assert len(rows) == 27, sorted(job["job_id"] for job in rows)
