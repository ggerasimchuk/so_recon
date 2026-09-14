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

from so_recon.paths import ProjectPaths
from so_recon.simulator.suite_record import (
    MANDATORY_CHECKS,
    JobOutcome,
    SuiteOutcome,
    evaluate_suite,
    load_job_plan,
)
from so_recon.validation.physics import PhysicsCheck

REPO = Path(__file__).resolve().parents[2]

#: The approved preset the P1 benchmark runs under, and the field of it that forbids the
#: thread comparison. `require_resource_profile` matches an approved preset byte-strict, so
#: a one- or two-thread worker needs a NEW preset — a COMPUTE §5 change, not an
#: implementer's choice.
REFUSING_PRESET = "P1_LOOP"


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


# --------------------------------------------------------------------------------------
# ruling G — a job the presets structurally forbid declares NOT_RUN, and says which preset
# --------------------------------------------------------------------------------------


def _threads_jobs() -> list[dict[str, Any]]:
    return [j for j in _plan()["suites"]["p1"]["jobs"] if j["job_id"].startswith("threads")]


def test_the_thread_comparison_declares_the_outcome_it_can_actually_have() -> None:
    """`expected_outcome: COMPLETE` for a job no approved preset can run is a false claim.

    It also had a consequence: `evaluate_suite` scored all four as attempts that missed
    their expected outcome, so a P1 session could never exit 0 on that account alone.
    """
    jobs = _threads_jobs()
    assert len(jobs) == 4
    for job in jobs:
        assert job["expected_outcome"] == "NOT_RUN", job["job_id"]


def test_each_forbidden_job_names_the_preset_that_refuses_it() -> None:
    """The named reason is the guard on this mechanism.

    Declaring `NOT_RUN` is how a job that CANNOT run stops being scored as a surprise. That
    is exactly the shape a job which genuinely should have completed could be declared away
    in, so the declaration has to name the approved preset that refuses it and the setting
    it refuses on.
    """
    for job in _threads_jobs():
        note = str(job["note"] or "")
        assert REFUSING_PRESET in note, (job["job_id"], note)
        assert "julia_threads" in note, (job["job_id"], note)
        assert str(job["julia_threads"]) in note, (job["job_id"], note)


def test_no_other_job_in_any_suite_declares_not_run() -> None:
    """Narrow by construction: only the jobs an approved preset structurally forbids."""
    plan = _plan()
    declared = [
        job["job_id"]
        for suite in plan["suites"].values()
        for job in suite["jobs"]
        if job["expected_outcome"] == "NOT_RUN"
    ]
    assert sorted(declared) == ["threads1_1", "threads1_2", "threads2_1", "threads2_2"]
    for job in _threads_jobs():
        assert job["julia_threads"] != 4, (
            f"{job['job_id']} asks for the thread count the preset already allows; "
            "then nothing forbids it and NOT_RUN is not the truth about it"
        )


def test_a_p1_session_that_only_misses_the_forbidden_jobs_is_not_scored_as_surprised() -> None:
    """The consequence the ruling is for, against the REAL plan and the REAL evaluator."""
    paths = ProjectPaths.default(REPO)
    plan = load_job_plan(paths.resolve("configs/e01_jobs.json"), paths).suites["p1"]

    def outcome(job_id: str, group: str, status: str, expected: str) -> JobOutcome:
        return JobOutcome.model_validate(
            {
                "job_id": job_id,
                "group": group,
                "kind": "benchmark",
                "profile": "P1_LOOP",
                "accounting": "ledger",
                "expected_outcome": expected,
                "status": status,
                "wall_s": 1.0,
                "cpu_s": 1.0,
                "peak_rss_bytes": 1,
                "output_bytes": 1,
                "native_chunk_calls": 1,
                "accepted_steps": 1,
                "cut_steps": 0,
                "nonlinear_iterations": 1,
                "retry_count": 0,
            }
        )

    # What a real P1 session produces: every job as the matrix expects it, and the four
    # thread jobs recorded NOT_RUN by `_thread_comparison` because the preset forbids them.
    jobs = tuple(
        outcome(
            job.job_id,
            job.group,
            "NOT_RUN" if job.job_id.startswith("threads") else job.expected_outcome,
            job.expected_outcome,
        )
        for job in plan.jobs
        if job.deferred_to is None
    )
    assert any(job.status == "NOT_RUN" for job in jobs)
    names = {job.scored_as for job in plan.jobs if job.scored_as in MANDATORY_CHECKS}
    checks = tuple(
        PhysicsCheck(
            name=name,
            status="PASS",
            metrics={"x": 0.0},
            thresholds={"x_max": 1.0},
            input_hashes={},
            evidence_paths=(),
            reason=None,
        )
        for name in sorted(names)
    ) + (
        PhysicsCheck(
            name="benchmark_threads",
            status="NOT_RUN",
            metrics={},
            thresholds={},
            input_hashes={},
            evidence_paths=(),
            reason="the approved preset forbids it",
        ),
    )
    code, limitations = evaluate_suite(SuiteOutcome(jobs=jobs, checks=checks), plan)
    assert code == 0, (code, limitations)
    assert any("benchmark_threads" in item for item in limitations), limitations
