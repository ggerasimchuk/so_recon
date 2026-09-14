"""E01.12.6 — what a resumed session must not do twice, and what one session must not spend.

Two review findings live here.

**A resume that reruns everything is not a resume.** 12.6 asks for `read ledger/hash lock/
config -> check completed outputs -> missing jobs in the previous order`, and for a
production repeat NOT to enter the solver when a valid immutable matching result already
exists. `--resume-ledger` used to validate the file and change nothing else.

**One ledger per job silently drops the session disk budget.** `BudgetLedger.reserve`
enforces the 5 GiB session cap through `disk_stop(session_output_bytes=...)`; with a fresh
ledger per job, every one of them sees a single entry and the cap never bites. The session
has to hold that number itself.

Nothing here launches Julia.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from so_recon.config.resources import resource_profile
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator import suites
from so_recon.simulator.budget import DISK_FAILURE_RESERVE_BYTES, BudgetStop, disk_stop
from so_recon.simulator.case_io import case_manifest_sha256, write_case
from so_recon.simulator.suite_record import (
    SUITE_REPORT_FILENAME,
    JobOutcome,
    PlannedJob,
    SuitePlan,
    SuiteReport,
)
from so_recon.simulator.suites import Session, SuiteRun, load_resume_state, run_suite_jobs
from so_recon.validation.physics import PhysicsCheck
from tests.forward_case import build_case, write_case_arrays

MIB = 1024 * 1024
GIB = 1024 * MIB


# --------------------------------------------------------------------------------------
# fixtures: a plan, a published parent session, and a SuiteRun that launches nothing
# --------------------------------------------------------------------------------------


def _planned(job_id: str, group: str, **extra: Any) -> PlannedJob:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "group": group,
        "kind": "fixture",
        "profile": "P0_VERIFY",
        "expected_outcome": "COMPLETE",
        "output_request": "states",
        "julia_threads": 4,
        "cache_bypass": False,
        "accounting": "launcher",
        "native_source": None,
        "scored_as": f"{group}_check",
        "declared_by_plan": True,
    }
    payload.update(extra)
    return PlannedJob.model_validate(payload)


def _plan() -> SuitePlan:
    return SuitePlan(
        profile="P0_VERIFY",
        wall_budget_s=600,
        declared_count_without_retry=3,
        groups=("analytic", "operations", "restart"),
        jobs=(
            _planned("closed_cell", "analytic"),
            _planned("closed_box", "analytic"),
            _planned("mixing", "operations"),
            _planned("restart_continuous", "restart", accounting="ledger"),
        ),
    )


def _outcome(job_id: str, group: str, status: str = "COMPLETE", **extra: Any) -> JobOutcome:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "group": group,
        "kind": "fixture",
        "profile": "P0_VERIFY",
        "accounting": "launcher",
        "expected_outcome": "COMPLETE",
        "status": status,
        "wall_s": 2.0,
        "cpu_s": None,
        "peak_rss_bytes": None,
        "output_bytes": None,
        "native_chunk_calls": 1,
        "accepted_steps": 1,
        "cut_steps": 0,
        "nonlinear_iterations": 1,
        "retry_count": 0,
    }
    payload.update(extra)
    return JobOutcome.model_validate(payload)


def _check(name: str, status: str = "PASS") -> PhysicsCheck:
    return PhysicsCheck(
        name=name,
        status=status,  # type: ignore[arg-type]
        metrics={},
        thresholds={},
        input_hashes={},
        evidence_paths=(),
        reason=None if status == "PASS" else f"{name}: {status}",
    )


def _publish_parent(
    paths: ProjectPaths,
    *,
    analytic_status: str = "COMPLETE",
    analytic_check: str = "PASS",
) -> Path:
    """A previous session's run directory, with the ledger path a command would print."""
    run_dir = paths.runs / "20260915T000000Z-verify-physics-parent"
    (run_dir / "ledgers").mkdir(parents=True, exist_ok=True)
    report = SuiteReport(
        suite="p0",
        profile="P0_VERIFY",
        exit_code=2,
        run_id=run_dir.name,
        command="verify-physics",
        started_at="2026-09-15T00:00:00+00:00",
        finished_at="2026-09-15T00:03:00+00:00",
        git_commit="d2b2247",
        git_dirty=False,
        environment_lock_hash="1" * 64,
        tolerances_path="configs/e01_tolerances.yml",
        tolerances_sha256="2" * 64,
        job_plan_path="configs/e01_jobs.json",
        job_plan_sha256="3" * 64,
        planned_job_ids=("closed_cell", "closed_box", "mixing", "restart_continuous"),
        deferred_job_ids=(),
        declared_count_without_retry=3,
        jobs=(
            _outcome("closed_cell", "analytic", analytic_status),
            _outcome("closed_box", "analytic", analytic_status),
        ),
        checks=(_check("analytic_check", analytic_check),),
        mandatory_checks=("analytic_check",),
        exploratory_checks=(),
        remaining_job_ids=("mixing", "restart_continuous"),
        stopped_reason="the session's 600 s wall budget was spent",
        limitations=(),
        benchmark={},
        artifacts={"case.closed_box": "artifacts/case-p0-closed_box.json"},
        launcher_forwards=2,
        ledger_forwards=0,
    )
    (run_dir / SUITE_REPORT_FILENAME).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    ledger_path = run_dir / "ledgers" / "closed_cell.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "ledger-1",
                "session_id": f"{run_dir.name}-closed_cell",
                "profile": resource_profile("P0_VERIFY").model_dump(mode="json"),
                "started_at": "2026-09-15T00:00:00+00:00",
                "entries": [],
                "inherited_entries": [],
                "parent_session_id": None,
                "parent_ledger_sha256": None,
                "stop_reason": None,
            }
        ),
        encoding="utf-8",
    )
    return ledger_path


def _session(paths: ProjectPaths, tmp_path: Path, **extra: Any) -> Session:
    return Session(
        julia=tmp_path / "julia-binary",
        project=paths.julia,
        session_dir=tmp_path,
        profile=extra.pop("profile", resource_profile("P0_VERIFY")),
        paths=paths,
        ledger_dir=tmp_path / "ledgers",
        session_id="session-under-test",
        **extra,
    )


def _suite_run(paths: ProjectPaths, tmp_path: Path, **extra: Any) -> SuiteRun:
    return SuiteRun(
        cfg=None,  # type: ignore[arg-type]
        paths=paths,
        ctx=None,  # type: ignore[arg-type]
        log=logging.getLogger("test"),
        suite="p0",
        plan=extra.pop("plan", _plan()),
        profile=resource_profile("P0_VERIFY"),
        tolerances={},
        julia=tmp_path / "julia-binary",
        session=_session(paths, tmp_path),
        deadline_s=float("inf"),
        **extra,
    )


def _paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return ProjectPaths.default(tmp_path)


# --------------------------------------------------------------------------------------
# finding 1 — the resume reads what the previous session finished
# --------------------------------------------------------------------------------------


def test_the_resume_state_reads_the_parent_session_behind_its_ledger(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths)
    state = load_resume_state(ledger_path, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert state.completed_groups == frozenset({"analytic"})
    assert {job.job_id for job in state.jobs_of("analytic")} == {"closed_cell", "closed_box"}
    assert state.artifacts["case.closed_box"] == "artifacts/case-p0-closed_box.json"


def test_a_group_whose_check_failed_is_not_complete(tmp_path: Path) -> None:
    """12.6: a new session budget is not permission to skip a failed case."""
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths, analytic_check="FAIL")
    state = load_resume_state(ledger_path, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert state.completed_groups == frozenset()


def test_a_group_whose_job_missed_its_expected_outcome_is_not_complete(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths, analytic_status="RESOURCE_FAILURE")
    state = load_resume_state(ledger_path, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert state.completed_groups == frozenset()


def test_a_resumed_session_runs_only_the_missing_groups_in_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths)
    state = load_resume_state(ledger_path, suite="p0", plan=_plan(), paths=paths)
    entered: list[str] = []

    def recorder(name: str) -> Any:
        def runner(run: SuiteRun) -> None:
            entered.append(name)

        return runner

    monkeypatch.setattr(
        suites, "GROUPS", {name: recorder(name) for name in ("analytic", "operations", "restart")}
    )
    run = _suite_run(paths, tmp_path, resume=state)
    outcome = run_suite_jobs(run)

    assert entered == ["operations", "restart"], "the completed group was re-entered"
    carried = {job.job_id for job in outcome.jobs}
    assert carried == {"closed_cell", "closed_box"}, carried
    assert [c.name for c in outcome.checks] == ["analytic_check"]
    assert outcome.artifacts["case.closed_box"] == "artifacts/case-p0-closed_box.json"


def test_replay_re_enters_every_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit replay flag is the only thing that makes a repeat run the solver again."""
    paths = _paths(tmp_path)
    state = load_resume_state(_publish_parent(paths), suite="p0", plan=_plan(), paths=paths)
    entered: list[str] = []
    names = ("analytic", "operations", "restart")
    monkeypatch.setattr(
        suites, "GROUPS", {name: (lambda run, n=name: entered.append(n)) for name in names}
    )
    run_suite_jobs(_suite_run(paths, tmp_path, resume=state, replay=True))
    assert entered == ["analytic", "operations", "restart"]


def test_a_matching_immutable_result_is_reused_without_entering_the_solver(
    tmp_path: Path,
) -> None:
    """`BudgetLedger.is_already_complete` gets the production caller it never had."""
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths)
    refs = write_case_arrays(paths)
    case = build_case(refs)
    digest = case_manifest_sha256(case)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["entries"] = [
        {
            "job_id": "parent-attempt-1",
            "model_hash": case.model_hash,
            "case_sha256": digest,
            "attempt": 1,
            "state": "COMPLETE",
            "reserved_s": 1.0,
            "reserved_bytes": 1024,
            "cost": None,
        }
    ]
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_resume_state(ledger_path, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert state.already_complete(model_hash=case.model_hash, case_sha256=digest)
    assert not state.already_complete(model_hash="f" * 64, case_sha256=digest)

    run = _suite_run(paths, tmp_path, resume=state)
    reused = run.reusable_outcome(run.planned("closed_cell"), case)
    assert reused is not None
    assert reused.status == "COMPLETE"
    assert reused.reused_from_run_id == "20260915T000000Z-verify-physics-parent"


def test_a_replayed_session_reuses_nothing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    state = load_resume_state(_publish_parent(paths), suite="p0", plan=_plan(), paths=paths)
    refs = write_case_arrays(paths)
    case = build_case(refs)
    run = _suite_run(paths, tmp_path, resume=state, replay=True)
    assert run.reusable_outcome(run.planned("closed_cell"), case) is None


def test_the_case_digest_matches_the_manifest_a_session_really_publishes(
    tmp_path: Path,
) -> None:
    """The reuse key is the digest of the published case manifest, byte for byte.

    `BudgetLedger` records `sha256_file(case_path)`; the reuse check has to produce the same
    number without publishing the case a second time, so this pins the two together.
    """
    paths = _paths(tmp_path)
    refs = write_case_arrays(paths)
    case = build_case(refs)
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    ref = write_case(case, paths, ctx)
    assert case_manifest_sha256(case) == sha256_file(paths.resolve(ref.path))


# --------------------------------------------------------------------------------------
# finding 2 — the session disk budget
# --------------------------------------------------------------------------------------


def test_a_fresh_per_job_ledger_never_sees_the_session_total() -> None:
    """Why the compensation is owed: each per-job account starts at zero bytes written."""
    profile = resource_profile("P0_VERIFY").model_copy(update={"disk_budget_bytes": 512 * MIB})
    assert (
        disk_stop(
            profile,
            session_output_bytes=0,
            predicted_bytes=100 * MIB,
            free_bytes=100 * GIB,
        )
        is None
    )


def test_the_session_disk_budget_bites(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    profile = resource_profile("P0_VERIFY").model_copy(update={"disk_budget_bytes": 512 * MIB})
    session = _session(paths, tmp_path, profile=profile)

    session.admit_output(predicted_bytes=100 * MIB)
    for _ in range(4):
        session.record_output(100 * MIB)
    assert session.output_bytes == 400 * MIB

    with pytest.raises(BudgetStop) as excinfo:
        session.admit_output(predicted_bytes=100 * MIB)
    assert excinfo.value.status == "INCOMPLETE_BUDGET"
    assert str(DISK_FAILURE_RESERVE_BYTES) in str(excinfo.value)


def test_the_session_counts_bytes_across_every_job(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    session = _session(paths, tmp_path)
    assert session.output_bytes == 0
    session.record_output(7)
    session.record_output(None)
    session.record_output(5)
    assert session.output_bytes == 12
