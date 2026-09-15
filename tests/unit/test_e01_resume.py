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
    load_job_plan,
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
    evidence = paths.root / "artifacts" / "case-p0-closed_box.json"
    evidence.write_text('{"diagnostic": "complete"}')
    report = report.model_copy(
        update={
            "resume_input_hash": suites.resume_input_hash(paths, _plan()),
            "resume_evidence_hashes": {paths.relative(evidence): sha256_file(evidence)},
        }
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


def test_a_complete_ledger_without_result_evidence_does_not_reuse(
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
    assert reused is None


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


# --------------------------------------------------------------------------------------
# C1 — a resumed group may not be "complete" on a verdict the parent never published
# --------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]


def _two_check_plan() -> SuitePlan:
    """One group whose two jobs are scored by two different checks."""
    return SuitePlan(
        profile="P0_VERIFY",
        wall_budget_s=600,
        declared_count_without_retry=2,
        groups=("analytic",),
        jobs=(
            _planned("closed_cell", "analytic", scored_as="analytic_check"),
            _planned("closed_box", "analytic", scored_as="second_check"),
        ),
    )


def test_a_group_is_not_complete_when_a_check_it_is_scored_by_was_never_published(
    tmp_path: Path,
) -> None:
    """The dead guard: `scored` was built with `if name in verdicts`, so the check the
    parent never published was dropped instead of blocking the group's completion.

    The parent below published `analytic_check` PASS and nothing for `second_check`. Its
    `analytic` group is therefore NOT finished, and a resume that skips it would carry a
    group forward on a verdict that does not exist.
    """
    paths = _paths(tmp_path)
    ledger_path = _publish_parent(paths)
    state = load_resume_state(ledger_path, suite="p0", plan=_two_check_plan(), paths=paths)
    assert state is not None
    assert state.completed_groups == frozenset()


def _publish_black_oil_parent(paths: ProjectPaths, plan: SuitePlan) -> Path:
    """A black-oil session that recorded every job row and only the `black_oil` verdict."""
    run_dir = paths.runs / "20260915T024238Z-verify-physics-parent"
    (run_dir / "ledgers").mkdir(parents=True, exist_ok=True)
    report = SuiteReport(
        suite="bo",
        profile="P0_VERIFY",
        exit_code=0,
        run_id=run_dir.name,
        command="verify-physics",
        started_at="2026-09-15T02:42:38+00:00",
        finished_at="2026-09-15T02:48:00+00:00",
        git_commit="4455aea",
        git_dirty=False,
        environment_lock_hash="1" * 64,
        tolerances_path="configs/e01_tolerances.yml",
        tolerances_sha256="2" * 64,
        job_plan_path="configs/e01_jobs.json",
        job_plan_sha256="3" * 64,
        planned_job_ids=tuple(job.job_id for job in plan.jobs),
        deferred_job_ids=(),
        declared_count_without_retry=plan.declared_count_without_retry,
        jobs=tuple(_outcome(job.job_id, job.group, job.expected_outcome) for job in plan.jobs),
        checks=(_check("black_oil"),),
        mandatory_checks=("black_oil",),
        exploratory_checks=(),
        remaining_job_ids=(),
        stopped_reason=None,
        limitations=(),
        benchmark={},
        artifacts={},
        launcher_forwards=2,
        ledger_forwards=2,
    )
    (run_dir / SUITE_REPORT_FILENAME).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    ledger_path = run_dir / "ledgers" / "bo_restart_prefix.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "ledger-1",
                "session_id": f"{run_dir.name}-bo_restart_prefix",
                "profile": resource_profile("P0_VERIFY").model_dump(mode="json"),
                "started_at": "2026-09-15T02:42:38+00:00",
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


def test_a_resumed_black_oil_group_is_not_carried_without_its_restart_verdict(
    tmp_path: Path,
) -> None:
    """The reviewer's route 1, against the REAL `bo` plan and the REAL resume loader.

    `so-recon verify-physics --suite bo --resume-ledger <parent>/ledgers/bo_restart_prefix.json`
    read a parent whose only verdict was `black_oil`, marked the one black-oil group complete
    and carried exactly that verdict forward. The restart round trip left the session, the
    exit code and the published page at once.
    """
    paths = _paths(tmp_path)
    repo_paths = ProjectPaths.default(REPO)
    plan = load_job_plan(repo_paths.resolve("configs/e01_jobs.json"), repo_paths).suites["bo"]
    ledger_path = _publish_black_oil_parent(paths, plan)

    state = load_resume_state(ledger_path, suite="bo", plan=plan, paths=paths)
    assert state is not None
    assert state.completed_groups == frozenset(), (
        "a group whose restart verdict the parent never published is not a finished group"
    )

    run = _suite_run(paths, tmp_path, plan=plan, resume=state)
    assert not run.skips("black_oil")


def test_a_group_that_dies_mid_scoring_still_leaves_the_verdict_it_owes(
    tmp_path: Path,
) -> None:
    """The second half of C1's route 2: the page has to say WHICH round trip is missing.

    `scored_as` alone makes the lost verdict an exit-2 `missing`, which is the gate. It does
    not put a row on the published page: `_blackoil_section` renders the checks a session
    appended, and a check nobody appended renders as nothing at all. The guard leaves a
    NOT_RUN carrying the exception, so an INCOMPLETE black-oil session names what it lost.
    """
    paths = _paths(tmp_path)
    run = _suite_run(paths, tmp_path)
    thresholds = {"blackoil_restart_pressure_relative_max": 1e-6}
    gates = (("blackoil_restart_pressure_relative", "blackoil_restart_pressure_relative_max"),)

    with (
        pytest.raises(RuntimeError, match="memory drift"),
        suites.verdict_owed(run, "black_oil_restart", thresholds, gates),
    ):
        raise RuntimeError("the worker stopped on memory drift")

    assert [c.name for c in run.checks] == ["black_oil_restart"]
    assert run.checks[0].status == "NOT_RUN"
    assert "memory drift" in str(run.checks[0].reason)


def test_the_guard_never_doubles_a_verdict_the_group_already_recorded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    run = _suite_run(paths, tmp_path)
    with suites.verdict_owed(run, "black_oil_restart", {}, ()):
        run.checks.append(_check("black_oil_restart"))
    assert [(c.name, c.status) for c in run.checks] == [("black_oil_restart", "PASS")]


# --------------------------------------------------------------------------------------
# I1 — the session disk cap counts what the session really wrote
# --------------------------------------------------------------------------------------


def test_the_session_charges_what_a_published_directory_really_holds(tmp_path: Path) -> None:
    """`record_output` takes a number somebody else measured; this measures it."""
    paths = _paths(tmp_path)
    session = _session(paths, tmp_path)
    result_dir = tmp_path / "results" / "p0-closed_cell"
    (result_dir / "nested").mkdir(parents=True)
    (result_dir / "states.h5").write_bytes(b"x" * 4096)
    (result_dir / "nested" / "monthly.parquet").write_bytes(b"y" * 512)

    session.record_directory(result_dir)
    assert session.output_bytes == 4096 + 512
    session.record_directory(tmp_path / "never-written")
    assert session.output_bytes == 4096 + 512


def test_a_launcher_published_fixture_is_charged_to_the_session_disk_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every `artifacts/results/p0-*` and `p1-*` directory used to cost the session nothing.

    `Session.admit_output` re-imposes COMPUTE §7's 5 GiB session cap, but only over what
    `record_output` was told about, and the launcher route told it nothing: the analytic,
    operational and refinement groups publish their fixtures through `_publish` and never
    charged a byte. The BO fixture loop did call `record_output` — with
    `result.cost.output_bytes`, which `publish_fixture` writes as a constant 0.
    """
    paths = _paths(tmp_path)
    run = _suite_run(paths, tmp_path)
    result_dir = paths.artifacts / "results" / "p0-closed_cell"
    result_dir.mkdir(parents=True)
    (result_dir / "states.h5").write_bytes(b"z" * 2048)

    def fake_publish(*_args: Any, **_kwargs: Any) -> tuple[None, Path]:
        return None, result_dir

    monkeypatch.setattr(suites, "publish_fixture", fake_publish)
    published = suites._publish(run, {"fixtures": {"closed_cell": {}}}, "closed_cell", "p0-x")

    assert published == result_dir
    assert run.session.output_bytes == 2048, (
        "a fixture the session published is bytes the session wrote"
    )


@pytest.mark.parametrize(
    "mutation", ["source", "plan", "tolerance", "environment", "missing", "corrupt", "cache_bypass"]
)
def test_resume_refuses_stale_or_unverifiable_group(tmp_path: Path, mutation: str) -> None:
    paths = _paths(tmp_path)
    ledger = _publish_parent(paths)
    plan = _plan()
    if mutation == "source":
        source = paths.root / "julia" / "adapter" / "model.jl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("changed native implementation")
    elif mutation == "plan":
        plan = plan.model_copy(update={"wall_budget_s": 599})
    elif mutation == "cache_bypass":
        plan = plan.model_copy(
            update={"jobs": tuple(j.model_copy(update={"cache_bypass": True}) for j in plan.jobs)}
        )
    elif mutation == "tolerance":
        (paths.root / "configs" / "e01_tolerances.yml").parent.mkdir(parents=True, exist_ok=True)
        (paths.root / "configs" / "e01_tolerances.yml").write_text("changed")
    elif mutation == "environment":
        (paths.root / "uv.lock").write_text("changed")
    else:
        evidence = paths.root / "artifacts" / "case-p0-closed_box.json"
        if mutation == "corrupt":
            evidence.write_text("corrupted")
        elif evidence.exists():
            evidence.unlink()
    state = load_resume_state(ledger, suite="p0", plan=plan, paths=paths)
    assert state is None or not state.completed_groups


@pytest.mark.parametrize("damage", [None, "missing", "corrupt"])
def test_resume_validates_result_outputs_not_only_record_hash(
    tmp_path: Path, damage: str | None
) -> None:
    from tests.unit.test_forward_results import _extraction, _publish

    paths, record = _publish(tmp_path, _extraction())
    ledger = _publish_parent(paths)
    report_path = ledger.parent.parent / SUITE_REPORT_FILENAME
    report = json.loads(report_path.read_text())
    report["jobs"][0]["accounting"] = "ledger"
    report["jobs"][0]["result_record_path"] = paths.relative(record)
    report["resume_evidence_hashes"][paths.relative(record)] = sha256_file(record)
    report_path.write_text(json.dumps(report))
    result = suites.load_forward_result(record, paths)
    if damage == "missing":
        paths.resolve(result.monthly_path).unlink()
    elif damage == "corrupt":
        paths.resolve(result.monthly_path).write_bytes(b"corrupted parquet")
    state = load_resume_state(ledger, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert ("analytic" in state.completed_groups) == (damage is None)
    if damage is None:
        case = suites.load_case(paths.artifacts / "case.json", paths)
        payload = json.loads(ledger.read_text())
        payload["entries"] = [
            {
                "job_id": "verified-parent",
                "model_hash": case.model_hash,
                "case_sha256": case_manifest_sha256(case),
                "attempt": 1,
                "state": "COMPLETE",
                "reserved_s": 1.0,
                "reserved_bytes": 1024,
                "cost": None,
            }
        ]
        ledger.write_text(json.dumps(payload))
        state = load_resume_state(ledger, suite="p0", plan=_plan(), paths=paths)
        run = _suite_run(paths, tmp_path, resume=state)
        assert run.reusable_outcome(run.planned("closed_cell"), case) is not None


def test_cache_bypass_group_is_repeated_even_with_matching_identity(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ledger = _publish_parent(paths)
    plan = _plan()
    plan = plan.model_copy(
        update={"jobs": tuple(job.model_copy(update={"cache_bypass": True}) for job in plan.jobs)}
    )
    report_path = ledger.parent.parent / SUITE_REPORT_FILENAME
    report = json.loads(report_path.read_text())
    report["resume_input_hash"] = suites.resume_input_hash(paths, plan)
    report_path.write_text(json.dumps(report))
    state = load_resume_state(ledger, suite="p0", plan=plan, paths=paths)
    assert state is not None
    assert not state.completed_groups


def test_legacy_report_without_identity_is_not_reused(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ledger = _publish_parent(paths)
    report_path = ledger.parent.parent / SUITE_REPORT_FILENAME
    report = json.loads(report_path.read_text())
    report.pop("resume_input_hash")
    report_path.write_text(json.dumps(report))
    state = load_resume_state(ledger, suite="p0", plan=_plan(), paths=paths)
    assert state is not None
    assert not state.completed_groups
    assert not state.outcomes
