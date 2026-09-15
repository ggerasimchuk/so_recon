"""E01.12.11 — the stage report may not say more than its artifacts do.

Four defects the review found, one test group each. Every one of them is a way the page
could read as more measured, more complete or more accepted than the run directories behind
it; each test below fails if the report goes back to overstating.

Nothing here runs a suite. The fixtures write the artifacts a real session would have left —
`run.json` and `e01_suite.json` — and the report is built from those.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from so_recon.paths import ProjectPaths
from so_recon.simulator.suite_record import (
    SUITE_REPORT_FILENAME,
    JobOutcome,
    SuiteReport,
)
from so_recon.validation.e01_report import (
    build_e01_report,
    render_e01_report,
    write_figure_provenance,
)
from so_recon.validation.physics import PhysicsCheck

P0_MANDATORY = (
    "closed_cell_pvt",
    "closed_box_pvt",
    "hydrostatic",
    "segregation",
    "bl",
    "operations",
    "controls_calendar",
    "restart_round_trip",
    "worker_isolation",
)
P1_MANDATORY = ("five_spot", "five_spot_refinement", "bl_refinement", "p1_worlds")


def _paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "manifests").mkdir(parents=True, exist_ok=True)
    return ProjectPaths.default(tmp_path)


def _check(name: str, status: str = "PASS") -> PhysicsCheck:
    ran = status != "NOT_RUN"
    return PhysicsCheck(
        name=name,
        status=status,  # type: ignore[arg-type]
        metrics={"balance_cumulative_relative": 1e-9} if ran else {},
        thresholds={"balance_cumulative_relative_max": 1e-3},
        input_hashes={},
        evidence_paths=(),
        reason=None if status == "PASS" else f"{name}: {status} in this fixture",
    )


def _job(
    job_id: str,
    *,
    status: str = "COMPLETE",
    accounting: str = "launcher",
    group: str = "analytic",
    **extra: Any,
) -> JobOutcome:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "group": group,
        "kind": "fixture",
        "profile": "P0_VERIFY",
        "accounting": accounting,
        "expected_outcome": "COMPLETE",
        "status": status,
        "wall_s": 1.5,
        "cpu_s": None,
        "peak_rss_bytes": None,
        "output_bytes": None,
        "native_chunk_calls": 3,
        "accepted_steps": 3,
        "cut_steps": 0,
        "nonlinear_iterations": 6,
        "retry_count": 0,
        "reason": None if status == "COMPLETE" else f"{job_id}: {status}",
        "measurement_method": "the whole diagnostic's wall, divided evenly across its forwards",
    }
    payload.update(extra)
    return JobOutcome.model_validate(payload)


def _write_run(
    paths: ProjectPaths,
    run_id: str,
    *,
    suite: str | None = None,
    checks: tuple[PhysicsCheck, ...] = (),
    jobs: tuple[JobOutcome, ...] = (),
    exit_code: int = 0,
    artifacts: dict[str, str] | None = None,
    git_commit: str | None = None,
    remaining_job_ids: tuple[str, ...] = (),
    stopped_reason: str | None = None,
) -> Path:
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "command": "verify-physics",
                "argv": ["so-recon", "verify-physics"],
                "status": "PASS" if exit_code == 0 else "FAIL",
                "created_at": "2026-09-15T00:00:00+00:00",
                "finished_at": "2026-09-15T00:01:00+00:00",
                "git_commit": git_commit or "d2b2247",
                "git_dirty": False,
                "spec_version": "4.0",
                "config_version": "E01.1",
                "resolved_config_hash": "0" * 64,
                "environment_lock_hash": "1" * 64,
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    if suite is not None:
        report = SuiteReport(
            suite=suite,
            profile="P0_VERIFY",
            exit_code=exit_code,
            run_id=run_id,
            command="verify-physics",
            started_at="2026-09-15T00:00:00+00:00",
            finished_at="2026-09-15T00:01:00+00:00",
            git_commit="d2b2247",
            git_dirty=False,
            environment_lock_hash="1" * 64,
            tolerances_path="configs/e01_tolerances.yml",
            tolerances_sha256="2" * 64,
            job_plan_path="configs/e01_jobs.json",
            job_plan_sha256="3" * 64,
            planned_job_ids=tuple(job.job_id for job in jobs),
            deferred_job_ids=(),
            declared_count_without_retry=len(jobs),
            jobs=jobs,
            checks=checks,
            mandatory_checks=tuple(sorted(c.name for c in checks)),
            exploratory_checks=(),
            remaining_job_ids=remaining_job_ids,
            stopped_reason=stopped_reason,
            limitations=(),
            benchmark={},
            artifacts=artifacts or {},
            launcher_forwards=len(jobs),
            ledger_forwards=0,
        )
        (run_dir / SUITE_REPORT_FILENAME).write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
    return run_dir


def _green_p0(paths: ProjectPaths, run_id: str = "20260915T000000Z-verify-physics-aaaa") -> Path:
    return _write_run(
        paths,
        run_id,
        suite="p0",
        checks=tuple(_check(name) for name in P0_MANDATORY),
        jobs=(_job("closed_cell"),),
    )


# --------------------------------------------------------------------------------------
# finding 3 — a RESOURCE_FAILURE nobody cited is still a RESOURCE_FAILURE
# --------------------------------------------------------------------------------------


def test_a_resource_failure_in_an_uncited_session_is_a_named_limitation(tmp_path: Path) -> None:
    """The operator chooses which runs to cite; the operator does not choose what happened.

    A session refused by the memory guard is invisible to a report built only from the
    suites the operator passed to `--runs`, which is exactly how a page comes to say "no
    resource failures" while the repository holds four of them.
    """
    paths = _paths(tmp_path)
    cited = _green_p0(paths)
    _write_run(
        paths,
        "20260915T010000Z-verify-physics-bbbb",
        suite="p1",
        checks=(_check("p1_worlds", "NOT_RUN"),),
        jobs=(
            _job("world42", status="RESOURCE_FAILURE", accounting="ledger", group="worlds"),
            _job("world43", status="RESOURCE_FAILURE", accounting="ledger", group="worlds"),
        ),
        exit_code=2,
    )
    report = build_e01_report((cited,), paths)
    text = "\n".join(report.limitations)
    assert "20260915T010000Z-verify-physics-bbbb" in text, report.limitations
    assert "world42" in text and "world43" in text, report.limitations
    assert "RESOURCE_FAILURE" in text
    assert "20260915T010000Z-verify-physics-bbbb" in render_e01_report(report)


def test_an_uncited_session_that_was_clean_adds_no_limitation(tmp_path: Path) -> None:
    """The sweep names refusals. It does not name every run that ever happened."""
    paths = _paths(tmp_path)
    cited = _green_p0(paths)
    _write_run(
        paths,
        "20260915T020000Z-verify-physics-cccc",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
    )
    report = build_e01_report((cited,), paths)
    assert not [item for item in report.limitations if "cccc" in item], report.limitations


def test_a_published_p1_manifest_that_accepts_nothing_contradicts_a_passing_p1_worlds(
    tmp_path: Path,
) -> None:
    """`reports/p1_suite_manifest.json` is single-path: it may describe another session.

    When it does, the page must say so beside the `p1_worlds` verdict rather than let a
    PASS stand next to a committed manifest whose rows are all RESOURCE_FAILURE.
    """
    paths = _paths(tmp_path)
    (paths.reports / "p1_suite_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "p1-suite-1",
                "fully_accepted": False,
                "n_parents": 5,
                "rows": [
                    {"parent_world_id": "p1-base-s0041", "status": None, "accepted": False},
                    {
                        "parent_world_id": "p1-base-s0042",
                        "status": "RESOURCE_FAILURE",
                        "accepted": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cited = _write_run(
        paths,
        "20260915T030000Z-verify-physics-dddd",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
    )
    report = build_e01_report((cited,), paths)
    text = "\n".join(report.limitations)
    assert "p1_suite_manifest.json" in text, report.limitations
    assert "fully_accepted" in text or "not accepted" in text, report.limitations


def test_a_committed_figure_drawn_for_another_run_is_named_as_such(tmp_path: Path) -> None:
    """The PNGs live at one committed path; a report that drew none must not claim them."""
    paths = _paths(tmp_path)
    figures = paths.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "e01_true_so.png").write_bytes(b"\x89PNG\r\n")
    cited = _green_p0(paths)
    write_figure_provenance(paths, run_id="an-earlier-report", cited=[paths.runs / "old"], drawn=[])
    report = build_e01_report((cited,), paths)
    text = "\n".join(report.limitations)
    assert "reports/figures/e01_true_so.png" in text, report.limitations
    assert "NOT drawn by this report" in text, report.limitations


def test_a_figure_this_report_drew_is_not_a_limitation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    figures = paths.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    drawn = figures / "e01_true_so.png"
    drawn.write_bytes(b"\x89PNG\r\n")
    cited = _green_p0(paths)
    write_figure_provenance(paths, run_id="this-report", cited=[cited], drawn=[drawn])
    report = build_e01_report((cited,), paths)
    assert not [item for item in report.limitations if "Figures:" in item], report.limitations


# --------------------------------------------------------------------------------------
# finding 5 — an unmeasured quantity is not a zero
# --------------------------------------------------------------------------------------


def test_an_unmeasured_job_quantity_renders_as_a_dash_not_as_zero(tmp_path: Path) -> None:
    """`peak RSS 0`, `out bytes 0` and `cpu s 0` read as measurements. They were not."""
    paths = _paths(tmp_path)
    cited = _green_p0(paths)
    report = build_e01_report((cited,), paths)
    text = render_e01_report(report)
    header = next(line for line in text.splitlines() if line.startswith("| job | group |"))
    columns = [c.strip() for c in header.strip("|").split("|")]
    row = next(line for line in text.splitlines() if line.startswith("| `closed_cell`"))
    cells = dict(zip(columns, (c.strip() for c in row.strip("|").split("|")), strict=True))
    for column in ("cpu s", "peak RSS", "out bytes"):
        assert cells[column] == "—", (column, row)
    # A count that really was measured stays a number, so the dash means what it says.
    assert cells["cut"] == "0" and cells["chunks"] == "3", row


def test_the_jobs_table_prints_how_a_launcher_row_was_measured(tmp_path: Path) -> None:
    """The per-job wall is the diagnostic's, divided evenly. The page has to say so."""
    paths = _paths(tmp_path)
    report = build_e01_report((_green_p0(paths),), paths)
    text = render_e01_report(report)
    assert "divided evenly" in text, "the measurement_method caveat is never printed"


# --------------------------------------------------------------------------------------
# finding 7 — the OW gate
# --------------------------------------------------------------------------------------


def test_the_ow_gate_is_not_pass_on_an_incomplete_matrix(tmp_path: Path) -> None:
    """A p0-only report has no five-spot and no worlds. That is not a passing OW gate."""
    paths = _paths(tmp_path)
    report = build_e01_report((_green_p0(paths),), paths)
    assert report.status == "FAIL", report.status_reason
    assert report.ow_gate != "PASS", (
        "the OW gate read PASS while the stage status read FAIL: it is computed from "
        "failed/unrun alone and ignores the missing and remaining branches"
    )


def test_the_ow_gate_passes_when_the_whole_matrix_does(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    p0 = _green_p0(paths)
    p1 = _write_run(
        paths,
        "20260915T040000Z-verify-physics-eeee",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
    )
    report = build_e01_report((p0, p1), paths)
    assert report.status == "PASS_WITH_LIMITATIONS", report.status_reason
    assert report.ow_gate == "PASS"


# --------------------------------------------------------------------------------------
# E01.13 — a black-oil capability that FAILED is a limitation, not a silent PASS
# --------------------------------------------------------------------------------------


def _green_ow(paths: ProjectPaths) -> tuple[Path, Path]:
    """A complete, passing oil-water matrix: p0 and p1, every mandatory check green."""
    p0 = _green_p0(paths)
    p1 = _write_run(
        paths,
        "20260915T040000Z-verify-physics-eeee",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
    )
    return p0, p1


#: What the black-oil group's own refusal really leaves behind. `_run_black_oil_restart`
#: raises `CommandError` when the two sides of the restart split disagree; `run_suite_jobs`
#: records that as a stop, the two restart jobs are never reached, and `evaluate_suite`
#: returns 2. A fixture that hard-coded an empty `remaining_job_ids` for an exit-2 session
#: would test `INCOMPLETE` in a shape no refusal ever produces.
BO_REFUSED_JOBS = ("bo_restart_prefix", "bo_restart_suffix_new_worker")
BO_REFUSED_REASON = (
    "group 'black_oil' failed: CommandError: the black-oil restart is split after report "
    "step 3 here and after step 4 in julia/verification/blackoil.jl"
)


def _bo_run(
    paths: ProjectPaths,
    *,
    exit_code: int,
    git_commit: str | None = None,
    refused: bool = False,
) -> Path:
    """A published black-oil session with the given exit code.

    `refused=True` gives it the shape the split-step guard really produces: declared jobs
    left unreached and a stop reason naming the refusal.
    """
    return _write_run(
        paths,
        "20260915T050000Z-verify-physics-ffff",
        suite="bo",
        checks=(_check("black_oil", "PASS" if exit_code == 0 else "FAIL"),),
        jobs=(_job("bo_closed", group="black_oil"),),
        exit_code=exit_code,
        git_commit=git_commit,
        remaining_job_ids=BO_REFUSED_JOBS if refused else (),
        stopped_reason=BO_REFUSED_REASON if refused else None,
    )


def test_a_black_oil_capability_that_failed_is_named_and_does_not_read_as_a_clean_pass(
    tmp_path: Path,
) -> None:
    """The three values of `bo_status`, against a complete and passing oil-water matrix.

    Plan 13.5 fixes both halves of this. A black-oil failure must NOT fail the stage — an
    oil-water deliverable is never removed or failed because a black-oil benchmark failed —
    and it must not be invisible either. Before Task 13 only the `NOT_RUN` branch could be
    reached, so `FAIL` fell through to `PASS` with a reason line reading "and black oil ran".
    """
    expected = {
        None: ("PASS_WITH_LIMITATIONS", "NOT_RUN"),
        0: ("PASS", None),
        1: ("PASS_WITH_LIMITATIONS", "FAIL"),
    }
    for exit_code, (status, named) in expected.items():
        paths = _paths(tmp_path / f"bo{exit_code}")
        runs = list(_green_ow(paths))
        if exit_code is not None:
            runs.append(_bo_run(paths, exit_code=exit_code))
        report = build_e01_report(tuple(runs), paths)
        assert report.status == status, (exit_code, report.status, report.status_reason)
        # The oil-water gate is untouched by the black-oil verdict either way (13.5).
        assert report.ow_gate == "PASS", (exit_code, report.ow_gate)
        if named is None:
            assert report.bo_status == "PASS", exit_code
            continue
        assert report.bo_status == named, exit_code
        assert named in report.status_reason, (
            f"the stage status reason for a {named} black-oil capability is "
            f"{report.status_reason!r}, which never mentions it"
        )


def test_a_failed_black_oil_capability_is_on_the_rendered_page(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    runs = (*_green_ow(paths), _bo_run(paths, exit_code=1))
    text = render_e01_report(build_e01_report(runs, paths))
    assert "BO status: FAIL" in text, text[:400]


def test_a_refused_black_oil_session_does_not_read_as_one_that_never_ran(
    tmp_path: Path,
) -> None:
    """`evaluate_suite` exit 2 is an INCOMPLETE session, not an absent one.

    The black-oil group refuses its own session with exactly that code when the two sides of
    the restart split disagree, so a capability that was attempted and REFUSED used to land
    on `NOT_RUN` — indistinguishable on the page from one nobody ever ran. It is a limitation
    either way and neither is a stage FAIL (13.5); what changed is that the page now says
    which of the two happened.
    """
    paths = _paths(tmp_path)
    # The shape a real refusal produces: exit 2, the two restart jobs never reached, and a
    # stop reason naming the CommandError. A session with an empty `remaining_job_ids` would
    # exercise INCOMPLETE in a shape the guard never produces.
    runs = (*_green_ow(paths), _bo_run(paths, exit_code=2, refused=True))
    report = build_e01_report(runs, paths)
    assert report.bo_status == "INCOMPLETE", report.bo_status
    assert report.status == "PASS_WITH_LIMITATIONS", report.status_reason
    assert report.ow_gate == "PASS"
    assert "INCOMPLETE" in report.status_reason, report.status_reason
    assert any("REFUSED or left incomplete" in item for item in report.limitations), (
        report.limitations
    )
    assert "BO status: INCOMPLETE" in render_e01_report(report)

    # 13.5, the half this fixture exists to prove: the capability's unreached jobs are named
    # everywhere a reader looks, and counted against NEITHER the stage status nor the OW gate.
    assert report.black_oil_remaining_job_ids == BO_REFUSED_JOBS
    assert set(BO_REFUSED_JOBS) <= set(report.remaining_job_ids)
    assert "never reached" not in report.status_reason, report.status_reason
    text = render_e01_report(report)
    for job_id in BO_REFUSED_JOBS:
        assert job_id in text, job_id
    assert any(all(job_id in item for job_id in BO_REFUSED_JOBS) for item in report.limitations), (
        report.limitations
    )


def test_an_oil_water_job_that_was_never_reached_still_fails_the_stage(
    tmp_path: Path,
) -> None:
    """The narrowing of round 3 is BLACK-OIL ONLY. The general rule has to keep biting.

    "Planned jobs were never reached" is what stops a half-run p0 or p1 from passing, and
    nothing about the black-oil capability may weaken it. Same complete matrix, same green
    checks, one unreached p1 job — and the stage must be FAIL with that job named.
    """
    paths = _paths(tmp_path)
    p0 = _green_p0(paths)
    p1 = _write_run(
        paths,
        "20260915T040000Z-verify-physics-eeee",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
        exit_code=2,
        remaining_job_ids=("world45",),
    )
    report = build_e01_report((p0, p1), paths)
    assert report.status == "FAIL", report.status_reason
    assert "planned jobs were never reached" in report.status_reason, report.status_reason
    assert "world45" in report.status_reason
    assert report.ow_gate == "FAIL"
    assert report.black_oil_remaining_job_ids == ()


def test_an_oil_water_job_unreached_beside_a_refused_capability_still_fails_the_stage(
    tmp_path: Path,
) -> None:
    """Both at once: the black-oil jobs are excused, the oil-water one is not."""
    paths = _paths(tmp_path)
    p0 = _green_p0(paths)
    p1 = _write_run(
        paths,
        "20260915T040000Z-verify-physics-eeee",
        suite="p1",
        checks=tuple(_check(name) for name in P1_MANDATORY),
        jobs=(_job("world41", accounting="ledger", group="worlds"),),
        exit_code=2,
        remaining_job_ids=("world45",),
    )
    bo = _bo_run(paths, exit_code=2, refused=True)
    report = build_e01_report((bo, p0, p1), paths)
    assert report.status == "FAIL", report.status_reason
    assert "world45" in report.status_reason
    for job_id in BO_REFUSED_JOBS:
        assert job_id not in report.status_reason, (job_id, report.status_reason)
    assert report.bo_status == "INCOMPLETE"


def test_the_black_oil_exit_code_mapping_is_covered_end_to_end(tmp_path: Path) -> None:
    """Every exit code a black-oil session can publish maps to a distinct, named status."""
    expected = {None: "NOT_RUN", 0: "PASS", 1: "FAIL", 2: "INCOMPLETE", 3: "INCOMPLETE"}
    seen: dict[str, int | None] = {}
    for exit_code, status in expected.items():
        paths = _paths(tmp_path / f"map{exit_code}")
        runs = list(_green_ow(paths))
        if exit_code is not None:
            runs.append(_bo_run(paths, exit_code=exit_code))
        report = build_e01_report(tuple(runs), paths)
        assert report.bo_status == status, (exit_code, report.bo_status)
        seen.setdefault(status, exit_code)
    # Distinct statuses, so none of the four cases can be read as another.
    assert set(seen) == {"NOT_RUN", "PASS", "FAIL", "INCOMPLETE"}


# --------------------------------------------------------------------------------------
# E01.13 fix round 2 — the page states which commit each cited run belongs to
# --------------------------------------------------------------------------------------


def test_a_report_whose_runs_span_commits_says_so_instead_of_naming_one(
    tmp_path: Path,
) -> None:
    """12.11 makes «проверенный commit/dirty» an element of this page.

    `git_commit` is the FIRST cited run's, and its sibling `git_dirty` aggregates over all of
    them with `any(...)` — which is what made a single commit read as a claim about the whole
    page. A report that cites a session republished after a later commit beside sessions from
    an earlier one has two, and before this the page printed one of them and showed nothing
    else anywhere.
    """
    paths = _paths(tmp_path)
    p0, p1 = _green_ow(paths)
    bo = _bo_run(paths, exit_code=0, git_commit="bbbbbbbbbbbbbbbb")
    report = build_e01_report((bo, p0, p1), paths)
    assert report.git_commits == ("bbbbbbbbbbbbbbbb", "d2b2247"), report.git_commits
    assert any("not all produced at one commit" in item for item in report.limitations), (
        report.limitations
    )
    text = render_e01_report(report)
    assert "mixed — the cited runs span" in text
    # And every run's own commit is on the page, in the Commands table.
    assert "| commit |" in text
    for commit in ("bbbbbbbbbbbb", "d2b2247"):
        assert commit in text, commit


def test_a_report_whose_runs_share_a_commit_states_that_one_commit_plainly(
    tmp_path: Path,
) -> None:
    """The common case must not grow a caveat it does not need."""
    paths = _paths(tmp_path)
    report = build_e01_report(_green_ow(paths), paths)
    assert report.git_commits == ("d2b2247",)
    assert not any("one commit" in item for item in report.limitations), report.limitations
    text = render_e01_report(report)
    assert "| git_commit | `d2b2247` |" in text
    assert "mixed" not in text


def test_the_rendered_page_prints_the_ow_gate(tmp_path: Path) -> None:
    """12.11 lists the OW gate as an element of the report. It was never rendered."""
    paths = _paths(tmp_path)
    text = render_e01_report(build_e01_report((_green_p0(paths),), paths))
    assert "OW gate" in text, "the OW gate is in StageReport and not on the page"


# --------------------------------------------------------------------------------------
# finding 8 — the seam between the command module and the report module
# --------------------------------------------------------------------------------------


def test_the_stage_report_does_not_import_the_command_module() -> None:
    """The schemas live in `simulator.suite_record`; only the commands live in `commands`."""
    source = Path(inspect.getsourcefile(build_e01_report) or "").read_text(encoding="utf-8")
    assert "so_recon.simulator.commands" not in source


def test_the_report_command_has_no_deferred_import() -> None:
    """A function-level import to dodge a circular import is the proof of a broken seam."""
    from so_recon.simulator.commands import run_e01_report

    body = inspect.getsource(run_e01_report)
    offending = [
        line.strip()
        for line in body.splitlines()
        if line.startswith("    ") and line.strip().startswith(("import ", "from "))
    ]
    assert not offending, offending


# --------------------------------------------------------------------------------------
# I1 — which session-scope caps are enforced, said on the page
# --------------------------------------------------------------------------------------


def test_the_page_names_which_session_scope_caps_are_enforced_and_which_are_not(
    tmp_path: Path,
) -> None:
    """A cap the code declares and does not enforce is not a cap the page may imply.

    COMPUTE §7's disk budget IS re-imposed at session scope (`Session.admit_output`). The
    session forward count, the session wall between group boundaries and SPEC §3.3's two
    attempts per model hash are NOT: the first has no session-scope caller at all, the
    second is checked only between groups, and the third is per-job-ledger because E01 opens
    one ledger per run of a model. They are latent rather than live — the declared matrices
    are fixed at 23/17/4 jobs against caps of 64 and 2000 — and the page says so rather than
    leaving a reader to assume all four bite.
    """
    paths = _paths(tmp_path)
    report = build_e01_report((_green_p0(paths),), paths)
    named = [item for item in report.limitations if "ession-scope budget caps" in item]
    assert named, report.limitations
    text = named[0].lower()
    assert "disk budget is re-imposed" in text
    assert "forward count" in text
    assert "wall" in text
    assert "two attempts per model hash is enforced per job ledger" in text
    assert named[0] in render_e01_report(report)


# --------------------------------------------------------------------------------------
# I2 — the summary column of a FAILING row is about the metric that failed
# --------------------------------------------------------------------------------------

#: The real `five_spot_refinement` verdict of run 20260915T021124Z-verify-physics-b9ec409b,
#: copied from its published `e01_suite.json`. Four of its five thresholds pair with no
#: metric at all under a `startswith` match — including the one that failed, whose metric is
#: `monthly_volume_near_zero_absolute_m3_sc` against
#: `refinement_monthly_volume_absolute_max_m3_sc`.
FAILING_REFINEMENT = PhysicsCheck(
    name="five_spot_refinement",
    status="FAIL",
    metrics={
        "coarse_cells": 256.0,
        "fine_cells": 2304.0,
        "inventory_relative": 9.231783393011368e-11,
        "monthly_volume_absolute_crossover_m3_sc": 4.9999999999999996e-05,
        "monthly_volume_absolute_m3_sc": 3.4947780576666254e-06,
        "monthly_volume_near_zero_absolute_m3_sc": 3.0304194259052176e-06,
        "monthly_volume_relative": 1.3821991187514615e-05,
        "monthly_volume_self_relative": 1.038466270256858,
        "n_months": 36.0,
        "n_zones": 64.0,
        "so_pv_mae": 0.015665273005009587,
        "so_zone_max_abs": 0.051251142022209706,
        "so_zone_range_coarse": 0.46102623103589047,
        "support_pore_volume_relative": 7.275957614183426e-16,
    },
    thresholds={
        "refinement_inventory_relative_max": 0.01,
        "refinement_monthly_volume_absolute_max_m3_sc": 1e-06,
        "refinement_monthly_volume_relative_max": 0.02,
        "refinement_so_pv_mae_max": 0.02,
        "support_pore_volume_relative_max": 1e-09,
    },
    input_hashes={},
    evidence_paths=(),
    reason=(
        "five_spot_refinement: monthly_volume_near_zero_absolute_m3_sc=3.03042e-06 exceeds "
        "refinement_monthly_volume_absolute_max_m3_sc=1e-06"
    ),
)


def _matrix_row(page: str, name: str) -> str:
    return next(line for line in page.splitlines() if line.startswith(f"| `{name}` |"))


def test_a_failing_row_does_not_advertise_a_metric_that_passed(tmp_path: Path) -> None:
    """The column paired metrics to thresholds by prefix and reported the largest ratio.

    On this row four of the five thresholds pair with nothing — including the one that
    failed, because its metric is named `monthly_volume_near_zero_absolute_m3_sc` and its
    threshold `refinement_monthly_volume_absolute_max_m3_sc`. The single incidental match
    was `support_pore_volume_relative`, which clears its gate by seven orders of magnitude,
    so a FAILING row advertised a passing measurement.
    """
    paths = _paths(tmp_path)
    run = _write_run(
        paths,
        "20260915T030000Z-verify-physics-dddd",
        suite="p1",
        checks=(FAILING_REFINEMENT,),
        jobs=(_job("five_spot_coarse"),),
        exit_code=1,
    )
    row = _matrix_row(render_e01_report(build_e01_report((run,), paths)), "five_spot_refinement")

    assert "| FAIL |" in row
    assert "monthly_volume_near_zero_absolute_m3_sc" in row, row
    assert "support_pore_volume_relative`=7.28e-16" not in row, row


def test_a_failing_row_never_advertises_a_measurement_inside_its_gate(tmp_path: Path) -> None:
    """A FAIL whose reason names nothing parseable prints an em dash, never a passing ratio."""
    paths = _paths(tmp_path)
    unparseable = FAILING_REFINEMENT.model_copy(
        update={"reason": "five_spot_refinement: the fine grid published no monthly table"}
    )
    run = _write_run(
        paths,
        "20260915T031000Z-verify-physics-eeee",
        suite="p1",
        checks=(unparseable,),
        jobs=(_job("five_spot_coarse"),),
        exit_code=1,
    )
    row = _matrix_row(render_e01_report(build_e01_report((run,), paths)), "five_spot_refinement")
    assert "| FAIL | yes | — |" in row, row


def test_a_passing_row_still_reports_the_measurement_closest_to_its_gate(tmp_path: Path) -> None:
    """The reading aid is unchanged where it was never misleading."""
    paths = _paths(tmp_path)
    run = _green_p0(paths)
    row = _matrix_row(render_e01_report(build_e01_report((run,), paths)), "hydrostatic")
    assert "balance_cumulative_relative" in row, row
