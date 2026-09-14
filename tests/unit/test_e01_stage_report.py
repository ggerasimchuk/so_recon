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
                "git_commit": "d2b2247",
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
            remaining_job_ids=(),
            stopped_reason=None,
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
