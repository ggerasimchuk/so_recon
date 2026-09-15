"""E01.12.1 — what the new commands do when they are asked for something impossible.

Every test here is about the MAPPING: a command line, a refusal, an exit code and a run
record. Nothing in this file launches Julia, runs physics or scores a tolerance — the real
matrix is integration evidence and lives in `tests/integration`, and a unit suite that
quietly started a 36-month trajectory would be a unit suite nobody could run.

The two rules the whole file exists to pin:

* a command NEVER exits 0 merely because it wrote something. A suite that wrote a partial
  report, a forward that wrote a checkpoint and a report command that rendered a page all
  still have to say what happened, and `0` is reserved for "what was asked for happened";
* every failure downstream of a usable repository root leaves a FAIL run record, because
  `execute_run` is the only thing that opens a run and it is the only thing that closes one
  (invariant I6).

The fake runner below is used ONLY where the subject is that mapping. It is never used to
manufacture a physics verdict.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from so_recon.cli import main
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import CASE_MANIFEST_FILENAME, write_case
from so_recon.simulator.suite_record import (
    MANDATORY_CHECKS,
    SUITE_REPORT_FILENAME,
    JobOutcome,
    SuiteOutcome,
    forward_exit_code,
    suite_exit_code,
)
from so_recon.simulator.suites import SuiteRun
from so_recon.synthetic.p1 import P1Design, render_p1
from so_recon.synthetic.world_io import build_p1_case
from so_recon.validation.physics import PhysicsCheck

REPO = Path(__file__).resolve().parents[2]

E01_YAML = """
spec_version: "4.0"
config_version: "TEST.E01"
project_name: SO-RECON
paths:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  artifacts: artifacts
  reports: reports
  configs: configs
  julia: julia
sources:
  files: []
julia:
  project: julia
  smoke_script: julia/smoke/smoke_case.jl
  timeout_s: 1800
resources:
  profile: P0_VERIFY
  soft_bytes: 12884901888
  hard_bytes: 17179869184
  reserve_bytes: 6442450944
  disk_budget_bytes: 5368709120
  wall_budget_s: 600
  job_timeout_s: 300
  startup_timeout_s: 300
  max_new_forward: 64
  julia_workers: 1
  julia_threads: 4
  blas_threads: 1
  poll_interval_s: 0.25
  max_swap_growth_bytes: 536870912
"""

#: The same block with one number moved. `require_resource_profile` refuses it: a block that
#: NAMES an approved preset has to carry that preset's limits.
UNAPPROVED_YAML = E01_YAML.replace("wall_budget_s: 600", "wall_budget_s: 6000")

#: A 3.0 configuration, which has no `resources` field at all. Legal for `manifest`; refused
#: by anything that runs physics.
NO_RESOURCES_YAML = """
spec_version: "3.0"
config_version: "TEST.E00"
paths: {raw: data/raw}
sources: {files: []}
julia: {project: julia, smoke_script: julia/smoke/smoke_case.jl, timeout_s: 1800}
"""

JULIA_MANIFEST = """
julia_version = "1.12.7"
manifest_format = "2.0"

[[deps.JSON]]
version = "1.1.2"

[[deps.Jutul]]
version = "0.4.31"

[[deps.JutulDarcy]]
version = "0.3.11"
"""


@pytest.fixture
def e01_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root carrying a real 4.0 E01 configuration and the frozen tolerances."""
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "e01.yml").write_text(E01_YAML, encoding="utf-8")
    (tmp_path / "configs" / "project.yml").write_text(NO_RESOURCES_YAML, encoding="utf-8")
    (tmp_path / "configs" / "unapproved.yml").write_text(UNAPPROVED_YAML, encoding="utf-8")
    for name in ("e01_tolerances.yml", "e01_jobs.json"):
        shutil.copy(REPO / "configs" / name, tmp_path / "configs" / name)
    (tmp_path / "julia" / "smoke").mkdir(parents=True)
    (tmp_path / "julia" / "smoke" / "smoke_case.jl").write_text("# fake\n", encoding="utf-8")
    (tmp_path / "julia" / "worker").mkdir(parents=True)
    (tmp_path / "julia" / "worker" / "main.jl").write_text("# fake\n", encoding="utf-8")
    (tmp_path / "julia" / "Manifest.toml").write_text(JULIA_MANIFEST, encoding="utf-8")
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n", encoding="utf-8")
    return tmp_path


def _runs(root: Path) -> list[Path]:
    """Run directories, oldest first BY RECORDED START TIME.

    Sorting by directory name orders two runs of the same second by the digest in their
    run id, and that digest is over the command, the resolved config hash and the commit
    (`registry.run.make_run_id`) — not over when the run started. A test that opens a
    context of its own and then invokes the CLI in the same second would therefore read
    back whichever digest happened to sort last, and adding a field to the configuration
    schema would silently flip it. `created_at` carries microseconds and orders them the
    way the tests mean.
    """

    def started_at(run_dir: Path) -> tuple[str, str]:
        record = run_dir / "run.json"
        if not record.is_file():
            return ("", run_dir.name)
        payload = json.loads(record.read_text(encoding="utf-8"))
        return (str(payload.get("created_at", "")), run_dir.name)

    return sorted((root / "artifacts" / "runs").iterdir(), key=started_at)


def _last_record(root: Path) -> dict[str, Any]:
    return json.loads((_runs(root)[-1] / "run.json").read_text(encoding="utf-8"))


def _argv(root: Path, *rest: str) -> list[str]:
    return ["--root", str(root), "--config", str(root / "configs" / "e01.yml"), *rest]


def _check(name: str, status: str) -> PhysicsCheck:
    return PhysicsCheck(
        name=name,
        status=status,  # type: ignore[arg-type]
        metrics={"balance_cumulative_relative": 1e-9} if status == "PASS" else {},
        thresholds={"balance_cumulative_relative_max": 1e-6},
        input_hashes={},
        evidence_paths=(),
        reason=None if status == "PASS" else f"{name}: did not run in this fake session",
    )


def _p0_mandatory() -> tuple[PhysicsCheck, ...]:
    """A PASS for every mandatory check the p0 plan says this suite produces.

    `evaluate_suite` requires the matrix to be COMPLETE, not merely free of failures: plan
    12.9 says a stage gate needs all its mandatory checks present and none of them unrun. A
    fake outcome that named one check would therefore be incomplete rather than clean, which
    is the rule working, so the "clean" fixture below names them all.
    """
    plan = json.loads((REPO / "configs" / "e01_jobs.json").read_text(encoding="utf-8"))
    names = {
        job["scored_as"]
        for job in plan["suites"]["p0"]["jobs"]
        if job.get("deferred_to") is None and job["scored_as"] in MANDATORY_CHECKS
    }
    return tuple(_check(name, "PASS") for name in sorted(names))


def _job(job_id: str, status: str = "COMPLETE") -> JobOutcome:
    return JobOutcome(
        job_id=job_id,
        group="fake",
        kind="fixture",
        profile="P0_VERIFY",
        accounting="launcher",
        expected_outcome="COMPLETE",
        status=status,
        wall_s=1.0,
        cpu_s=1.0,
        peak_rss_bytes=1,
        output_bytes=1,
        native_chunk_calls=1,
        accepted_steps=1,
        cut_steps=0,
        nonlinear_iterations=1,
        retry_count=0,
        reason=None,
    )


# --------------------------------------------------------------------------------------
# 12.1 the exit-code mapping itself
# --------------------------------------------------------------------------------------


def test_incomplete_result_is_nonzero_exit() -> None:
    assert forward_exit_code("COMPLETE") == 0
    assert forward_exit_code("INCOMPLETE_BUDGET") == 2
    assert forward_exit_code("RESOURCE_FAILURE") == 2


@pytest.mark.parametrize(
    "status",
    ["INVALID_INPUT", "PHYSICALLY_INVALID", "NUMERICAL_FAILURE", "CONTROL_INFEASIBLE", "TIMEOUT"],
)
def test_every_refusal_is_a_nonzero_exit(status: str) -> None:
    """There is exactly one status that means "what was asked for happened"."""
    assert forward_exit_code(status) == 2


# --------------------------------------------------------------------------------------
# 12.1 a written artifact is not a success
# --------------------------------------------------------------------------------------


def test_a_partial_suite_report_exits_2_and_names_what_is_left(e01_project: Path) -> None:
    """The wall stop. The jobs that ran keep their rows; the ones that did not are named."""

    def stopped(run: SuiteRun) -> SuiteOutcome:
        return SuiteOutcome(
            jobs=(_job("closed_cell"), _job("closed_box")),
            checks=(_check("closed_cell_pvt", "PASS"),),
            remaining_job_ids=("hydrostatic", "gravity_segregation"),
            stopped_reason="the session's 600 s wall budget was spent after two jobs",
        )

    code = main(_argv(e01_project, "verify-physics", "--suite", "p0"), suite_runner=stopped)
    assert code == 2
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    report = json.loads(
        (_runs(e01_project)[-1] / SUITE_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert report["exit_code"] == 2
    assert report["remaining_job_ids"] == ["hydrostatic", "gravity_segregation"]
    assert [job["job_id"] for job in report["jobs"]] == ["closed_cell", "closed_box"]
    assert "wall budget" in report["stopped_reason"]


def test_a_failed_check_exits_1_even_though_the_report_was_written(e01_project: Path) -> None:
    def failed(run: SuiteRun) -> SuiteOutcome:
        return SuiteOutcome(
            jobs=(_job("closed_cell"),),
            checks=(_check("closed_cell_pvt", "FAIL"),),
            remaining_job_ids=(),
            stopped_reason=None,
        )

    assert main(_argv(e01_project, "verify-physics", "--suite", "p0"), suite_runner=failed) == 1
    assert _last_record(e01_project)["status"] == "FAIL"
    report = json.loads(
        (_runs(e01_project)[-1] / SUITE_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert report["exit_code"] == 1
    assert report["checks"][0]["status"] == "FAIL"


def test_an_unrun_check_is_not_a_pass(e01_project: Path) -> None:
    """Plan 12.9: the stage matrix requires zero unrun checks, so NOT_RUN is not green."""

    def unrun(run: SuiteRun) -> SuiteOutcome:
        checks = tuple(c for c in _p0_mandatory() if c.name != "bl") + (_check("bl", "NOT_RUN"),)
        return SuiteOutcome(
            jobs=(_job("closed_cell"),), checks=checks, remaining_job_ids=(), stopped_reason=None
        )

    assert main(_argv(e01_project, "verify-physics", "--suite", "p0"), suite_runner=unrun) == 2


def test_a_clean_suite_exits_0(e01_project: Path) -> None:
    def clean(run: SuiteRun) -> SuiteOutcome:
        return SuiteOutcome(
            jobs=(_job("closed_cell"),),
            checks=_p0_mandatory(),
            remaining_job_ids=(),
            stopped_reason=None,
        )

    assert main(_argv(e01_project, "verify-physics", "--suite", "p0"), suite_runner=clean) == 0
    record = _last_record(e01_project)
    assert record["status"] == "PASS"
    assert suite_exit_code(_runs(e01_project)[-1]) == 0


def test_a_job_that_missed_its_expected_outcome_is_not_a_pass(e01_project: Path) -> None:
    surprise = _job("bhp_infeasible", status="COMPLETE").model_copy(
        update={"expected_outcome": "CONTROL_INFEASIBLE"}
    )

    def surprising(run: SuiteRun) -> SuiteOutcome:
        return SuiteOutcome(
            jobs=(surprise,),
            checks=_p0_mandatory(),
            remaining_job_ids=(),
            stopped_reason=None,
        )

    assert main(_argv(e01_project, "verify-physics", "--suite", "p0"), suite_runner=surprising) == 2


# --------------------------------------------------------------------------------------
# 12.1 the refusals
# --------------------------------------------------------------------------------------


def test_unknown_suite_is_refused_before_any_run(e01_project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(_argv(e01_project, "verify-physics", "--suite", "p7"))
    assert exc.value.code == 2
    assert not (e01_project / "artifacts" / "runs").exists()


def test_a_config_without_resources_records_a_fail(e01_project: Path) -> None:
    code = main(
        [
            "--root",
            str(e01_project),
            "--config",
            str(e01_project / "configs" / "project.yml"),
            "verify-physics",
            "--suite",
            "p0",
        ]
    )
    assert code != 0
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    assert any("resource profile" in note for note in record["notes"])


def test_an_unapproved_profile_records_a_fail(e01_project: Path) -> None:
    code = main(
        [
            "--root",
            str(e01_project),
            "--config",
            str(e01_project / "configs" / "unapproved.yml"),
            "verify-physics",
            "--suite",
            "p0",
        ]
    )
    assert code != 0
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    assert any("approved" in note for note in record["notes"])


def test_a_missing_julia_records_a_fail(e01_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SO_RECON_JULIA", str(e01_project / "no-such-julia"))
    code = main(_argv(e01_project, "verify-physics", "--suite", "p0"))
    assert code != 0
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    assert any("julia" in note.lower() for note in record["notes"])


def test_a_case_path_outside_the_root_records_a_fail(e01_project: Path) -> None:
    code = main(_argv(e01_project, "forward", "--case", "../outside/case.json"))
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


def test_a_manifest_that_is_not_a_case_records_a_fail(e01_project: Path) -> None:
    case_dir = e01_project / "artifacts" / "cases"
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(
        json.dumps({"case_id": "c", "model_hash": "0" * 64}), encoding="utf-8"
    )
    code = main(_argv(e01_project, "forward", "--case", "artifacts/cases/case.json"))
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


def test_a_case_whose_bytes_do_not_match_its_model_hash_records_a_fail(
    e01_project: Path,
) -> None:
    """The wrong case hash: a REAL case manifest whose declared hash is no longer its own.

    The case is rendered by the production generator and published by the production
    writer, so what is tampered with here is one field of a manifest that was valid a line
    earlier — exactly the drift `load_case` exists to catch. No Julia is launched: the
    refusal happens before any work is booked.
    """
    paths = ProjectPaths.default(e01_project)
    paths.ensure_dirs()
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    case = build_p1_case(render_p1(41, P1Design()), paths, ctx)
    write_case(case, paths, ctx)
    manifest = ctx.run_dir / CASE_MANIFEST_FILENAME
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["model_hash"] == case.model_hash
    payload["model_hash"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    code = main(_argv(e01_project, "forward", "--case", paths.relative(manifest)))
    assert code != 0
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    assert any("model_hash" in note for note in record["notes"])


def test_a_restart_manifest_that_is_not_one_records_a_fail(e01_project: Path) -> None:
    """The reused result with the wrong SHA: a checkpoint that is not this model's."""
    case_dir = e01_project / "artifacts" / "cases"
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps({"case_id": "c"}), encoding="utf-8")
    (case_dir / "restart.json").write_text(
        json.dumps({"model_hash": "1" * 64, "completed_report_step": 1}), encoding="utf-8"
    )
    code = main(
        _argv(
            e01_project,
            "forward-resume",
            "--case",
            "artifacts/cases/case.json",
            "--restart",
            "artifacts/cases/restart.json",
        )
    )
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


def test_e01_report_refuses_a_run_directory_that_does_not_exist(e01_project: Path) -> None:
    code = main(_argv(e01_project, "e01-report", "--runs", "artifacts/runs/nope"))
    assert code != 0
    record = _last_record(e01_project)
    assert record["status"] == "FAIL"
    # And nothing was published: a stage report is never written from evidence that is not
    # there (plan 12.11).
    assert not (e01_project / "reports" / "stages" / "E01.md").exists()


def test_synthetic_p1_refuses_a_negative_seed(e01_project: Path) -> None:
    code = main(_argv(e01_project, "synthetic-p1", "--seeds", "-1"))
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


def test_benchmark_refuses_fewer_than_five_warm_runs(e01_project: Path) -> None:
    """12.5's p50/p90 is computed over five completed warm timings or it is not computed."""
    code = main(
        _argv(
            e01_project,
            "benchmark-forward",
            "--case",
            "artifacts/cases/case.json",
            "--warm-runs",
            "2",
        )
    )
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


# --------------------------------------------------------------------------------------
# 12.3 the global options keep their place
# --------------------------------------------------------------------------------------


def test_global_options_still_come_before_the_subcommand(e01_project: Path) -> None:
    """`--root/--config` are global and precede the subcommand; the new commands do not move
    them. A command line that puts them after the subcommand is rejected by argparse."""
    with pytest.raises(SystemExit):
        main(["verify-physics", "--suite", "p0", "--root", str(e01_project)])


@pytest.mark.parametrize(
    "command",
    ["forward", "forward-resume", "verify-physics", "synthetic-p1", "benchmark-forward"],
)
def test_every_new_command_is_registered(command: str, e01_project: Path) -> None:
    """Registered, and refusing on its own terms rather than on argparse's."""
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(e01_project), command, "--help"])
    assert exc.value.code == 0


# --------------------------------------------------------------------------------------
# 12.6 a resume ledger that is not one
# --------------------------------------------------------------------------------------


def test_a_resume_ledger_that_is_not_a_ledger_records_a_fail(e01_project: Path) -> None:
    bad = e01_project / "artifacts" / "ledger.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{}", encoding="utf-8")
    code = main(_argv(e01_project, "verify-physics", "--suite", "p0", "--resume-ledger", str(bad)))
    assert code != 0
    assert _last_record(e01_project)["status"] == "FAIL"


def test_replay_is_registered_and_does_not_disturb_the_option_order(e01_project: Path) -> None:
    """12.6's explicit replay: the one thing that makes a resumed session run it all again."""

    def clean(run: SuiteRun) -> SuiteOutcome:
        return SuiteOutcome(
            jobs=(_job("closed_cell"),),
            checks=_p0_mandatory(),
            remaining_job_ids=(),
            stopped_reason=None,
        )

    argv = _argv(e01_project, "verify-physics", "--suite", "p0", "--replay")
    assert main(argv, suite_runner=clean) == 0
    recorded = _last_record(e01_project)["argv"]
    assert recorded[:2] == ["so-recon", "--root"] and recorded[3] == "--config"
    assert recorded[5] == "verify-physics" and "--replay" in recorded


def test_suite_exit_code_of_a_directory_with_no_report_is_not_zero(tmp_path: Path) -> None:
    assert suite_exit_code(tmp_path) != 0
