import json
import logging
from pathlib import Path

import pytest

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import RunRecordUnavailableError, execute_run


def _setup(tmp_project: Path) -> tuple[object, ProjectPaths]:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    return cfg, ProjectPaths.from_config(tmp_project, cfg.paths)


def test_execute_run_records_pass(tmp_project: Path) -> None:
    cfg, paths = _setup(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        log.info("working")
        return "PASS", ["done"]

    ctx = execute_run(command="x", argv=["so-recon", "x"], cfg=cfg, paths=paths, body=body)
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    # Not `record["notes"] == ["done"]`: tmp_project is never inside a real git repository,
    # so RunContext.start (Task 9, unchanged here) always prepends its own
    # "git commit unavailable" note (see test_missing_git_metadata_is_recorded_as_a_note in
    # tests/unit/test_run_registry.py). The invariant this test checks is that the body's
    # own note reaches the record, not that it is the only one.
    assert "done" in record["notes"]
    assert (ctx.run_dir / "run.log").is_file()


def test_execute_run_converts_any_exception_into_a_fail_record(tmp_project: Path) -> None:
    """Invariant I6: an unexpected exception must still leave a FAIL run record."""
    cfg, paths = _setup(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        raise RuntimeError("unexpected boom")

    ctx = execute_run(command="x", argv=["so-recon", "x"], cfg=cfg, paths=paths, body=body)
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert any("unexpected boom" in n for n in record["notes"])
    assert "RuntimeError" in (ctx.run_dir / "run.log").read_text(encoding="utf-8")


def test_unusable_runs_directory_is_a_clean_named_failure(tmp_project: Path) -> None:
    """Invariant I6's boundary: when the artifacts tree itself is unusable no record can
    exist, but the caller must get a named error rather than a raw traceback."""
    cfg, paths = _setup(tmp_project)
    # reports/ as a regular file makes ensure_dirs fail on reports/manifests.
    (tmp_project / "reports").write_text("not a directory\n")

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        raise AssertionError("body must never run when the run cannot be opened")

    with pytest.raises(RunRecordUnavailableError, match="cannot open a run record"):
        execute_run(command="manifest", argv=[], cfg=cfg, paths=paths, body=body)


@pytest.mark.parametrize("body_raises", [False, True])
def test_a_failure_while_closing_the_record_is_a_named_error(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, body_raises: bool
) -> None:
    """Closing is a write too (I6's boundary), and it must not escape as a traceback.

    Both branches are covered: the PASS path and the FAIL path each call finish(), so a
    guard on only one of them would leave the other open.
    """
    cfg, paths = _setup(tmp_project)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(RunContext, "finish", boom)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        if body_raises:
            raise RuntimeError("inner boom")
        return "PASS", []

    with pytest.raises(RunRecordUnavailableError, match="cannot close the run record"):
        execute_run(command="x", argv=["so-recon", "x"], cfg=cfg, paths=paths, body=body)


def test_execute_run_works_without_config(tmp_project: Path) -> None:
    paths = ProjectPaths.default(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        return "FAIL", ["config error"]

    ctx = execute_run(command="x", argv=[], cfg=None, paths=paths, body=body)
    assert json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))["status"] == "FAIL"
