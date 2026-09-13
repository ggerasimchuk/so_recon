import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from so_recon.config.schema import ProjectConfig, SourcesConfig
from so_recon.paths import ProjectPaths
from so_recon.registry import run as run_module
from so_recon.registry.artifact import write_json_artifact
from so_recon.registry.run import UNAVAILABLE, RunContext, environment_lock_hash, make_run_id

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)


def _cfg() -> ProjectConfig:
    return ProjectConfig(spec_version="3.0", config_version="t.1", sources=SourcesConfig(files=[]))


def test_make_run_id_format_and_determinism() -> None:
    a = make_run_id("smoke", "c" * 64, "abc123", NOW)
    b = make_run_id("smoke", "c" * 64, "abc123", NOW)
    assert a == b
    assert re.fullmatch(r"20260913T120000Z-smoke-[0-9a-f]{8}", a)
    assert make_run_id("smoke", "d" * 64, "abc123", NOW) != a
    assert make_run_id("smoke", "c" * 64, None, NOW) != a


def test_environment_lock_hash_marks_missing_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    h_missing = environment_lock_hash(paths)
    (tmp_path / "uv.lock").write_text("lock")
    h_with_lock = environment_lock_hash(paths)
    assert h_missing != h_with_lock
    assert h_with_lock == environment_lock_hash(paths)
    (tmp_path / "julia").mkdir()
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n")
    assert environment_lock_hash(paths) != h_with_lock


def test_run_context_writes_lineage_files(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(
        command="smoke",
        argv=["so-recon", "smoke"],
        cfg=_cfg(),
        paths=paths,
        raw_input_hashes={"coords": "a" * 64},
        now=NOW,
    )
    assert ctx.run_dir == paths.runs / ctx.run_id
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "RUNNING"
    assert record["spec_version"] == "3.0"
    assert record["resolved_config_hash"]
    assert record["raw_input_hashes"] == {"coords": "a" * 64}
    assert record["created_at"] == "2026-09-13T12:00:00+00:00"
    resolved = json.loads((ctx.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["config_version"] == "t.1"

    ref = write_json_artifact(
        ctx.run_dir / "thing.json",
        {"x": 1},
        paths,
        schema_version="1",
        producer_run_id=ctx.run_id,
        now=NOW,
    )
    ctx.update(julia_version="1.12.7")
    ctx.finish("PASS", outputs={"thing": ref}, notes=["ok"])
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["julia_version"] == "1.12.7"
    assert record["outputs"]["thing"]["sha256"] == ref.sha256
    assert record["outputs"]["thing"]["producer_run_id"] == ctx.run_id
    assert record["finished_at"] is not None


def test_run_context_without_config_still_records_a_run(tmp_path: Path) -> None:
    """Degraded mode: a config that failed to load must not prevent a FAIL record."""
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(
        command="manifest", argv=["so-recon", "manifest"], cfg=None, paths=paths, now=NOW
    )
    ctx.finish("FAIL", notes=["config error: boom"])
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert record["config_version"] == UNAVAILABLE
    assert record["resolved_config_hash"] == UNAVAILABLE
    assert any("boom" in n for n in record["notes"])
    assert not (ctx.run_dir / "resolved_config.json").exists()


def test_update_rejects_an_invalid_status(tmp_path: Path) -> None:
    """Lineage must never be silently corrupt: an unvalidated update would write a bogus
    status straight into run.json."""
    paths = ProjectPaths.default(tmp_path)
    ctx = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    with pytest.raises(ValidationError):
        ctx.update(status="PASSED")  # not a member of RunStatus
    on_disk = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "RUNNING", "a refused update must not reach the file"


def test_missing_git_metadata_is_recorded_as_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare null git_commit cannot be told apart from a broken git; say which."""
    paths = ProjectPaths.default(tmp_path)
    monkeypatch.setattr(run_module, "git_commit", lambda root: None)
    ctx = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    assert ctx.record.git_commit is None
    assert any("git commit unavailable" in n for n in ctx.record.notes)


def test_failed_start_leaves_no_empty_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant I6: a claimed directory with no run.json is a run that happened and
    cannot be recorded. The claim must be released instead."""
    paths = ProjectPaths.default(tmp_path)

    def boom(path: Path, obj: object, **kwargs: object) -> bytes:
        raise OSError("disk full")

    monkeypatch.setattr(run_module, "write_json_atomic", boom)
    with pytest.raises(OSError, match="disk full"):
        RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    assert list(paths.runs.iterdir()) == [], "the claimed run directory was not released"


def test_run_context_suffixes_same_second_runs(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    first = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    second = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    third = RunContext.start(command="x", argv=[], cfg=_cfg(), paths=paths, now=NOW)
    assert second.run_id == f"{first.run_id}-01"
    assert third.run_id == f"{first.run_id}-02"
    assert second.record.run_id == second.run_id
    assert (paths.runs / second.run_id / "run.json").is_file()
