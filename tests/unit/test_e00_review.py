"""Regression cases for E00 invariants I1, I4 and I6."""

import json
from pathlib import Path

import pytest

from so_recon.cli import main
from so_recon.config.schema import PathsConfig, SourceFileSpec, SourcesConfig
from so_recon.paths import PathEscapeError, ProjectPaths, validate_relative_path
from so_recon.registry import run as run_module
from so_recon.registry.source_manifest import build_source_manifest


@pytest.mark.parametrize("filename", ["resolved_config.json", "run.log"])
def test_startup_io_failure_closes_existing_record(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    # Fail only the affected file; run.json remains writable throughout.
    if filename == "run.log":
        import so_recon.runner as runner

        def fail_logging(*args: object, **kwargs: object):
            raise OSError("injected startup write failure")

        monkeypatch.setattr(runner, "configure_logging", fail_logging)
    else:
        original_write = run_module.write_json_atomic

        def fail_write(path: Path, obj: object):
            if path.name == filename:
                raise OSError("injected startup write failure")
            return original_write(path, obj)

        monkeypatch.setattr(run_module, "write_json_atomic", fail_write)
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    records = list((tmp_project / "artifacts" / "runs").glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "FAIL"
    assert record["finished_at"] is not None
    assert any("injected startup write failure" in note for note in record["notes"])


@pytest.mark.parametrize("field", ["artifacts", "reports", "configs", "interim", "processed"])
def test_output_directories_cannot_overlap_raw(tmp_path: Path, field: str) -> None:
    cfg = PathsConfig(**{field: "data/raw/output"})
    with pytest.raises(PathEscapeError):
        ProjectPaths.from_config(tmp_path, cfg).ensure_dirs()
    assert not (tmp_path / "data" / "raw").exists()


def test_runtime_subdirectory_symlink_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts" / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathEscapeError):
        ProjectPaths.default(root).ensure_dirs()
    assert list(outside.iterdir()) == []


def test_manifest_cannot_hash_arbitrary_repository_files(tmp_path: Path) -> None:
    (tmp_path / "private.csv").write_text("private\n")
    source = SourceFileSpec(
        name="private", path="private.csv", encoding="utf-8", delimiter=",", decimal="."
    )
    with pytest.raises(PathEscapeError):
        build_source_manifest(
            SourcesConfig(files=[source]), ProjectPaths.default(tmp_path), config_version="test"
        )


@pytest.mark.parametrize("path", ["C:data/raw", "z:secret.csv"])
def test_windows_drive_relative_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_path(path)


def test_source_symlink_cannot_alias_an_output_file(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    paths.raw.mkdir()
    output = paths.manifests / "source_manifest.json"
    output.write_text("original source content\n")
    (paths.raw / "source.csv").symlink_to(output)
    source = SourceFileSpec(
        name="source", path="data/raw/source.csv", encoding="utf-8", delimiter=",", decimal="."
    )
    with pytest.raises(PathEscapeError):
        build_source_manifest(SourcesConfig(files=[source]), paths, config_version="test")
    assert output.read_text() == "original source content\n"


def test_source_must_be_in_configured_raw_directory(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    (paths.processed / "source.csv").write_text("h\n1\n")
    source = SourceFileSpec(
        name="source",
        path="data/processed/source.csv",
        encoding="utf-8",
        delimiter=",",
        decimal=".",
    )
    with pytest.raises(PathEscapeError):
        build_source_manifest(SourcesConfig(files=[source]), paths, config_version="test")


def test_reports_failure_still_records_failed_run(tmp_project: Path) -> None:
    (tmp_project / "reports").write_text("not a directory\n")
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    records = list((tmp_project / "artifacts" / "runs").glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "FAIL"
    assert record["finished_at"] is not None


def test_unreadable_lock_still_records_failed_run(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(paths: ProjectPaths) -> str:
        raise PermissionError("cannot read lock")

    monkeypatch.setattr(run_module, "environment_lock_hash", unreadable)
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    records = list((tmp_project / "artifacts" / "runs").glob("*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "FAIL"
    assert record["environment_lock_hash"] == "unavailable"
    assert any("cannot read lock" in note for note in record["notes"])
