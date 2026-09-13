import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactImmutabilityError,
    register_artifact,
    write_artifact,
    write_json_artifact,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths.default(tmp_path)


def test_write_artifact_returns_full_lineage_ref(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ref = write_artifact(
        tmp_path / "reports" / "x.bin",
        b"payload",
        paths,
        schema_version="1",
        producer_run_id="run-1",
        media_type="application/octet-stream",
        parent_artifact_ids=["parent-sha"],
        now=NOW,
    )
    assert ref.sha256 == hashlib.sha256(b"payload").hexdigest()
    assert ref.artifact_id == ref.sha256
    assert ref.path == "reports/x.bin"
    assert ref.size_bytes == 7
    assert ref.schema_version == "1"
    assert ref.producer_run_id == "run-1"
    assert ref.parent_artifact_ids == ["parent-sha"]
    assert ref.created_at == "2026-09-13T12:00:00+00:00"


def test_rewriting_identical_content_is_idempotent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "reports" / "x.bin"
    a = write_artifact(
        target,
        b"same",
        paths,
        schema_version="1",
        producer_run_id="r1",
        media_type="application/octet-stream",
        now=NOW,
    )
    b = write_artifact(
        target,
        b"same",
        paths,
        schema_version="1",
        producer_run_id="r2",
        media_type="application/octet-stream",
        now=NOW,
    )
    assert a.sha256 == b.sha256
    assert b.producer_run_id == "r2"
    assert target.read_bytes() == b"same"


def test_rewriting_with_different_content_is_refused(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "reports" / "x.bin"
    write_artifact(
        target,
        b"first",
        paths,
        schema_version="1",
        producer_run_id="r1",
        media_type="application/octet-stream",
        now=NOW,
    )
    with pytest.raises(ArtifactImmutabilityError):
        write_artifact(
            target,
            b"second",
            paths,
            schema_version="1",
            producer_run_id="r2",
            media_type="application/octet-stream",
            now=NOW,
        )
    assert target.read_bytes() == b"first", "the existing artifact must survive a refused write"


def test_write_json_artifact_is_deterministic(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    a = write_json_artifact(
        tmp_path / "reports" / "a.json",
        {"b": 1, "a": 2},
        paths,
        schema_version="1",
        producer_run_id="r1",
        now=NOW,
    )
    b = write_json_artifact(
        tmp_path / "reports" / "b.json",
        {"a": 2, "b": 1},
        paths,
        schema_version="1",
        producer_run_id="r1",
        now=NOW,
    )
    assert a.sha256 == b.sha256
    assert a.media_type == "application/json"


def test_register_artifact_hashes_existing_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    target = tmp_path / "artifacts" / "t.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"parquet-bytes")
    ref = register_artifact(
        target,
        paths,
        schema_version="1",
        producer_run_id="r1",
        media_type="application/vnd.apache.parquet",
        now=NOW,
    )
    assert ref.sha256 == hashlib.sha256(b"parquet-bytes").hexdigest()
    assert ref.path == "artifacts/t.parquet"


def test_artifact_outside_repository_root_is_refused(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path / "repo")
    with pytest.raises(ValueError):
        write_artifact(
            tmp_path / "elsewhere.bin",
            b"x",
            paths,
            schema_version="1",
            producer_run_id="r1",
            media_type="application/octet-stream",
            now=NOW,
        )
