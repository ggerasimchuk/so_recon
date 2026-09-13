import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.config.schema import SourceFileSpec, SourcesConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.source_manifest import (
    MANIFEST_SCHEMA_VERSION,
    MissingSourceError,
    build_source_manifest,
    hash_and_count,
    load_source_manifest,
    manifest_bytes,
    write_source_manifest,
)

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


def _spec(name: str, required: bool = True, header_lines: int = 1) -> SourceFileSpec:
    return SourceFileSpec(
        name=name,
        path=f"data/raw/{name}.csv",
        encoding="utf-8",
        delimiter=";",
        decimal=",",
        header_lines=header_lines,
        required=required,
    )


def _paths(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    paths.raw.mkdir(parents=True, exist_ok=True)  # tests create the source dir explicitly
    return paths


def test_hash_and_count_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"a;b\r\n1;2\r\n")
    sha, size, physical = hash_and_count(p, chunk_size=2)
    assert sha == hashlib.sha256(b"a;b\r\n1;2\r\n").hexdigest()
    assert (size, physical) == (10, 2)


def test_hash_and_count_without_trailing_newline(tmp_path: Path) -> None:
    """plastoper.csv ends without a newline: the last partial line still counts."""
    p = tmp_path / "f.csv"
    p.write_bytes(b"a;b\r\n1;2\r\n3;4")
    sha, size, physical = hash_and_count(p, chunk_size=2)
    assert sha == hashlib.sha256(b"a;b\r\n1;2\r\n3;4").hexdigest()
    assert (size, physical) == (13, 3)


def test_hash_and_count_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"")
    assert hash_and_count(p) == (hashlib.sha256(b"").hexdigest(), 0, 0)


def test_manifest_splits_physical_lines_from_data_rows(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"\xef\xbb\xbfh1;h2\n1;2\n3;4\n")
    (paths.raw / "tail.csv").write_bytes(b"h1;h2\n1;2")  # no trailing newline
    sources = SourcesConfig(files=[_spec("mer"), _spec("tail")])
    m = build_source_manifest(sources, paths, config_version="t")
    by_name = {s.name: s for s in m.sources}
    assert (by_name["mer"].physical_line_count, by_name["mer"].data_rows) == (3, 2)
    assert (by_name["tail"].physical_line_count, by_name["tail"].data_rows) == (2, 1)
    assert by_name["mer"].header_lines == 1


def test_manifest_has_no_timestamps_or_git_state(tmp_path: Path) -> None:
    """Invariant I5: the committed manifest must be byte-identical across runs."""
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"h\n1\n")
    sources = SourcesConfig(files=[_spec("mer")])
    first = manifest_bytes(build_source_manifest(sources, paths, config_version="t"))
    second = manifest_bytes(build_source_manifest(sources, paths, config_version="t"))
    assert first == second
    payload = json.loads(first)
    assert payload["manifest_version"] == MANIFEST_SCHEMA_VERSION
    assert set(payload) == {"manifest_version", "spec_version", "config_version", "sources"}


def test_missing_required_files_raise_with_full_list(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    sources = SourcesConfig(files=[_spec("mer"), _spec("gis"), _spec("opt", required=False)])
    with pytest.raises(MissingSourceError) as exc:
        build_source_manifest(sources, paths, config_version="t")
    assert exc.value.missing == ["data/raw/mer.csv", "data/raw/gis.csv"]


def test_optional_missing_file_is_skipped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"1\n")
    sources = SourcesConfig(files=[_spec("mer"), _spec("opt", required=False)])
    m = build_source_manifest(sources, paths, config_version="t")
    assert [s.name for s in m.sources] == ["mer"]


def test_sources_are_never_modified(tmp_path: Path) -> None:
    """Invariant I1: hashing must not touch mtime, size or content of the sources."""
    paths = _paths(tmp_path)
    src = paths.raw / "mer.csv"
    src.write_bytes(b"h\n1\n")
    before = (src.read_bytes(), src.stat().st_size, src.stat().st_mtime_ns)
    build_source_manifest(SourcesConfig(files=[_spec("mer")]), paths, config_version="t")
    after = (src.read_bytes(), src.stat().st_size, src.stat().st_mtime_ns)
    assert before == after
    assert sorted(p.name for p in paths.raw.iterdir()) == ["mer.csv"]


def test_write_publishes_identical_bytes_and_roundtrips(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.raw / "mer.csv").write_bytes(b"h\n1\n")
    m = build_source_manifest(SourcesConfig(files=[_spec("mer")]), paths, config_version="t")
    run_dir = paths.runs / "run-1"
    run_dir.mkdir(parents=True)
    published = paths.manifests / "source_manifest.json"
    ref, published_path = write_source_manifest(
        m, paths, run_dir=run_dir, published_path=published, producer_run_id="run-1", now=NOW
    )
    assert published_path == published
    assert (run_dir / "source_manifest.json").read_bytes() == published.read_bytes()
    assert ref.sha256 == hashlib.sha256(published.read_bytes()).hexdigest()
    assert ref.producer_run_id == "run-1"
    assert load_source_manifest(published) == m
