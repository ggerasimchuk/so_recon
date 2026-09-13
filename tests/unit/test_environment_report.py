import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from so_recon.environment.report import (
    ENVIRONMENT_SCHEMA_VERSION,
    check_locked_versions,
    collect_environment,
    parse_julia_manifest,
    read_locked_versions,
    render_markdown,
    report_json_bytes,
    write_environment_report,
)
from so_recon.paths import PathEscapeError, ProjectPaths

MANIFEST = """
julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "abc"

[[deps.JSON]]
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.1.2"

[[deps.Jutul]]
uuid = "2b460a1a-8a2b-45b2-b125-b5c536396eb9"
version = "0.4.40"

[[deps.JutulDarcy]]
uuid = "82210473-ab04-4dce-b31b-11573c4f8e0a"
version = "0.3.11"

[[deps.Other]]
version = "9.9.9"
"""


def _probe(cmd: list[str]) -> str | None:
    if cmd[0] == "uv":
        return "uv 0.12.7 (Homebrew)\n"
    if cmd[0].endswith("julia"):
        return "julia version 1.12.7\n"
    return None


def _with_locks(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "uv.lock").write_text("lock")
    (tmp_path / "julia").mkdir(exist_ok=True)
    (tmp_path / "julia" / "Manifest.toml").write_text(MANIFEST)
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n")
    return ProjectPaths.default(tmp_path)


def test_parse_julia_manifest(tmp_path: Path) -> None:
    p = tmp_path / "Manifest.toml"
    p.write_text(MANIFEST)
    julia_version, pkgs = parse_julia_manifest(p)
    assert julia_version == "1.12.7"
    assert pkgs == {"JSON": "1.1.2", "Jutul": "0.4.40", "JutulDarcy": "0.3.11"}


def test_locked_versions_match_is_empty(tmp_path: Path) -> None:
    locked = read_locked_versions(_with_locks(tmp_path))
    assert locked.julia_pinned == "1.12.7"
    assert (
        check_locked_versions(
            locked, julia_version="1.12.7", jutul_version="0.4.40", jutuldarcy_version="0.3.11"
        )
        == []
    )


@pytest.mark.parametrize(
    ("julia", "jutul", "darcy", "needle"),
    [
        ("1.12.8", "0.4.40", "0.3.11", "julia"),
        ("1.12.7", "0.4.41", "0.3.11", "Jutul"),
        ("1.12.7", "0.4.40", "0.3.12", "JutulDarcy"),
    ],
)
def test_version_drift_against_lock_is_a_mismatch(
    tmp_path: Path, julia: str, jutul: str, darcy: str, needle: str
) -> None:
    """Amendment 9: drift from the lock is a failure, not an informational note."""
    locked = read_locked_versions(_with_locks(tmp_path))
    mismatches = check_locked_versions(
        locked, julia_version=julia, jutul_version=jutul, jutuldarcy_version=darcy
    )
    assert mismatches
    assert any(needle in m for m in mismatches)


def test_missing_lock_is_a_mismatch(tmp_path: Path) -> None:
    locked = read_locked_versions(ProjectPaths.default(tmp_path))
    mismatches = check_locked_versions(
        locked, julia_version="1.12.7", jutul_version="0.4.40", jutuldarcy_version="0.3.11"
    )
    assert any("missing" in m for m in mismatches)


def test_inconsistent_julia_locks_are_a_mismatch(tmp_path: Path) -> None:
    """.julia-version and Manifest.toml disagreeing is itself drift, whichever one the
    running version happens to match."""
    paths = _with_locks(tmp_path)
    (tmp_path / "julia" / ".julia-version").write_text("1.12.9\n")
    mismatches = check_locked_versions(
        read_locked_versions(paths),
        julia_version="1.12.9",
        jutul_version="0.4.40",
        jutuldarcy_version="0.3.11",
    )
    assert any("inconsistent" in m for m in mismatches)


def test_a_single_missing_package_lock_is_a_mismatch(tmp_path: Path) -> None:
    paths = _with_locks(tmp_path)
    without_jutul = "\n".join(
        block for block in MANIFEST.split("\n\n") if "deps.Jutul]]" not in block
    )
    (tmp_path / "julia" / "Manifest.toml").write_text(without_jutul)
    mismatches = check_locked_versions(
        read_locked_versions(paths),
        julia_version="1.12.7",
        jutul_version="0.4.40",
        jutuldarcy_version="0.3.11",
    )
    assert any("Jutul version lock missing" in m for m in mismatches)
    assert not any("JutulDarcy version lock missing" in m for m in mismatches)


def test_publishing_outside_the_repository_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_bytes_atomic does not validate paths, so the caller must (invariant I4)."""
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "repo"
    paths = ProjectPaths.default(root)
    rep = collect_environment(paths, probe=_probe)
    run_dir = paths.runs / "run-1"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "elsewhere.md"
    with pytest.raises(PathEscapeError):
        write_environment_report(
            rep,
            paths,
            run_dir=run_dir,
            md_path=outside,
            json_path=paths.manifests / "environment.json",
            producer_run_id="run-1",
            now=datetime(2026, 9, 13, tzinfo=UTC),
        )
    assert not outside.exists()


def test_collect_environment_with_and_without_lock_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = ProjectPaths.default(tmp_path)
    rep = collect_environment(paths, probe=_probe)
    assert rep.uv_lock_sha256 is None
    assert rep.julia_manifest_version is None
    assert rep.julia_pinned_version is None
    assert rep.julia_executable_version is None
    assert rep.uv_version == "0.12.7"

    rep2 = collect_environment(_with_locks(tmp_path), probe=_probe)
    assert rep2.uv_lock_sha256 is not None
    assert rep2.julia_manifest_version == "1.12.7"
    assert rep2.julia_pinned_version == "1.12.7"
    assert rep2.julia_packages["JutulDarcy"] == "0.3.11"
    assert rep2.environment_lock_hash != rep.environment_lock_hash


def test_report_is_deterministic_and_carries_no_time_or_git_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant I5: repeated gate runs must not produce a git diff."""
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = _with_locks(tmp_path)
    first = report_json_bytes(collect_environment(paths, probe=_probe))
    second = report_json_bytes(collect_environment(paths, probe=_probe))
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    for forbidden in ("created_at", "git_commit", "git_dirty", "run_id"):
        assert forbidden not in payload


def test_render_and_write_publishes_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = _with_locks(tmp_path)
    rep = collect_environment(paths, probe=_probe)
    md = render_markdown(rep)
    assert "# Environment report" in md
    assert "environment_lock_hash" in md
    assert "created_at" not in md
    run_dir = paths.runs / "run-1"
    run_dir.mkdir(parents=True)
    md_path = paths.reports / "environment_report.md"
    json_path = paths.manifests / "environment.json"
    md_ref, json_ref = write_environment_report(
        rep,
        paths,
        run_dir=run_dir,
        md_path=md_path,
        json_path=json_path,
        producer_run_id="run-1",
        now=datetime(2026, 9, 13, tzinfo=UTC),
    )
    assert md_path.read_text(encoding="utf-8") == md
    assert (run_dir / "environment_report.md").read_bytes() == md_path.read_bytes()
    assert (run_dir / "environment.json").read_bytes() == json_path.read_bytes()
    assert json.loads(json_path.read_text(encoding="utf-8"))["python_version"] == rep.python_version
    assert md_ref.producer_run_id == "run-1" and json_ref.producer_run_id == "run-1"
