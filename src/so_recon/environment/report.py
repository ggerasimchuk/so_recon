"""Environment report: versions and lock hashes (SPEC 19.12; STAGES E00 output).

The committed report is deterministic ACROSS MACHINES, not merely across runs. It carries
no timestamps, run ids or git state, and no machine fingerprint either: `os`, `arch`,
`uv_version` and `julia_executable_version` describe the host that happened to run the
command, not the environment the project is pinned to, and leaving them in the committed
file made `make gate` report a false drift the first time a collaborator ran it on another
OS (invariant I5, amendment 10 — «чтобы повторный gate не загрязнял git»).

What stays committed is exactly what the locks fix: `python_version` (pinned by
`.python-version`, so a difference there IS real drift the gate should fail on), the
uv/Julia lock hashes, the Julia versions read out of `Manifest.toml`, and the combined
`environment_lock_hash`. The host facts are not lost — they are recorded per run in
EnvironmentStamp, written into the run directory and never committed.
"""

from __future__ import annotations

import json
import platform
import subprocess
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_artifact
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia

ENVIRONMENT_SCHEMA_VERSION = "3"
TRACKED_JULIA_PACKAGES = ("JutulDarcy", "Jutul", "JSON")
JULIA_VERSION_FILE = ".julia-version"

VersionProbe = Callable[[list[str]], str | None]


def subprocess_probe(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def parse_julia_manifest(path: Path) -> tuple[str | None, dict[str, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    julia_version = data.get("julia_version")
    deps = data.get("deps", {})
    packages: dict[str, str] = {}
    for name in TRACKED_JULIA_PACKAGES:
        entries = deps.get(name)
        if entries and isinstance(entries, list) and "version" in entries[0]:
            packages[name] = str(entries[0]["version"])
    return (str(julia_version) if julia_version else None), packages


class LockedVersions(StrictModel):
    """Versions the project is pinned to. Runtime drift from these is a failure."""

    julia_pinned: str | None
    julia_manifest: str | None
    packages: dict[str, str]


def read_locked_versions(paths: ProjectPaths) -> LockedVersions:
    pin_file = paths.julia / JULIA_VERSION_FILE
    pinned = pin_file.read_text(encoding="utf-8").strip() if pin_file.is_file() else None
    manifest = paths.julia / "Manifest.toml"
    manifest_version: str | None = None
    packages: dict[str, str] = {}
    if manifest.is_file():
        manifest_version, packages = parse_julia_manifest(manifest)
    return LockedVersions(
        julia_pinned=pinned or None, julia_manifest=manifest_version, packages=packages
    )


def check_locked_versions(
    locked: LockedVersions, *, julia_version: str, jutul_version: str, jutuldarcy_version: str
) -> list[str]:
    """Return the drift between the running stack and the lock. Non-empty means FAIL."""
    mismatches: list[str] = []
    expected_julia = locked.julia_pinned or locked.julia_manifest
    if expected_julia is None:
        mismatches.append("julia version lock missing (julia/.julia-version and Manifest.toml)")
    elif expected_julia != julia_version:
        mismatches.append(f"julia version: running {julia_version}, locked {expected_julia}")
    if (
        locked.julia_pinned
        and locked.julia_manifest
        and locked.julia_pinned != locked.julia_manifest
    ):
        mismatches.append(
            f"julia lock is inconsistent: .julia-version {locked.julia_pinned}, "
            f"Manifest.toml {locked.julia_manifest}"
        )
    for name, running in (("Jutul", jutul_version), ("JutulDarcy", jutuldarcy_version)):
        expected = locked.packages.get(name)
        if expected is None:
            mismatches.append(f"{name} version lock missing in julia/Manifest.toml")
        elif expected != running:
            mismatches.append(f"{name} version: running {running}, locked {expected}")
    return mismatches


class EnvironmentReport(StrictModel):
    """Committed report. Every field here is fixed by a tracked lock file, so a diff in it
    is real drift — never a consequence of who ran the command or on what machine."""

    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    python_version: str
    uv_lock_sha256: str | None
    julia_pinned_version: str | None
    julia_manifest_version: str | None
    julia_manifest_sha256: str | None
    julia_packages: dict[str, str]
    environment_lock_hash: str


class EnvironmentStamp(StrictModel):
    """Run-scoped facts kept out of the committed report (invariant I5).

    This is where the host fingerprint lives: nothing is lost by keeping it out of the
    committed report, because every run records it here, next to the sha256 of the exact
    report bytes that run produced.
    """

    schema_version: str
    report_sha256: str
    run_id: str
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    os: str
    arch: str
    uv_version: str | None
    julia_executable_version: str | None


def _second_token(text: str | None) -> str | None:
    if not text:
        return None
    parts = text.split()
    return parts[1] if len(parts) >= 2 else None


def _julia_exe_version(probe: VersionProbe) -> str | None:
    try:
        exe = find_julia()
    except JuliaNotFoundError:
        return None
    out = probe([str(exe), "--version"])  # "julia version 1.12.7"
    if not out:
        return None
    parts = out.split()
    return parts[2] if len(parts) >= 3 else None


def collect_environment(paths: ProjectPaths) -> EnvironmentReport:
    """Read the committed report straight out of the tracked lock files.

    No probe parameter: nothing here shells out any more. Everything that needed a probe
    (uv and the julia executable) describes the host and now belongs to the stamp.
    """
    uv_lock = paths.root / "uv.lock"
    manifest = paths.julia / "Manifest.toml"
    locked = read_locked_versions(paths)
    return EnvironmentReport(
        python_version=platform.python_version(),
        uv_lock_sha256=sha256_file(uv_lock) if uv_lock.is_file() else None,
        julia_pinned_version=locked.julia_pinned,
        julia_manifest_version=locked.julia_manifest,
        julia_manifest_sha256=sha256_file(manifest) if manifest.is_file() else None,
        julia_packages=locked.packages,
        environment_lock_hash=environment_lock_hash(paths),
    )


def render_markdown(report: EnvironmentReport) -> str:
    rows = [
        ("schema_version", report.schema_version),
        ("python_version", report.python_version),
        ("uv_lock_sha256", report.uv_lock_sha256),
        ("julia_pinned_version", report.julia_pinned_version),
        ("julia_manifest_version", report.julia_manifest_version),
        ("julia_manifest_sha256", report.julia_manifest_sha256),
        ("environment_lock_hash", report.environment_lock_hash),
    ]
    lines = [
        "# Environment report",
        "",
        "Детерминированный отчёт: только то, что зафиксировано lock-файлами — без",
        "timestamps, git-состояния и отпечатка машины, чтобы повторный gate не создавал",
        "diff ни при повторе, ни на другой машине. Время запуска, git-состояние, `os`,",
        "`arch`, `uv_version` и `julia_executable_version` — в",
        "`artifacts/runs/<run_id>/environment_stamp.json`.",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | `{v}` |" for k, v in rows]
    lines += ["", "## Julia packages (locked)", "", "| package | version |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in sorted(report.julia_packages.items())]
    return "\n".join(lines) + "\n"


def report_json_bytes(report: EnvironmentReport) -> bytes:
    payload = report.model_dump(mode="json")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def report_markdown_bytes(report: EnvironmentReport) -> bytes:
    return render_markdown(report).encode("utf-8")


def write_environment_report(
    report: EnvironmentReport,
    paths: ProjectPaths,
    *,
    run_dir: Path,
    md_path: Path,
    json_path: Path,
    producer_run_id: str,
    now: datetime,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Immutable run artifacts first, then byte-identical published copies."""
    md_payload = report_markdown_bytes(report)
    json_payload = report_json_bytes(report)
    md_ref = write_artifact(
        run_dir / "environment_report.md",
        md_payload,
        paths,
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        producer_run_id=producer_run_id,
        media_type="text/markdown",
        now=now,
    )
    json_ref = write_artifact(
        run_dir / "environment.json",
        json_payload,
        paths,
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        producer_run_id=producer_run_id,
        media_type="application/json",
        now=now,
    )
    # write_bytes_atomic performs no validation, so prove containment here: a
    # mis-constructed path would otherwise write outside the repository (I4).
    paths.relative(md_path)
    paths.relative(json_path)
    write_bytes_atomic(md_path, md_payload)
    write_bytes_atomic(json_path, json_payload)
    return md_ref, json_ref


def write_environment_stamp(stamp: EnvironmentStamp, path: Path) -> None:
    write_json_atomic(path, stamp.model_dump(mode="json"))


def build_environment_stamp(
    paths: ProjectPaths,
    *,
    report_sha256: str,
    run_id: str,
    now: datetime,
    probe: VersionProbe = subprocess_probe,
) -> EnvironmentStamp:
    return EnvironmentStamp(
        schema_version=ENVIRONMENT_SCHEMA_VERSION,
        report_sha256=report_sha256,
        run_id=run_id,
        created_at=now.isoformat(),
        git_commit=git_commit(paths.root),
        git_dirty=git_is_dirty(paths.root),
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        uv_version=_second_token(probe(["uv", "--version"])),
        julia_executable_version=_julia_exe_version(probe),
    )
