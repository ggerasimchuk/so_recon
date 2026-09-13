"""Manifest of raw source files: SHA-256, size and line counts (SPEC 7.1, 19.12).

The sources are immutable: every file is opened read-only and is never moved, renamed
or rewritten (invariant I1). The manifest itself is deterministic — it carries no
timestamps, run ids or git state, so a repeated gate run produces no git diff
(invariant I5). Those run-scoped facts live in SourceManifestStamp instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from so_recon import SpecVersion
from so_recon.config.schema import SourcesConfig, StrictModel
from so_recon.paths import PathEscapeError, ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_artifact
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic

log = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "2"
MANIFEST_MEDIA_TYPE = "application/json"


class MissingSourceError(FileNotFoundError):
    def __init__(self, missing: Sequence[str]) -> None:
        super().__init__(f"required source files are missing: {list(missing)}")
        self.missing = list(missing)


def hash_and_count(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int, int]:
    """Return (sha256, size_bytes, physical_line_count) in a single read-only pass.

    physical_line_count counts newline bytes plus a final partial line when the file
    does not end with a newline. It is a physical count, not a parsed-record count:
    E00 does not parse CSV, so newlines inside quoted fields are not accounted for.
    """
    digest = hashlib.sha256()
    size = 0
    newlines = 0
    last = b""
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
    physical = newlines + (1 if size > 0 and last != b"\n" else 0)
    return digest.hexdigest(), size, physical


class SourceEntry(StrictModel):
    name: str
    path: str
    sha256: str
    size_bytes: int
    physical_line_count: int
    data_rows: int
    encoding: str
    delimiter: str
    decimal: str
    header_lines: int
    required: bool


class SourceManifest(StrictModel):
    manifest_version: Literal["2"] = "2"
    spec_version: str
    config_version: str
    sources: list[SourceEntry]

    def hashes(self) -> dict[str, str]:
        return {s.name: s.sha256 for s in self.sources}


class SourceManifestStamp(StrictModel):
    """Run-scoped facts kept out of the committed manifest (invariant I5)."""

    manifest_version: str
    manifest_sha256: str
    run_id: str
    created_at: str
    git_commit: str | None
    git_dirty: bool | None


def build_source_manifest(
    sources: SourcesConfig,
    paths: ProjectPaths,
    *,
    config_version: str,
    spec_version: SpecVersion = "3.0",
) -> SourceManifest:
    """Hash the configured sources into a deterministic manifest.

    `spec_version` defaults to 3.0 so that the E00 manifests stay byte-identical; every
    new caller passes the version from its validated config explicitly.
    """
    entries: list[SourceEntry] = []
    missing: list[str] = []
    for spec in sources.files:
        full = paths.resolve(spec.path)  # validates form and containment (invariant I4)
        data_root = paths.resolve("data")
        if not full.is_relative_to(data_root) or not full.is_relative_to(paths.raw.resolve()):
            raise PathEscapeError(
                f"source {spec.path!r} must stay inside the configured raw directory under data/"
            )
        if not full.is_file():
            if spec.required:
                missing.append(spec.path)
            else:
                log.warning("optional source %s not found at %s; skipped", spec.name, spec.path)
            continue
        sha, size, physical = hash_and_count(full)
        data_rows = max(physical - spec.header_lines, 0)
        log.info(
            "hashed %s: %d bytes, %d physical lines, %d data rows",
            spec.name,
            size,
            physical,
            data_rows,
        )
        entries.append(
            SourceEntry(
                name=spec.name,
                path=spec.path,
                sha256=sha,
                size_bytes=size,
                physical_line_count=physical,
                data_rows=data_rows,
                encoding=spec.encoding,
                delimiter=spec.delimiter,
                decimal=spec.decimal,
                header_lines=spec.header_lines,
                required=spec.required,
            )
        )
    if missing:
        raise MissingSourceError(missing)
    return SourceManifest(spec_version=spec_version, config_version=config_version, sources=entries)


def manifest_bytes(manifest: SourceManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_source_manifest(
    manifest: SourceManifest,
    paths: ProjectPaths,
    *,
    run_dir: Path,
    published_path: Path,
    producer_run_id: str,
    now: datetime,
) -> tuple[ArtifactRef, Path]:
    """Write the immutable run artifact, then publish byte-identical committed copy."""
    payload = manifest_bytes(manifest)
    ref = write_artifact(
        run_dir / "source_manifest.json",
        payload,
        paths,
        schema_version=MANIFEST_SCHEMA_VERSION,
        producer_run_id=producer_run_id,
        media_type=MANIFEST_MEDIA_TYPE,
        now=now,
    )
    # write_bytes_atomic performs no validation, so prove containment here: a
    # mis-constructed published_path would otherwise write outside the repository (I4).
    paths.relative(published_path)
    write_bytes_atomic(published_path, payload)
    return ref, published_path


def write_manifest_stamp(stamp: SourceManifestStamp, path: Path) -> None:
    write_json_atomic(path, stamp.model_dump(mode="json"))


def load_source_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
