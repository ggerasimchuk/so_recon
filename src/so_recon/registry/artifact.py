"""Immutable, content-addressed artifact references (SPEC 19.12, invariants I2/I3).

An artifact is identified by the SHA-256 of its bytes. Writing the same bytes to the same
path again is idempotent; writing different bytes to an existing path is refused, so a
result that has been handed to a downstream stage can never change underneath it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import sha256_bytes, sha256_file

JSON_MEDIA_TYPE = "application/json"


class ArtifactImmutabilityError(RuntimeError):
    """An existing artifact would have been overwritten with different content."""


class ArtifactRef(StrictModel):
    artifact_id: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str
    schema_version: str
    producer_run_id: str
    parent_artifact_ids: list[str]
    created_at: str


def _ref(
    *,
    repo_relative: str,
    digest: str,
    size_bytes: int,
    media_type: str,
    schema_version: str,
    producer_run_id: str,
    parent_artifact_ids: Sequence[str],
    now: datetime,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=digest,
        path=repo_relative,
        sha256=digest,
        size_bytes=size_bytes,
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=list(parent_artifact_ids),
        created_at=now.isoformat(),
    )


def register_artifact(
    path: Path,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    media_type: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    """Describe a file that another library already wrote (for example Parquet)."""
    repo_relative = paths.relative(path)
    return _ref(
        repo_relative=repo_relative,
        digest=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )


def write_artifact(
    path: Path,
    data: bytes,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    media_type: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    repo_relative = paths.relative(path)  # also proves containment inside the repository
    digest = sha256_bytes(data)
    if path.exists():
        existing = sha256_file(path)
        if existing != digest:
            raise ArtifactImmutabilityError(
                f"refusing to overwrite {repo_relative}: on disk {existing}, new {digest}"
            )
    else:
        write_bytes_atomic(path, data)
    return _ref(
        repo_relative=repo_relative,
        digest=digest,
        size_bytes=len(data),
        media_type=media_type,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )


def write_json_artifact(
    path: Path,
    obj: object,
    paths: ProjectPaths,
    *,
    schema_version: str,
    producer_run_id: str,
    parent_artifact_ids: Sequence[str] = (),
    now: datetime,
) -> ArtifactRef:
    payload = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return write_artifact(
        path,
        payload,
        paths,
        schema_version=schema_version,
        producer_run_id=producer_run_id,
        media_type=JSON_MEDIA_TYPE,
        parent_artifact_ids=parent_artifact_ids,
        now=now,
    )
