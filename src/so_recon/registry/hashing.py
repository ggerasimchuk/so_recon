"""Content hashing helpers shared by manifests, artifacts, run records and fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's SHA-256. Read-only: never mutates the source (invariant I1)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: object) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 text, NaN/Inf forbidden."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_json(obj: object) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
