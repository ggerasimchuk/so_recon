"""Separate immutable caches for physical forwards F and likelihood evaluations L."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import sha256_file, sha256_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE = re.compile(r"^[a-z][a-z0-9_-]*$")


class CacheIntegrityError(RuntimeError):
    """A cache entry or the immutable artifact it names no longer matches its digest."""


def _digest(value: str, *, label: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256, got {value!r}")
    return value


def forward_key(
    *,
    physical: Mapping[str, object],
    solver_hash: str,
    output_request: Mapping[str, object],
    adapter_hash: str,
    environment_lock_hash: str,
) -> str:
    """Identity of F: physics, requested outputs, solver, adapter and pinned environment.

    Particle ids, proposal RNG and nuisance parameters are deliberately absent.  A caller
    passes only the physical payload that the solver reads; any observation or noise change
    therefore leaves this key reusable while the L key below changes.
    """
    return sha256_json(
        {
            "schema_version": "e02-forward-key-1",
            "physical": dict(physical),
            "solver_hash": _digest(solver_hash, label="solver_hash"),
            "output_request": dict(output_request),
            "adapter_hash": _digest(adapter_hash, label="adapter_hash"),
            "environment_lock_hash": _digest(environment_lock_hash, label="environment_lock_hash"),
        }
    )


def likelihood_key(
    *,
    forward_hash: str,
    observation_hash: str,
    operator_hash: str,
    noise_hash: str,
) -> str:
    """Identity of L conditional on one already-computed physical forward."""
    payload = {
        "schema_version": "e02-likelihood-key-1",
        "forward_hash": _digest(forward_hash, label="forward_hash"),
        "observation_hash": _digest(observation_hash, label="observation_hash"),
        "operator_hash": _digest(operator_hash, label="operator_hash"),
        "noise_hash": _digest(noise_hash, label="noise_hash"),
    }
    return sha256_json(payload)


class ArtifactCache:
    """A small on-disk index from a semantic key to one immutable ``ArtifactRef``.

    The index is not the artifact.  Every hit re-resolves the target within the repository,
    re-hashes its bytes and checks its size.  Missing entries are misses; missing or corrupt
    targets are integrity failures and cannot silently cause a recomputation under a key
    that still claims the old value.
    """

    def __init__(self, paths: ProjectPaths, namespace: str) -> None:
        if not _NAMESPACE.fullmatch(namespace):
            raise ValueError(f"invalid cache namespace {namespace!r}")
        self.paths = paths
        self.namespace = namespace
        self.root = paths.artifacts / "cache" / namespace

    def _entry_path(self, key: str) -> Path:
        return self.root / f"{_digest(key, label='cache key')}.json"

    def get(self, key: str) -> ArtifactRef | None:
        entry = self._entry_path(key)
        if not entry.is_file():
            return None
        try:
            ref = ArtifactRef.model_validate(json.loads(entry.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError) as exc:
            raise CacheIntegrityError(
                f"cache entry {entry} is not a valid ArtifactRef: {exc}"
            ) from exc
        try:
            target = self.paths.resolve(ref.path)
        except ValueError as exc:
            raise CacheIntegrityError(f"cache entry {entry} names an unusable path: {exc}") from exc
        if not target.is_file():
            raise CacheIntegrityError(f"cache entry {entry} names missing artifact {ref.path}")
        actual = sha256_file(target)
        if actual != ref.sha256:
            raise CacheIntegrityError(
                f"artifact {ref.path} sha256 is {actual}, cache entry declares {ref.sha256}"
            )
        size = target.stat().st_size
        if size != ref.size_bytes:
            raise CacheIntegrityError(
                f"artifact {ref.path} has {size} bytes, cache entry declares {ref.size_bytes}"
            )
        return ref

    def put(self, key: str, ref: ArtifactRef) -> None:
        entry = self._entry_path(key)
        if entry.exists():
            existing = self.get(key)
            if existing != ref:
                raise CacheIntegrityError(
                    f"cache key {key} already maps to {existing and existing.path}, refusing "
                    f"a different value {ref.path}"
                )
            return
        # Prove the value before publishing an index that claims it.
        target = self.paths.resolve(ref.path)
        if (
            not target.is_file()
            or sha256_file(target) != ref.sha256
            or target.stat().st_size != ref.size_bytes
        ):
            raise CacheIntegrityError(
                f"refusing to cache {ref.path}: its bytes do not match the declared sha256 and size"
            )
        entry.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(entry, ref.model_dump(mode="json"))


__all__ = [
    "ArtifactCache",
    "CacheIntegrityError",
    "forward_key",
    "likelihood_key",
]
