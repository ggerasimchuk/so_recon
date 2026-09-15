"""F and L have different semantic keys and immutable cache entries."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from so_recon.inference.cache import (
    ArtifactCache,
    CacheIntegrityError,
    forward_key,
    likelihood_key,
)
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_artifact

SHA = {letter: letter * 64 for letter in "abcdef"}


def test_noise_change_invalidates_likelihood_only() -> None:
    common = dict(forward_hash=SHA["a"], observation_hash=SHA["b"], operator_hash=SHA["c"])
    assert likelihood_key(**common, noise_hash=SHA["d"]) != likelihood_key(
        **common, noise_hash=SHA["e"]
    )
    assert forward_key(
        physical={"grid": [2, 2, 1], "controls": [1.0]},
        solver_hash=SHA["a"],
        output_request={"state_times_s": [0.0]},
        adapter_hash=SHA["b"],
        environment_lock_hash=SHA["c"],
    ) == forward_key(
        physical={"grid": [2, 2, 1], "controls": [1.0]},
        solver_hash=SHA["a"],
        output_request={"state_times_s": [0.0]},
        adapter_hash=SHA["b"],
        environment_lock_hash=SHA["c"],
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"grid": [3, 2, 1], "controls": [1.0]},
        {"grid": [2, 2, 1], "controls": [2.0]},
        {"grid": [2, 2, 1], "controls": [1.0], "fluids": "different"},
    ],
)
def test_a_physical_change_invalidates_the_forward_key(changed: dict[str, object]) -> None:
    common: dict[str, Any] = dict(
        solver_hash=SHA["a"],
        output_request={"state_times_s": [0.0, 1.0]},
        adapter_hash=SHA["b"],
        environment_lock_hash=SHA["c"],
    )
    original = forward_key(physical={"grid": [2, 2, 1], "controls": [1.0]}, **common)
    assert forward_key(physical=changed, **common) != original


def test_solver_output_and_lock_changes_each_invalidate_f() -> None:
    base: dict[str, Any] = dict(
        physical={"case": SHA["a"]},
        solver_hash=SHA["b"],
        output_request={"state_times_s": [0.0]},
        adapter_hash=SHA["c"],
        environment_lock_hash=SHA["d"],
    )
    original = forward_key(**base)
    for field, value in (
        ("solver_hash", SHA["e"]),
        ("output_request", {"state_times_s": [0.0, 1.0]}),
        ("adapter_hash", SHA["e"]),
        ("environment_lock_hash", SHA["e"]),
    ):
        assert forward_key(**{**base, field: value}) != original


def test_invalid_digest_is_refused_instead_of_becoming_a_key() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        likelihood_key(
            forward_hash="not-a-sha",
            observation_hash=SHA["b"],
            operator_hash=SHA["c"],
            noise_hash=SHA["d"],
        )


def test_cache_reuses_only_the_verified_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    artifact_path = root / "artifacts" / "value.json"
    ref = write_artifact(
        artifact_path,
        b'{"value":1}\n',
        paths,
        schema_version="test-1",
        producer_run_id="run-1",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    cache = ArtifactCache(paths, "forward")
    key = SHA["a"]
    assert cache.get(key) is None
    cache.put(key, ref)
    assert cache.get(key) == ref

    artifact_path.write_bytes(b'{"value":2}\n')
    with pytest.raises(CacheIntegrityError, match="sha256"):
        cache.get(key)


def test_cache_refuses_a_reference_with_the_wrong_size(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    ref = write_artifact(
        root / "artifacts" / "value",
        b"value",
        paths,
        schema_version="test-1",
        producer_run_id="run-1",
        media_type="application/octet-stream",
        now=datetime.now(UTC),
    ).model_copy(update={"size_bytes": 999})
    with pytest.raises(CacheIntegrityError, match="sha256 and size"):
        ArtifactCache(paths, "forward").put(SHA["a"], ref)


def test_cache_refuses_a_different_value_for_an_existing_key(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    first = write_artifact(
        root / "artifacts" / "first",
        b"first",
        paths,
        schema_version="test-1",
        producer_run_id="run-1",
        media_type="application/octet-stream",
        now=datetime.now(UTC),
    )
    second = write_artifact(
        root / "artifacts" / "second",
        b"second",
        paths,
        schema_version="test-1",
        producer_run_id="run-2",
        media_type="application/octet-stream",
        now=datetime.now(UTC),
    )
    cache = ArtifactCache(paths, "likelihood")
    cache.put(SHA["a"], first)
    with pytest.raises(CacheIntegrityError, match="already maps"):
        cache.put(SHA["a"], second)
