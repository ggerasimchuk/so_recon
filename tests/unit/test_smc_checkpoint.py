"""Registered SMC checkpoints validate every shard before restoring a state."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from so_recon.config.inference import InferenceConfig
from so_recon.inference.checkpoint import CheckpointIntegrityError, load_state, save_state
from so_recon.inference.contracts import SMCState
from so_recon.inference.smc import infer
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.run import RunContext
from tests.unit.test_smc_engine import SCHEMA, never_stop, problem


class Context:
    def __init__(self, root: Path) -> None:
        self.run_id = "checkpoint-test"
        self.run_dir = root / "artifacts" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.record = SimpleNamespace(outputs={})

    def add_output(self, key: str, ref: ArtifactRef) -> None:
        self.record.outputs[key] = ref


def completed_state(tmp_path: Path) -> tuple[SMCState, object, object, InferenceConfig]:
    target, proposal, config = problem()
    state = infer(target, proposal, SCHEMA, config, tmp_path / "working", never_stop)
    return state, target, proposal, config


def expected(state: SMCState, config: InferenceConfig) -> dict[str, str]:
    from so_recon.registry.hashing import sha256_json

    return {
        "target_hash": state.target_hash,
        "proposal_hash": state.proposal_hash,
        "basis_hash": SCHEMA.basis_hash,
        "config_hash": sha256_json(config.model_dump(mode="json")),
    }


def test_checkpoint_round_trip_validates_manifest_parquet_and_hdf5(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    state, _, _, config = completed_state(tmp_path)
    ctx = Context(root)
    ref = save_state(
        state,
        ctx.run_dir / "checkpoint-001" / "manifest.json",
        paths,
        cast(RunContext, ctx),
    )
    restored = load_state(ref, paths, expected(state, config))
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    assert set(ctx.record.outputs) == {"smc_checkpoint"}


def test_corrupt_shard_is_never_loaded_from_a_valid_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    state, _, _, config = completed_state(tmp_path)
    ctx = Context(root)
    ref = save_state(
        state,
        ctx.run_dir / "checkpoint-001" / "manifest.json",
        paths,
        cast(RunContext, ctx),
    )
    particles = ref.path.replace("manifest.json", "particles.parquet")
    paths.resolve(particles).write_bytes(b"corrupt")
    with pytest.raises(CheckpointIntegrityError, match="sha256"):
        load_state(ref, paths, expected(state, config))


@pytest.mark.parametrize("field", ["target_hash", "proposal_hash", "basis_hash", "config_hash"])
def test_incompatible_checkpoint_identity_is_refused(tmp_path: Path, field: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    state, _, _, config = completed_state(tmp_path)
    ctx = Context(root)
    ref = save_state(
        state,
        ctx.run_dir / "checkpoint-001" / "manifest.json",
        paths,
        cast(RunContext, ctx),
    )
    mismatched = expected(state, config)
    mismatched[field] = "f" * 64
    with pytest.raises(CheckpointIntegrityError, match=field):
        load_state(ref, paths, mismatched)


def test_partial_next_snapshot_does_not_damage_previous_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    state, _, _, config = completed_state(tmp_path)
    ctx = Context(root)
    first = save_state(
        state,
        ctx.run_dir / "checkpoint-001" / "manifest.json",
        paths,
        cast(RunContext, ctx),
    )
    partial = ctx.run_dir / "checkpoint-002"
    partial.mkdir()
    (partial / "particles.parquet").write_bytes(b"partial-uncommitted-shard")
    assert not (partial / "manifest.json").exists()
    restored = load_state(first, paths, expected(state, config))
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")


def test_an_accepted_snapshot_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    state, _, _, _ = completed_state(tmp_path)
    ctx = Context(root)
    manifest = ctx.run_dir / "checkpoint-001" / "manifest.json"
    save_state(state, manifest, paths, cast(RunContext, ctx))
    changed = state.model_copy(update={"level": state.level + 1})
    with pytest.raises(Exception, match="overwrite"):
        save_state(changed, manifest, paths, cast(RunContext, ctx))
