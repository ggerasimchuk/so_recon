"""Immutable registered SMC checkpoints with independently verified data shards."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from so_recon.inference.contracts import SMCState
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactRef,
    register_artifact,
    write_artifact,
    write_json_artifact,
)
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import write_arrays

CHECKPOINT_SCHEMA_VERSION = "e02-smc-checkpoint-1"
PARTICLES_SCHEMA_VERSION = "e02-smc-particles-1"
ARRAYS_SCHEMA_VERSION = "e02-smc-arrays-1"


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint manifest, shard or declared run identity failed verification."""


def _verify_ref(ref: ArtifactRef, paths: ProjectPaths, *, role: str) -> Path:
    try:
        path = paths.resolve(ref.path)
    except ValueError as exc:
        raise CheckpointIntegrityError(f"{role} path is unusable: {exc}") from exc
    if not path.is_file():
        raise CheckpointIntegrityError(f"{role} {ref.path} is missing")
    actual = sha256_file(path)
    if actual != ref.sha256:
        raise CheckpointIntegrityError(
            f"{role} {ref.path} sha256 is {actual}, manifest declares {ref.sha256}"
        )
    if path.stat().st_size != ref.size_bytes:
        raise CheckpointIntegrityError(f"{role} {ref.path} size does not match {ref.size_bytes}")
    return path


def _particle_bytes(state: SMCState) -> bytes:
    rows = [
        {
            "particle_id": particle.particle_id,
            "ancestor_id": particle.ancestor_id,
            "particle_json": json.dumps(
                particle.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
        for particle in state.particles
    ]
    schema = pa.schema(
        [
            pa.field("particle_id", pa.int64(), nullable=False),
            pa.field("ancestor_id", pa.int64(), nullable=False),
            pa.field("particle_json", pa.string(), nullable=False),
        ]
    )
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink, compression="snappy")
    payload: bytes = sink.getvalue().to_pybytes()
    return payload


def _arrays(state: SMCState) -> dict[str, tuple[np.ndarray, str, tuple[str, ...]]]:
    n = len(state.particles)
    n_v = len(state.particles[0].evaluation.theta.v) if n else 0
    n_z = len(state.particles[0].evaluation.theta.z_perp) if n else 0
    v = np.asarray(
        [particle.evaluation.theta.v for particle in state.particles], dtype=np.float64
    ).reshape(n, n_v)
    z = np.asarray(
        [particle.evaluation.theta.z_perp for particle in state.particles],
        dtype=np.float64,
    ).reshape(n, n_z)
    log_values = np.asarray(
        [
            (
                particle.evaluation.log_p0,
                particle.evaluation.log_l,
                particle.evaluation.log_r,
            )
            for particle in state.particles
        ],
        dtype=np.float64,
    ).reshape(n, 3)
    return {
        "v": (v, "1", ("particle", "v_coordinate")),
        "z_perp": (z, "1", ("particle", "residual_coordinate")),
        "log_values": (log_values, "1", ("particle", "log_component")),
        "log_weights": (
            np.asarray(state.log_weights, dtype=np.float64),
            "1",
            ("particle",),
        ),
    }


def save_state(
    state: SMCState,
    path: Path,
    paths: ProjectPaths,
    ctx: RunContext,
) -> ArtifactRef:
    """Write Parquet and HDF5 first, then atomically publish their manifest last."""
    if path.name != "manifest.json":
        raise ValueError("a checkpoint path must end in manifest.json")
    root = path.parent
    particles_ref = write_artifact(
        root / "particles.parquet",
        _particle_bytes(state),
        paths,
        schema_version=PARTICLES_SCHEMA_VERSION,
        producer_run_id=ctx.run_id,
        media_type="application/vnd.apache.parquet",
        now=datetime.now(UTC),
    )
    write_arrays(root / "arrays.h5", _arrays(state), paths=paths)
    arrays_ref = register_artifact(
        root / "arrays.h5",
        paths,
        schema_version=ARRAYS_SCHEMA_VERSION,
        producer_run_id=ctx.run_id,
        media_type="application/x-hdf5",
        now=datetime.now(UTC),
    )
    prior = ctx.record.outputs.get("smc_checkpoint")
    identities = {
        "target_hash": state.target_hash,
        "proposal_hash": state.proposal_hash,
        "basis_hash": state.diagnostics.get("basis_hash"),
        "config_hash": state.diagnostics.get("config_hash"),
        **dict(state.diagnostics.get("target_components", {})),
    }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": state.model_dump(mode="json"),
        "identities": identities,
        "shards": {
            "particles": particles_ref.model_dump(mode="json"),
            "arrays": arrays_ref.model_dump(mode="json"),
        },
        "previous_checkpoint_artifact_id": prior.artifact_id if prior else None,
    }
    parent_ids = [particles_ref.artifact_id, arrays_ref.artifact_id]
    if prior is not None:
        parent_ids.append(prior.artifact_id)
    manifest_ref = write_json_artifact(
        path,
        payload,
        paths,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=parent_ids,
        now=datetime.now(UTC),
    )
    ctx.add_output("smc_checkpoint", manifest_ref)
    return manifest_ref


def _validate_particles(path: Path, state: SMCState) -> None:
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise CheckpointIntegrityError(f"particles shard cannot be read: {exc}") from exc
    if len(rows) != len(state.particles):
        raise CheckpointIntegrityError(
            f"particles shard has {len(rows)} rows for {len(state.particles)} state particles"
        )
    for expected, row in zip(state.particles, rows, strict=True):
        try:
            actual = json.loads(str(row["particle_json"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(f"invalid particle row: {exc}") from exc
        if actual != expected.model_dump(mode="json"):
            raise CheckpointIntegrityError(
                f"particle row {row.get('particle_id')} disagrees with manifest state"
            )
        if (
            row.get("particle_id") != expected.particle_id
            or row.get("ancestor_id") != expected.ancestor_id
        ):
            raise CheckpointIntegrityError(
                f"particle metadata columns disagree for particle {expected.particle_id}"
            )


def _validate_arrays(path: Path, state: SMCState) -> None:
    expected = _arrays(state)
    try:
        with h5py.File(path, "r") as handle:
            if set(handle) != set(expected):
                raise CheckpointIntegrityError(
                    f"arrays shard datasets {sorted(handle)} do not match {sorted(expected)}"
                )
            for name, (values, _, _) in expected.items():
                actual = np.asarray(handle[name])
                if actual.shape != values.shape or not np.array_equal(actual, values):
                    raise CheckpointIntegrityError(
                        f"arrays shard dataset {name!r} disagrees with manifest state"
                    )
    except CheckpointIntegrityError:
        raise
    except Exception as exc:
        raise CheckpointIntegrityError(f"arrays shard cannot be read: {exc}") from exc


def load_state(
    ref: ArtifactRef,
    paths: ProjectPaths,
    expected_hashes: dict[str, str],
) -> SMCState:
    """Verify manifest identity and every referenced byte before returning a state."""
    manifest_path = _verify_ref(ref, paths, role="checkpoint manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                f"unsupported checkpoint schema {payload.get('schema_version')!r}"
            )
        state = SMCState.model_validate(payload["state"])
        identities = payload["identities"]
        particle_ref = ArtifactRef.model_validate(payload["shards"]["particles"])
        arrays_ref = ArtifactRef.model_validate(payload["shards"]["arrays"])
    except CheckpointIntegrityError:
        raise
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointIntegrityError(f"invalid checkpoint manifest: {exc}") from exc
    for name, expected in expected_hashes.items():
        actual = identities.get(name)
        if actual != expected:
            raise CheckpointIntegrityError(
                f"{name} mismatch: checkpoint {actual!r}, expected {expected!r}"
            )
    state_identities = {
        "target_hash": state.target_hash,
        "proposal_hash": state.proposal_hash,
        "basis_hash": state.diagnostics.get("basis_hash"),
        "config_hash": state.diagnostics.get("config_hash"),
        **dict(state.diagnostics.get("target_components", {})),
    }
    if identities != state_identities:
        raise CheckpointIntegrityError(
            "checkpoint identities disagree with the hashes carried by its state"
        )
    particle_path = _verify_ref(particle_ref, paths, role="particles shard")
    arrays_path = _verify_ref(arrays_ref, paths, role="arrays shard")
    _validate_particles(particle_path, state)
    _validate_arrays(arrays_path, state)
    return state


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointIntegrityError",
    "load_state",
    "save_state",
]
