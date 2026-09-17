"""E03 Task 05 — non-pickle frozen checkpoints (plan §7.5): safetensors + JSON.

The reproducible scientific checkpoint of the learned proposal is three byte streams:
Float64 tensor weights in `weights.safetensors`, the flow/encoder architecture in
`flow_config.json` and the feature scaler in `scaler.json`. Nothing here is pickled, so
nothing here can smuggle executable state, and every stream is identified by the SHA-256
of its bytes — the hashes a `ProposalManifest` vouches for and `bind_frozen_proposal`
demands back.

Float32-trained parameters are exported AS Float64: the operational law is the fully
pinned Float64 model (§7.5), and a training-time optimizer state, if one is ever kept,
is a runtime artifact beside these files, never instead of them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load as load_tensors
from safetensors.numpy import save as save_tensors

from so_recon.registry.hashing import sha256_bytes

WEIGHTS_SCHEMA = "e03-nsf-weights-1"
FLOW_CONFIG_SCHEMA = "e03-flow-config-1"
SCALER_SCHEMA = "e03-scaler-1"

WEIGHTS_FILENAME = "weights.safetensors"
FLOW_CONFIG_FILENAME = "flow_config.json"
SCALER_FILENAME = "scaler.json"


class CheckpointMismatch(ValueError):
    """A checkpoint stream does not carry the digest it was vouched for."""


@dataclass(frozen=True)
class CheckpointHashes:
    """The digests of the three checkpoint byte streams."""

    weights_sha256: str
    flow_config_sha256: str
    scaler_sha256: str


@dataclass(frozen=True)
class FrozenCheckpoint:
    """The loaded, hash-attributed checkpoint: Float64 tensors, config, scaler."""

    tensors: dict[str, np.ndarray]
    flow_config: dict[str, Any]
    scaler_payload: dict[str, Any]
    hashes: CheckpointHashes


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _module_tensors(module: torch.nn.Module) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for name, tensor in module.state_dict().items():
        array = tensor.detach().cpu().numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float64)
        tensors[str(name)] = np.ascontiguousarray(array)
    return tensors


def serialize_checkpoint(
    module: torch.nn.Module,
    *,
    flow_config: Mapping[str, Any],
    scaler_payload: Mapping[str, Any],
) -> tuple[bytes, bytes, bytes]:
    """The three checkpoint byte streams: weights, flow config, scaler."""
    return (
        save_tensors(_module_tensors(module)),
        _canonical_json_bytes(dict(flow_config)),
        _canonical_json_bytes(dict(scaler_payload)),
    )


def checkpoint_hashes(weights: bytes, flow_config: bytes, scaler: bytes) -> CheckpointHashes:
    return CheckpointHashes(
        weights_sha256=sha256_bytes(weights),
        flow_config_sha256=sha256_bytes(flow_config),
        scaler_sha256=sha256_bytes(scaler),
    )


def save_checkpoint(
    root: Path,
    *,
    module: torch.nn.Module,
    flow_config: Mapping[str, Any],
    scaler_payload: Mapping[str, Any],
) -> CheckpointHashes:
    """Write `weights.safetensors`, `flow_config.json`, `scaler.json` under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    weights, config_bytes, scaler_bytes = serialize_checkpoint(
        module, flow_config=flow_config, scaler_payload=scaler_payload
    )
    (root / WEIGHTS_FILENAME).write_bytes(weights)
    (root / FLOW_CONFIG_FILENAME).write_bytes(config_bytes)
    (root / SCALER_FILENAME).write_bytes(scaler_bytes)
    return checkpoint_hashes(weights, config_bytes, scaler_bytes)


def checkpoint_from_bytes(
    weights: bytes,
    flow_config: bytes,
    scaler: bytes,
    *,
    expected: CheckpointHashes | None = None,
) -> FrozenCheckpoint:
    """Load a checkpoint from in-memory bytes, verifying digests when `expected` is given."""
    hashes = checkpoint_hashes(weights, flow_config, scaler)
    if expected is not None:
        _verify(hashes, expected)
    return FrozenCheckpoint(
        tensors=load_tensors(weights),
        flow_config=json.loads(flow_config.decode("utf-8")),
        scaler_payload=json.loads(scaler.decode("utf-8")),
        hashes=hashes,
    )


def load_checkpoint(root: Path, *, expected: CheckpointHashes | None = None) -> FrozenCheckpoint:
    """Read the three streams from `root`, verifying digests when `expected` is given."""
    return checkpoint_from_bytes(
        (root / WEIGHTS_FILENAME).read_bytes(),
        (root / FLOW_CONFIG_FILENAME).read_bytes(),
        (root / SCALER_FILENAME).read_bytes(),
        expected=expected,
    )


def _verify(actual: CheckpointHashes, expected: CheckpointHashes) -> None:
    labels = (
        ("weights", actual.weights_sha256, expected.weights_sha256),
        ("flow config", actual.flow_config_sha256, expected.flow_config_sha256),
        ("scaler", actual.scaler_sha256, expected.scaler_sha256),
    )
    for name, got, want in labels:
        if got != want:
            raise CheckpointMismatch(
                f"the {name} stream fails its published digest (got {got}, vouched for "
                f"{want}): the checkpoint was modified after publication"
            )


__all__ = [
    "CheckpointHashes",
    "CheckpointMismatch",
    "FLOW_CONFIG_FILENAME",
    "FLOW_CONFIG_SCHEMA",
    "FrozenCheckpoint",
    "SCALER_FILENAME",
    "SCALER_SCHEMA",
    "WEIGHTS_FILENAME",
    "WEIGHTS_SCHEMA",
    "checkpoint_from_bytes",
    "checkpoint_hashes",
    "load_checkpoint",
    "save_checkpoint",
    "serialize_checkpoint",
]
