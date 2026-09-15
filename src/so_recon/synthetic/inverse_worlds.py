"""Leakage-safe construction of the first E02 exploratory inverse world."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from so_recon.geology.conditional import LOG_K_QUANTITY, p1_prior_context
from so_recon.inference.contracts import ObservationBundle, PriorContext
from so_recon.observation.bins import rounding_grid
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.synthetic.p1 import P1Design, render_p1
from so_recon.synthetic.world_io import write_world

INFERENCE_INPUT_VERSION = "e02-inference-input-1"
INFERENCE_ALLOWLIST = frozenset({"context", "G", "U", "observations", "density_schema", "basis"})
FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_ref",
        "theta",
        "theta_ref",
        "seed",
        "parent_seed",
        "initial_pressure_pa",
        "full_permeability",
        "full_porosity",
        "full_state",
    }
)


def _refuse_hidden_state(value: object, *, location: str = "input") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"{location}.{key}: truth or hidden generator state is forbidden")
            _refuse_hidden_state(item, location=f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _refuse_hidden_state(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and any(
        marker in value.lower().replace("\\", "/")
        for marker in ("/truth/", "/theta.json", "/states.h5")
    ):
        raise ValueError(f"{location}: truth artifact path is forbidden")


def inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete encoder/inference input against a recursive allowlist."""
    _refuse_hidden_state(payload)
    unexpected = sorted(set(payload) - INFERENCE_ALLOWLIST)
    if unexpected:
        raise ValueError(f"unexpected inference input keys: {unexpected}")
    return payload


def make_inverse_world(
    design_id: str,
    seed: int,
    paths: ProjectPaths,
    ctx: RunContext,
) -> tuple[PriorContext, ObservationBundle, ArtifactRef]:
    """Publish T1 generator truth while returning only conditional context and observations.

    Dynamic history is generated only after the truth forward runs; the bundle returned here
    is therefore explicitly empty rather than carrying guessed watercut.  The truth artifact
    is a separate evaluator output and is never embedded in that bundle.
    """
    if design_id != "e02-t1-v1":
        raise ValueError(
            f"design {design_id!r} has no registered renderer; available generator is 'e02-t1-v1'"
        )
    design = P1Design(family="base")
    world = render_p1(seed, design)
    log_k = {
        str(row["observation_id"]): float(row["value"])
        for row in world.static_observations.to_pylist()
        if row["quantity"] == LOG_K_QUANTITY
    }
    context = p1_prior_context(design, log_k)
    grid = rounding_grid(0.01)
    observations = ObservationBundle(
        history=(),
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(value) for value in grid.edges)},
        cutoff_s=design.report_edges_s[-1],
        information_hash=context.information_hash,
        observation_hash=sha256_json(
            {
                "status": "DYNAMIC_HISTORY_PENDING_TRUTH_FORWARD",
                "design_id": design_id,
                "information_hash": context.information_hash,
            }
        ),
    )
    truth_ref = write_world(world, None, paths, ctx)
    return context, observations, truth_ref


__all__ = [
    "INFERENCE_ALLOWLIST",
    "INFERENCE_INPUT_VERSION",
    "inference_payload",
    "make_inverse_world",
]
