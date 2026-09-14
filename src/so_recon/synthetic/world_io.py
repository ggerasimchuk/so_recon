"""E01.11 — a P1 world on disk: the case, the truth, the view, and the boundary between them.

`p1.py` renders a world. This module is where it becomes files, and it exists because the
separation between the SOLVER's input and the INVERSE PROBLEM's input is a property of the
file layout, not of anyone's discipline:

    artifacts/worlds/<parent_world_id>/
        grid.h5                          geometry: public
        truth/geology.h5                 full-field porosity, permeability, latent field
        truth/initial.h5                 the initial pressure and saturation
        truth/theta.json                 the generator's parameters
        truth/support_truth.parquet      the NOISELESS support averages
        truth/truth_manifest.json        the evaluator's entry point
        view/<derived_view_id>/observations.parquet   sparse, noisy G
        view/<derived_view_id>/controls.json          the policy that was run
        view/<derived_view_id>/context.json           what an encoder may read
        world_manifest.json

The `CaseBundle` the solver is given references `truth/geology.h5` and `truth/initial.h5`,
because a forward operator cannot run without the full rock and the full initial state.
That bundle is never handed to an encoder. What an encoder is handed is `context.json`, and
`context_payload` is the ONLY way its content is produced: it is an allowlist projection of
the world manifest, so a field that is not in `CONTEXT_ALLOWLIST` cannot reach the file
however the manifest grows. `_refuse_truth_reference` is the second lock — it re-reads the
bytes about to be written and refuses any that name the truth directory.

Immutability is the registry's, not a second one: every array goes through
`case_io.write_arrays` and every table and record through `registry.artifact.write_artifact`,
both of which reuse identical bytes and refuse different ones at a path that is taken. That
is what «checksum reuse only for identical arrays and config» means here — a repeat of the
same seed and design rewrites nothing and a changed one cannot claim the same name, because
the name carries a digest of the generator, the design and the seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactRef,
    register_artifact,
    write_artifact,
    write_json_artifact,
)
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import canonical_json, sha256_bytes
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import (
    CASE_MANIFEST_FILENAME,
    cartesian_neighbors,
    compute_model_hash,
    validate_case,
    write_arrays,
)
from so_recon.simulator.contracts import (
    CELL_AXES,
    CELL_DIM_AXES,
    DIM_CELL_AXES,
    FACE_AXES,
    ArrayRef,
    CaseBundle,
    ForwardResult,
    ForwardStatus,
    GridSpec,
    InitialStateSpec,
    ObservationSpec,
    RockSpec,
)
from so_recon.simulator.results import RESULT_FILENAME
from so_recon.synthetic.p1 import (
    GENERATOR_VERSION,
    RenderedWorld,
    control_segments,
    observation_masks,
    support_truth,
)

WORLD_SCHEMA_VERSION: Literal["world-1"] = "world-1"
TRUTH_SCHEMA_VERSION = "p1-truth-1"
CONTROLS_SCHEMA_VERSION = "p1-controls-1"
CONTEXT_SCHEMA_VERSION = "p1-context-1"
SUITE_SCHEMA_VERSION = "p1-suite-1"

WORLD_MANIFEST_FILENAME = "world_manifest.json"
TRUTH_MANIFEST_FILENAME = "truth_manifest.json"
CONTEXT_FILENAME = "context.json"
CONTROLS_FILENAME = "controls.json"
OBSERVATIONS_FILENAME = "observations.parquet"
SUPPORT_TRUTH_FILENAME = "support_truth.parquet"
THETA_FILENAME = "theta.json"
GRID_FILENAME = "grid.h5"
GEOLOGY_FILENAME = "geology.h5"
INITIAL_FILENAME = "initial.h5"
SUITE_MANIFEST_FILENAME = "p1_suite_manifest.json"

WORLDS_DIRNAME = "worlds"
TRUTH_DIRNAME = "truth"
VIEW_DIRNAME = "view"

#: The view every parent of this plan is rendered into, and the split it belongs to. A
#: prefix of the calendar or another noise realisation would be a DIFFERENT derived view
#: over the SAME parent, which is why neither takes part in `parent_world_id`.
DERIVED_VIEW_ID = "full36-v1"
SPLIT = "exploratory"

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE = "application/json"

#: The whole of what a context file may carry (plan 11.7), verbatim. This tuple is the
#: MECHANISM and not a comment: `context_payload` projects onto it, and `write_world` has no
#: other way to produce the bytes it writes, so a manifest field that is not named here is
#: structurally unable to reach an encoder.
CONTEXT_ALLOWLIST: tuple[str, ...] = (
    "parent_world_id",
    "design_id",
    "derived_view_id",
    "split",
    "observation_ref",
    "control_ref",
    "observation_masks",
    "cutoff",
)


def context_payload(world_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse problem's whole input, projected out of the world manifest.

    A `KeyError` for a missing allowed key is deliberate: a context that silently dropped
    its observation reference would be a context an encoder could not use, and discovering
    that in E02 is worse than discovering it here.
    """
    return {key: world_manifest[key] for key in CONTEXT_ALLOWLIST}


class FileRef(StrictModel):
    """A published file, by repo-relative path and by the digest of its bytes."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class WorldManifest(StrictModel):
    """One world's identity, its references and the outcome of its forward.

    Everything above `provenance` is determined by `(generator version, design, seed)` and
    by what the forward published. `provenance` is the run: a run id, a wall-clock stamp and
    the library versions. Plan §3.1 keeps time, RAM and hostname out of a model hash, and
    they are kept out of the identity here for the same reason.
    """

    schema_version: Literal["world-1"] = WORLD_SCHEMA_VERSION
    spec_version: Literal["4.0"] = "4.0"
    generator_version: str
    parent_world_id: str
    design_id: str
    derived_view_id: str
    split: str
    family: str
    seed: int = Field(ge=0)
    case_id: str
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_date: str
    cutoff: str
    observation_ref: FileRef
    control_ref: FileRef
    truth_ref: FileRef
    case_ref: FileRef | None
    observation_masks: dict[str, Any]
    forward_status: ForwardStatus | None
    forward_reason: str | None
    forward_result_ref: FileRef | None
    forward_outputs: dict[str, str]
    cost: dict[str, Any] | None
    #: The forward completed and published every output its request asked for. The PHYSICS
    #: gates are scored separately and recorded on the suite row: a world can be `accepted`
    #: here and still fail a balance threshold, and saying so needs both numbers visible.
    accepted: bool
    provenance: dict[str, Any]


class WorldLocations:
    """Where one world's files live. Constructed from the project paths and the parent id."""

    def __init__(self, paths: ProjectPaths, parent_world_id: str, view_id: str) -> None:
        self.root = paths.artifacts / WORLDS_DIRNAME / parent_world_id
        self.truth = self.root / TRUTH_DIRNAME
        self.view = self.root / VIEW_DIRNAME / view_id
        self.grid_file = self.root / GRID_FILENAME
        self.geology_file = self.truth / GEOLOGY_FILENAME
        self.initial_file = self.truth / INITIAL_FILENAME
        self.theta_file = self.truth / THETA_FILENAME
        self.support_truth_file = self.truth / SUPPORT_TRUTH_FILENAME
        self.truth_manifest_file = self.truth / TRUTH_MANIFEST_FILENAME
        self.observations_file = self.view / OBSERVATIONS_FILENAME
        self.controls_file = self.view / CONTROLS_FILENAME
        self.context_file = self.view / CONTEXT_FILENAME
        self.manifest_file = self.root / WORLD_MANIFEST_FILENAME


def world_locations(
    paths: ProjectPaths, parent_world_id: str, *, view_id: str = DERIVED_VIEW_ID
) -> WorldLocations:
    return WorldLocations(paths, parent_world_id, view_id)


# --------------------------------------------------------------------------------------
# publication helpers
# --------------------------------------------------------------------------------------


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression="snappy")
    payload: bytes = buffer.getvalue().to_pybytes()
    return payload


def _publish_parquet(
    path: Path,
    table: pa.Table,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    key: str,
    schema_version: str,
    now: datetime,
) -> FileRef:
    payload = _parquet_bytes(table)
    ref = write_artifact(
        path,
        payload,
        paths,
        schema_version=schema_version,
        producer_run_id=ctx.run_id,
        media_type=PARQUET_MEDIA_TYPE,
        now=now,
    )
    ctx.add_output(key, ref)
    return FileRef(path=ref.path, sha256=ref.sha256, size_bytes=ref.size_bytes)


def _publish_json(
    path: Path,
    payload: object,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    key: str,
    schema_version: str,
    now: datetime,
) -> tuple[ArtifactRef, FileRef]:
    ref = write_json_artifact(
        path, payload, paths, schema_version=schema_version, producer_run_id=ctx.run_id, now=now
    )
    ctx.add_output(key, ref)
    return ref, FileRef(path=ref.path, sha256=ref.sha256, size_bytes=ref.size_bytes)


def _publish_run_record(
    path: Path,
    payload: object,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    key: str,
    schema_version: str,
    now: datetime,
) -> tuple[ArtifactRef, FileRef]:
    """Write a record that is a RUN's, not a world's, and register the bytes it left.

    The deterministic files of a world go through the content-addressed writer above and are
    reused byte for byte when the same seed and design are rendered again. Two records
    cannot: `world_manifest.json` carries the run id and the wall-clock stamp, and
    `truth_manifest.json` names the outputs of whichever forward produced them. Publishing
    those through the immutable writer would make a second run of the same world fail on the
    stamp rather than on anything scientific, so they are written in place and registered.
    Everything an encoder ever sees — `context.json` included — is on the immutable side.
    """
    write_json_atomic(path, payload)
    ref = register_artifact(
        path,
        paths,
        schema_version=schema_version,
        producer_run_id=ctx.run_id,
        media_type=JSON_MEDIA_TYPE,
        now=now,
    )
    ctx.add_output(key, ref)
    return ref, FileRef(path=ref.path, sha256=ref.sha256, size_bytes=ref.size_bytes)


def _file_ref(path: Path, paths: ProjectPaths) -> FileRef | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    return FileRef(path=paths.relative(path), sha256=sha256_bytes(payload), size_bytes=len(payload))


# --------------------------------------------------------------------------------------
# the solver's case
# --------------------------------------------------------------------------------------


def build_p1_case(world: RenderedWorld, paths: ProjectPaths, ctx: RunContext) -> CaseBundle:
    """Publish the world's arrays and assemble the `CaseBundle` the solver is given.

    Idempotent by construction: every file goes through the content-addressed writers, so
    calling this twice for one world rewrites nothing and returns the same model hash.

    This bundle carries the WHOLE truth — full permeability, full porosity, the full initial
    pressure and saturation — because the forward operator reads all of it. It is the object
    that must never be handed to an encoder, and the encoder's own input is built by
    `write_world` out of `context_payload` instead.
    """
    design = world.design
    now = datetime.now(UTC)
    loc = world_locations(paths, world.parent_world_id)
    arrays = world.arrays

    grid = write_arrays(
        loc.grid_file,
        {
            "cell_centers_m": (arrays["cell_centers_m"], "m", CELL_DIM_AXES),
            "cell_volume_m3": (arrays["cell_volume_m3"], "m3", CELL_AXES),
            "neighbors": (cartesian_neighbors(design.shape), "1", FACE_AXES),
        },
        paths=paths,
    )
    geology = write_arrays(
        loc.geology_file,
        {
            "porosity": (arrays["porosity"], "1", CELL_AXES),
            "permeability_m2": (arrays["permeability_m2"], "m2", DIM_CELL_AXES),
            # A log permeability is dimensionless as far as the exchange's unit vocabulary
            # goes; the quantity it is the log OF is stated in the dataset's own name.
            "log_permeability_m2": (arrays["log_permeability_m2"], "1", CELL_AXES),
            "generator_field": (arrays["generator_field"], "1", CELL_AXES),
        },
        paths=paths,
    )
    initial = write_arrays(
        loc.initial_file,
        {
            "pressure_pa": (arrays["pressure_pa"], "Pa", CELL_AXES),
            "sw": (arrays["sw"], "1", CELL_AXES),
        },
        paths=paths,
    )
    observations = _publish_parquet(
        loc.observations_file,
        world.static_observations,
        paths,
        ctx,
        key=f"world.{world.parent_world_id}.observations",
        schema_version=CONTEXT_SCHEMA_VERSION,
        now=now,
    )

    fields = world.case_fields
    case = CaseBundle(
        case_id=str(fields["case_id"]),
        world_id=str(fields["world_id"]),
        start_date=str(fields["start_date"]),
        cutoff=str(fields["cutoff"]),
        report_edges_s=tuple(fields["report_edges_s"]),
        grid=GridSpec(
            shape=design.shape,
            extent_m=design.extent_m,
            cell_centers_m=grid["cell_centers_m"],
            cell_volume_m3=grid["cell_volume_m3"],
            neighbors=grid["neighbors"],
        ),
        rock=RockSpec(porosity=geology["porosity"], permeability_m2=geology["permeability_m2"]),
        fluids=fields["fluids"],
        wells=tuple(fields["wells"]),
        controls=tuple(fields["controls"]),
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=initial["pressure_pa"],
            sw=initial["sw"],
            # Generated, not observed: this is the state the world STARTS from, and it is
            # not a late-life field oil saturation wearing that name (plan 11.4).
            meaning="synthetic_initial",
        ),
        boundary=fields["boundary"],
        observations=ObservationSpec(
            table_path=observations.path,
            sha256=observations.sha256,
            dynamic_channels=tuple(fields["dynamic_channels"]),
            pressure_available=bool(fields["pressure_available"]),
        ),
        renderer_version=str(fields["renderer_version"]),
        units=dict(fields["units"]),
        seeds=dict(fields["seeds"]),
        source_hashes=dict(fields["source_hashes"]),
        model_hash="e" * 64,
    )
    case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    report = validate_case(case, paths)
    if not report.valid:
        raise ValueError(
            f"the rendered world {world.parent_world_id} does not produce a valid case: "
            + "; ".join(report.errors)
        )
    return case


# --------------------------------------------------------------------------------------
# the world on disk
# --------------------------------------------------------------------------------------


def _controls_payload(world: RenderedWorld) -> dict[str, Any]:
    """The policy that was run, as data. It is context: an operator knows its own schedule."""
    design = world.design
    return {
        "schema_version": CONTROLS_SCHEMA_VERSION,
        "parent_world_id": world.parent_world_id,
        "design_id": design.design_id,
        "start_date": design.start_date,
        "cutoff": design.cutoff,
        "report_edges_s": list(design.report_edges_s),
        "units": {"control_rate": "m3_sc/day", "pressure": "Pa", "time": "s"},
        "wells": [
            {
                "well_id": well.well_id,
                "cells": list(well.cells),
                "radius_m": well.radius_m,
                "reference_depth_m": well.reference_depth_m,
                "model": well.model,
                "allow_crossflow": well.allow_crossflow,
            }
            for well in world.case_fields["wells"]
        ],
        "segments": [segment.model_dump(mode="json") for segment in control_segments(design)],
    }


def _forward_outputs(forward: ForwardResult | None) -> dict[str, str]:
    """The published artifacts of a forward, by role. Empty when nothing was published."""
    if forward is None:
        return {}
    outputs: dict[str, str] = {}
    for role, value in (
        ("monthly", forward.monthly_path),
        ("connections", forward.connections_path),
        ("balances", forward.balances_path),
    ):
        if value is not None:
            outputs[role] = value
    for name, ref in sorted(forward.states.items()):
        outputs[f"state.{name}"] = f"{ref.path}#{ref.dataset}"
    if forward.restart is not None:
        outputs["restart_manifest"] = forward.restart.manifest_path
    return outputs


def _truth_payload(
    world: RenderedWorld,
    case: CaseBundle,
    forward: ForwardResult | None,
    theta_ref: FileRef,
    support_ref: FileRef,
) -> dict[str, Any]:
    def described(ref: ArrayRef) -> dict[str, Any]:
        return {
            "path": ref.path,
            "dataset": ref.dataset,
            "sha256": ref.sha256,
            "shape": list(ref.shape),
            "unit": ref.unit,
        }

    pressure, sw = case.initial.pressure_pa, case.initial.sw
    if pressure is None or sw is None:
        raise ValueError(
            f"{world.parent_world_id}: a P1 case carries an EXPLICIT initial state; the truth "
            "manifest has nothing to point the evaluator at without it"
        )
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "warning": (
            "This is the TRUTH of a synthetic world. It is the evaluator's reference and is "
            "never handed to an encoder or to a data loader; the inverse problem's input is "
            "the view's context.json (ObservationSpec.truth_access is 'forbidden')"
        ),
        "generator_version": GENERATOR_VERSION,
        "parent_world_id": world.parent_world_id,
        "design_id": world.design.design_id,
        "family": world.design.family,
        "seed": world.seed,
        "case_id": case.case_id,
        "model_hash": case.model_hash,
        "theta_ref": theta_ref.model_dump(mode="json"),
        "support_truth_ref": support_ref.model_dump(mode="json"),
        "arrays": {
            "porosity": described(case.rock.porosity),
            "permeability_m2": described(case.rock.permeability_m2),
            "initial_pressure_pa": described(pressure),
            "initial_sw": described(sw),
            "cell_centers_m": described(case.grid.cell_centers_m),
        },
        "forward": {
            "status": None if forward is None else forward.status,
            "reason": None if forward is None else forward.reason,
            "times_s": [] if forward is None else list(forward.times_s),
            "outputs": _forward_outputs(forward),
        },
    }


def _refuse_truth_reference(payload: Mapping[str, Any], world: RenderedWorld) -> None:
    """Refuse a context that names the truth, whatever produced it.

    The allowlist already makes this unreachable through `context_payload`. It is checked
    anyway, on the values and not only on the keys: a reference is a path, and an allowed
    key holding a path into `truth/` would be a leak the key names could not show.
    """
    text = canonical_json(payload)
    for forbidden in (f"/{TRUTH_DIRNAME}/", THETA_FILENAME, GEOLOGY_FILENAME, INITIAL_FILENAME):
        if forbidden in text:
            raise ValueError(
                f"the context of {world.parent_world_id} names {forbidden!r}; the inverse "
                "problem's input may not reference the truth"
            )
    if set(payload) != set(CONTEXT_ALLOWLIST):
        raise ValueError(
            f"the context of {world.parent_world_id} carries {sorted(payload)}, and the "
            f"allowlist is {sorted(CONTEXT_ALLOWLIST)}"
        )


def write_world(
    world: RenderedWorld,
    forward: ForwardResult | None,
    paths: ProjectPaths,
    ctx: RunContext,
) -> ArtifactRef:
    """Publish a world: its case, its truth, its view and its manifest, in that order.

    `forward` is the result of running the case, or `None` when it has not been run. A
    result that is NOT `COMPLETE` is published exactly like one that is: the status and the
    reason go on the manifest and the row is kept, because a world that failed is an outcome
    and deleting it would make the suite look better than it is (SPEC §18.4, plan 11.9).

    The manifest is written LAST, after every file it names, so a process lost half-way
    leaves unfinished staging rather than a world that claims to be complete.
    """
    now = datetime.now(UTC)
    loc = world_locations(paths, world.parent_world_id)
    case = build_p1_case(world, paths, ctx)
    key = f"world.{world.parent_world_id}"

    _, theta_ref = _publish_json(
        loc.theta_file,
        world.theta,
        paths,
        ctx,
        key=f"{key}.theta",
        schema_version=TRUTH_SCHEMA_VERSION,
        now=now,
    )
    support_ref = _publish_parquet(
        loc.support_truth_file,
        support_truth(world.arrays, world.design),
        paths,
        ctx,
        key=f"{key}.support_truth",
        schema_version=TRUTH_SCHEMA_VERSION,
        now=now,
    )
    _, control_ref = _publish_json(
        loc.controls_file,
        _controls_payload(world),
        paths,
        ctx,
        key=f"{key}.controls",
        schema_version=CONTROLS_SCHEMA_VERSION,
        now=now,
    )
    _, truth_ref = _publish_run_record(
        loc.truth_manifest_file,
        _truth_payload(world, case, forward, theta_ref, support_ref),
        paths,
        ctx,
        key=f"{key}.truth",
        schema_version=TRUTH_SCHEMA_VERSION,
        now=now,
    )

    table_path, table_sha = case.observations.table_path, case.observations.sha256
    if table_path is None or table_sha is None:
        raise ValueError(
            f"{world.parent_world_id}: the case declares no observations table, so the view "
            "has nothing to hand an encoder"
        )
    observation_ref = FileRef(
        path=table_path,
        sha256=table_sha,
        size_bytes=paths.resolve(table_path).stat().st_size,
    )
    result_record = (
        None
        if forward is None or forward.monthly_path is None
        else _file_ref(paths.resolve(forward.monthly_path).parent / RESULT_FILENAME, paths)
    )
    manifest = WorldManifest(
        generator_version=GENERATOR_VERSION,
        parent_world_id=world.parent_world_id,
        design_id=world.design.design_id,
        derived_view_id=DERIVED_VIEW_ID,
        split=SPLIT,
        family=world.design.family,
        seed=world.seed,
        case_id=case.case_id,
        model_hash=case.model_hash,
        start_date=world.design.start_date,
        cutoff=world.design.cutoff,
        observation_ref=observation_ref,
        control_ref=control_ref,
        truth_ref=truth_ref,
        case_ref=_file_ref(ctx.run_dir / CASE_MANIFEST_FILENAME, paths),
        observation_masks=observation_masks(world.design),
        forward_status=None if forward is None else forward.status,
        forward_reason=None if forward is None else forward.reason,
        forward_result_ref=result_record,
        forward_outputs=_forward_outputs(forward),
        cost=None if forward is None else forward.cost.model_dump(mode="json"),
        accepted=forward is not None and forward.status == "COMPLETE",
        provenance={
            "run_id": ctx.run_id,
            "created_at": now.isoformat(),
            "numpy_version": np.__version__,
            "pyarrow_version": pa.__version__,
        },
    )
    payload = manifest.model_dump(mode="json")
    context = context_payload(payload)
    _refuse_truth_reference(context, world)
    _publish_json(
        loc.context_file,
        context,
        paths,
        ctx,
        key=f"{key}.context",
        schema_version=CONTEXT_SCHEMA_VERSION,
        now=now,
    )
    ref, _ = _publish_run_record(
        loc.manifest_file,
        payload,
        paths,
        ctx,
        key=key,
        schema_version=WORLD_SCHEMA_VERSION,
        now=now,
    )
    return ref


# --------------------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------------------


def world_row(
    world: RenderedWorld,
    forward: ForwardResult | None,
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | None = None,
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One outcome row of the P1 suite, failures included.

    `gates` is what the physics evaluation concluded about this world — the balance numbers,
    the BHP behaviour, the state checks. It is a separate field from `accepted` on purpose:
    a forward can complete and still miss a threshold, and a row that folded the two into
    one verdict would hide which of them happened.
    """
    return {
        "parent_world_id": world.parent_world_id,
        "seed": world.seed,
        "family": world.design.family,
        "design_id": world.design.design_id,
        "derived_view_id": manifest["derived_view_id"],
        "split": manifest["split"],
        "model_hash": manifest["model_hash"],
        "status": manifest["forward_status"],
        "reason": manifest["forward_reason"],
        "accepted": manifest["accepted"],
        "cost": None if forward is None else forward.cost.model_dump(mode="json"),
        "manifest_path": manifest_path,
        "truth_ref": manifest["truth_ref"]["path"],
        "context_ref": manifest["observation_ref"]["path"],
        "gates": dict(gates) if gates is not None else None,
    }


def write_suite_manifest(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    now: datetime | None = None,
) -> ArtifactRef:
    """Publish every outcome row of the parent set, including the ones that failed.

    `fully_accepted` is the conjunction of the rows and is stated rather than inferred: plan
    11.9 requires a suite with a failed world to SAY that it was not fully accepted, and a
    manifest that simply omitted the row would say the opposite by omission.
    """
    stamp = now or datetime.now(UTC)
    payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "derived_view_id": DERIVED_VIEW_ID,
        "split": SPLIT,
        "n_parents": len(rows),
        "fully_accepted": bool(rows) and all(bool(row["accepted"]) for row in rows),
        "note": (
            "Every parent of the first P1 set has a row here, whatever its outcome. A row is "
            "never removed to make the suite pass (SPEC 18.4)"
        ),
        "rows": [dict(row) for row in rows],
    }
    ref, _ = _publish_run_record(
        path, payload, paths, ctx, key="p1_suite", schema_version=SUITE_SCHEMA_VERSION, now=stamp
    )
    return ref
