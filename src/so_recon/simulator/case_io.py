"""Validation and the safe Python <-> HDF5 <-> Julia exchange for forward cases.

Three jobs, in the order a case meets them:

1. **Check before spending anything.** A path that escapes the repository, a digest that
   does not match the bytes, a NaN permeability or a producer controlled on a phase rate
   is refused here, in Python, before any Julia process is started. `validate_case`
   collects everything it finds into a `ValidationReport` instead of raising on the first
   problem, because an operator fixing a case wants the whole list.
2. **Publish immutably, in one sequence.** `write_arrays` writes one or more arrays into a
   temporary neighbour file with a single writer, flushes, closes, hashes, compares against
   whatever already occupies the destination, and only then renames into place;
   `write_array` is the one-dataset case of it. `write_case` then
   runs the rest of that sequence end to end: it registers every array of the case through
   the existing artifact registry and into the run's outputs, and publishes the JSON
   manifest LAST. Losing the process halfway therefore leaves unfinished staging rather
   than a case that looks complete, and the manifest's `parent_artifact_ids` always name
   artifacts the run really recorded. Writing different bytes to a path that already holds
   an artifact is refused by that registry — no second registry is created here.
3. **Survive the round trip.** HDF5 states are `(n_times, n_cells)` with `axis_order`
   recorded as a dataset attribute. Julia is column-major and its HDF5 bindings hand back
   reversed dimensions, so the attribute is what lets the two sides agree; `read_array`
   refuses an array whose on-disk `axis_order` is not the one the reference declares.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactImmutabilityError,
    ArtifactRef,
    json_artifact_bytes,
    register_artifact,
    write_json_artifact,
)
from so_recon.registry.atomic import fsync_dir, stage_path
from so_recon.registry.hashing import sha256_bytes, sha256_file, sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.contracts import (
    SATURATION_SUM_TOLERANCE,
    ArrayRef,
    CaseBundle,
)

HDF5_MEDIA_TYPE = "application/x-hdf5"
ARRAY_SCHEMA_VERSION = "array-1"
CASE_MANIFEST_FILENAME = "case.json"

UNIT_ATTRIBUTE = "unit"
AXIS_ORDER_ATTRIBUTE = "axis_order"

#: A Cartesian grid's total volume must equal the volume of the box that encloses it.
VOLUME_CLOSURE_RTOL = 1e-9

#: Fields of a `CaseBundle` that determine the forward operator F, and therefore the model
#: hash. The array digests reach the hash through the `ArrayRef`s these fields carry.
#:
#: Deliberately absent: `case_id`, `world_id`, `sector_id`, `start_date` and `cutoff` are
#: identity and calendar labels, so two cases with the same physics must hash alike;
#: `observations` is what a later stage conditions on, not an input F reads;
#: `source_hashes` records where the case came from, which changes when an identical case
#: is regenerated; and `model_hash` is the result itself. Wall time, RAM and hostname have
#: no field here at all — they live in `CostRecord` (plan 3.1).
MODEL_HASH_FIELDS = (
    "schema_version",
    "spec_version",
    "information_mode",
    "report_edges_s",
    "grid",
    "rock",
    "fluids",
    "wells",
    "controls",
    "initial",
    "boundary",
    "gravity_m_s2",
    "renderer_version",
    "units",
    "seeds",
)

#: What `compute_static_hash` leaves out of `MODEL_HASH_FIELDS`: the schedule. Both are
#: covered by a checkpoint's own schedule-prefix digest instead, which is what lets a
#: continuation change the future without being refused as a different model.
STATIC_HASH_EXCLUDED = frozenset({"controls", "report_edges_s"})

_NUMPY_DTYPES: dict[str, np.dtype[Any]] = {
    "float64": np.dtype(np.float64),
    "int64": np.dtype(np.int64),
    "bool": np.dtype(np.bool_),
}


class CaseIntegrityError(RuntimeError):
    """A case file does not hash to what it claims, so it cannot be trusted."""


class ValidationReport(StrictModel):
    """The outcome of checking a case. Each error names the field and the reason."""

    valid: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _verdict_matches_the_errors(self) -> ValidationReport:
        if self.valid != (not self.errors):
            raise ValueError(f"valid={self.valid} contradicts {len(self.errors)} recorded errors")
        return self


# --------------------------------------------------------------------------------------
# numeric checks
# --------------------------------------------------------------------------------------


def validate_numeric_array(
    name: str,
    values: NDArray[np.float64],
    *,
    positive: bool = False,
    fraction: bool = False,
) -> None:
    """Reject a physically impossible array. Invalid input is refused, never clipped."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: nonfinite values")
    if positive and not (values > 0).all():
        raise ValueError(f"{name}: must be positive")
    if fraction and not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"{name}: outside [0,1]")


def validate_saturations(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    *,
    tolerance: float = SATURATION_SUM_TOLERANCE,
) -> None:
    """Sw + So = 1 to within `tolerance`; a larger drift is an error, not a renormalisation."""
    total = sw + so
    if not np.isfinite(total).all():
        raise ValueError("saturations: nonfinite values")
    deviation = float(np.max(np.abs(total - 1.0)))
    if deviation > tolerance:
        raise ValueError(
            f"saturations: sw + so must sum to 1 within {tolerance:g}, "
            f"largest deviation is {deviation:g}"
        )


def validate_range(name: str, values: NDArray[np.float64], low: float, high: float) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: nonfinite values")
    if not ((values >= low) & (values <= high)).all():
        raise ValueError(f"{name}: outside the allowed range [{low:g}, {high:g}]")


def cartesian_neighbors(shape: tuple[int, int, int]) -> NDArray[np.int64]:
    """The face list of a Cartesian grid, zero-based, `cell_id = i + nx*(j + ny*k)`.

    Each face appears once, as the ordered pair (lower cell id, higher cell id).
    """
    nx, ny, nz = shape
    faces: list[tuple[int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = i + nx * (j + ny * k)
                if i + 1 < nx:
                    faces.append((cell, cell + 1))
                if j + 1 < ny:
                    faces.append((cell, cell + nx))
                if k + 1 < nz:
                    faces.append((cell, cell + nx * ny))
    return np.array(faces, dtype=np.int64).reshape(len(faces), 2)


# --------------------------------------------------------------------------------------
# HDF5 exchange
# --------------------------------------------------------------------------------------


def _dtype_name(dtype: np.dtype[Any]) -> str:
    for name, candidate in _NUMPY_DTYPES.items():
        if dtype == candidate:
            return name
    raise ValueError(f"dtype {dtype} is not exchangeable; use one of {sorted(_NUMPY_DTYPES)}")


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_arrays(
    path: Path,
    datasets: Mapping[str, tuple[NDArray[Any], str, tuple[str, ...]]],
    *,
    paths: ProjectPaths,
) -> dict[str, ArrayRef]:
    """Publish several arrays into ONE immutable HDF5 file and describe each of them.

    One file, one writer, one rename: the datasets are written together by a single handle
    and the whole file is staged, flushed, closed, hashed and only then renamed into place.
    Every `ArrayRef` this returns therefore carries the SAME `sha256`, which is the digest of
    the file (plan 3.1: «Artifact SHA — байты файла»), and differs only in `dataset`. A
    reader proves the file first and the dataset second.

    `track_times=False` is not cosmetic: with the HDF5 default the file embeds the moment it
    was created, so writing the same array twice produces different bytes and the
    content-addressed artifact contract would be unenforceable.
    """
    repo_relative = paths.relative(path)  # also proves containment inside the repository
    if not datasets:
        raise ValueError(f"{repo_relative}: an HDF5 file is written with at least one dataset")
    prepared: dict[str, tuple[NDArray[Any], str, str, tuple[str, ...]]] = {}
    for name, (values, unit, axis_order) in datasets.items():
        array = np.asarray(values)
        dtype = _dtype_name(array.dtype)
        if len(axis_order) != array.ndim:
            raise ValueError(
                f"{repo_relative}: {name}: axis_order {axis_order} does not describe a "
                f"{array.ndim}-dimensional array"
            )
        prepared[name] = (array, dtype, unit, tuple(axis_order))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stage_path(path)
    try:
        with h5py.File(tmp, "w") as handle:
            for name, (array, _, unit, axis_order) in prepared.items():
                written = handle.create_dataset(name, data=array, track_times=False)
                written.attrs[UNIT_ATTRIBUTE] = unit
                written.attrs.create(
                    AXIS_ORDER_ATTRIBUTE,
                    list(axis_order),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            handle.flush()
        digest = sha256_file(tmp)
        if path.exists():
            if sha256_file(path) != digest:
                raise ArtifactImmutabilityError(f"refusing to overwrite {path}: different content")
            tmp.unlink()
        else:
            tmp.replace(path)
            fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {
        name: ArrayRef(
            path=repo_relative,
            dataset=name,
            sha256=digest,
            shape=tuple(int(n) for n in array.shape),
            dtype=dtype,
            unit=unit,
            axis_order=axis_order,
        )
        for name, (array, dtype, unit, axis_order) in prepared.items()
    }


def write_array(
    path: Path,
    dataset: str,
    values: NDArray[Any],
    *,
    unit: str,
    axis_order: tuple[str, ...],
    paths: ProjectPaths,
) -> ArrayRef:
    """Publish one array as an immutable single-dataset HDF5 file and describe it."""
    return write_arrays(path, {dataset: (np.asarray(values), unit, axis_order)}, paths=paths)[
        dataset
    ]


def read_array(ref: ArrayRef, paths: ProjectPaths) -> NDArray[Any]:
    """Read an array back, proving it is the one the reference describes."""
    path = paths.resolve(ref.path)  # form, then containment after symlink resolution
    if not path.is_file():
        raise FileNotFoundError(f"{ref.path}: no such file")
    digest = sha256_file(path)
    if digest != ref.sha256:
        raise ValueError(f"{ref.path}: sha256 {digest} does not match the declared {ref.sha256}")
    with h5py.File(path, "r") as handle:
        if ref.dataset not in handle:
            raise ValueError(f"{ref.path}: dataset {ref.dataset!r} is missing")
        dataset = handle[ref.dataset]
        shape = tuple(int(n) for n in dataset.shape)
        if shape != ref.shape:
            raise ValueError(f"{ref.path}: shape {shape} does not match the declared {ref.shape}")
        axis_order = tuple(_decode(a) for a in dataset.attrs.get(AXIS_ORDER_ATTRIBUTE, ()))
        if axis_order != ref.axis_order:
            raise ValueError(
                f"{ref.path}: axis_order {axis_order} does not match the declared {ref.axis_order}"
            )
        unit = _decode(dataset.attrs.get(UNIT_ATTRIBUTE, ""))
        if unit != ref.unit:
            raise ValueError(f"{ref.path}: unit {unit!r} does not match the declared {ref.unit!r}")
        if _dtype_name(dataset.dtype) != ref.dtype:
            raise ValueError(
                f"{ref.path}: dtype {dataset.dtype} does not match the declared {ref.dtype}"
            )
        values: NDArray[Any] = np.asarray(dataset[()])
    return values


def register_array(
    ref: ArrayRef,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    key: str,
    now: datetime | None = None,
) -> ArtifactRef:
    """Record a published array in the run's lineage through the existing registry."""
    artifact = register_artifact(
        paths.resolve(ref.path),
        paths,
        schema_version=ARRAY_SCHEMA_VERSION,
        producer_run_id=ctx.run_id,
        media_type=HDF5_MEDIA_TYPE,
        now=now or datetime.now(UTC),
    )
    if artifact.sha256 != ref.sha256:
        raise ValueError(
            f"{ref.path}: the file hashes to {artifact.sha256}, but the reference declares "
            f"{ref.sha256}"
        )
    ctx.add_output(key, artifact)
    return artifact


# --------------------------------------------------------------------------------------
# case validation
# --------------------------------------------------------------------------------------


def case_array_refs(case: CaseBundle) -> dict[str, ArrayRef]:
    """Every array a case depends on, labelled by its position in the record."""
    refs: dict[str, ArrayRef] = {
        "grid.cell_centers_m": case.grid.cell_centers_m,
        "grid.cell_volume_m3": case.grid.cell_volume_m3,
        "grid.neighbors": case.grid.neighbors,
        "rock.porosity": case.rock.porosity,
        "rock.permeability_m2": case.rock.permeability_m2,
    }
    if case.initial.pressure_pa is not None:
        refs["initial.pressure_pa"] = case.initial.pressure_pa
    if case.initial.sw is not None:
        refs["initial.sw"] = case.initial.sw
    return refs


def _check_neighbors(case: CaseBundle, faces: NDArray[np.int64], errors: list[str]) -> None:
    label = "grid.neighbors"
    n_cells = case.grid.n_cells
    if faces.min(initial=0) < 0 or faces.max(initial=0) >= n_cells:
        errors.append(f"{label}: cell id outside [0, {n_cells}) in the face list")
        return
    if bool((faces[:, 0] == faces[:, 1]).any()):
        errors.append(f"{label}: self edge, a face cannot join a cell to itself")
    edges = [frozenset((int(a), int(b))) for a, b in faces]
    unique = set(edges)
    if len(unique) != len(edges):
        errors.append(f"{label}: duplicate edge, every face must appear exactly once")
    expected = {frozenset((int(a), int(b))) for a, b in cartesian_neighbors(case.grid.shape)}
    if unique != expected:
        errors.append(
            f"{label}: not the symmetric Cartesian topology of {case.grid.shape}; "
            f"{len(expected - unique)} faces missing, {len(unique - expected)} unexpected"
        )


def _check_geometry(
    case: CaseBundle, centers: NDArray[np.float64], volumes: NDArray[np.float64], errors: list[str]
) -> None:
    try:
        validate_numeric_array("grid.cell_centers_m", centers)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if not (centers[:, 2] > 0.0).all():
            errors.append("grid.cell_centers_m: z is depth, positive down; got non-positive z")
    try:
        validate_numeric_array("grid.cell_volume_m3", volumes, positive=True)
    except ValueError as exc:
        errors.append(str(exc))
        return
    enclosing = float(np.prod(np.asarray(case.grid.extent_m, dtype=np.float64)))
    total = float(volumes.sum())
    if abs(total - enclosing) > VOLUME_CLOSURE_RTOL * enclosing:
        errors.append(
            f"grid.cell_volume_m3: cells total {total:g} m3 but extent_m encloses {enclosing:g} m3"
        )


def validate_case(case: CaseBundle, paths: ProjectPaths) -> ValidationReport:
    """Check a case against its own files. Returns every problem; raises for none of them."""
    errors: list[str] = []
    arrays: dict[str, NDArray[Any]] = {}
    for label, ref in case_array_refs(case).items():
        try:
            arrays[label] = read_array(ref, paths)
        except (ValueError, OSError) as exc:
            errors.append(f"{label}: {exc}")

    if "grid.neighbors" in arrays:
        _check_neighbors(case, arrays["grid.neighbors"].astype(np.int64), errors)
    if "grid.cell_centers_m" in arrays and "grid.cell_volume_m3" in arrays:
        _check_geometry(case, arrays["grid.cell_centers_m"], arrays["grid.cell_volume_m3"], errors)

    for label, positive, fraction in (
        ("rock.porosity", False, True),
        ("rock.permeability_m2", True, False),
        ("initial.pressure_pa", True, False),
        ("initial.sw", False, True),
    ):
        if label in arrays:
            try:
                validate_numeric_array(label, arrays[label], positive=positive, fraction=fraction)
            except ValueError as exc:
                errors.append(str(exc))

    if "initial.sw" in arrays:
        low, high = case.fluids.mobile_saturation_range
        try:
            validate_range("initial.sw", arrays["initial.sw"], low, high)
            validate_saturations(arrays["initial.sw"], 1.0 - arrays["initial.sw"])
        except ValueError as exc:
            errors.append(str(exc))

    errors.extend(_check_observations(case, paths))

    recomputed = compute_model_hash(case)
    if recomputed != case.model_hash:
        errors.append(
            f"model_hash: declared {case.model_hash}, but the case content hashes to {recomputed}"
        )
    return ValidationReport(valid=not errors, errors=tuple(errors))


def _check_observations(case: CaseBundle, paths: ProjectPaths) -> list[str]:
    if case.observations.table_path is None:
        return []
    try:
        table = paths.resolve(case.observations.table_path)
    except ValueError as exc:
        return [f"observations.table_path: {exc}"]
    if not table.is_file():
        return [f"observations.table_path: no such file {case.observations.table_path}"]
    digest = sha256_file(table)
    if digest != case.observations.sha256:
        return [
            f"observations.sha256: the table hashes to {digest}, but the case declares "
            f"{case.observations.sha256}"
        ]
    return []


# --------------------------------------------------------------------------------------
# model hash and publication
# --------------------------------------------------------------------------------------


def model_hash_payload(case: CaseBundle) -> dict[str, Any]:
    dumped = case.model_dump(mode="json")
    return {name: dumped[name] for name in MODEL_HASH_FIELDS}


def compute_model_hash(case: CaseBundle) -> str:
    """Canonical JSON of every input that determines F, including the array digests."""
    return sha256_json(model_hash_payload(case))


def compute_static_hash(case: CaseBundle) -> str:
    """Everything that determines F EXCEPT the schedule: the model a continuation shares.

    A continuation may change what happens after its checkpoint — that is what a resume's
    `future_policy` is for — so its model hash legitimately differs from the one the
    checkpoint was written under, and comparing the two would refuse every policy change.
    What it may NOT change is the grid, the rock, the fluids, the wells, the initial state or
    the boundary: a restart that resumed a different reservoir would be continuing a run that
    never happened. This digest is that half; the schedule half is proved separately by the
    checkpoint's prefix digest, which covers the report edges and controls up to it.
    """
    dumped = case.model_dump(mode="json")
    return sha256_json(
        {name: dumped[name] for name in MODEL_HASH_FIELDS if name not in STATIC_HASH_EXCLUDED}
    )


def case_manifest_sha256(case: CaseBundle) -> str:
    """The digest the published manifest of THIS case has, without publishing it again.

    `JobDescriptor.case_sha256` — and therefore every `BudgetLedger` entry — records
    `sha256_file` of the manifest `write_case` wrote, and that manifest is exactly
    `json_artifact_bytes(case.model_dump(mode="json"))`: no timestamp, no run id, nothing
    that moves between sessions. So a resumed session can ask whether a previous one already
    completed this exact case without republishing it, and
    `tests/unit/test_e01_resume.py::test_the_case_digest_matches_the_manifest_a_session_really_publishes`
    pins the two together.
    """
    return sha256_bytes(json_artifact_bytes(case.model_dump(mode="json")))


def write_case(
    case: CaseBundle,
    paths: ProjectPaths,
    ctx: RunContext,
    *,
    now: datetime | None = None,
) -> ArtifactRef:
    """Publish a case: register every array it depends on, then the manifest, last.

    This is the whole of the publication sequence, not its final step. The arrays are
    already staged, hashed and renamed into place by `write_array`; here each one is
    recorded in the run's lineage through the existing registry, and only then is the JSON
    manifest written. That ordering is the point: a process lost part-way leaves registered
    arrays and no manifest, which is unfinished staging rather than a case that claims to be
    complete — and it means the `parent_artifact_ids` the manifest names always resolve to
    artifacts the run actually recorded.

    Each array is keyed in `ctx.outputs` by its position in the record, so
    `rock.permeability_m2` is findable as such rather than as an anonymous digest.
    """
    report = validate_case(case, paths)
    if not report.valid:
        raise ValueError(f"refusing to publish case {case.case_id}: " + "; ".join(report.errors))
    stamp = now or datetime.now(UTC)
    registered = [
        register_array(ref, paths, ctx, key=label, now=stamp)
        for label, ref in sorted(case_array_refs(case).items())
    ]
    ref = write_json_artifact(
        ctx.run_dir / CASE_MANIFEST_FILENAME,
        case.model_dump(mode="json"),
        paths,
        schema_version=case.schema_version,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=sorted({artifact.artifact_id for artifact in registered}),
        now=stamp,
    )
    ctx.add_output("case", ref)
    return ref


def load_case(path: Path, paths: ProjectPaths) -> CaseBundle:
    """Read a case manifest and prove it still hashes to the model it names."""
    relative = paths.relative(path)  # containment; a manifest outside the repo is refused
    payload = json.loads(path.read_text(encoding="utf-8"))
    case = CaseBundle.model_validate(payload)
    recomputed = compute_model_hash(case)
    if recomputed != case.model_hash:
        raise CaseIntegrityError(
            f"{relative}: model_hash {case.model_hash} does not match the case content, "
            f"which hashes to {recomputed}"
        )
    return case
