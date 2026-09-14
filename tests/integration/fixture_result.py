"""Turn one Julia verification fixture into a real, published `ForwardResult`.

Every julia-marked verification suite in this tree does the same three things with what the
diagnostic hands back: it rebuilds the fixture's case through the PRODUCTION contract
(`CaseBundle`, `FluidSpec`, `WellSpec`, `ControlSegment`, `validate_case`), it publishes the
native extraction through the PRODUCTION publisher (`publish_forward_result`), and it reads
the record back with every declared digest, unit, shape and physical range re-proved. That
sequence is what makes the evaluator's verdict a verdict about bytes a real run would leave
behind rather than about a dictionary a test built.

It lives here because Task 9's analytic fixtures and Task 10's operational ones both need it
and neither owns it. Nothing in this module asserts a STATUS: a fixture that is meant to come
back `CONTROL_INFEASIBLE` is as legitimate as one that completes, and deciding which is the
caller's business.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.case_io import (
    cartesian_neighbors,
    compute_model_hash,
    validate_case,
    write_array,
)
from so_recon.simulator.contracts import (
    CELL_AXES,
    CELL_DIM_AXES,
    DIM_CELL_AXES,
    FACE_AXES,
    ArrayRef,
    BoundarySpec,
    CaseBundle,
    ControlSegment,
    CostRecord,
    FluidSpec,
    ForwardResult,
    GridSpec,
    InitialStateSpec,
    JobDescriptor,
    ObservationSpec,
    OutputRequest,
    RockSpec,
    WellSpec,
)
from so_recon.simulator.results import (
    RESULT_FILENAME,
    load_forward_result,
    publish_forward_result,
    write_forward_result,
)

#: The seed every verification fixture is published under. These cases are deterministic and
#: carry no random field at all; the seed is recorded because a case records one.
FIXTURE_SEED = 20260914


def expected_cell_centers(
    shape: tuple[int, int, int], extent: tuple[float, float, float], datum: float = 1000.0
) -> NDArray[np.float64]:
    """The fixture's geometry, recomputed here rather than trusted from the report."""
    nx, ny, nz = shape
    dx, dy, dz = (e / n for e, n in zip(extent, shape, strict=True))
    out = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                out[i + nx * (j + ny * k)] = (
                    dx * i + 0.5 * dx,
                    dy * j + 0.5 * dy,
                    datum + dz * k + 0.5 * dz,
                )
    return out


def build_case(
    fixture: dict[str, Any], paths: ProjectPaths, label: str, *, world: str
) -> tuple[CaseBundle, Path]:
    """Rebuild one Julia fixture as a real `CaseBundle` and prove it against its own files.

    Every typed record is parsed through the production contract rather than trusted: an
    illegal fluid, an initial saturation outside the mobile range the fluid declares, a
    control with a role its target does not belong to, a boundary that names no pressure or a
    geometry that is not the uniform grid its shape implies would not survive these lines.
    """
    case_json = fixture["case"]
    arrays = fixture["arrays"]
    shape = tuple(int(n) for n in case_json["grid"]["shape"])
    extent = tuple(float(e) for e in case_json["grid"]["extent_m"])
    n_cells = shape[0] * shape[1] * shape[2]
    centers = np.asarray(arrays["cell_centers_m"], dtype=np.float64)
    if not np.allclose(centers, expected_cell_centers(shape, extent), rtol=0.0, atol=1e-9):
        raise AssertionError(f"{label}: the fixture's cell centres are not its own uniform grid")
    cell_volume = float(np.prod(np.asarray(extent))) / n_cells

    written: dict[str, ArrayRef] = {}
    for name, values, unit, axes in (
        ("cell_centers_m", centers, "m", CELL_DIM_AXES),
        ("cell_volume_m3", np.full(n_cells, cell_volume, dtype=np.float64), "m3", CELL_AXES),
        ("neighbors", cartesian_neighbors(shape), "1", FACE_AXES),
        ("porosity", np.asarray(arrays["porosity"], dtype=np.float64), "1", CELL_AXES),
        (
            "permeability_m2",
            np.asarray(arrays["permeability_m2"], dtype=np.float64),
            "m2",
            DIM_CELL_AXES,
        ),
        ("pressure_pa", np.asarray(arrays["pressure_pa"], dtype=np.float64), "Pa", CELL_AXES),
        ("sw", np.asarray(arrays["sw"], dtype=np.float64), "1", CELL_AXES),
    ):
        written[name] = write_array(
            paths.artifacts / f"arrays-{label}" / f"{name}.h5",
            "values",
            values,
            unit=unit,
            axis_order=axes,
            paths=paths,
        )

    case = CaseBundle(
        case_id=str(case_json["case_id"]),
        world_id=f"world-{world}-{label}",
        start_date=str(case_json["start_date"]),
        cutoff="2029-12-31",
        report_edges_s=tuple(float(e) for e in case_json["report_edges_s"]),
        grid=GridSpec(
            shape=shape,  # type: ignore[arg-type]
            extent_m=extent,  # type: ignore[arg-type]
            cell_centers_m=written["cell_centers_m"],
            cell_volume_m3=written["cell_volume_m3"],
            neighbors=written["neighbors"],
        ),
        rock=RockSpec(porosity=written["porosity"], permeability_m2=written["permeability_m2"]),
        fluids=FluidSpec.model_validate(case_json["fluids"]),
        wells=tuple(WellSpec.model_validate(w) for w in case_json["wells"]),
        controls=tuple(ControlSegment.model_validate(c) for c in case_json["controls"]),
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=written["pressure_pa"],
            sw=written["sw"],
            meaning="synthetic_initial",
        ),
        boundary=BoundarySpec.model_validate(case_json["boundary"]),
        observations=ObservationSpec(dynamic_channels=(), pressure_available=False),
        renderer_version=f"e01-{world}",
        units=dict(case_json["units"]),
        seeds={"fixture": FIXTURE_SEED},
        source_hashes={"generator": "c" * 64},
        model_hash="e" * 64,
    )
    case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    report = validate_case(case, paths)
    if not report.valid:
        raise AssertionError(f"{label}: {report.errors}")

    case_path = paths.artifacts / f"case-{label}.json"
    case_path.write_text(json.dumps(case.model_dump(mode="json")), encoding="utf-8")
    return case, case_path


def solver_cost(extraction: dict[str, Any]) -> CostRecord:
    """The cost record a fixture's own solver counters describe."""
    solver = extraction["solver"]
    return CostRecord(
        wall_s=0.0,
        cpu_s=0.0,
        peak_rss_bytes=0,
        output_bytes=0,
        accepted_steps=int(solver["accepted_steps"]),
        cut_steps=int(solver["cut_steps"]),
        nonlinear_iterations=int(solver["nonlinear_iterations"]),
        retry_count=0,
        measurement_method="solver report counters from the native extraction",
    )


def publish_fixture(
    fixture: dict[str, Any],
    paths: ProjectPaths,
    label: str,
    report: dict[str, Any],
    *,
    world: str,
) -> tuple[ForwardResult, Path]:
    """Publish one fixture as a real `ForwardResult`, read it back, and return both.

    The status is NOT asserted here. `publish_forward_result` decides it from the extraction —
    a fixture whose wells could not reach their demanded control publishes `CONTROL_INFEASIBLE`
    with its evidence, which is a result and not a failure — so the caller says which status
    its fixture is supposed to have.
    """
    case, case_path = build_case(fixture, paths, label, world=world)
    solver_path = paths.artifacts / "solver" / "e01_solver.json"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    if not solver_path.exists():
        solver_path.write_text(
            json.dumps({"max_timestep_days": 1, "max_nonlinear_iterations": 30}), encoding="utf-8"
        )
    job = JobDescriptor(
        job_id=f"job-e01-{world}-{label}",
        case_path=paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash=case.model_hash,
        solver_config_path=paths.relative(solver_path),
        solver_config_sha256=sha256_file(solver_path),
        output_request=OutputRequest(
            state_times_s=tuple(case.report_edges_s), keep_native_restart=False
        ),
        seed=FIXTURE_SEED,
        result_dir=f"artifacts/results/{label}",
        attempt=1,
    )
    extraction = fixture["extraction"]
    result = publish_forward_result(
        job,
        case,
        extraction,
        paths,
        cost=solver_cost(extraction),
        solver_metadata={
            "version.Jutul": report["jutul_version"],
            "version.JutulDarcy": report["jutuldarcy_version"],
        },
    )
    result_dir = paths.resolve(job.result_dir)
    record_path = result_dir / RESULT_FILENAME
    write_forward_result(result, record_path)
    # Every declared digest, shape, unit and physical range re-proved from the bytes on disk.
    reloaded = load_forward_result(record_path, paths)
    if reloaded != result:
        raise AssertionError(f"{label}: the record read back is not the one published")
    # A classified result — `CONTROL_INFEASIBLE`, say — publishes no outputs and therefore no
    # time axis, which is the point of classifying it: there is nothing to compare.
    if result.status == "COMPLETE" and reloaded.times_s != tuple(extraction["states"]["times_s"]):
        raise AssertionError(f"{label}: the published time axis is not the extraction's")
    return result, result_dir
