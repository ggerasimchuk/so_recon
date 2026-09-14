"""E01.7: the accepted-substep integrals of a real solver run, and the result they publish.

ONE real Julia process. `julia/verification/fixtures.jl --test-outputs` runs four small
verification forwards — the two-month calendar fixture, the same fixture under deliberate
timestep cuts, a fully masked well and a well on an unreachable rate — asserts what only
Julia can see about them, and hands back everything this side needs: the case it simulated,
its arrays, and the extraction of each run. Python then does what Julia cannot: it builds
the same case through the real `CaseBundle` contract, integrates the accepted substeps into
monthly volumes, evaluates the component balance and publishes a `ForwardResult` that is
read back and re-verified.

The cross-check that matters is the one the plan names: the summed standard phase volumes
are compared against the NATIVE MASS BALANCE — the component masses of the reservoir and the
wellbores, divided by the surface densities — and not against a second run of this side's
own aggregator. The two quantities come from different native objects (a surface volumetric
rate target and `TotalMasses`), so agreement between them is evidence rather than tautology.

This build still does not claim an HDF5 restart. SPEC §17.1 — «HDF5 summary не объявляется
полноценным restart без round-trip test» — and the states file published here is a summary:
`ForwardResult.restart` stays null.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P0_VERIFY_PROFILE
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
    SECONDS_PER_DAY,
    ArrayRef,
    BoundarySpec,
    CaseBundle,
    ControlSegment,
    CostRecord,
    FluidSpec,
    GridSpec,
    InitialStateSpec,
    JobDescriptor,
    ObservationSpec,
    OutputRequest,
    RockSpec,
    WellSpec,
)
from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    SubprocessJuliaLauncher,
    find_julia,
)
from so_recon.simulator.results import (
    RESULT_FILENAME,
    connection_step_table,
    extraction_balance,
    extraction_balances,
    integrate_monthly,
    load_forward_result,
    publish_forward_result,
    well_step_table,
    write_forward_result,
)
from so_recon.simulator.schedule import compile_schedule, month_edges_s
from so_recon.validation import (
    CUMULATIVE_RELATIVE_TOLERANCE,
    MEDIAN_STEP_RELATIVE_TOLERANCE,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_JL = ROOT / "julia" / "verification" / "fixtures.jl"

DAY = SECONDS_PER_DAY

#: `verification_case(:two_interval_controls)`: January and the leap February of 2020, a
#: completion event on day 45 and a field shutdown on day 52, both wells on 2 m³_sc/day
#: while they run on a rate.
CONTROLS_START = date(2020, 1, 1)
CONTROLS_MONTHS = 2
CONTROLS_RATE_M3_DAY = 2.0

#: The fixture's box: 4 x 1 x 2 cells of 10 m cube, 200 m³ of pore volume each.
FIXTURE_SHAPE = (4, 1, 2)
FIXTURE_N_CELLS = 8
CELL_EDGE_M = 10.0
DATUM_M = 1000.0
RHO_W_SC, RHO_O_SC = 1000.0, 800.0


def _skip_unless_julia_is_installed() -> Path:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        return find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")


def _expected_cell_centers() -> NDArray[np.float64]:
    """The fixture's geometry, recomputed here rather than trusted from the report."""
    nx, ny, nz = FIXTURE_SHAPE
    out = np.zeros((FIXTURE_N_CELLS, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                out[i + nx * (j + ny * k)] = (
                    CELL_EDGE_M * i + 0.5 * CELL_EDGE_M,
                    CELL_EDGE_M * j + 0.5 * CELL_EDGE_M,
                    DATUM_M + CELL_EDGE_M * k + 0.5 * CELL_EDGE_M,
                )
    return out


def _publish_fixture_case(report: dict[str, Any], paths: ProjectPaths) -> tuple[CaseBundle, Path]:
    """Build the Julia fixture as a real `CaseBundle`, from the case Julia simulated.

    The arrays come back from the diagnostic in the exchange's own semantic layout and are
    written here through `write_array`, so the case this side reasons about is pinned by the
    same digests the adapter would have re-hashed. What the fixture cannot send — the cell
    volumes and the face list, which Julia derives from the mesh — is computed here and
    checked against the geometry the case declares.
    """
    case_json = report["case"]
    arrays = report["arrays"]
    assert tuple(case_json["grid"]["shape"]) == FIXTURE_SHAPE
    centers = np.asarray(arrays["cell_centers_m"], dtype=np.float64)
    assert centers == pytest.approx(_expected_cell_centers(), rel=0, abs=0)

    written: dict[str, ArrayRef] = {}
    for name, values, unit, axes in (
        ("cell_centers_m", centers, "m", CELL_DIM_AXES),
        (
            "cell_volume_m3",
            np.full(FIXTURE_N_CELLS, CELL_EDGE_M**3, dtype=np.float64),
            "m3",
            CELL_AXES,
        ),
        ("neighbors", cartesian_neighbors(FIXTURE_SHAPE), "1", FACE_AXES),
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
            paths.artifacts / "fixture-arrays" / f"{name}.h5",
            "values",
            values,
            unit=unit,
            axis_order=axes,
            paths=paths,
        )

    wells = tuple(
        WellSpec(
            well_id=str(w["well_id"]),
            cells=tuple(int(c) for c in w["cells"]),
            radius_m=float(w["radius_m"]),
            reference_depth_m=float(w["reference_depth_m"]),
            model=str(w["model"]),  # type: ignore[arg-type]
            allow_crossflow=bool(w["allow_crossflow"]),
        )
        for w in case_json["wells"]
    )
    # Parsing the fixture's own control segments through the real contract is itself a
    # check: an illegal role, target or rate would not survive this line.
    controls = tuple(ControlSegment.model_validate(c) for c in case_json["controls"])
    case = CaseBundle(
        case_id=str(case_json["case_id"]),
        world_id="world-e01-outputs",
        start_date=str(case_json["start_date"]),
        cutoff="2020-02-29",
        report_edges_s=tuple(float(e) for e in case_json["report_edges_s"]),
        grid=GridSpec(
            shape=FIXTURE_SHAPE,
            extent_m=tuple(float(e) for e in case_json["grid"]["extent_m"]),  # type: ignore[arg-type]
            cell_centers_m=written["cell_centers_m"],
            cell_volume_m3=written["cell_volume_m3"],
            neighbors=written["neighbors"],
        ),
        rock=RockSpec(porosity=written["porosity"], permeability_m2=written["permeability_m2"]),
        fluids=FluidSpec(),
        wells=wells,
        controls=controls,
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=written["pressure_pa"],
            sw=written["sw"],
            meaning="synthetic_initial",
        ),
        boundary=BoundarySpec(kind="closed", cells=()),
        observations=ObservationSpec(dynamic_channels=(), pressure_available=False),
        renderer_version="e01.7",
        units=dict(case_json["units"]),
        seeds={"fixture": 20260913},
        source_hashes={"generator": "c" * 64},
        model_hash="e" * 64,
    )
    case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    report_validation = validate_case(case, paths)
    assert report_validation.valid, report_validation.errors

    case_path = paths.artifacts / "fixture-case.json"
    case_path.write_text(json.dumps(case.model_dump(mode="json")), encoding="utf-8")
    return case, case_path


def _descriptor(case: CaseBundle, case_path: Path, paths: ProjectPaths) -> JobDescriptor:
    solver_path = paths.artifacts / "solver" / "e01_solver.json"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    solver_path.write_text(
        json.dumps({"max_timestep_days": 5, "max_nonlinear_iterations": 15}), encoding="utf-8"
    )
    return JobDescriptor(
        job_id="job-e01-outputs",
        case_path=paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash=case.model_hash,
        solver_config_path=paths.relative(solver_path),
        solver_config_sha256=sha256_file(solver_path),
        # The times the extraction was asked for. Whether the result really carries them is
        # a CROSS-record check — it needs the request as well as the result — and belongs to
        # the stage that drives the chunks; `load_forward_result` proves only what the
        # result alone can prove.
        output_request=OutputRequest(
            state_times_s=tuple(case.report_edges_s), keep_native_restart=False
        ),
        seed=20260913,
        result_dir="artifacts/results/job-e01-outputs",
        attempt=1,
    )


def _cost(extraction: dict[str, Any]) -> CostRecord:
    """The solver counters the extraction measured, in the record that carries them."""
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


@pytest.mark.julia
def test_accepted_substeps_integrate_into_a_complete_verifiable_forward(
    tmp_project: Path,
) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    out_path = paths.artifacts / "verification" / "outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    launcher = SubprocessJuliaLauncher(
        julia_exe, ROOT / "julia", timeout_s=P0_VERIFY_PROFILE.job_timeout_s
    )
    launcher.launch(FIXTURES_JL, ["--test-outputs"], out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert (report["jutul_version"], report["jutuldarcy_version"]) == ("0.4.31", "0.3.11")

    extraction = report["extraction"]
    chunk = extraction["chunk"]

    # --- 7.3 the accepted substeps tile the chunk, and there are more of them than steps --
    dt = chunk["dt_s"]
    assert all(value > 0.0 for value in dt)
    assert sum(dt) == pytest.approx(60.0 * DAY, rel=1e-12)
    assert len(dt) == extraction["solver"]["accepted_steps"]
    assert len(dt) > len(chunk["edges_s"]) - 1
    # The choice of `result.result` over the high-level `result.states` is what makes any of
    # this possible: the latter carries the reservoir and nothing else.
    assert "Facility" not in report["state_keys"]["high_level"]
    assert "PRO1" not in report["state_keys"]["high_level"]
    assert {"Facility", "PRO1", "Reservoir"} <= set(report["state_keys"]["raw"])

    # --- the calendar the case declares reached the integral ------------------------------
    case, case_path = _publish_fixture_case(report, paths)
    schedule = compile_schedule(case.report_edges_s, case.controls)
    expected_edges = month_edges_s(CONTROLS_START, CONTROLS_MONTHS)
    assert case.report_edges_s == expected_edges
    assert schedule.month_edges_s == expected_edges
    assert schedule.edges_s == pytest.approx(chunk["edges_s"], rel=0, abs=0)
    assert schedule.month_index == (0, 1, 1, 1)
    # No accepted substep crosses an event or a month boundary: each one belongs to exactly
    # one compiled interval, and the aggregator below would refuse it otherwise.
    for start, end, interval in zip(
        chunk["start_s"], chunk["end_s"], chunk["interval_index"], strict=True
    ):
        assert schedule.edges_s[interval] <= start < end <= schedule.edges_s[interval + 1]

    # --- the result itself ----------------------------------------------------------------
    job = _descriptor(case, case_path, paths)
    result = publish_forward_result(
        job,
        case,
        extraction,
        paths,
        cost=_cost(extraction),
        solver_metadata={
            "version.Jutul": report["jutul_version"],
            "version.JutulDarcy": report["jutuldarcy_version"],
        },
    )
    assert result.status == "COMPLETE", result.reason
    record_path = paths.resolve(job.result_dir) / RESULT_FILENAME
    write_forward_result(result, record_path)
    reloaded = load_forward_result(record_path, paths)
    assert reloaded == result
    # SPEC 17.1: an HDF5 summary is not a restart until a round-trip test says so.
    assert reloaded.restart is None
    assert reloaded.times_s == tuple(extraction["states"]["times_s"])
    assert reloaded.completed_time_s == 60.0 * DAY
    for ref in reloaded.states.values():
        assert ref.shape == (len(reloaded.times_s), FIXTURE_N_CELLS)
        assert ref.axis_order == ("time", "cell")

    # --- 7.5 the monthly volumes, against a number the control pins -----------------------
    monthly = pq.read_table(paths.resolve(str(reloaded.monthly_path))).to_pylist()
    producer = {row["month_index"]: row for row in monthly if row["well_id"] == "PRO1"}
    injector = {row["month_index"]: row for row in monthly if row["well_id"] == "INJ1"}
    assert sorted(producer) == [0, 1] and sorted(injector) == [0, 1]

    # January is one interval, entirely on the liquid-rate control, so the total standard
    # liquid volume is the rate times the uptime EXACTLY — 2 m³_sc/day for 31 days — while
    # the phase split is what the forward model predicted. February mixes a rate interval, a
    # shut interval and a bottom-hole one, so no volume there is pinned by a control.
    assert schedule.monthly_uptime_s("PRO1") == (31.0 * DAY, 22.0 * DAY)
    assert producer[0]["liquid_prod_m3_sc"] == pytest.approx(CONTROLS_RATE_M3_DAY * 31.0, rel=1e-9)
    assert producer[0]["oil_prod_m3_sc"] > 0.0
    assert producer[0]["water_prod_m3_sc"] > 0.0
    assert 0.0 < producer[0]["fw"] < 1.0
    assert producer[0]["fw_valid"] is True
    assert producer[0]["fw"] == pytest.approx(
        producer[0]["water_prod_m3_sc"] / producer[0]["liquid_prod_m3_sc"]
    )
    # The injector never produced, so its production columns stay zero and it reports no
    # water cut at all — and its injected water is NOT subtracted from anybody's production.
    for month in (0, 1):
        assert injector[month]["oil_prod_m3_sc"] == 0.0
        assert injector[month]["water_prod_m3_sc"] == 0.0
        assert injector[month]["fw"] is None
        assert injector[month]["fw_valid"] is False
    assert injector[0]["water_inj_m3_sc"] == pytest.approx(CONTROLS_RATE_M3_DAY * 31.0, rel=1e-9)

    # --- 7.8 the summed standard volumes against the NATIVE mass balance -------------------
    # The monthly volumes come from the surface volumetric rate targets; the inventory comes
    # from `TotalMasses` divided by the surface densities. Two different native objects, so
    # this is a comparison rather than the aggregator checked against itself.
    inventory = np.asarray(extraction["inventory_m3_sc"], dtype=np.float64)
    native_change = inventory[-1] - inventory[0]
    water_in = sum(row["water_inj_m3_sc"] - row["water_prod_m3_sc"] for row in monthly)
    oil_in = -sum(row["oil_prod_m3_sc"] for row in monthly)
    assert water_in == pytest.approx(native_change[0], rel=1e-6)
    assert oil_in == pytest.approx(native_change[1], rel=1e-6)
    # And the comparison is not vacuous: both components really moved.
    assert native_change[0] > 100.0
    assert native_change[1] < -100.0

    # --- 7.6 both component balances, against SPEC 23.1 ------------------------------------
    # The whole model against the surface flux, and the reservoir alone against the native
    # connection flux built from geometry, state and PVT. Two statements about the same run,
    # both published so that the stage validator of plan 12.9 can read them off the result
    # rather than re-run this suite.
    metrics = extraction_balances(extraction)
    assert sorted(metrics) == ["full_system_surface", "reservoir_connections"]
    for name, one in sorted(metrics.items()):
        assert one.components == ("water", "oil"), name
        assert max(one.cumulative_relative) <= CUMULATIVE_RELATIVE_TOLERANCE, name
        assert max(one.median_step_relative) <= MEDIAN_STEP_RELATIVE_TOLERANCE, name
        assert one.within_spec_tolerance, name
    assert extraction_balance(extraction) == metrics["full_system_surface"]
    assert reloaded.solver_metadata["balance_within_spec_tolerance"] == "true"
    assert reloaded.solver_metadata["balance.reservoir_connections_within_spec_tolerance"] == "true"
    assert reloaded.solver_metadata["balance_headline"] == "full_system_surface"

    balances = pq.read_table(paths.resolve(str(reloaded.balances_path))).to_pylist()
    assert [(row["balance"], row["component"]) for row in balances] == [
        ("full_system_surface", "water"),
        ("full_system_surface", "oil"),
        ("reservoir_connections", "water"),
        ("reservoir_connections", "oil"),
    ]
    assert {row["source_term"] for row in balances} == {
        "surface component flux (q_t * mix) + boundary influx",
        "reservoir-well connection flux (geometry/state/PVT) + boundary influx",
    }
    # This fixture is closed: it has no `FlowBoundaryCondition` at all, so the boundary part
    # of both net sources is an exact zero and the extraction publishes no boundary row.
    assert [row["boundary_source_m3_sc"] for row in balances] == [0.0, 0.0, 0.0, 0.0]
    assert extraction["boundary"] == []
    assert np.asarray(extraction["net_boundary_source_m3_sc"], dtype=np.float64).max() == 0.0
    # The absolute residual and the throughput ratio are on the record beside the relative
    # one, so a large standing inventory cannot hide a small offtake error.
    assert all(row["absolute_residual_m3_sc"] >= 0.0 for row in balances)
    assert all("throughput_relative" in row for row in balances)

    # And the two really are different statements on this run, not one number written twice:
    # the surface flux is NOT the sum of the connection fluxes at every instant, because the
    # wellbores are filling and emptying.
    surface_source = np.asarray(extraction["net_surface_source_m3_sc"], dtype=np.float64)
    connection_source = np.asarray(extraction["net_connection_source_m3_sc"], dtype=np.float64)
    assert np.abs(surface_source - connection_source).max() > 1e-3
    published = {row["balance"]: row for row in balances if row["component"] == "water"}
    assert published["full_system_surface"]["net_source_m3_sc"] != pytest.approx(
        published["reservoir_connections"]["net_source_m3_sc"], rel=1e-9
    )
    # The reservoir-only statement, recomputed here from the raw extraction, agrees with the
    # published one.
    reservoir = np.asarray(extraction["reservoir_inventory_m3_sc"], dtype=np.float64)
    connection_residual = np.abs(np.diff(reservoir, axis=0) - connection_source)
    assert connection_residual.max() < 1e-2
    assert published["reservoir_connections"]["max_step_absolute_m3_sc"] == pytest.approx(
        connection_residual[:, 0].max(), rel=1e-9
    )

    # --- 7.4 connection diagnostics: the mask, not a kh split ------------------------------
    connections = pq.read_table(paths.resolve(str(reloaded.connections_path))).to_pylist()
    producer_cells = {
        (row["connection_id"], row["cell_id"]) for row in connections if row["well_id"] == "PRO1"
    }
    assert producer_cells == {(0, 3), (1, 7)}
    # The completion event closes PRO1's LOWER connection for part of February, so the two
    # connections of one well have different open times in that month. A surface rate split
    # by well index could not produce that.
    february = {
        row["connection_id"]: row
        for row in connections
        if row["well_id"] == "PRO1" and row["month_index"] == 1
    }
    assert february[0]["open_s"] > february[1]["open_s"]
    assert february[1]["open_s"] == pytest.approx(8.0 * DAY)
    assert february[0]["bhp_mean_pa"] > 0.0

    # A `SimpleWell`'s connection flux depends on an EXTRA STATE FIELD — its explicit
    # connection pressure drop — which JutulDarcy 0.3.11 does not store by default and which
    # the native perforation flux reads in preference to the density head. Asking the well
    # submodel to record it (a change to what is stored, not to what is solved) takes the
    # reservoir balance from 6.8 m³_sc of mass unaccounted for over one day to 9.3e-5.
    assert report["simple_well_extraction"]["stored_extra_state_fields"] == {
        "PRO1": ["ConnectionPressureDrop"]
    }
    assert report["simple_well_residual_m3_sc"] < 1e-2
    assert all(value != 0.0 for value in report["simple_well_connection_pressure_drop_pa"])
    # A multisegment well has no such field, so nothing is asked for and nothing is stored.
    assert extraction["stored_extra_state_fields"] == {"INJ1": [], "PRO1": []}
    # And the guard behind it is still a guard: a substate without the field is refused
    # rather than read off the re-initialised zeros.
    refusal = report["simple_well_refusal"]
    assert "ConnectionPressureDrop" in refusal
    assert "PRO1" in refusal and "request_extra_outputs!" in refusal

    # A fully masked well moves exactly nothing: not a tolerance, a multiplication by zero.
    masked = report["masked_connections"]
    assert masked and all(not row["connection_open"] for row in masked)
    assert all(row["total_mass_kg_s"] == 0.0 for row in masked)
    # The same well with its completions open carries a real flux, so the zero is the mask.
    assert all(abs(row["total_mass_kg_s"]) > 1e-3 for row in report["open_connections"])

    # --- 7.3 the same integral survives real timestep cuts ---------------------------------
    cuts = report["extraction_with_cuts"]
    assert cuts["solver"]["cut_steps"] > 0
    assert len(cuts["chunk"]["dt_s"]) == cuts["solver"]["accepted_steps"]
    assert all(value > 0.0 for value in cuts["chunk"]["dt_s"])
    assert sum(cuts["chunk"]["dt_s"]) == pytest.approx(60.0 * DAY, rel=1e-12)
    cut_monthly = integrate_monthly(well_step_table(cuts), schedule.month_edges_s).to_pylist()
    cut_producer = {row["month_index"]: row for row in cut_monthly if row["well_id"] == "PRO1"}
    # The rate-controlled month integrates to the same volume on a run with a completely
    # different step history: the control pins it, and the cut trial states are not in the
    # sum. The bottom-hole month does NOT have to agree, and is not asserted to.
    assert cut_producer[0]["liquid_prod_m3_sc"] == pytest.approx(
        producer[0]["liquid_prod_m3_sc"], rel=1e-6
    )
    assert len(cuts["chunk"]["dt_s"]) != len(dt)
    for name, one in sorted(extraction_balances(cuts).items()):
        assert one.within_spec_tolerance, name

    # --- 7.8 a control nobody could hold is CONTROL_INFEASIBLE, with the evidence ----------
    assert "requested lrat=100000.0" in report["infeasible_reason"]
    assert "operated on bhp" in report["infeasible_reason"]
    infeasible = publish_forward_result(
        job,
        case,
        {
            "schema_version": extraction["schema_version"],
            "status": "COMPLETE",
            "control_evidence": report["infeasible_evidence"],
            "control_infeasible_reason": report["infeasible_reason"],
        },
        paths,
        cost=_cost(extraction),
        solver_metadata={},
    )
    assert infeasible.status == "CONTROL_INFEASIBLE"
    assert infeasible.reason == report["infeasible_reason"]
    assert infeasible.monthly_path is None

    # --- the isolated run, read through the same table builders ----------------------------
    # An isolated well still has a surface rate — the wellbore's own decompression — so the
    # tables are built and read rather than assumed empty, and every connection in them is
    # shut.
    isolated = report["isolated_extraction"]
    isolated_steps = well_step_table(isolated)
    assert isolated_steps.num_rows == len(isolated["chunk"]["dt_s"])
    isolated_connections = connection_step_table(isolated)
    assert isolated_connections.num_rows > 0
    assert not any(isolated_connections.column("connection_open").to_pylist())
    assert not any(isolated_connections.column("total_mass_kg_s").to_pylist())
