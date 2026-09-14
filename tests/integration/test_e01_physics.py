"""E01.0: the native JutulDarcy oil-water model, and the worker's route into it.

Three real Julia processes, no more. The cold start plus the first JutulDarcy specialisation
is the expensive part of every one of these tests, so each launch is made to carry as many
independent claims as it can:

* `test_the_native_constructor_matches_numbers_computed_here` runs the standalone
  constructor diagnostic (`julia/verification/fixtures.jl --test-model`) through the
  project's own launcher, under the P0_VERIFY job timeout. Julia asserts the constructor
  against the native API and reports what it measured; this side recomputes the same
  physics from the plan's §3.1 numbers — the pore volume, both phase densities at two
  pressures, the face gravity of a vertical connection and the viscosity ratio — and
  compares. A number that both sides derive from the same wrong assumption would agree; a
  number one side reads out of JutulDarcy and the other computes from `rho_sc*exp(c*Δp)`
  agrees only if the model really is the one the plan specified. The diagnostic builds
  models and takes no time steps, so it is not a forward and does not spend the forward
  budget; what bounds it is the launcher's timeout.
* `test_the_worker_reaches_the_adapter_and_rebuilds_every_job` drives one persistent worker
  through jobs A, B and A again, every one of them reserved and resolved on a real
  `BudgetLedger`. It proves the wiring (a job reaches the constructor, runs and comes back
  with a published result describing the model that was built), and it proves the isolation
  the wiring is for: B's model must not change what the second A reports. The isolation of
  the RESULTS — that the second A reproduces the first's states and volumes, not merely its
  pore volume — is `test_e01_restart.py`'s A -> B -> A.
* `test_the_native_controls_follow_the_calendar_the_case_declares` runs the controls
  diagnostic (`julia/verification/fixtures.jl --test-controls`) the same way, and compares
  what Julia measured against what `so_recon.simulator.schedule` computes here: the real
  month lengths, the compiled intervals, the uptime, and the monthly volumes the rate
  controls imply. That diagnostic is the one place in this build where time is actually
  integrated — seven small verification forwards (a four-interval fixture, two infeasible
  rates, two isolation runs and two crossflow runs), all inside ONE launcher process bounded
  by the P0_VERIFY job timeout, against a session allowance of 64.

What this file deliberately does NOT test is the forward result itself: the accepted-substep
integrals are `test_e01_outputs.py`'s subject and the native restart is
`test_e01_restart.py`'s. Here a completed job is evidence that the constructor was reached
with the case that was named, and nothing more is read out of it.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P0_VERIFY_PROFILE, P1_LOOP_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import CASE_MANIFEST_FILENAME, write_case
from so_recon.simulator.contracts import (
    CONTROL_RATE_UNIT,
    MILLIDARCY_M2,
    SECONDS_PER_DAY,
    STANDARD_GRAVITY_M_S2,
    CaseBundle,
    ControlSegment,
    FluidSpec,
    ForwardResult,
    JobDescriptor,
    OutputRequest,
)
from so_recon.simulator.forward import forward_handoff
from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    SubprocessJuliaLauncher,
    find_julia,
)
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    CONNECTIONS_FILENAME,
    MONTHLY_FILENAME,
    STATES_FILENAME,
    extraction_balances,
)
from so_recon.simulator.schedule import compile_schedule, month_edges_s
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.validation.physics import (
    DEFAULT_TOLERANCES_RELPATH,
    CommonSupport,
    bl_cell_average,
    cartesian_zone_ids,
    compare_refinement,
    evaluate_physics,
    load_tolerances,
)
from tests.forward_case import (
    N_CELLS,
    SHAPE,
    build_case,
    cell_centers,
    write_case_arrays,
)
from tests.integration.fixture_result import publish_fixture

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_JL = ROOT / "julia" / "verification" / "fixtures.jl"

#: The §3.1 educational numbers, written here as the independent side of the comparison.
P_SC_PA = 101325.0
P_RESERVOIR_PA = 1.5e7
RHO_W_SC, RHO_O_SC = 1000.0, 800.0
C_W, C_O = 4e-10, 1e-9
MU_W, MU_O = 0.001, 0.003

#: `:closed_cell` is one 10 m cube at porosity 0.2, with its top face at the fixture datum.
CLOSED_CELL_EDGE_M = 10.0
CLOSED_CELL_POROSITY = 0.2
DATUM_M = 1000.0

#: `tests.forward_case` lays its 2x2x2 box out with 10 m layers below the same datum, so its
#: cell centres are at 1005 m and 1015 m (see `forward_case.cell_centers`).
FIXTURE_DEPTHS_M = [1005.0, 1015.0]

#: The 2x2x2 exchange fixture: 100 x 100 x 20 m of box at the porosity its arrays carry.
FIXTURE_BULK_VOLUME_M3 = 100.0 * 100.0 * 20.0
POROSITY_A = 0.25
POROSITY_B = 0.5


def _skip_unless_julia_is_installed() -> Path:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        return find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")


def _density(rho_sc: float, compressibility: float, pressure_pa: float) -> float:
    """rho(p) = rho_sc * exp(c * (p - p_sc)) — the plan's §3.1 relation, by hand."""
    return rho_sc * math.exp(compressibility * (pressure_pa - P_SC_PA))


@pytest.mark.julia
def test_the_native_constructor_matches_numbers_computed_here(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    out_path = paths.artifacts / "verification" / "constructor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    launcher = SubprocessJuliaLauncher(
        julia_exe, ROOT / "julia", timeout_s=P0_VERIFY_PROFILE.job_timeout_s
    )
    launcher.launch(FIXTURES_JL, ["--test-model"], out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    # The pinned environment, read out of the process that did the work.
    assert (report["jutul_version"], report["jutuldarcy_version"]) == ("0.4.31", "0.3.11")
    assert report["gravity_constant"] == STANDARD_GRAVITY_M_S2

    # The Julia fixture spells the §3.1 fluid out because it cannot import a Pydantic
    # default. That duplication is checked here rather than trusted: field by field, the
    # fixture is `FluidSpec()`.
    declared = FluidSpec().model_dump(mode="json")
    reported = report["fluids"]
    assert set(reported) == set(declared)
    for name, value in declared.items():
        expected = list(value) if isinstance(value, tuple) else value
        assert reported[name] == expected, name

    # --- geometry and the initial state -------------------------------------------------
    closed = report["closed_cell"]
    assert closed["n_cells"] == 1
    assert closed["phases"] == ["AqueousPhase", "LiquidPhase"]
    assert closed["pore_volume_m3"] == pytest.approx(
        CLOSED_CELL_EDGE_M**3 * CLOSED_CELL_POROSITY, rel=1e-12
    )
    assert closed["pressure_pa"] == [P_RESERVOIR_PA]
    # Water first, oil second, summing to one: a swapped phase order also sums to one.
    assert closed["saturations"] == [[0.3], [0.7]]
    assert closed["permeability_m2"] == pytest.approx([100.0 * MILLIDARCY_M2] * 3, rel=1e-12)
    # The model is built at the datum the case declares, not at the mesh's own origin: a
    # grid origined at zero would put this centre at 5 m instead of 1005 m.
    assert closed["declared_cell_centers_m"] == [[5.0, 5.0, DATUM_M + 5.0]]
    assert closed["mesh_cell_centers_m"] == closed["declared_cell_centers_m"]

    # And the other direction: a declared centre the shape and extent cannot produce is
    # refused by name. Ignoring `cell_centers_m` would make this silently succeed.
    refusal = report["rejected_geometry_message"]
    assert "grid.cell_centers_m" in refusal
    assert "zero-based cell 5" in refusal and "x axis" in refusal
    assert "17.5 m" in refusal and "15.0 m" in refusal and "2.5 m" in refusal

    # --- PVT, against the exponential the plan specifies --------------------------------
    pvt = report["pvt"]
    assert pvt["p_pa"] == [P_SC_PA, P_RESERVOIR_PA]
    expected_w = [_density(RHO_W_SC, C_W, p) for p in pvt["p_pa"]]
    expected_o = [_density(RHO_O_SC, C_O, p) for p in pvt["p_pa"]]
    assert pvt["rho_w_kg_m3"] == pytest.approx(expected_w, rel=1e-12)
    assert pvt["rho_o_kg_m3"] == pytest.approx(expected_o, rel=1e-12)
    # B = rho_sc/rho: exactly one at the reference pressure, and below one above it.
    assert (pvt["b_w"][0], pvt["b_o"][0]) == (1.0, 1.0)
    assert pvt["b_w"][1] < 1.0 and pvt["b_o"][1] < 1.0
    # Density grows with pressure, and stays positive.
    assert pvt["rho_w_kg_m3"][1] > pvt["rho_w_kg_m3"][0] > 0.0
    assert pvt["rho_o_kg_m3"][1] > pvt["rho_o_kg_m3"][0] > 0.0
    # Oil is the lighter and the more viscous phase; neither pair has been swapped.
    assert pvt["rho_w_kg_m3"][0] > pvt["rho_o_kg_m3"][0]
    assert pvt["viscosity_pa_s"] == [MU_W, MU_O]
    assert pvt["viscosity_ratio"] == 3.0

    # --- viscosity on every submodel, and native face gravity ---------------------------
    probe = report["wellbore_probe"]
    # Two layers at the case's datum, and a well datum that is the top of the box rather
    # than a number 985 m away from the cells it belongs to.
    assert probe["cell_center_depth_m"] == [DATUM_M + 5.0, DATUM_M + 15.0]
    assert probe["well_reference_depth_m"] == DATUM_M
    assert sorted(probe["submodel_viscosities_pa_s"]) == ["INJ1", "PRO1", "Reservoir"]
    assert all(v == [MU_W, MU_O] for v in probe["submodel_viscosities_pa_s"].values())
    # The filter that picks those submodels is a filter: the facility carries no viscosity.
    assert probe["submodels_without_viscosity"] == ["Facility"]
    # The parameter JutulDarcy will actually use IS its own compute_face_gdz.
    assert probe["face_gdz"] == probe["native_face_gdz"]
    # z is depth, positive down, and gdz = -g*(z_r - z_l): one cell edge of head downwards
    # across a vertical connection, and none at all inside a layer.
    assert probe["vertical_face_gdz"] == pytest.approx(
        [-STANDARD_GRAVITY_M_S2 * CLOSED_CELL_EDGE_M] * len(probe["vertical_face_gdz"]),
        rel=1e-12,
    )
    assert probe["vertical_face_gdz"] and probe["horizontal_face_gdz"]
    assert all(value == 0.0 for value in probe["horizontal_face_gdz"])
    # Pore volume is a parameter and does not follow pressure: 8 cells of 10 m cube at 0.2.
    expected_pv = 8 * CLOSED_CELL_EDGE_M**3 * CLOSED_CELL_POROSITY
    assert probe["pore_volume_m3"] == pytest.approx(expected_pv, rel=1e-12)
    assert probe["pore_volume_after_pressure_change_m3"] == probe["pore_volume_m3"]

    # --- and a model that is rebuilt, not reused ----------------------------------------
    rebuild = report["rebuild"]
    assert rebuild["pore_volume_a1_m3"] == rebuild["pore_volume_a2_m3"] == 200.0
    assert rebuild["pore_volume_b_m3"] == 400.0


def _unloaded_machine(session_dir: Path) -> ResourceSnapshot:
    """A real snapshot with only its memory fields pinned.

    Whether this host has memory to spare is Task 3's subject, measured by Task 3's own
    tests; making a physics test fail because something else was running would say nothing
    about the model. The clock, the CPU time, the free disk and the measurement method all
    still come from the probe, so the cost record remains a measurement.
    """
    return probe_resources(None, session_dir).model_copy(
        update={
            "total_bytes": 64 * 1024**3,
            "available_bytes": 32 * 1024**3,
            "process_rss_bytes": 1024**3,
            "swap_used_bytes": 0,
        }
    )


def _publish_case(
    paths: ProjectPaths, *, label: str, porosity: float, **overrides: NDArray[np.float64]
) -> tuple[CaseBundle, Path]:
    """Write one complete case, with its own arrays, and return it with its manifest path."""
    refs = write_case_arrays(
        paths,
        subdir=f"arrays-{label}",
        porosity=np.full(N_CELLS, porosity, dtype=np.float64),
        **overrides,
    )
    case = build_case(refs, case_id=f"case-e01-{label}")
    ctx = RunContext.start(command=f"physics-{label}", argv=[], cfg=None, paths=paths)
    write_case(case, paths, ctx)
    return case, ctx.run_dir / CASE_MANIFEST_FILENAME


def _displaced_cell_centers() -> NDArray[np.float64]:
    """The fixture's geometry with one cell moved off the uniform grid it declares.

    Python's own `validate_case` accepts this — it checks that centres are finite and that z
    is a depth, not that they reproduce `shape` and `extent_m` — so the array reaches Julia
    intact and the adapter is the side that has to notice.
    """
    centers = cell_centers(SHAPE).copy()
    centers[5, 0] += 7.0
    return centers


@pytest.mark.julia
def test_the_worker_reaches_the_adapter_and_rebuilds_every_job(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()

    case_a, path_a = _publish_case(paths, label="a", porosity=POROSITY_A)
    case_b, path_b = _publish_case(paths, label="b", porosity=POROSITY_B)
    # Same physics as A, but its declared geometry no longer describes the grid it names.
    case_c, path_c = _publish_case(
        paths, label="c", porosity=POROSITY_A, cell_centers_m=_displaced_cell_centers()
    )
    assert len({case_a.model_hash, case_b.model_hash, case_c.model_hash}) == 3

    solver_path = paths.artifacts / "solver" / "e01_solver.json"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    solver_path.write_text(
        json.dumps({"max_timestep_days": 30, "max_nonlinear_iterations": 15}), encoding="utf-8"
    )
    session_dir = paths.artifacts / "session"

    def probe() -> ResourceSnapshot:
        return _unloaded_machine(session_dir)

    ledger = BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=paths.artifacts / "ledger.json",
        session_id="session-e01-physics",
        probe=probe,
    )

    def descriptor(job_id: str, case: CaseBundle, case_path: Path) -> JobDescriptor:
        return JobDescriptor(
            job_id=job_id,
            case_path=paths.relative(case_path),
            case_sha256=sha256_file(case_path),
            model_hash=case.model_hash,
            solver_config_path=paths.relative(solver_path),
            solver_config_sha256=sha256_file(solver_path),
            output_request=OutputRequest(
                state_times_s=(30.0 * SECONDS_PER_DAY,), keep_native_restart=False
            ),
            seed=20260913,
            result_dir=f"artifacts/results/{job_id}",
            attempt=1,
        )

    def published_record(result: ForwardResult) -> dict[str, Any]:
        """Read back the record the worker published, proving its digest on the way."""
        record_path = paths.resolve(result.solver_metadata["result_path"])
        assert record_path.parent == paths.root / f"artifacts/results/{result.job_id}"
        assert sha256_file(record_path) == result.solver_metadata["result_sha256"]
        parsed: dict[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
        return parsed

    with PersistentJuliaWorker(
        julia_exe,
        ROOT / "julia",
        session_dir,
        P0_VERIFY_PROFILE,
        paths=paths,
        probe=probe,
    ) as worker:

        def run(job_id: str, case: CaseBundle, case_path: Path) -> ForwardResult:
            job = descriptor(job_id, case, case_path)
            return worker.submit(job, ledger, handoff=forward_handoff(job, case, paths))

        results = [
            run("job-e01-a1", case_a, path_a),
            run("job-e01-b1", case_b, path_b),
            run("job-e01-a2", case_a, path_a),
        ]
        displaced = run("job-e01-c1", case_c, path_c)
        assert worker.pid == worker.handshake.pid  # one process behind all four jobs

    records = [published_record(result) for result in results]

    for result, record in zip(results, records, strict=True):
        # The adapter was reached, the case was built into a model and the forward ran to
        # the end of its schedule. What was BUILT is what this test reads; what was
        # integrated belongs to the output and restart suites.
        assert result.status == "COMPLETE", result.reason
        assert result.physics_class == "OW"
        assert record["status"] == result.status
        built = record["model"]
        assert built["n_cells"] == N_CELLS
        # The case's own datum reached the model: 1005 m and 1015 m, not 5 m and 15 m.
        assert built["cell_center_depth_m"] == FIXTURE_DEPTHS_M
        assert built["phases"] == ["AqueousPhase", "LiquidPhase"]
        assert built["reference_densities_kg_m3"] == [RHO_W_SC, RHO_O_SC]
        assert built["gravity_m_s2"] == STANDARD_GRAVITY_M_S2
        # Both wells of the case reached the model, and both carry the case's viscosities.
        assert built["wells"] == ["INJ1", "PRO1"]
        assert sorted(built["viscosities_pa_s"]) == ["INJ1", "PRO1", "Reservoir"]
        assert all(v == [MU_W, MU_O] for v in built["viscosities_pa_s"].values())

    # A -> B -> A: the second A is the first A again, and none of B is left in it.
    volumes = [record["model"]["pore_volume_m3"] for record in records]
    assert volumes[0] == pytest.approx(FIXTURE_BULK_VOLUME_M3 * POROSITY_A, rel=1e-12)
    assert volumes[1] == pytest.approx(FIXTURE_BULK_VOLUME_M3 * POROSITY_B, rel=1e-12)
    assert volumes[2] == volumes[0]

    # A declared geometry that does not describe its own grid is refused, by name, before
    # anything is simulated — and no model record is published for it.
    assert displaced.status == "INVALID_INPUT"
    assert displaced.reason is not None
    assert "grid.cell_centers_m" in displaced.reason
    assert "zero-based cell 5" in displaced.reason and "x axis" in displaced.reason
    assert published_record(displaced)["model"] is None

    # Four attempts, all of them paid for.
    assert [entry.job_id for entry in ledger.record.entries] == [
        "job-e01-a1",
        "job-e01-b1",
        "job-e01-a2",
        "job-e01-c1",
    ]
    assert [entry.state for entry in ledger.record.entries] == [
        "COMPLETE",
        "COMPLETE",
        "COMPLETE",
        "FAILED",
    ]
    assert [entry.status for entry in ledger.record.entries] == [
        "COMPLETE",
        "COMPLETE",
        "COMPLETE",
        "INVALID_INPUT",
    ]


#: The fixture of `verification_case(:two_interval_controls)`: January and the leap
#: February of 2020, split by a completion event on day 45 and a field shutdown on day 52.
CONTROLS_START = date(2020, 1, 1)
CONTROLS_MONTHS = 2
CONTROLS_RATE_M3_DAY = 2.0
CONTROLS_PRODUCER_BHP_FLOOR_PA = 5.0e6
CONTROLS_PRODUCER_BHP_TARGET_PA = 1.9e7
CONTROLS_INJECTOR_BHP_CEILING_PA = 4.0e7
CONTROLS_INJECTOR_BHP_TARGET_PA = 2.2e7


@pytest.mark.julia
def test_the_native_controls_follow_the_calendar_the_case_declares(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    out_path = paths.artifacts / "verification" / "controls.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    launcher = SubprocessJuliaLauncher(
        julia_exe, ROOT / "julia", timeout_s=P0_VERIFY_PROFILE.job_timeout_s
    )
    launcher.launch(FIXTURES_JL, ["--test-controls"], out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert (report["jutul_version"], report["jutuldarcy_version"]) == ("0.4.31", "0.3.11")

    fixture = report["two_interval_controls"]
    assert fixture["start_date"] == CONTROLS_START.isoformat()
    assert fixture["control_rate_unit"] == CONTROL_RATE_UNIT

    # --- the calendar is the real one ----------------------------------------------------
    # The case's report edges are January and the LEAP February, recomputed here from the
    # calendar rather than read back from the file that produced them. Twelve 30-day months
    # would put the second edge 60 days out instead of 60 ... which is the point: the third
    # edge lands on 60 days only because 31 + 29 is 60, and on 61 in 2021.
    expected_edges = month_edges_s(CONTROLS_START, CONTROLS_MONTHS)
    assert fixture["report_edges_s"] == pytest.approx(list(expected_edges), rel=0, abs=0)
    assert expected_edges[1] == 31.0 * SECONDS_PER_DAY
    assert expected_edges[2] - expected_edges[1] == 29.0 * SECONDS_PER_DAY
    assert month_edges_s(date(2021, 1, 1), CONTROLS_MONTHS)[2] == 59.0 * SECONDS_PER_DAY

    # --- the two schedule compilers agree -------------------------------------------------
    # Julia grouped the case's flat control segments into intervals; Python compiles the
    # same segments with `compile_schedule`. Parsing them into `ControlSegment` also puts
    # the fixture through the real contract: an illegal role, target or rate would not
    # survive this line.
    segments = tuple(ControlSegment.model_validate(c) for c in fixture["controls"])
    schedule = compile_schedule(expected_edges, segments)
    assert schedule.edges_s == pytest.approx(fixture["edges_s"], rel=0, abs=0)
    assert list(schedule.month_index) == fixture["month_index"]
    assert schedule.wells == ("INJ1", "PRO1")
    # Four intervals, three of them inside February: the two events are not rounded away.
    assert schedule.month_index == (0, 1, 1, 1)
    assert schedule.edges_s == tuple(d * SECONDS_PER_DAY for d in (0.0, 31.0, 45.0, 52.0, 60.0))
    for interval, group in enumerate(fixture["controls_by_interval"]):
        for control in group:
            compiled = schedule.control_for(interval, control["well_id"])
            assert compiled == ControlSegment.model_validate(control)

    # --- roles, masks and limits are restated on every interval ---------------------------
    assert fixture["control_types"]["PRO1"] == [
        "ProducerControl",
        "ProducerControl",
        "DisabledControl",
        "ProducerControl",
    ]
    assert fixture["control_types"]["INJ1"] == [
        "InjectorControl",
        "InjectorControl",
        "DisabledControl",
        "InjectorControl",
    ]
    # The producer's lower completion closes on interval 1, everything closes on interval 2,
    # and interval 3 is fully open AGAIN: a mask that was inherited could not come back.
    assert fixture["masks"]["PRO1"] == [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    assert fixture["masks"]["INJ1"] == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

    # Only the limits the case recorded. A bhp control and a shut well carry none, and
    # JutulDarcy's own defaults (which WOULD have added a rate floor, as the diagnostic
    # reports) are switched off.
    assert fixture["recorded_bhp_limits_pa"]["PRO1"] == [
        CONTROLS_PRODUCER_BHP_FLOOR_PA,
        CONTROLS_PRODUCER_BHP_FLOOR_PA,
        None,
        None,
    ]
    # The injector's ceiling is recorded while it runs on a rate; a well already ON bhp
    # carries no limit, for either role.
    assert fixture["recorded_bhp_limits_pa"]["INJ1"] == [
        CONTROLS_INJECTOR_BHP_CEILING_PA,
        CONTROLS_INJECTOR_BHP_CEILING_PA,
        None,
        None,
    ]
    assert "rate_lower" in fixture["native_default_limit_for_bhp_producer"]

    # --- the injected stream, as BUILT on every branch that constructs one ---------------
    # A wrong mixture or surface density in `InjectorControl` is silent wrong physics, and
    # the bhp branch builds its own rather than inheriting one: a rate control that later
    # hits its bhp limit switches through `replace_target`, which copies the mixture and
    # density across and so could never reveal a mistake in that constructor.
    assert fixture["injector_targets"] == [
        "SurfaceWaterRateTarget",
        "SurfaceWaterRateTarget",
        "DisabledTarget",
        "BottomHolePressureTarget",
    ]
    # Pure water by mass, water first — on the bhp branch too, not only on the rate one.
    assert fixture["injector_mixtures"] == [[1.0, 0.0], [1.0, 0.0], None, [1.0, 0.0]]
    # And the surface density is the fluid's own water density, never Jutul's default 1.0.
    assert fixture["injector_densities_kg_m3"] == [RHO_W_SC, RHO_W_SC, None, RHO_W_SC]
    assert RHO_W_SC != 1.0

    # --- the native sign convention --------------------------------------------------------
    producer = fixture["evidence"]["PRO1"]
    injector = fixture["evidence"]["INJ1"]
    assert [s["operating_target"] for s in producer] == ["lrat", "lrat", "disabled", "bhp"]
    assert all(s["honoured"] for s in producer)
    # Production is negative natively and positive in the public number beside it; the
    # native rate is the day rate divided by 86400 and nothing else.
    assert producer[0]["native_lrat_m3_s"] == pytest.approx(
        -CONTROLS_RATE_M3_DAY / SECONDS_PER_DAY, rel=1e-12
    )
    assert producer[0]["liquid_rate_m3_day"] == pytest.approx(CONTROLS_RATE_M3_DAY, rel=1e-12)
    # Injection is the other direction under the same production-positive convention.
    assert injector[0]["water_rate_m3_day"] == pytest.approx(-CONTROLS_RATE_M3_DAY, rel=1e-12)
    assert producer[3]["bhp_pa"] == pytest.approx(CONTROLS_PRODUCER_BHP_TARGET_PA, rel=1e-12)
    # The injector's bhp interval really ran, on the bhp its own constructor was given.
    assert [s["operating_target"] for s in injector] == ["wrat", "wrat", "disabled", "bhp"]
    assert injector[3]["bhp_pa"] == pytest.approx(CONTROLS_INJECTOR_BHP_TARGET_PA, rel=1e-12)
    assert injector[3]["water_rate_m3_day"] < 0.0

    # --- uptime, and Vo + Vw = q_liquid * uptime ------------------------------------------
    # The uptime Julia used is the open part of each interval; the same number falls out of
    # the compiled schedule here, aggregated to the months the case reports on.
    interval_uptime = fixture["uptime_s"]
    assert interval_uptime == pytest.approx(
        [d * SECONDS_PER_DAY for d in (31.0, 14.0, 0.0, 8.0)], rel=0, abs=0
    )
    monthly = [0.0] * CONTROLS_MONTHS
    for month, seconds in zip(fixture["month_index"], interval_uptime, strict=True):
        monthly[month] += seconds
    assert tuple(monthly) == schedule.monthly_uptime_s("PRO1")
    assert schedule.monthly_uptime_s("PRO1") == (31.0 * SECONDS_PER_DAY, 22.0 * SECONDS_PER_DAY)

    volumes = fixture["liquid_volume_m3"]
    for interval in (0, 1):
        expected = CONTROLS_RATE_M3_DAY * interval_uptime[interval] / SECONDS_PER_DAY
        assert volumes[interval] == pytest.approx(expected, rel=1e-10)
        # Both phases really flow, so the identity is not "oil only" wearing a total's name.
        assert producer[interval]["oil_rate_m3_day"] > 0.0
        assert producer[interval]["water_rate_m3_day"] > 0.0
        assert producer[interval]["oil_rate_m3_day"] + producer[interval][
            "water_rate_m3_day"
        ] == pytest.approx(CONTROLS_RATE_M3_DAY, rel=1e-10)
    # No uptime, no volume, and no phase split to report.
    assert interval_uptime[2] == 0.0
    assert volumes[2] == 0.0
    assert (producer[2]["oil_rate_m3_day"], producer[2]["water_rate_m3_day"]) == (0.0, 0.0)
    schedule.reject_flow_without_uptime("PRO1", (volumes[0], sum(volumes[1:])))

    # --- an unreachable rate is CONTROL_INFEASIBLE, with the target it really ran on -------
    infeasible = report["infeasible"]["evidence"]
    assert infeasible["requested_target"] == "lrat"
    assert infeasible["requested_value"] == 1.0e5
    assert infeasible["honoured"] is False
    assert infeasible["operating_target"] == "bhp"
    assert infeasible["bhp_pa"] == pytest.approx(CONTROLS_PRODUCER_BHP_FLOOR_PA, rel=1e-12)
    # The well kept producing; the achieved rate is four orders below what was demanded.
    assert 0.0 < infeasible["liquid_rate_m3_day"] < 1.0e-2 * infeasible["requested_value"]
    reason = report["infeasible"]["reason"]
    assert "requested lrat=100000.0" in reason and "operated on bhp" in reason
    assert "PRO1" in reason

    # The same recorded `bhp_limit_pa` is a FLOOR under a producer and a CEILING over an
    # injector; one `:bhp` key does both because the native check reads it off the control.
    over = report["infeasible_injector"]["evidence"]
    assert over["requested_target"] == "wrat"
    assert over["honoured"] is False
    assert over["operating_target"] == "bhp"
    assert over["bhp_pa"] == pytest.approx(CONTROLS_INJECTOR_BHP_CEILING_PA, rel=1e-12)
    # Injection is negative under the production-positive convention, and far short of the
    # 100 000 m3_sc/day demanded.
    assert -1.0e-2 * over["requested_value"] < over["water_rate_m3_day"] < 0.0
    assert "requested wrat=100000.0" in report["infeasible_injector"]["reason"]

    # --- full isolation is zero connection mass flux, exactly ------------------------------
    isolation = report["isolation"]
    assert isolation["isolated"]["max_abs_dp_pa"] == 0.0
    assert isolation["isolated"]["max_abs_dsw"] == 0.0
    # Not vacuous: the same well with its completions open empties the box.
    assert isolation["open"]["max_abs_dp_pa"] > 1.0e6
    assert isolation["open"]["liquid_rate_m3_day"] > 1.0
    # What still crosses the isolated well's surface is the wellbore's own decompression,
    # two orders below the open well and a small part of the mass standing in the bore.
    assert (
        isolation["isolated"]["liquid_rate_m3_day"] < 0.01 * isolation["open"]["liquid_rate_m3_day"]
    )
    assert (
        isolation["isolated"]["surface_mass_over_one_day_kg"]
        < 0.05 * isolation["isolated"]["well_mass_kg"]
    )

    # --- a shut surface is not a shut well --------------------------------------------------
    crossflow = report["crossflow"]
    for run in crossflow.values():
        assert run["operating_target"] == "disabled"
        # DisabledControl IS the native zero-net-surface-rate formulation: exactly zero.
        assert run["surface_mass_rate_kg_s"] == 0.0
        assert run["liquid_rate_m3_day"] == 0.0
    # And it is not an isolated well: with both completions open the wellbore carried fluid
    # from the deep, higher-pressure connection to the shallow one while the surface stayed
    # at zero. Closing the completions is what takes that away.
    assert (
        crossflow["coupled"]["upper_perforated_pressure_pa"]
        > crossflow["isolated"]["upper_perforated_pressure_pa"]
    )
    assert (
        crossflow["coupled"]["lower_perforated_pressure_pa"]
        < crossflow["isolated"]["lower_perforated_pressure_pa"]
    )
    coupled = abs(crossflow["coupled"]["well_segment_mass_flux_kg_s"][1])
    isolated = abs(crossflow["isolated"]["well_segment_mass_flux_kg_s"][1])
    assert coupled > 50.0 * isolated

    # --- the refusals name what they refuse -----------------------------------------------
    # Every guard in `controls.jl` that stands between a malformed control and silent wrong
    # physics. The mask-length one matters most: `apply_perforation_mask!` iterates
    # `eachindex(mask)`, so a short mask would leave the remaining perforations fully open
    # with no complaint from the backend, and Python cannot catch it — a `ControlSegment`
    # never sees the model's connection count.
    refusals = report["refusals"]
    assert set(refusals) == {
        "missing_well",
        "duplicate_well",
        "unknown_well",
        "mask_too_short",
        "mask_too_long",
        "mask_not_boolean",
        "producer_phase_rate",
        "zero_rate",
        "non_positive_bhp_limit",
        "unknown_boundary_kind",
        "closed_boundary_with_cells",
    }
    assert "1 entries for 2 connections" in refusals["mask_too_short"]
    assert "3 entries for 2 connections" in refusals["mask_too_long"]
    assert "not partially open" in refusals["mask_not_boolean"]
    assert "connection_open[1] is 0.5" in refusals["mask_not_boolean"]
    assert "two controls on one interval" in refusals["duplicate_well"]
    assert "GHOST" in refusals["unknown_well"]
    assert "the model does not have" in refusals["unknown_well"]
    assert "SPEC 9.1" in refusals["producer_phase_rate"]
    assert "role='shut'" in refusals["zero_rate"]
    assert "must be positive when given" in refusals["non_positive_bhp_limit"]
    # Task 10.6 built `pressure_water`, so what is refused by KIND now is anything else; the
    # refusals for an incompletely specified `pressure_water` boundary are asserted in the
    # operational suite below, where that boundary is actually used.
    assert "aquifer" in refusals["unknown_boundary_kind"]
    assert "'closed' and 'pressure_water'" in refusals["unknown_boundary_kind"]
    assert "closed boundary names no cells" in refusals["closed_boundary_with_cells"]
    # Every one of them names the well or the field it is about, never just "invalid".
    for label in ("missing_well", "duplicate_well", "mask_too_short", "mask_not_boolean"):
        assert "INJ1" in refusals[label] or "PRO1" in refusals[label], label

    hole = report["missing_control_message"]
    assert "INJ1" in hole and "never inherited" in hole
    crossflow_refusal = report["crossflow_refusal_message"]
    assert "allow_crossflow=false" in crossflow_refusal
    assert "0.3.11" in crossflow_refusal


# ======================================================================================
# E01.10 — the operational fixtures: mixing, crossflow, roles, boundaries
# ======================================================================================

OPERATIONS_JL = ROOT / "julia" / "verification" / "operations.jl"
REFINEMENT_JL = ROOT / "julia" / "verification" / "refinement.jl"
TOLERANCES_PATH = ROOT / DEFAULT_TOLERANCES_RELPATH

#: The 10.4 sector, restated here so the published case is checked against the plan's own
#: numbers rather than against the file that produced them.
TWO_LAYER_SHAPE = (4, 4, 2)
TWO_LAYER_KH_MD = (200.0, 50.0)
TWO_LAYER_POROSITY = (0.25, 0.15)
TWO_LAYER_SW = (0.25, 0.65)
#: The producer's two connections: column (3, 3) of each layer, zero-based.
TWO_LAYER_PERFORATED_CELLS = (15, 31)
MIXING_LIQUID_RATE_M3_DAY = 4.0

#: The 10.6 sector: two 10 m cubes at 100 mD, so the native face transmissibility is
#: `A*k/dx = 100 m² * 9.869233e-14 m² / 10 m`. Recomputed here and compared against the
#: model's own `Transmissibilities` parameter.
SECTOR_FACE_TRANSMISSIBILITY = 10.0**2 * 100.0 * MILLIDARCY_M2 / 10.0


def _launch_json(
    julia_exe: Path, paths: ProjectPaths, script: Path, flag: str, timeout_s: int, label: str
) -> dict[str, Any]:
    """Run one verification diagnostic in its own process and read the payload it wrote."""
    out_path = paths.artifacts / "verification" / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    launcher = SubprocessJuliaLauncher(julia_exe, ROOT / "julia", timeout_s=timeout_s)
    launcher.launch(script, [flag], out_path)
    payload: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert (payload["jutul_version"], payload["jutuldarcy_version"]) == ("0.4.31", "0.3.11")
    return payload


def _connection_rows(
    extraction: dict[str, Any], well_id: str, connection_id: int
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in extraction["connections"]
        if row["well_id"] == well_id and row["connection_id"] == connection_id
    ]
    return sorted(rows, key=lambda row: row["step"])


def _standard_connection_split(
    extraction: dict[str, Any], well_id: str, connection_id: int, step: int
) -> tuple[float, float]:
    """One connection's water and oil flux at one substep, as a STANDARD volume rate.

    The published cross term is a component MASS flux in kg/s; dividing by the phase's own
    reference density is what makes two components comparable and is the same conversion the
    inventory uses. Positive is out of the reservoir, so a producing connection is positive.
    """
    rho_w, rho_o = extraction["reference_densities_kg_m3"]
    row = _connection_rows(extraction, well_id, connection_id)[step]
    return (float(row["water_mass_kg_s"]) / rho_w, float(row["oil_mass_kg_s"]) / rho_o)


def _within_spec(balances: dict[str, Any]) -> None:
    for name, metrics in balances.items():
        assert metrics.within_spec_tolerance, f"{name}: {metrics.cumulative_relative}"


@pytest.mark.julia
def test_the_operational_fixtures_mix_crossflow_isolate_and_are_supported(
    tmp_project: Path,
) -> None:
    """E01.10.4-10.6 on the pinned solver, in ONE P0_VERIFY-bounded Julia process.

    Nine small forwards: a two-layer sector produced through one well with a completion in
    each layer, the same sector shut in with its wellbore open and then with it closed, a well
    that produces, is shut and comes back as an injector inside one calendar month, two
    bottom-hole probes, and a produced sector under three treatments of its outer face. Every
    one of them is at or below 128 cells, 2 layers, 3 wells and 12 report intervals, which is
    what P0_VERIFY allows.

    Julia measures what only Julia can see — the native connection flux through each
    perforation, the native boundary flux, the native face transmissibility, the control a
    well really operated on. This side recomputes what it can from geometry and the §3.1
    numbers, publishes the results the production path would publish, and scores them against
    `configs/e01_tolerances.yml`.
    """
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    report = _launch_json(
        julia_exe,
        paths,
        OPERATIONS_JL,
        "--test-operations",
        P0_VERIFY_PROFILE.job_timeout_s,
        "operations",
    )
    assert report["gravity_constant"] == STANDARD_GRAVITY_M_S2
    fixtures = report["fixtures"]
    tolerances = load_tolerances(TOLERANCES_PATH)
    closed_flux_max = tolerances["closed_connection_mass_kg_s_max"]
    rate_tolerance = tolerances["rate_control_relative_max"]

    # ---- 10.4 the two-layer sector is the sector the plan describes ---------------------
    mixing = fixtures["mixing"]
    assert mixing["status"] == "COMPLETE"
    case = mixing["case"]
    assert tuple(case["grid"]["shape"]) == TWO_LAYER_SHAPE
    arrays = mixing["arrays"]
    porosity = np.asarray(arrays["porosity"], dtype=np.float64)
    permeability = np.asarray(arrays["permeability_m2"], dtype=np.float64)
    sw0 = np.asarray(arrays["sw"], dtype=np.float64)
    upper = np.arange(16)
    lower = np.arange(16, 32)
    for cells, phi, kh, sw in zip(
        (upper, lower), TWO_LAYER_POROSITY, TWO_LAYER_KH_MD, TWO_LAYER_SW, strict=True
    ):
        assert porosity[cells] == pytest.approx(phi)
        assert permeability[0][cells] == pytest.approx(kh * MILLIDARCY_M2, rel=1e-12)
        assert sw0[cells] == pytest.approx(sw)
    # A producer with two connections, one per layer, and one injector in each layer: three
    # wells, which is the P0_VERIFY ceiling.
    wells = {w["well_id"]: w for w in case["wells"]}
    assert sorted(wells) == ["INJ_LOWER", "INJ_UPPER", "PRO1"]
    assert tuple(wells["PRO1"]["cells"]) == TWO_LAYER_PERFORATED_CELLS

    # ---- 10.4 the surface composition is NATIVE MIXING ----------------------------------
    extraction = mixing["extraction"]
    rho_w, rho_o = extraction["reference_densities_kg_m3"]
    producer = extraction["wells"]["PRO1"]
    n_steps = len(extraction["chunk"]["dt_s"])
    # SPEC 9.1: the case prescribes a TOTAL standard liquid rate and never two phase rates.
    targets = {c["target"] for c in case["controls"] if c["well_id"] == "PRO1"}
    assert targets == {"liquid_rate"}
    for step in range(n_steps):
        qw = -float(producer["surface_water_m3_s"][step])
        qo = -float(producer["surface_oil_m3_s"][step])
        assert (qw + qo) * SECONDS_PER_DAY == pytest.approx(
            MIXING_LIQUID_RATE_M3_DAY, rel=rate_tolerance
        )
        # Both phases really flow, so the split below is a split and not "oil only".
        assert qw > 0.0 and qo > 0.0
    last = n_steps - 1
    upper_w, upper_o = _standard_connection_split(extraction, "PRO1", 0, last)
    lower_w, lower_o = _standard_connection_split(extraction, "PRO1", 1, last)
    upper_fw = upper_w / (upper_w + upper_o)
    lower_fw = lower_w / (lower_w + lower_o)
    # The two layers deliver very different fluid: 200 mD at Sw = 0.25 is nearly dry oil and
    # 50 mD at Sw = 0.65 is nearly pure water, which is the §3.1 Corey pair's own answer.
    assert upper_fw < 0.05 and lower_fw > 0.95
    surface_fw = -float(producer["surface_water_m3_s"][last]) / (
        -float(producer["surface_water_m3_s"][last]) - float(producer["surface_oil_m3_s"][last])
    )
    # The stream that reaches the surface is what the two connections delivered, mixed in the
    # wellbore — strictly between them, and equal to their sum rather than to an assumed split.
    assert upper_fw < surface_fw < lower_fw
    connection_fw = (upper_w + lower_w) / (upper_w + lower_w + upper_o + lower_o)
    assert surface_fw == pytest.approx(connection_fw, rel=1e-5)
    _within_spec(extraction_balances(extraction))

    # ---- 10.4 a shut SURFACE with an open wellbore: opposed connection fluxes ------------
    crossflow = fixtures["crossflow_open"]["extraction"]
    assert fixtures["crossflow_open"]["status"] == "COMPLETE"
    for step in range(len(crossflow["chunk"]["dt_s"])):
        assert crossflow["wells"]["XF1"]["operating_target"][step] == "disabled"
        # `DisabledControl` IS the native zero-net-surface formulation: an exact zero.
        assert [c[step] for c in crossflow["wells"]["XF1"]["surface_component_mass_kg_s"]] == [
            0.0,
            0.0,
        ]
    shallow = _connection_rows(crossflow, "XF1", 0)[0]
    deep = _connection_rows(crossflow, "XF1", 1)[0]
    assert shallow["connection_open"] and deep["connection_open"]
    # OPPOSED, and neither of them small: the wellbore's segments carry mass out of the
    # higher-potential layer and into the lower-potential one while nothing leaves the well.
    assert shallow["total_mass_kg_s"] > 1e-3
    assert deep["total_mass_kg_s"] < -1e-3
    # What they differ by is what the wellbore itself stored, and that is a small part of it.
    stored = abs(shallow["total_mass_kg_s"] + deep["total_mass_kg_s"])
    assert stored < 0.2 * max(abs(shallow["total_mass_kg_s"]), abs(deep["total_mass_kg_s"]))
    # CONSERVATION INCLUDING WELL STORAGE. The reservoir alone is NOT conserved here — it
    # gave up oil and took water through the connections — and the reservoir plus the wellbore
    # is, to nine significant figures, with no surface flux at all to account for it.
    inventory = np.asarray(crossflow["inventory_m3_sc"], dtype=np.float64)
    reservoir = np.asarray(crossflow["reservoir_inventory_m3_sc"], dtype=np.float64)
    total_drift = np.abs(inventory[-1] - inventory[0]) / inventory[0]
    assert total_drift.max() < 1e-6
    reservoir_change = reservoir[-1] - reservoir[0]
    assert np.abs(reservoir_change).max() > 0.1
    connection_source = np.asarray(crossflow["net_connection_source_m3_sc"], dtype=np.float64).sum(
        axis=0
    )
    assert reservoir_change == pytest.approx(connection_source, rel=1e-3)
    _within_spec(extraction_balances(crossflow))

    # ---- 10.4 the SAME surface condition with both completions closed --------------------
    isolated = fixtures["crossflow_closed"]["extraction"]
    assert fixtures["crossflow_closed"]["status"] == "COMPLETE"
    for connection_id in (0, 1):
        rows = _connection_rows(isolated, "XF1", connection_id)
        assert rows and all(not row["connection_open"] for row in rows)
        assert max(abs(float(row["total_mass_kg_s"])) for row in rows) <= closed_flux_max
    # A well that stopped producing is not a well that was isolated, and the difference is
    # visible in the reservoir: the open wellbore equalised the layers it was shut in across,
    # and the closed one left them where they were put.
    gap = report["crossflow_layer_gap_pa"]
    assert gap["initial"] == pytest.approx(4.0e6)
    assert gap["open_final"] < 0.05 * gap["isolated_final"]
    assert gap["isolated_final"] > 0.5 * gap["initial"]

    # ---- 10.5 one immutable case: producer -> shut -> injector --------------------------
    roles = fixtures["roles"]
    assert roles["status"] == "COMPLETE"
    roles_case = roles["case"]
    roles_extraction = roles["extraction"]
    # The event days are INSIDE a real calendar month: February 2020 carries all three roles.
    edges_days = [e / SECONDS_PER_DAY for e in roles_extraction["chunk"]["edges_s"]]
    assert edges_days == pytest.approx([0.0, 31.0, 45.0, 52.0, 60.0, 91.0])
    month_edges = month_edges_s(date(2020, 1, 1), 3)
    assert [e / SECONDS_PER_DAY for e in month_edges] == pytest.approx([0.0, 31.0, 60.0, 91.0])
    # Requested and actual role, restated on every interval and never inherited.
    segments = tuple(ControlSegment.model_validate(c) for c in roles_case["controls"])
    assert [s.role for s in segments] == ["producer", "shut", "injector"]
    assert [list(s.connection_open) for s in segments] == [
        [True, True],
        [True, False],
        [True, True],
    ]
    evidence = roles_extraction["control_evidence"]["OPS1"]
    assert all(step["honoured"] for step in evidence)
    operating = roles_extraction["wells"]["OPS1"]["operating_target"]
    assert set(operating) == {"lrat", "disabled", "wrat"}
    assert operating[0] == "lrat" and operating[-1] == "wrat"
    shut_steps = [k for k, target in enumerate(operating) if target == "disabled"]
    assert shut_steps
    # ISOLATION AND SHUTDOWN, on the same well at the same instant. The closed completion
    # carries nothing at all; the open one, under the very same shut surface, does not meet
    # the isolation gate — which is precisely why they are not one claim.
    for step in shut_steps:
        closed_row = _connection_rows(roles_extraction, "OPS1", 1)[step]
        open_row = _connection_rows(roles_extraction, "OPS1", 0)[step]
        assert not closed_row["connection_open"] and open_row["connection_open"]
        assert abs(float(closed_row["total_mass_kg_s"])) <= closed_flux_max
        assert abs(float(open_row["total_mass_kg_s"])) > closed_flux_max
    _within_spec(extraction_balances(roles_extraction))

    # The published monthly volumes keep production and injection APART. February holds both,
    # and a single signed number for that month would report their difference as production.
    _, roles_dir = publish_fixture(roles, paths, "ops-roles", report, world="operations")
    monthly = pq.read_table(roles_dir / "monthly.parquet").to_pylist()
    by_month = {int(row["month_index"]): row for row in monthly if row["well_id"] == "OPS1"}
    assert sorted(by_month) == [0, 1, 2]
    assert by_month[0]["liquid_prod_m3_sc"] > 0.0 and by_month[0]["water_inj_m3_sc"] == 0.0
    assert by_month[1]["liquid_prod_m3_sc"] > 0.0 and by_month[1]["water_inj_m3_sc"] > 0.0
    assert by_month[2]["liquid_prod_m3_sc"] == 0.0 and by_month[2]["water_inj_m3_sc"] > 0.0
    # 14 days of production and 8 of injection at 0.5 m3_sc/day, on the same month's row.
    assert by_month[1]["liquid_prod_m3_sc"] == pytest.approx(7.0, rel=1e-6)
    assert by_month[1]["water_inj_m3_sc"] == pytest.approx(4.0, rel=1e-6)

    # ---- 10.5 a bottom-hole limit that is reached, and one that is not -------------------
    feasible = fixtures["bhp_feasible"]["extraction"]
    assert feasible["control_infeasible_reason"] is None
    for step in feasible["control_evidence"]["OPS1"]:
        assert step["honoured"] and step["operating_target"] == "lrat"
        assert step["liquid_rate_m3_day"] == pytest.approx(
            step["requested_value"], rel=rate_tolerance
        )
    feasible_result, _ = publish_fixture(
        fixtures["bhp_feasible"], paths, "ops-bhp-ok", report, world="operations"
    )
    assert feasible_result.status == "COMPLETE", feasible_result.reason

    infeasible = fixtures["bhp_infeasible"]["extraction"]
    steps = infeasible["control_evidence"]["OPS1"]
    assert all(not step["honoured"] for step in steps)
    for step in steps:
        assert step["requested_target"] == "lrat"
        assert step["requested_value"] == pytest.approx(1.0e5)
        # THE LIMIT WAS REACHED and the well went onto it, at the pressure THIS CASE recorded.
        assert step["operating_target"] == "bhp"
        assert step["bhp_pa"] == pytest.approx(CONTROLS_PRODUCER_BHP_FLOOR_PA, rel=1e-12)
        # And the achieved rate is nothing like the demanded one.
        assert 0.0 < step["liquid_rate_m3_day"] < 1e-3 * step["requested_value"]
    # `set_default_limits = false`: JutulDarcy's own convenience floor of one atmosphere was
    # NOT silently substituted for the limit the case wrote down.
    assert CONTROLS_PRODUCER_BHP_FLOOR_PA != 101325.0
    infeasible_result, _ = publish_fixture(
        fixtures["bhp_infeasible"], paths, "ops-bhp-bad", report, world="operations"
    )
    assert infeasible_result.status == "CONTROL_INFEASIBLE"
    assert infeasible_result.reason is not None
    assert "OPS1" in infeasible_result.reason
    assert "requested lrat=100000.0" in infeasible_result.reason
    assert "operated on bhp" in infeasible_result.reason
    # A classified result publishes no outputs: there is no partial integral for a case whose
    # wells did not do what the case asked (SPEC 18.4).
    assert infeasible_result.monthly_path is None

    # ---- 10.6 a closed benchmark, an infinite support, and a finite store -----------------
    native_trans = report["sector_face_transmissibility"]
    # The conductance the boundary was given IS the sector's own face transmissibility, and
    # `A*k/dx` recomputed here from the geometry agrees with the model's own parameter.
    assert native_trans["native"] == pytest.approx(SECTOR_FACE_TRANSMISSIBILITY, rel=1e-12)
    assert native_trans["declared_boundary_trans_flow"] == pytest.approx(
        native_trans["native"], rel=1e-12
    )

    drops: dict[str, float] = {}
    published_boundary: dict[str, Path] = {}
    for name in ("boundary_closed", "boundary_pressure_water", "boundary_finite_buffer"):
        fixture = fixtures[name]
        assert fixture["status"] == "COMPLETE"
        states = fixture["extraction"]["states"]
        pressure = np.asarray(states["pressure_pa"], dtype=np.float64)
        drops[name] = float(pressure[0][0] - pressure[-1][0])
        _within_spec(extraction_balances(fixture["extraction"]))
        _, result_dir = publish_fixture(fixture, paths, f"ops-{name}", report, world="operations")
        published_boundary[name] = result_dir

    # THE ORDERING IS THE MEASUREMENT. A closed sector gives up about 9 bar over three days;
    # the same sector behind a fixed-pressure boundary of that conductance gives up almost
    # nothing; a finite buffer of a known pore volume behind the same conductance is in
    # between, because it runs down while it supports.
    #
    # The closed case is stated as the no-support reference and not as a controlled contrast:
    # the buffer case has a third cell, so some of its smaller drawdown is simply more rock
    # in the system. What IS controlled is the pair that shares a geometry and a conductance —
    # the fixed-pressure boundary against the finite store — and the buffer gives up more than
    # twenty times as much pressure as the boundary does, because only one of them is finite.
    assert drops["boundary_pressure_water"] < drops["boundary_finite_buffer"]
    assert drops["boundary_finite_buffer"] < drops["boundary_closed"]
    assert drops["boundary_finite_buffer"] > 20.0 * drops["boundary_pressure_water"]

    # The boundary influx is IN the balance and is named there, so nobody attributes an
    # aquifer's water to a well. Water only: the boundary's `fractional_flow` is (1, 0).
    def _boundary_source(result_dir: Path) -> dict[str, float]:
        rows = pq.read_table(result_dir / "balances.parquet").to_pylist()
        return {
            str(row["component"]): float(row["boundary_source_m3_sc"])
            for row in rows
            if row["balance"] == "reservoir_connections"
        }

    supported = _boundary_source(published_boundary["boundary_pressure_water"])
    assert supported["water"] > 0.2
    assert supported["oil"] == 0.0
    for name in ("boundary_closed", "boundary_finite_buffer"):
        assert _boundary_source(published_boundary[name]) == {"water": 0.0, "oil": 0.0}

    # EXCHANGE SIGN. The native boundary flux is positive OUT of the reservoir, so a support
    # that is feeding the sector is negative on every substep, and the source it becomes is
    # positive into it.
    boundary_rows = fixtures["boundary_pressure_water"]["extraction"]["boundary"]
    assert boundary_rows and all(row["cell_id"] == 1 for row in boundary_rows)
    assert all(row["water_mass_kg_s"] < 0.0 for row in boundary_rows)
    assert all(row["oil_mass_kg_s"] == 0.0 for row in boundary_rows)
    assert all(
        row["trans_flow"] == pytest.approx(SECTOR_FACE_TRANSMISSIBILITY, rel=1e-12)
        for row in boundary_rows
    )
    boundary_source = np.asarray(
        fixtures["boundary_pressure_water"]["extraction"]["net_boundary_source_m3_sc"],
        dtype=np.float64,
    )
    assert (boundary_source[:, 0] > 0.0).all()
    assert (boundary_source[:, 1] == 0.0).all()
    # The other two cases have no boundary at all, and publish that as an exact zero rather
    # than as an absent field.
    for name in ("boundary_closed", "boundary_finite_buffer"):
        assert fixtures[name]["extraction"]["boundary"] == []
        assert not np.asarray(
            fixtures[name]["extraction"]["net_boundary_source_m3_sc"], dtype=np.float64
        ).any()

    # A FINITE STORE RUNS DOWN, which is the whole difference from the fixed-pressure model.
    buffer_states = fixtures["boundary_finite_buffer"]["extraction"]["states"]
    pore = np.asarray(buffer_states["pore_volume_m3"], dtype=np.float64)
    sw = np.asarray(buffer_states["sw"], dtype=np.float64)
    bw = np.asarray(buffer_states["bw"], dtype=np.float64)
    pressure = np.asarray(buffer_states["pressure_pa"], dtype=np.float64)
    # The buffer's pore volume is a number the case wrote down: a 10 m cube at porosity 0.4.
    assert pore[0][2] == pytest.approx(400.0, rel=1e-12)
    buffer_water = sw[:, 2] * pore[:, 2] / bw[:, 2]
    assert buffer_water[-1] < buffer_water[0]
    assert pressure[-1][2] < pressure[0][2]
    # Under the fixed-pressure boundary the support pressure is a constant of the run, by
    # construction; that is what makes it an infinite store and not an aquifer with a size.
    assert all(row["pressure_pa"] == pytest.approx(1.5e7, rel=1e-12) for row in boundary_rows)

    # ---- 10.6 a boundary nobody fully specified is refused, by name -----------------------
    refusals = report["boundary_refusals"]
    assert set(refusals) == {
        "unknown_kind",
        "no_cells",
        "no_pressure",
        "no_trans_flow",
        "cell_outside_grid",
        "duplicate_cell",
        "fractional_flow_not_a_split",
    }
    assert "aquifer" in refusals["unknown_kind"]
    assert "never defaulted" in refusals["no_pressure"]
    assert "never guessed" in refusals["no_trans_flow"]
    assert "outside the grid" in refusals["cell_outside_grid"]
    assert "named twice" in refusals["duplicate_cell"]
    assert "sum to exactly 1" in refusals["fractional_flow_not_a_split"]


# ======================================================================================
# E01.10.7-10.8 — the five-spot, and the grid/time refinement study
# ======================================================================================

#: The 10.7 five-spot, restated here from the plan rather than from the file that built it.
FIVE_SPOT_SHAPE = (16, 16, 1)
FIVE_SPOT_EXTENT_M = (400.0, 400.0, 10.0)
FIVE_SPOT_MONTHS = 36
FIVE_SPOT_INJECTOR_IJ = ((2, 2), (2, 13), (13, 2), (13, 13))
FIVE_SPOT_PRODUCER_IJ = ((7, 7), (7, 8), (8, 7), (8, 8))
FIVE_SPOT_INJECTION_M3_DAY = 10.0
FIVE_SPOT_PRODUCTION_M3_DAY = 40.0
SUPPORT_SIDE = 8

#: The Buckley-Leverett core of plan 9.6, which 10.8 refines in space AND time: 0.2 pore
#: volumes of water over one day, so the analytic reference is evaluated at t_pvi = 0.2.
BL_INJECTED_PV = 0.2


def _cell(i: int, j: int, nx: int) -> int:
    return i + nx * j


def _five_spot_outputs(result_dir: Path) -> dict[str, Path]:
    return {
        "states": result_dir / STATES_FILENAME,
        "balances": result_dir / BALANCES_FILENAME,
        "connections": result_dir / CONNECTIONS_FILENAME,
    }


def _refinement_outputs(result_dir: Path) -> dict[str, Path]:
    return {"states": result_dir / STATES_FILENAME, "monthly": result_dir / MONTHLY_FILENAME}


@pytest.mark.julia
def test_the_five_spot_is_symmetric_and_survives_refinement(tmp_project: Path) -> None:
    """E01.10.7-10.8 under P1_LOOP: 256 and 1024 cells, 5 wells, 36 calendar months.

    Four forwards in one Julia process: Buckley-Leverett at 64 cells with the report step and
    at 128 with half of it, and the five-spot at 16x16 and at 32x32. The first pair has an
    analytic answer and is scored against it; the second has none and is scored on AGREEMENT,
    over a fixed 8x8 support of physical zones that both meshes tile exactly.

    Nothing here claims the fine grid is right. What is measured is how much of the answer
    moved when the mesh and the timestep were refined — a discretization sensitivity — and the
    verdict `compare_refinement` returns says exactly that and no more.

    Five wells and 36 report intervals put this outside P0_VERIFY, whose ceiling is three
    wells and twelve intervals, so the launcher is bounded by the P1_LOOP job timeout.
    """
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    report = _launch_json(
        julia_exe,
        paths,
        REFINEMENT_JL,
        "--test-refinement",
        P1_LOOP_PROFILE.job_timeout_s,
        "refinement",
    )
    fixtures = report["fixtures"]
    tolerances = load_tolerances(TOLERANCES_PATH)

    # ---- 10.7 the pattern is the pattern the plan describes ------------------------------
    coarse = fixtures["five_spot_16"]
    assert coarse["status"] == "COMPLETE"
    case = coarse["case"]
    assert tuple(case["grid"]["shape"]) == FIVE_SPOT_SHAPE
    assert tuple(case["grid"]["extent_m"]) == FIVE_SPOT_EXTENT_M
    assert len(case["report_edges_s"]) == FIVE_SPOT_MONTHS + 1
    assert [e / SECONDS_PER_DAY for e in case["report_edges_s"]] == pytest.approx(
        [e / SECONDS_PER_DAY for e in month_edges_s(date(2020, 1, 1), FIVE_SPOT_MONTHS)]
    )
    wells = {w["well_id"]: w for w in case["wells"]}
    assert len(wells) == 5
    assert sorted(wells["PRO1"]["cells"]) == sorted(
        _cell(i, j, 16) for i, j in FIVE_SPOT_PRODUCER_IJ
    )
    # The producer's four connections are EQUAL: none of the four central cells is "the
    # centre", because an even grid has none, and a `SimpleWell` is one node so all four are
    # equidistant from it.
    assert wells["PRO1"]["model"] == "simple"
    injector_cells = sorted(
        tuple(w["cells"]) for name, w in wells.items() if name.startswith("INJ")
    )
    assert injector_cells == sorted((_cell(i, j, 16),) for i, j in FIVE_SPOT_INJECTOR_IJ)
    # GRAVITY IS PRESENT — the case declares the native 9.80665 — and the horizontal faces of
    # a single layer carry a gravity head of exactly zero. The case does not switch anything
    # off; the native parameter says the head is zero because the depths are equal.
    assert case["gravity_m_s2"] == STANDARD_GRAVITY_M_S2
    native = coarse["native"]
    assert native["two_point_gravity_difference"]
    assert set(native["two_point_gravity_difference"]) == {0.0}
    assert len(set(native["cell_center_depth_m"])) == 1
    rates = {(c["well_id"].startswith("INJ"), c["target"], c["value"]) for c in case["controls"]}
    assert rates == {
        (False, "liquid_rate", FIVE_SPOT_PRODUCTION_M3_DAY),
        (True, "water_rate", FIVE_SPOT_INJECTION_M3_DAY),
    }

    # ---- 10.7 the symmetry, scored against the tolerance fixed before it ran --------------
    _, coarse_dir = publish_fixture(coarse, paths, "five-spot-16", report, world="refinement")
    check = evaluate_physics("five_spot", _five_spot_outputs(coarse_dir), tolerances)
    assert check.status == "PASS", check.reason
    assert check.unrun_metrics == ()
    # Julia measured the same two reflections on its own copy of the field; they agree.
    julia_symmetry = report["five_spot_symmetry"]["five_spot_16"]
    assert check.metrics["five_spot_symmetry_x_abs"] == pytest.approx(
        julia_symmetry["so_mirror_x_abs"], rel=1e-9
    )
    assert check.metrics["five_spot_symmetry_y_abs"] == pytest.approx(
        julia_symmetry["so_mirror_y_abs"], rel=1e-9
    )
    # And the field it is symmetric about really varies over half a saturation unit.
    assert check.metrics["five_spot_so_range"] > 0.5

    # ---- 10.8 the same continuous problem on a finer grid ---------------------------------
    fine = fixtures["five_spot_32"]
    assert fine["status"] == "COMPLETE"
    fine_case = fine["case"]
    assert tuple(fine_case["grid"]["shape"]) == (32, 32, 1)
    # THE SAME CONTINUOUS FIELD, not a second realisation: the same extent, the same
    # homogeneous porosity and permeability, and the same total pore volume.
    assert tuple(fine_case["grid"]["extent_m"]) == FIVE_SPOT_EXTENT_M
    for label, fixture in (("16", coarse), ("32", fine)):
        assert set(np.asarray(fixture["arrays"]["porosity"], dtype=np.float64)) == {0.2}, label
    coarse_pv = float(np.asarray(coarse["native"]["pore_volume_m3"], dtype=np.float64).sum())
    fine_pv = float(np.asarray(fine["native"]["pore_volume_m3"], dtype=np.float64).sum())
    assert fine_pv == pytest.approx(coarse_pv, rel=1e-12)
    # The wells are at the same PHYSICAL locations: what perforated one coarse cell perforates
    # the 2x2 block of fine cells that replaced it.
    fine_wells = {w["well_id"]: w for w in fine_case["wells"]}
    assert sorted(fine_wells["INJ_SW"]["cells"]) == sorted(
        _cell(i, j, 32) for i in (4, 5) for j in (4, 5)
    )
    # The native well index is RECOMPUTED from the same radius in the refined geometry, never
    # carried over and never tuned back to reproduce the coarse answer.
    wi = report["five_spot_well_index_total"]
    assert wi["32"]["INJ_SW"] != wi["16"]["INJ_SW"]
    assert wi["32"]["INJ_SW"] > wi["16"]["INJ_SW"]

    _, fine_dir = publish_fixture(fine, paths, "five-spot-32", report, world="refinement")
    support = CommonSupport(
        name="five_spot_refinement",
        n_zones=SUPPORT_SIDE**2,
        coarse_zone_id=cartesian_zone_ids(16, 16, SUPPORT_SIDE),
        fine_zone_id=cartesian_zone_ids(32, 32, SUPPORT_SIDE),
    )
    # The support Julia used and the one recomputed here are the same partition.
    assert support.coarse_zone_id.tolist() == report["support_zone_ids"]["five_spot_16"]
    assert support.fine_zone_id.tolist() == report["support_zone_ids"]["five_spot_32"]

    refinement = compare_refinement(
        _refinement_outputs(coarse_dir), _refinement_outputs(fine_dir), support, tolerances
    )
    assert refinement.status == "PASS", refinement.reason
    assert refinement.unrun_metrics == ()
    # The mapping is pore-volume-conservative to round-off, which is what makes the saturation
    # comparison legal at all.
    assert refinement.metrics["support_pore_volume_relative"] < 1e-12
    assert refinement.metrics["coarse_cells"] == 256.0
    assert refinement.metrics["fine_cells"] == 1024.0
    assert refinement.metrics["n_months"] == float(FIVE_SPOT_MONTHS)
    # NOT VACUOUS: the two grids really do give different answers, and the difference is a
    # discretisation sensitivity rather than a posterior.
    assert refinement.metrics["so_pv_mae"] > 0.0
    assert refinement.metrics["so_zone_max_abs"] > refinement.metrics["so_pv_mae"]

    # ---- 10.8 Buckley-Leverett: 64 cells at dt, 128 at dt/2, against the formula ----------
    errors: dict[int, float] = {}
    for n_cells in (64, 128):
        fixture = fixtures[f"bl_{n_cells}"]
        assert fixture["status"] == "COMPLETE"
        states = fixture["extraction"]["states"]
        sw = np.asarray(states["sw"][-1], dtype=np.float64)
        pore = np.asarray(states["pore_volume_m3"][-1], dtype=np.float64)
        assert sw.size == n_cells
        total = float(pore.sum())
        edges = np.concatenate([[0.0], np.cumsum(pore) / total])
        reference = bl_cell_average(edges, BL_INJECTED_PV)
        errors[n_cells] = float(np.sum(np.abs(sw - reference) * (pore / total)))
    # The timestep really was refined with the grid: twice as many accepted substeps.
    substeps = {n: len(fixtures[f"bl_{n}"]["extraction"]["chunk"]["dt_s"]) for n in (64, 128)}
    assert substeps[128] == 2 * substeps[64]
    # The error against the analytic solution does not GROW under refinement, which is the
    # plan's gate, and in fact it falls.
    ratio = errors[128] / errors[64]
    assert ratio <= tolerances["bl_refinement_ratio_max"]
    assert errors[128] < errors[64]
