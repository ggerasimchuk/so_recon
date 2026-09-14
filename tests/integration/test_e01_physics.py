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
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P0_VERIFY_PROFILE
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
from so_recon.simulator.schedule import compile_schedule, month_edges_s
from so_recon.simulator.worker import PersistentJuliaWorker
from tests.forward_case import (
    N_CELLS,
    SHAPE,
    build_case,
    cell_centers,
    write_case_arrays,
)

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
        "unimplemented_boundary",
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
    assert "Tasks 9-10" in refusals["unimplemented_boundary"]
    assert "closed boundary names no cells" in refusals["closed_boundary_with_cells"]
    # Every one of them names the well or the field it is about, never just "invalid".
    for label in ("missing_well", "duplicate_well", "mask_too_short", "mask_not_boolean"):
        assert "INJ1" in refusals[label] or "PRO1" in refusals[label], label

    hole = report["missing_control_message"]
    assert "INJ1" in hole and "never inherited" in hole
    crossflow_refusal = report["crossflow_refusal_message"]
    assert "allow_crossflow=false" in crossflow_refusal
    assert "0.3.11" in crossflow_refusal
