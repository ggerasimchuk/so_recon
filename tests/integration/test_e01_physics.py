"""E01.0: the native JutulDarcy oil-water model, and the worker's route into it.

Two real Julia processes, no more. The cold start plus the first JutulDarcy specialisation
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
  `BudgetLedger`. It proves the wiring (a job now reaches the constructor and comes back
  describing the model that was built), and it proves the isolation the wiring is for: B's
  model must not change what the second A reports.

Neither test claims a COMPLETE forward. This build constructs a model and an initial state;
integrating the requested time axis and publishing its outputs is a later stage, and a
worker that cannot produce states says so instead of reporting success.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import CASE_MANIFEST_FILENAME, write_case
from so_recon.simulator.contracts import (
    MILLIDARCY_M2,
    SECONDS_PER_DAY,
    STANDARD_GRAVITY_M_S2,
    CaseBundle,
    FluidSpec,
    ForwardResult,
    JobDescriptor,
    OutputRequest,
)
from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    SubprocessJuliaLauncher,
    find_julia,
)
from so_recon.simulator.worker import PersistentJuliaWorker
from tests.forward_case import N_CELLS, build_case, write_case_arrays

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_JL = ROOT / "julia" / "verification" / "fixtures.jl"

#: The §3.1 educational numbers, written here as the independent side of the comparison.
P_SC_PA = 101325.0
P_RESERVOIR_PA = 1.5e7
RHO_W_SC, RHO_O_SC = 1000.0, 800.0
C_W, C_O = 4e-10, 1e-9
MU_W, MU_O = 0.001, 0.003

#: `:closed_cell` is one 10 m cube at porosity 0.2.
CLOSED_CELL_EDGE_M = 10.0
CLOSED_CELL_POROSITY = 0.2

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


def _publish_case(paths: ProjectPaths, *, label: str, porosity: float) -> tuple[CaseBundle, Path]:
    """Write one complete case, with its own arrays, and return it with its manifest path."""
    refs = write_case_arrays(
        paths,
        subdir=f"arrays-{label}",
        porosity=np.full(N_CELLS, porosity, dtype=np.float64),
    )
    case = build_case(refs, case_id=f"case-e01-{label}")
    ctx = RunContext.start(command=f"physics-{label}", argv=[], cfg=None, paths=paths)
    write_case(case, paths, ctx)
    return case, ctx.run_dir / CASE_MANIFEST_FILENAME


@pytest.mark.julia
def test_the_worker_reaches_the_adapter_and_rebuilds_every_job(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()

    case_a, path_a = _publish_case(paths, label="a", porosity=POROSITY_A)
    case_b, path_b = _publish_case(paths, label="b", porosity=POROSITY_B)
    assert case_a.model_hash != case_b.model_hash

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
        results = [
            worker.submit(descriptor("job-e01-a1", case_a, path_a), ledger),
            worker.submit(descriptor("job-e01-b1", case_b, path_b), ledger),
            worker.submit(descriptor("job-e01-a2", case_a, path_a), ledger),
        ]
        assert worker.pid == worker.handshake.pid  # one process behind all three jobs

    records = [published_record(result) for result in results]

    for result, record in zip(results, records, strict=True):
        # The adapter was reached: the case was built into a model, and the answer names
        # what is missing rather than claiming a forward nobody integrated.
        assert result.status == "INVALID_INPUT"
        assert result.reason is not None
        assert result.reason.startswith("outputs unavailable")
        assert result.physics_class == "OW"
        assert record["status"] == result.status
        built = record["model"]
        assert built["n_cells"] == N_CELLS
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

    # Three attempts, all of them paid for and none of them a success.
    assert [entry.job_id for entry in ledger.record.entries] == [
        "job-e01-a1",
        "job-e01-b1",
        "job-e01-a2",
    ]
    assert all(entry.state == "FAILED" for entry in ledger.record.entries)
    assert {entry.status for entry in ledger.record.entries} == {"INVALID_INPUT"}
