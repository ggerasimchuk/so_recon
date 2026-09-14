"""E01.9: the analytic oil-water fixtures, run on the pinned solver and SCORED.

ONE real Julia process. `julia/verification/analytic.jl --test-analytic` builds and runs
seven small verification forwards through the same `build_ow`/`build_forces`/`run_forward`
the worker reaches — a one-cell and an eight-cell closed system, three Buckley-Leverett
grids, a hydrostatic column and a gravity-segregation column — plus one deliberately broken
PVT that must never reach a solver at all. Python then does what Julia cannot: it rebuilds
each case through the real `CaseBundle` contract, PUBLISHES a `ForwardResult` with its
states, balances and connection tables, reads the record back with every digest re-proved,
and hands the published files to `so_recon.validation.physics.evaluate_physics` against
`configs/e01_tolerances.yml`.

That last step is the point of the file. Plan §12.9 has the stage validator read the
PUBLISHED artifacts rather than re-run the suite, so the evaluator here is given exactly the
bytes a real run leaves behind, and the verdicts it returns are the ones a stage report would
cite. The tolerances are the ones fixed in Task 9.1 before any of these numbers existed;
nothing in this file loosens one.

What each side is allowed to claim is kept apart on purpose. Julia measures what only Julia
can see — the native `TwoPointGravityDifference` parameter, the model's own phase densities
at the initial state, and the exception a non-physical PVT raises inside the constructor —
and Python recomputes the same hydrostatic face residual from the PUBLISHED `bw`, so the two
numbers come from different objects and agreeing means something.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.paths import ProjectPaths
from so_recon.simulator.case_io import write_arrays
from so_recon.simulator.contracts import (
    CELL_AXES,
    STANDARD_GRAVITY_M_S2,
    TIME_CELL_AXES,
    FluidSpec,
    ForwardStatus,
)
from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    SubprocessJuliaLauncher,
    find_julia,
)
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    CONNECTIONS_FILENAME,
    STATES_FILENAME,
)
from so_recon.validation.physics import (
    DEFAULT_TOLERANCES_RELPATH,
    bl_front_position,
    evaluate_physics,
    load_tolerances,
)
from tests.integration.fixture_result import publish_fixture

ROOT = Path(__file__).resolve().parents[2]
ANALYTIC_JL = ROOT / "julia" / "verification" / "analytic.jl"
TOLERANCES_PATH = ROOT / DEFAULT_TOLERANCES_RELPATH

#: The column both 9.7 fixtures use: two 50 m cells below the 1000 m fixture datum. Stated
#: here as well as in the evaluator's registry so the published case is checked against it.
COLUMN_DEPTHS_M = (1025.0, 1075.0)
RHO_W_SC = 1000.0

#: The BL core: 100 m of 1 m² at porosity 0.2, and the 0.2 pore volumes injected over a day.
BL_PORE_VOLUME_M3 = 100.0 * 1.0 * 0.2
BL_INJECTED_PV = 0.2
BL_GRIDS = (32, 64, 128)


def _skip_unless_julia_is_installed() -> Path:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        return find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")


def _publish_result(
    fixture: dict[str, Any], paths: ProjectPaths, label: str, report: dict[str, Any]
) -> Path:
    """Publish one fixture as a real `ForwardResult`, read it back, and return its directory.

    `tests.integration.fixture_result` does the work — rebuilding the case through the
    production contract, publishing the extraction through the production publisher and
    re-proving every digest on the way back — and is shared with Task 10's operational suite.
    Every fixture of THIS file is supposed to complete, so that is asserted here.
    """
    result, result_dir = publish_fixture(fixture, paths, label, report, world="analytic")
    assert result.status == "COMPLETE", result.reason
    return result_dir


def _outputs(result_dir: Path) -> dict[str, Path]:
    return {
        "states": result_dir / STATES_FILENAME,
        "balances": result_dir / BALANCES_FILENAME,
        "connections": result_dir / CONNECTIONS_FILENAME,
    }


@pytest.mark.julia
def test_the_analytic_fixtures_pass_the_tolerances_fixed_before_they_ran(
    tmp_project: Path,
) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    out_path = paths.artifacts / "verification" / "analytic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    launcher = SubprocessJuliaLauncher(
        julia_exe, ROOT / "julia", timeout_s=P0_VERIFY_PROFILE.job_timeout_s
    )
    launcher.launch(ANALYTIC_JL, ["--test-analytic"], out_path)
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert (report["jutul_version"], report["jutuldarcy_version"]) == ("0.4.31", "0.3.11")
    assert report["gravity_constant"] == STANDARD_GRAVITY_M_S2
    fixtures = report["fixtures"]
    tolerances = load_tolerances(TOLERANCES_PATH)

    checks: dict[str, Any] = {}
    published: dict[str, Path] = {}

    # ---- 9.5 the closed / PVT fixtures -------------------------------------------------
    for name in ("closed_cell_pvt", "closed_box_pvt"):
        fixture = fixtures[name]
        assert fixture["status"] == "COMPLETE"
        # No source anywhere: the monitoring well is shut and every completion is closed.
        assert all(not c["connection_open"] for c in fixture["extraction"]["connections"])
        result_dir = _publish_result(fixture, paths, name, report)
        published[name] = result_dir
        check = evaluate_physics(name, _outputs(result_dir), tolerances)
        assert check.status == "PASS", check.reason
        # A closed system at rest does not move at all, and the tolerance is not what makes
        # that true: both drifts are exact zeros, three report steps in.
        assert check.metrics["state_saturation_drift"] == 0.0
        assert check.metrics["pressure_relative_drift"] == 0.0
        assert check.metrics["connection_mass_kg_s"] == 0.0
        assert check.metrics["n_published_times"] == 4.0
        assert check.metrics["balance_cumulative_target_met"] == 1.0
        checks[name] = check

    # ---- 9.5 a PVT that cannot describe a fluid never reaches the solver ----------------
    broken = fixtures["broken_pvt"]
    assert broken["phase"] == "build"
    assert broken["status"] == "PHYSICALLY_INVALID"
    # The status the fixture reports is a status of the contract, produced by the same
    # classifier the worker uses (`SOReconAdapter.failure_status`).
    assert broken["status"] in ForwardStatus.__args__  # type: ignore[attr-defined]
    assert broken["is_invalid_case_input"] is False
    assert "density_sc_kg_m3" in broken["reason"]
    # It failed at construction, so there is no extraction and no partial integral for it.
    assert "extraction" not in broken
    # And the same PVT cannot even be expressed through the Python contract.
    with pytest.raises(ValueError, match="must be positive for both phases"):
        FluidSpec.model_validate(broken["case"]["fluids"])

    # ---- 9.6 Buckley-Leverett on three grids -------------------------------------------
    bl_outputs: dict[str, Path] = {}
    for n_cells in BL_GRIDS:
        fixture = fixtures[f"bl_{n_cells}"]
        assert fixture["status"] == "COMPLETE"
        assert fixture["extraction"]["control_infeasible_reason"] is None
        result_dir = _publish_result(fixture, paths, f"bl_{n_cells}", report)
        bl_outputs[f"states_{n_cells}"] = result_dir / STATES_FILENAME
        if n_cells == 128:
            bl_outputs["balances_128"] = result_dir / BALANCES_FILENAME
            bl_128_dir = result_dir
    bl = evaluate_physics("bl", bl_outputs, tolerances)
    assert bl.status == "PASS", bl.reason
    assert bl.metrics["bl_t_pvi"] == pytest.approx(BL_INJECTED_PV, rel=1e-12)
    assert bl.metrics["bl_front_position_pv"] == pytest.approx(bl_front_position(BL_INJECTED_PV))
    # Pre-breakthrough: the shock is well inside the core, so the reference is the one the
    # analytic solution is valid for and the outlet has seen no water.
    assert bl.metrics["bl_front_position_pv"] < 0.3
    errors = [bl.metrics[f"bl_pv_l1_at_{n}"] for n in BL_GRIDS]
    assert errors[-1] <= tolerances["bl_pv_l1_max_at_128"]
    # The error does not GROW under refinement, which is the plan's gate, and it does in fact
    # fall on both refinements.
    for coarse, fine in zip(errors, errors[1:], strict=False):
        assert fine < coarse
    # The quadrature that turns the pointwise reference into a cell average contributes
    # nothing to any of that: 64 against 128 subpoints, against the L1 tolerance itself.
    for n_cells in BL_GRIDS:
        assert (
            bl.metrics[f"bl_quadrature_delta_64_vs_128_at_{n_cells}"]
            < 0.01 * tolerances["bl_pv_l1_max_at_128"]
        )
        # A first-order upwind answer is monotone in x; overshoot would be a scheme this
        # fixture does not have.
        assert bl.metrics[f"bl_profile_monotonicity_violation_at_{n_cells}"] == 0.0
    checks["bl"] = bl

    # The injected volume the reference is written for really crossed the sand face.
    monthly = _read_monthly(bl_128_dir)
    injected = sum(row["water_inj_m3_sc"] for row in monthly if row["well_id"] == "INJ1")
    assert injected == pytest.approx(BL_INJECTED_PV * BL_PORE_VOLUME_M3, rel=1e-6)
    assert sum(row["water_prod_m3_sc"] for row in monthly) == pytest.approx(0.0, abs=1e-9)

    # ---- 9.7 the hydrostatic column ----------------------------------------------------
    column = fixtures["hydrostatic"]
    assert column["status"] == "COMPLETE"
    native = column["native"]
    assert tuple(native["cell_center_depth_m"]) == COLUMN_DEPTHS_M
    face = native["faces"][0]
    # The native face gravity is `-g*(z_r - z_l)` (`Jutul.compute_face_gdz`), so a column
    # that had been origined at zero or measured upwards would disagree here.
    assert face["gdz"] == pytest.approx(-STANDARD_GRAVITY_M_S2 * 50.0, rel=1e-12)
    # z is depth, positive down: dp/dz is positive and is the water density times g.
    gradient = face["pressure_difference_pa"] / face["dz_m"]
    assert gradient > 0.0
    assert gradient == pytest.approx(
        face["face_density_kg_m3"][0] * STANDARD_GRAVITY_M_S2, rel=1e-6
    )
    result_dir = _publish_result(column, paths, "hydrostatic", report)
    check = evaluate_physics("hydrostatic", _outputs(result_dir), tolerances)
    assert check.status == "PASS", check.reason
    # Python's residual is computed from the PUBLISHED `bw`; Julia's from the model's own
    # `TwoPointGravityDifference` parameter and `PhaseMassDensities`. They are the same
    # number because the model really is in discrete hydrostatic balance.
    native_residual = abs(face["potential_residual_pa"][0])
    assert check.metrics["hydrostatic_face_residual_pa"] == pytest.approx(native_residual, rel=1e-6)
    assert check.metrics["pressure_depth_gradient_pa_m"] == pytest.approx(gradient, rel=1e-12)
    # A real residual, measured before the first timestep: small because the column was
    # initialised from the CONTINUOUS hydrostatic solution of the same rho(p), not from the
    # discrete balance, which would have made it zero by construction.
    assert 0.0 < native_residual < 1.0
    # The extent of the single-phase solver defect the fixture documents, on the record and
    # asserted. At exactly So = 0 the two-phase Newton update goes non-finite; Jutul catches
    # it under `failure_cuts_timestep`, discards the partial increment and halves the step,
    # so no NaN ever reaches a published state and an exhausted cut budget would be a hard
    # failure. What could rot silently is the COUNT — `run_forward` runs at info_level -1 and
    # the launcher discards stderr — so the recovery is bounded here by number.
    degeneracy = report["single_phase_degeneracy"]
    assert degeneracy["fixture"] == "hydrostatic"
    assert degeneracy["report_steps"] == 3
    assert degeneracy["cut_steps"] == degeneracy["expected_cut_steps"] == 2
    assert degeneracy["accepted_steps"] == degeneracy["expected_accepted_steps"] == 4
    assert column["extraction"]["solver"]["cut_steps"] == 2
    # Every other fixture pays nothing for it: only this one sits at the single-phase limit.
    for name in ("closed_cell_pvt", "closed_box_pvt", "segregation", "bl_32", "bl_64", "bl_128"):
        assert fixtures[name]["extraction"]["solver"]["cut_steps"] == 0, name
    checks["hydrostatic"] = check

    # ---- 9.7 gravity segregation -------------------------------------------------------
    segregation = fixtures["segregation"]
    assert segregation["status"] == "COMPLETE"
    result_dir = _publish_result(segregation, paths, "segregation", report)
    check = evaluate_physics("segregation", _outputs(result_dir), tolerances)
    assert check.status == "PASS", check.reason
    # DEMONSTRATED: the heavy phase started above the light one and its centre of mass moved
    # down. Nothing in the fixture fixes the answer — the column is closed and free.
    # Sw = (0.7, 0.3) over two equal pore volumes at 1025 m and 1075 m puts the water's
    # centre of mass at 1040 m; the published number differs from that by the millimetre the
    # deeper cell's compressed water is worth, because the weight is a STANDARD volume.
    assert check.metrics["water_mean_depth_start_m"] == pytest.approx(1040.0, abs=0.01)
    assert check.metrics["water_mean_depth_increase_m"] > 1.0
    assert check.metrics["water_center_of_height_change_m"] < -1.0
    # The water in the column is conserved: weighted by its STANDARD volume the closed column
    # holds the same water to 3e-8 relative, which is JutulDarcy's own per-step mass-balance
    # tolerance (`tol_mb = 1e-7`). The RESERVOIR volume alone moves by 1.6e-5 over the same
    # thirty days, because the water is compressible and the pressure redistributes as the
    # column segregates; measuring the wrong one of those two would look like a leak.
    assert check.metrics["water_volume_relative_change"] < 1e-6
    sw = _read_states_field(result_dir, "sw")
    assert sw[-1][0] < sw[0][0] and sw[-1][1] > sw[0][1]  # water leaves the top cell
    checks["segregation"] = check

    # ---- 9.8 every metric on the record, and nothing unrun -----------------------------
    assert sorted(checks) == [
        "bl",
        "closed_box_pvt",
        "closed_cell_pvt",
        "hydrostatic",
        "segregation",
    ]
    for name, check in checks.items():
        assert check.status == "PASS", f"{name}: {check.reason}"
        assert check.unrun_metrics == (), name
        assert check.metrics["saturation_sum_abs"] <= tolerances["saturation_sum_abs_max"]
        assert check.input_hashes and all(len(d) == 64 for d in check.input_hashes.values())

    # ---- 9.2 the evaluator can turn a REAL published result red ------------------------
    # One cell's water saturation at the last report step, moved by a thousand times the
    # drift tolerance; every other published byte identical. An evaluator that cannot fail
    # is not a check, and feeding it only good numbers never finds out which of the two it
    # is. No new forward is simulated for this: the mutation is applied to the states file
    # the closed box already published.
    _mutation_turns_the_closed_box_red(published["closed_box_pvt"], paths, tolerances)


def _mutation_turns_the_closed_box_red(
    result_dir: Path, paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    clean = evaluate_physics("closed_box_pvt", _outputs(result_dir), tolerances)
    assert clean.status == "PASS", clean.reason

    fields = {
        name: _read_states_field(result_dir, name)
        for name in ("pressure_pa", "sw", "so", "pore_volume_m3", "bw", "bo", "time_s", "cell_id")
    }
    drift = 1000 * tolerances["closed_state_saturation_drift_max"]
    fields["sw"][-1, 0] += drift
    fields["so"][-1, 0] -= drift
    mutated_dir = paths.artifacts / "mutated"
    write_arrays(
        mutated_dir / STATES_FILENAME,
        {
            "pressure_pa": (fields["pressure_pa"], "Pa", TIME_CELL_AXES),
            "sw": (fields["sw"], "1", TIME_CELL_AXES),
            "so": (fields["so"], "1", TIME_CELL_AXES),
            "pore_volume_m3": (fields["pore_volume_m3"], "m3", TIME_CELL_AXES),
            "bw": (fields["bw"], "1", TIME_CELL_AXES),
            "bo": (fields["bo"], "1", TIME_CELL_AXES),
            "time_s": (fields["time_s"], "s", ("time",)),
            "cell_id": (fields["cell_id"].astype(np.int64), "1", CELL_AXES),
        },
        paths=paths,
    )
    mutated = dict(_outputs(result_dir))
    mutated["states"] = mutated_dir / STATES_FILENAME
    verdict = evaluate_physics("closed_box_pvt", mutated, tolerances)
    assert verdict.status == "FAIL"
    assert verdict.reason is not None and "state_saturation_drift" in verdict.reason
    assert verdict.metrics["state_saturation_drift"] == pytest.approx(drift, rel=1e-9)
    # The bytes it scored are not the bytes the clean verdict scored, and the record says so.
    assert verdict.input_hashes["states"] != clean.input_hashes["states"]
    assert not math.isnan(verdict.metrics["saturation_sum_abs"])


def _read_monthly(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(result_dir / "monthly.parquet").to_pylist()
    return rows


def _read_states_field(result_dir: Path, name: str) -> NDArray[np.float64]:
    with h5py.File(result_dir / STATES_FILENAME, "r") as handle:
        return np.asarray(handle[name][()], dtype=np.float64)
