"""E01.9: the physics evaluator itself — its tolerances, its analytic reference, and proof
that it can turn a fixture red.

Three groups of claims, and the third is the one that matters.

* `load_tolerances` is the single source of truth for what the evaluator scores, so it is
  checked against the plan's own numbers and against the three constants `validation.balance`
  already carried before this task existed. Two files stating the same tolerance is a defect
  in waiting; the test below is what keeps them from drifting apart in silence.
* `bl_saturation` is the Buckley-Leverett reference in the limit the fixture declares
  (Corey n=2, equal viscosity, Swc=Sor=0, incompressible, horizontal). It is checked at
  t=0, at the injection boundary, behind the front and ahead of the shock, and against the
  Welge construction and the volume it must carry — none of which is how it is computed.
* `evaluate_physics` is checked BOTH ways. A clean fixture passes, and then one inventory,
  one connection flux and one saturation are perturbed by hand and the evaluator is
  required to go red. An evaluator that cannot fail is not a check, and a suite that only
  ever feeds it good numbers never finds out which of the two it has.

The published artifacts here are synthesised, not simulated: these are the evaluator's own
unit tests and they must run without Julia. `tests/integration/test_e01_analytic.py` is the
other half — the same evaluator against the artifacts a real pinned solver published.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.simulator.case_io import write_arrays
from so_recon.simulator.contracts import CELL_AXES, SECONDS_PER_DAY, TIME_CELL_AXES
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    CONNECTION_MONTHLY_SCHEMA,
    CONNECTIONS_FILENAME,
    STATES_FILENAME,
    balance_table,
)
from so_recon.validation.balance import (
    BALANCE_FLOOR_M3_SC,
    CUMULATIVE_RELATIVE_TOLERANCE,
    MEDIAN_STEP_RELATIVE_TOLERANCE,
    component_balance,
)
from so_recon.validation.physics import (
    BL_SHOCK_SATURATION,
    DEFAULT_TOLERANCES_RELPATH,
    TOLERANCE_SCHEMA_VERSION,
    PhysicsCheck,
    bl_cell_average,
    bl_front_position,
    bl_saturation,
    evaluate_physics,
    load_tolerances,
)

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES_PATH = ROOT / DEFAULT_TOLERANCES_RELPATH

#: The plan's Task 9.1 block, restated here so the config is compared against the plan and
#: not against itself. A change to `configs/e01_tolerances.yml` that nobody decided fails
#: this test, which is the whole point of fixing the tolerances before the first scoring run.
PLAN_TOLERANCES: dict[str, float] = {
    "balance_cumulative_relative_max": 1.0e-3,
    "balance_step_median_relative_max": 1.0e-5,
    "balance_cumulative_target": 1.0e-5,
    "balance_absolute_floor_m3_sc": 1.0e-6,
    "saturation_sum_abs_max": 1.0e-10,
    "saturation_bound_slack": 1.0e-8,
    "closed_connection_mass_kg_s_max": 1.0e-10,
    "closed_state_saturation_drift_max": 1.0e-8,
    "closed_pressure_relative_drift_max": 1.0e-7,
    "hydrostatic_gradient_relative_max": 1.0e-5,
    "hydrostatic_saturation_drift_max": 1.0e-6,
    "hydrostatic_pressure_relative_drift_max": 1.0e-6,
    "restart_saturation_abs_max": 1.0e-6,
    "restart_pressure_relative_max": 1.0e-6,
    "restart_volume_relative_max": 1.0e-5,
    "restart_inventory_relative_max": 1.0e-5,
    "rate_control_relative_max": 1.0e-4,
    "bl_pv_l1_max_at_128": 0.08,
    "bl_refinement_ratio_max": 1.1,
    "five_spot_symmetry_abs_max": 1.0e-4,
    "refinement_so_pv_mae_max": 0.02,
    "refinement_inventory_relative_max": 0.01,
    "refinement_monthly_volume_relative_max": 0.02,
}


# --------------------------------------------------------------------------------------
# 9.1 the tolerance config
# --------------------------------------------------------------------------------------


def test_the_published_config_is_the_tolerance_block_the_plan_fixed() -> None:
    assert load_tolerances(TOLERANCES_PATH) == PLAN_TOLERANCES


def _write_tolerances(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path


def _valid_payload() -> dict[str, Any]:
    return {"schema_version": TOLERANCE_SCHEMA_VERSION, **PLAN_TOLERANCES}


def test_a_missing_tolerance_is_a_refusal_and_never_a_default(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["bl_pv_l1_max_at_128"]
    path = _write_tolerances(tmp_path / "t.yml", payload)
    with pytest.raises(ValueError, match="bl_pv_l1_max_at_128"):
        load_tolerances(path)


def test_an_unknown_tolerance_is_a_refusal(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["bl_pv_l1_max_at_256"] = 0.04
    path = _write_tolerances(tmp_path / "t.yml", payload)
    with pytest.raises(ValueError, match="bl_pv_l1_max_at_256"):
        load_tolerances(path)


def test_a_config_from_another_schema_is_a_refusal(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["schema_version"] = "e01-tolerances-2"
    path = _write_tolerances(tmp_path / "t.yml", payload)
    with pytest.raises(ValueError, match="e01-tolerances-2"):
        load_tolerances(path)


def test_a_tolerance_that_is_not_a_positive_number_is_a_refusal(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["saturation_sum_abs_max"] = 0.0
    with pytest.raises(ValueError, match="saturation_sum_abs_max"):
        load_tolerances(_write_tolerances(tmp_path / "zero.yml", payload))
    payload["saturation_sum_abs_max"] = "1e-10"
    with pytest.raises(ValueError, match="saturation_sum_abs_max"):
        load_tolerances(_write_tolerances(tmp_path / "text.yml", payload))


def test_the_config_and_the_balance_module_state_the_same_three_numbers() -> None:
    """The duplication `validation.balance` already carried, pinned instead of removed.

    Task 7 fixed `CUMULATIVE_RELATIVE_TOLERANCE`, `MEDIAN_STEP_RELATIVE_TOLERANCE` and
    `BALANCE_FLOOR_M3_SC` as module constants and its own tests assert against them, so
    deleting them here would change a number Task 7 already pinned. The config is the
    single source of truth for what the EVALUATOR scores; these three constants stay as
    the defaults of the low-level helper, and this test is what stops the two from drifting.
    """
    tolerances = load_tolerances(TOLERANCES_PATH)
    assert tolerances["balance_cumulative_relative_max"] == CUMULATIVE_RELATIVE_TOLERANCE
    assert tolerances["balance_step_median_relative_max"] == MEDIAN_STEP_RELATIVE_TOLERANCE
    assert tolerances["balance_absolute_floor_m3_sc"] == BALANCE_FLOOR_M3_SC


# --------------------------------------------------------------------------------------
# 9.2/9.4 the Buckley-Leverett reference
# --------------------------------------------------------------------------------------


def test_bl_front_has_known_shock_speed() -> None:
    # Corey n=2, equal viscosity, Swc=Sor=0, incompressible, horizontal.
    s = bl_saturation(np.array([0.0, 0.1, 0.9]), t_pvi=0.2)
    assert s[0] == 1.0
    assert s[1] > 1 / np.sqrt(2)
    assert s[2] == 0.0


def test_bl_at_zero_time_is_a_step_at_the_injection_face() -> None:
    s = bl_saturation(np.array([0.0, 1e-9, 0.5, 1.0]), t_pvi=0.0)
    assert s.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_bl_refuses_a_negative_coordinate_or_a_negative_time() -> None:
    with pytest.raises(ValueError, match="negative BL coordinate/time"):
        bl_saturation(np.array([0.1]), t_pvi=-1e-9)
    with pytest.raises(ValueError, match="negative BL coordinate/time"):
        bl_saturation(np.array([-1e-9, 0.1]), t_pvi=0.2)


def test_bl_is_a_monotone_rarefaction_behind_the_front_and_dry_ahead_of_it() -> None:
    t = 0.2
    front = bl_front_position(t)
    x = np.linspace(0.0, front * 0.999, 200)
    s = bl_saturation(x, t_pvi=t)
    assert s[0] == 1.0
    assert np.all(np.diff(s) <= 1e-12)  # non-increasing away from the injection face
    assert s.min() > BL_SHOCK_SATURATION
    assert bl_saturation(np.array([front * 1.001, 0.99]), t_pvi=t).tolist() == [0.0, 0.0]


def test_the_shock_sits_where_the_welge_construction_puts_it() -> None:
    """`x_shock = t * f'(S*)` with `S* = 1/sqrt(2)`, from the tangent condition itself.

    The kernel computes the front from a tabulated derivative; this recomputes both the
    Welge tangent `f(S*)/S* == f'(S*)` and the position it implies, from the fractional
    flow written out by hand.
    """

    def fractional_flow(s: float) -> float:
        return s**2 / (s**2 + (1 - s) ** 2)

    star = 1 / math.sqrt(2)
    derivative = (fractional_flow(star + 1e-7) - fractional_flow(star - 1e-7)) / 2e-7
    assert fractional_flow(star) / star == pytest.approx(derivative, rel=1e-6)
    for t in (0.05, 0.2, 0.4):
        assert bl_front_position(t) == pytest.approx(t * derivative, rel=1e-4)


def test_the_cell_averaged_reference_carries_exactly_the_injected_volume() -> None:
    """A conservation check on the reference itself: `∫ S dx_D = t_pvi`.

    Nothing in `bl_saturation` enforces this — it interpolates a tabulated characteristic
    speed — so the integral coming out at the injected pore volume is evidence that the
    rarefaction and the shock were assembled into the right weak solution.
    """
    for t in (0.05, 0.2, 0.4):
        edges = np.linspace(0.0, 1.0, 129)
        averaged = bl_cell_average(edges, t_pvi=t, subpoints=64)
        volume = float(np.sum(averaged * np.diff(edges)))
        assert volume == pytest.approx(t, abs=2e-4)


def test_the_cell_average_quadrature_is_converged_at_64_subpoints() -> None:
    """64 against 128 subpoints, against the numerical tolerance the result is scored by.

    A quasi-pointwise reference must not be compared against a cell average without the
    shock being accounted for, which is what the quadrature is for; this is the evidence
    that the quadrature itself contributes nothing to the L1 the evaluator gates on.
    """
    tolerances = load_tolerances(TOLERANCES_PATH)
    for n_cells in (32, 64, 128):
        edges = np.linspace(0.0, 1.0, n_cells + 1)
        coarse = bl_cell_average(edges, t_pvi=0.2, subpoints=64)
        fine = bl_cell_average(edges, t_pvi=0.2, subpoints=128)
        delta = float(np.sum(np.abs(coarse - fine) * np.diff(edges)))
        assert delta < 0.01 * tolerances["bl_pv_l1_max_at_128"]


# --------------------------------------------------------------------------------------
# 9.2 the evaluator: synthesised published artifacts
# --------------------------------------------------------------------------------------

#: The registered hydrostatic column: two 50 m cells below the fixture datum.
HYDROSTATIC_DEPTHS_M = (1025.0, 1075.0)
RHO_W_SC = 1000.0
G = 9.80665


def _states_path(paths: ProjectPaths, label: str) -> Path:
    return paths.artifacts / label / STATES_FILENAME


def _write_states(
    paths: ProjectPaths,
    label: str,
    *,
    times_s: NDArray[np.float64],
    pressure_pa: NDArray[np.float64],
    sw: NDArray[np.float64],
    pore_volume_m3: NDArray[np.float64] | None = None,
    bw: NDArray[np.float64] | None = None,
    so: NDArray[np.float64] | None = None,
) -> Path:
    """One states.h5 in exactly the shape `write_forward_outputs` publishes."""
    n_times, n_cells = pressure_pa.shape
    so = 1.0 - sw if so is None else so
    pore = np.ones_like(pressure_pa) if pore_volume_m3 is None else pore_volume_m3
    bw_values = np.ones_like(pressure_pa) if bw is None else bw
    path = _states_path(paths, label)
    write_arrays(
        path,
        {
            "pressure_pa": (pressure_pa, "Pa", TIME_CELL_AXES),
            "sw": (sw, "1", TIME_CELL_AXES),
            "so": (so, "1", TIME_CELL_AXES),
            "pore_volume_m3": (pore, "m3", TIME_CELL_AXES),
            "bw": (bw_values, "1", TIME_CELL_AXES),
            "bo": (np.ones_like(pressure_pa), "1", TIME_CELL_AXES),
            "time_s": (times_s, "s", ("time",)),
            "cell_id": (np.arange(n_cells, dtype=np.int64), "1", CELL_AXES),
        },
        paths=paths,
    )
    assert n_times == len(times_s)
    return path


def _write_balances(paths: ProjectPaths, label: str, *, residual_m3_sc: float = 0.0) -> Path:
    """A published balances.parquet, from a real inventory/source pair.

    `residual_m3_sc` is added to the LAST inventory entry of water only: that is the
    mutation, and it is applied to the inventory rather than to the metric, so the residual
    the evaluator reads is one `component_balance` really computed.
    """
    inventory = np.array([[100.0, 200.0], [90.0, 200.0], [80.0, 200.0]], dtype=np.float64)
    source = np.diff(inventory, axis=0).copy()
    inventory[-1, 0] += residual_m3_sc
    metrics = {
        name: component_balance(inventory, source, components=("water", "oil"))
        for name in ("full_system_surface", "reservoir_connections")
    }
    path = paths.artifacts / label / BALANCES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(balance_table(metrics), path)
    return path


def _write_connections(paths: ProjectPaths, label: str, *, total_mass_kg: float = 0.0) -> Path:
    rows = [
        {
            "well_id": "MON1",
            "connection_id": index,
            "cell_id": index,
            "month_index": 0,
            "start_s": 0.0,
            "end_s": SECONDS_PER_DAY,
            "water_mass_kg": total_mass_kg,
            "oil_mass_kg": 0.0,
            "total_mass_kg": total_mass_kg,
            "open_s": 0.0,
            "bhp_min_pa": 1.5e7,
            "bhp_mean_pa": 1.5e7,
            "bhp_max_pa": 1.5e7,
        }
        for index in (0, 1)
    ]
    path = paths.artifacts / label / CONNECTIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CONNECTION_MONTHLY_SCHEMA), path)
    return path


def _closed_outputs(
    paths: ProjectPaths,
    *,
    label: str = "closed",
    sw_drift: float = 0.0,
    pressure_drift_pa: float = 0.0,
    residual_m3_sc: float = 0.0,
    connection_mass_kg: float = 0.0,
) -> dict[str, Path]:
    times = np.array([0.0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY], dtype=np.float64)
    sw = np.full((3, 2), 0.3)
    sw[-1, 0] += sw_drift
    pressure = np.full((3, 2), 1.5e7)
    pressure[-1, 1] += pressure_drift_pa
    return {
        "states": _write_states(paths, label, times_s=times, pressure_pa=pressure, sw=sw),
        "balances": _write_balances(paths, label, residual_m3_sc=residual_m3_sc),
        "connections": _write_connections(paths, label, total_mass_kg=connection_mass_kg),
    }


def _hydrostatic_outputs(
    paths: ProjectPaths, *, label: str = "hydro", pressure_offset_pa: float = 0.0
) -> dict[str, Path]:
    """A column whose published pressures really are in discrete hydrostatic balance.

    `p[r] - p[l] = g * (z[r] - z[l]) * (rho[l] + rho[r])/2` is the native two-point
    statement; with `bw = 1` both densities are `rho_w_sc`, so the head is exact and any
    departure the test introduces is the departure the evaluator has to see.
    """
    top, bottom = HYDROSTATIC_DEPTHS_M
    p_top = 1.5e7
    p_bottom = p_top + G * (bottom - top) * RHO_W_SC + pressure_offset_pa
    times = np.array([0.0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY], dtype=np.float64)
    pressure = np.tile(np.array([p_top, p_bottom]), (3, 1))
    sw = np.ones((3, 2))
    return {
        "states": _write_states(
            paths, label, times_s=times, pressure_pa=pressure, sw=sw, so=np.zeros((3, 2))
        ),
        "balances": _write_balances(paths, label),
        "connections": _write_connections(paths, label),
    }


def _segregation_outputs(
    paths: ProjectPaths, *, label: str = "segregation", sinks: bool = True
) -> dict[str, Path]:
    top, bottom = HYDROSTATIC_DEPTHS_M
    start = np.array([0.7, 0.3])
    end = np.array([0.5, 0.5]) if sinks else np.array([0.9, 0.1])
    sw = np.vstack([start, end])
    times = np.array([0.0, 10 * SECONDS_PER_DAY], dtype=np.float64)
    pressure = np.tile(np.array([1.5e7, 1.5e7 + G * (bottom - top) * 940.0]), (2, 1))
    return {
        "states": _write_states(
            paths,
            label,
            times_s=times,
            pressure_pa=pressure,
            sw=sw,
            pore_volume_m3=np.full((2, 2), 1000.0),
        ),
        "balances": _write_balances(paths, label),
        "connections": _write_connections(paths, label),
    }


def _bl_outputs(
    paths: ProjectPaths, *, label: str = "bl", coarse_answer: bool = False
) -> dict[str, Path]:
    """Three BL grids whose published saturation IS the analytic answer, cell-averaged.

    A perfect numerical answer makes the L1 zero, which is the wrong thing to test alone:
    `coarse_answer` replaces the 128-cell profile with the 32-cell one resampled, so the
    fine grid is no better than the coarse one and the refinement gate has to notice.
    """
    outputs: dict[str, Path] = {}
    profiles: dict[int, NDArray[np.float64]] = {}
    for n_cells in (32, 64, 128):
        edges = np.linspace(0.0, 1.0, n_cells + 1)
        profiles[n_cells] = bl_cell_average(edges, t_pvi=0.2, subpoints=64)
    if coarse_answer:
        centers = np.linspace(0.0, 1.0, 129)[:-1] + 0.5 / 128
        coarse_edges = np.linspace(0.0, 1.0, 33)
        index = np.clip(np.searchsorted(coarse_edges, centers) - 1, 0, 31)
        profiles[128] = profiles[32][index]
    for n_cells in (32, 64, 128):
        sw = np.vstack([np.zeros(n_cells), profiles[n_cells]])
        outputs[f"states_{n_cells}"] = _write_states(
            paths,
            f"{label}-{n_cells}",
            times_s=np.array([0.0, SECONDS_PER_DAY]),
            pressure_pa=np.full((2, n_cells), 1.5e7),
            sw=sw,
            pore_volume_m3=np.full((2, n_cells), 20.0 / n_cells),
        )
    outputs["balances_128"] = _write_balances(paths, f"{label}-128")
    return outputs


@pytest.fixture
def paths(tmp_project: Path) -> ProjectPaths:
    project = ProjectPaths.default(tmp_project)
    project.ensure_dirs()
    return project


@pytest.fixture
def tolerances() -> dict[str, float]:
    return load_tolerances(TOLERANCES_PATH)


# --------------------------------------------------------------------------------------
# 9.2 the evaluator: it passes on a clean fixture, and it FAILS on a perturbed one
# --------------------------------------------------------------------------------------


def test_a_clean_closed_fixture_passes_and_records_what_it_scored(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    outputs = _closed_outputs(paths)
    check = evaluate_physics("closed_box_pvt", outputs, tolerances)
    assert isinstance(check, PhysicsCheck)
    assert check.status == "PASS", check.reason
    assert check.reason is None
    # Every gated threshold is on the record, with the metric it gated.
    assert "closed_state_saturation_drift_max" in check.thresholds
    assert check.metrics["state_saturation_drift"] == 0.0
    assert check.metrics["connection_mass_kg_s"] == 0.0
    assert check.metrics["balance_cumulative_relative"] == pytest.approx(0.0, abs=1e-15)
    # The stricter SPEC 8.3 aim is reported beside the 23.1 gate, never as the gate.
    assert check.thresholds["balance_cumulative_target"] == 1.0e-5
    assert check.metrics["balance_cumulative_target_met"] == 1.0
    # Which bytes were scored, and where they are.
    assert set(check.input_hashes) == set(outputs)
    assert all(len(digest) == 64 for digest in check.input_hashes.values())
    assert len(check.evidence_paths) == len(outputs)


def test_one_saturation_moved_in_a_closed_system_turns_the_fixture_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    drift = 10 * tolerances["closed_state_saturation_drift_max"]
    check = evaluate_physics("closed_box_pvt", _closed_outputs(paths, sw_drift=drift), tolerances)
    assert check.status == "FAIL"
    assert check.metrics["state_saturation_drift"] == pytest.approx(drift, rel=1e-9)
    assert check.reason is not None and "state_saturation_drift" in check.reason


def test_one_pressure_moved_in_a_closed_system_turns_the_fixture_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, pressure_drift_pa=100.0), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "pressure_relative_drift" in check.reason


def test_one_inventory_moved_turns_the_balance_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, residual_m3_sc=1.0), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "balance_cumulative_relative" in check.reason
    assert (
        check.metrics["balance_cumulative_relative"] > tolerances["balance_cumulative_relative_max"]
    )


def test_one_connection_carrying_mass_turns_the_closed_fixture_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, connection_mass_kg=1.0), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "connection_mass_kg_s" in check.reason
    assert check.metrics["connection_mass_kg_s"] == pytest.approx(1.0 / SECONDS_PER_DAY)


def test_an_artifact_that_is_missing_is_reported_unrun_and_never_omitted(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    outputs = _closed_outputs(paths)
    del outputs["connections"]
    check = evaluate_physics("closed_box_pvt", outputs, tolerances)
    assert check.status == "NOT_RUN"
    assert check.reason is not None and "connections" in check.reason
    assert check.metrics == {}
    # The thresholds it WOULD have been scored against are still on the record, so a report
    # of every metric including the unrun ones can be written from this alone.
    assert "closed_connection_mass_kg_s_max" in check.thresholds


def test_an_artifact_the_evaluator_cannot_read_is_reported_unrun(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    outputs = _closed_outputs(paths)
    outputs["connections"] = paths.artifacts / "closed" / "absent.parquet"
    check = evaluate_physics("closed_box_pvt", outputs, tolerances)
    assert check.status == "NOT_RUN"
    assert check.reason is not None and "absent.parquet" in check.reason


def test_a_fixture_nobody_registered_is_refused_rather_than_scored(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="five_spot"):
        evaluate_physics("five_spot", _closed_outputs(paths), tolerances)


def test_an_incomplete_tolerance_mapping_is_refused(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    del tolerances["closed_pressure_relative_drift_max"]
    with pytest.raises(ValueError, match="closed_pressure_relative_drift_max"):
        evaluate_physics("closed_box_pvt", _closed_outputs(paths), tolerances)


def test_a_column_in_discrete_hydrostatic_balance_passes(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    check = evaluate_physics("hydrostatic", _hydrostatic_outputs(paths), tolerances)
    assert check.status == "PASS", check.reason
    assert check.metrics["hydrostatic_face_residual_relative"] < 1e-12
    # z is depth, positive down: the published column really gets heavier downwards.
    assert check.metrics["pressure_depth_gradient_pa_m"] == pytest.approx(RHO_W_SC * G, rel=1e-9)


def test_a_column_that_is_not_in_hydrostatic_balance_turns_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    # 100 Pa on a 490 kPa head is 2e-4 relative, twenty times the 1e-5 gate.
    check = evaluate_physics(
        "hydrostatic", _hydrostatic_outputs(paths, pressure_offset_pa=100.0), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "hydrostatic_face_residual_relative" in check.reason


def test_water_that_sinks_passes_segregation_and_water_that_rises_fails_it(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    sinking = evaluate_physics("segregation", _segregation_outputs(paths, sinks=True), tolerances)
    assert sinking.status == "PASS", sinking.reason
    assert sinking.metrics["water_mean_depth_increase_m"] > 0.0
    assert sinking.metrics["water_center_of_height_change_m"] < 0.0

    rising = evaluate_physics(
        "segregation", _segregation_outputs(paths, label="rise", sinks=False), tolerances
    )
    assert rising.status == "FAIL"
    assert rising.reason is not None and "water_mean_depth_increase_m" in rising.reason


def test_the_bl_refinement_passes_on_the_analytic_answer_and_fails_on_a_coarse_one(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    good = evaluate_physics("bl", _bl_outputs(paths), tolerances)
    assert good.status == "PASS", good.reason
    assert good.metrics["bl_pv_l1_at_128"] < tolerances["bl_pv_l1_max_at_128"]
    assert good.metrics["bl_front_position_pv"] == pytest.approx(bl_front_position(0.2))

    bad = evaluate_physics("bl", _bl_outputs(paths, label="blc", coarse_answer=True), tolerances)
    assert bad.status == "FAIL"
    assert bad.reason is not None and "bl_refinement_ratio_64_to_128" in bad.reason
