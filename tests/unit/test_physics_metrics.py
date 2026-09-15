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
    MONTHLY_SCHEMA,
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
    CommonSupport,
    PhysicsCheck,
    aggregate_so,
    bl_cell_average,
    bl_front_position,
    bl_saturation,
    cartesian_zone_ids,
    compare_refinement,
    evaluate_physics,
    load_tolerances,
    mirror_symmetry_abs,
)

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES_PATH = ROOT / DEFAULT_TOLERANCES_RELPATH

#: Task 9.1 plus the explicit v2 (2026-09-14) near-zero 1 ml phase tolerance; compare
#: against this independent list. A change to `configs/e01_tolerances.yml` that nobody decided fails
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
    "refinement_monthly_volume_absolute_max_m3_sc": 1.0e-6,
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
    payload["schema_version"] = "e01-tolerances-1"
    path = _write_tolerances(tmp_path / "t.yml", payload)
    with pytest.raises(ValueError, match="e01-tolerances-1"):
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
# 10.1 the support aggregator
# --------------------------------------------------------------------------------------


def test_support_average_uses_pore_volume() -> None:
    mean = aggregate_so(np.array([0.2, 0.8]), np.array([1.0, 3.0]), np.array([0, 0]), 1)
    np.testing.assert_allclose(mean, [0.65])


def test_a_zone_that_collected_no_pore_volume_is_nan_and_never_zero() -> None:
    # Zone 1 has a cell, and that cell has no pore volume; zone 2 has no cell at all. Both are
    # UNMEASURED, and an unmeasured saturation must not be depicted as a dry one: 0.0 is a
    # number somebody will plot beside real saturations and average further.
    mean = aggregate_so(np.array([0.5, 0.9]), np.array([2.0, 0.0]), np.array([0, 1]), 3)
    assert mean[0] == pytest.approx(0.5)
    assert math.isnan(mean[1])
    assert math.isnan(mean[2])


def test_zones_are_aggregated_apart_and_weighted_within_themselves() -> None:
    mean = aggregate_so(
        np.array([0.2, 0.8, 1.0, 0.0]),
        np.array([1.0, 3.0, 1.0, 1.0]),
        np.array([0, 0, 1, 1]),
        2,
    )
    np.testing.assert_allclose(mean, [0.65, 0.5])
    # And the weighting really is by pore volume: the unweighted mean of zone 0 is 0.5.
    assert mean[0] != pytest.approx(0.5)


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


def _write_balances(
    paths: ProjectPaths,
    label: str,
    *,
    residual_m3_sc: float = 0.0,
    step_residual_m3_sc: float = 0.0,
) -> Path:
    """A published balances.parquet, from a real inventory/source pair.

    Both knobs move an INVENTORY entry of water and leave the source alone, so the residual
    the evaluator reads is one `component_balance` really computed rather than a metric typed
    in by hand.

    `residual_m3_sc` moves the LAST entry, which breaks the cumulative balance.
    `step_residual_m3_sc` moves the MIDDLE one, which breaks two consecutive steps by equal
    and opposite amounts and therefore leaves the cumulative balance intact: that is the only
    way to make `balance_step_median_relative` the gate that fires, and without it a test
    could never tell the two balance gates apart.
    """
    inventory = np.array([[100.0, 200.0], [90.0, 200.0], [80.0, 200.0]], dtype=np.float64)
    source = np.diff(inventory, axis=0).copy()
    inventory[1, 0] += step_residual_m3_sc
    inventory[-1, 0] += residual_m3_sc
    metrics = {
        name: component_balance(inventory, source, components=("water", "oil"))
        for name in ("full_system_surface", "reservoir_connections")
    }
    path = paths.artifacts / label / BALANCES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(balance_table(metrics, {"water": 0.0, "oil": 0.0}), path)
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
    step_residual_m3_sc: float = 0.0,
    connection_mass_kg: float = 0.0,
    saturation_sum_excess: float = 0.0,
    saturation_overshoot: float = 0.0,
) -> dict[str, Path]:
    """The closed fixture's published artifacts, with one thing at a time moved.

    `saturation_sum_excess` and `saturation_overshoot` break the saturation field in the two
    independent ways the evaluator gates on, and both are applied to EVERY published time so
    that the drift gate stays quiet and the failure is unambiguously the one under test:
    an excess adds to `so` alone, so `sw + so` leaves 1 while both stay inside [0,1]; an
    overshoot pushes `sw` above 1 and `so` symmetrically below 0, so the sum stays exactly 1
    and only the bound is broken.
    """
    times = np.array([0.0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY], dtype=np.float64)
    sw = np.full((3, 2), 0.3)
    sw[-1, 0] += sw_drift
    if saturation_overshoot:
        sw = np.full((3, 2), 1.0 + saturation_overshoot)
    so = 1.0 - sw + saturation_sum_excess
    pressure = np.full((3, 2), 1.5e7)
    pressure[-1, 1] += pressure_drift_pa
    return {
        "states": _write_states(paths, label, times_s=times, pressure_pa=pressure, sw=sw, so=so),
        "balances": _write_balances(
            paths,
            label,
            residual_m3_sc=residual_m3_sc,
            step_residual_m3_sc=step_residual_m3_sc,
        ),
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
    paths: ProjectPaths,
    *,
    label: str = "bl",
    coarse_answer: bool = False,
    published_t_pvi: float = 0.2,
) -> dict[str, Path]:
    """Three BL grids whose published saturation IS the analytic answer, cell-averaged.

    A perfect numerical answer makes the L1 zero, which is the wrong thing to test alone, so
    there are two independent ways of breaking it and each one fires a DIFFERENT gate.

    `coarse_answer` replaces the 128-cell profile with the 32-cell one resampled: the fine
    grid is then no better than the coarse one and `bl_refinement_ratio_64_to_128` has to
    notice, while the L1 of the 64-cell grid stays at zero.

    `published_t_pvi` publishes every grid's profile at a DIFFERENT injected volume from the
    one the fixture's schedule implies (the `times_s` below still say one day, so the
    evaluator still builds its reference at 0.2 PV). The front is then in the wrong place by
    the same amount on all three grids, so the refinement ratio stays at one and it is
    `bl_pv_l1_at_128` that has to fire — the gate `coarse_answer` cannot reach.
    """
    outputs: dict[str, Path] = {}
    profiles: dict[int, NDArray[np.float64]] = {}
    for n_cells in (32, 64, 128):
        edges = np.linspace(0.0, 1.0, n_cells + 1)
        profiles[n_cells] = bl_cell_average(edges, t_pvi=published_t_pvi, subpoints=64)
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


def test_a_saturation_pair_that_does_not_sum_to_one_turns_the_fixture_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    """`saturation_sum_abs` has its own gate, so it needs its own red case.

    Every other fixture in this file publishes `so = 1 - sw`, which makes the sum identically
    one and the gate unfalsifiable: a sign error in `_saturation_metrics` would pass forever.
    Here `so` alone is moved, so `sw + so` leaves 1 while both stay inside [0,1] and the
    saturations stand still — the sum is the only thing wrong.
    """
    excess = 10 * tolerances["saturation_sum_abs_max"]
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, saturation_sum_excess=excess), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "saturation_sum_abs" in check.reason
    assert check.metrics["saturation_sum_abs"] == pytest.approx(excess, rel=1e-9)
    # ...and nothing else: the bounds are intact and the state never moved.
    assert check.metrics["saturation_bound_violation"] == 0.0
    assert check.metrics["state_saturation_drift"] == 0.0


def test_a_saturation_outside_zero_to_one_turns_the_fixture_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    """`saturation_bound_violation` likewise. `sw` goes above 1 and `so` symmetrically below
    0, so the pair still sums to exactly 1 and the bound is the only gate that can fire."""
    overshoot = 10 * tolerances["saturation_bound_slack"]
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, saturation_overshoot=overshoot), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "saturation_bound_violation" in check.reason
    assert check.metrics["saturation_bound_violation"] == pytest.approx(overshoot, rel=1e-6)
    assert check.metrics["saturation_sum_abs"] <= tolerances["saturation_sum_abs_max"]


def test_a_per_step_residual_that_cancels_over_the_horizon_still_turns_the_balance_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    """The median-per-step gate is not the cumulative gate, and this is what tells them apart.

    Moving the MIDDLE inventory entry breaks two consecutive steps by equal and opposite
    amounts, so the cumulative balance closes exactly and only SPEC 23.1's per-step half has
    anything to say. A suite that only ever breaks the cumulative balance would never find out
    whether the median gate works at all.
    """
    check = evaluate_physics(
        "closed_box_pvt", _closed_outputs(paths, step_residual_m3_sc=0.01), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None
    assert "balance_step_median_relative" in check.reason
    assert "balance_cumulative_relative" not in check.reason
    assert (
        check.metrics["balance_step_median_relative"]
        > tolerances["balance_step_median_relative_max"]
    )
    assert (
        check.metrics["balance_cumulative_relative"]
        <= tolerances["balance_cumulative_relative_max"]
    )


def test_a_front_in_the_wrong_place_on_every_grid_turns_the_bl_l1_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    """`bl_pv_l1_max_at_128` needs a red case the refinement ratio cannot claim.

    `coarse_answer` fires the ratio because the 64-cell error is zero; it says nothing about
    the L1 threshold. Publishing every grid's profile at 0.35 PV against a reference built at
    0.2 PV displaces the front by the same amount everywhere: the ratio stays at one and the
    0.08 L1 gate is the only thing left to fail. The L1 is then the extra water itself, since
    the BL solution is monotone in time before breakthrough.
    """
    check = evaluate_physics(
        "bl", _bl_outputs(paths, label="blf", published_t_pvi=0.35), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "bl_pv_l1_at_128" in check.reason
    assert "bl_refinement_ratio_64_to_128" not in check.reason
    assert check.metrics["bl_pv_l1_at_128"] > tolerances["bl_pv_l1_max_at_128"]
    assert check.metrics["bl_pv_l1_at_128"] == pytest.approx(0.35 - 0.2, abs=1e-3)
    assert check.metrics["bl_refinement_ratio_64_to_128"] <= tolerances["bl_refinement_ratio_max"]


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
    with pytest.raises(ValueError, match="nine_spot"):
        evaluate_physics("nine_spot", _closed_outputs(paths), tolerances)


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


# --------------------------------------------------------------------------------------
# 10.7 the five-spot: a symmetry that has to be a symmetry of something
# --------------------------------------------------------------------------------------

#: The registered five-spot: 16 x 16 cells in one layer over 400 x 400 x 10 m at porosity 0.2.
FIVE_SPOT_NX = 16
FIVE_SPOT_EXTENT_M = 400.0
FIVE_SPOT_THICKNESS_M = 10.0
FIVE_SPOT_POROSITY = 0.2
SUPPORT_SIDE = 8


def _areal_so(nx: int, *, uniform: bool = False) -> NDArray[np.float64]:
    """An oil saturation that is symmetric about BOTH grid axes, by construction.

    The field depends only on the distance of a cell centre from the pattern's own centre in
    each direction, so reflecting `i -> nx-1-i` or `j -> ny-1-j` leaves it unchanged. That is
    what makes the asymmetry a test introduces the ONLY asymmetry in the file.
    """
    if uniform:
        return np.full(nx * nx, 0.55, dtype=np.float64)
    axis = np.abs(np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) / ((nx - 1) / 2.0)
    field = 0.3 + 0.25 * (axis[None, :] + axis[:, None]) / 2.0
    return np.asarray(field.ravel(), dtype=np.float64)


def _five_spot_outputs(
    paths: ProjectPaths,
    *,
    label: str = "fivespot",
    asymmetry: float = 0.0,
    uniform: bool = False,
) -> dict[str, Path]:
    """A published five-spot result whose final oil saturation is the field above.

    `asymmetry` is added to ONE cell of the final state, which breaks the reflection about
    both axes by exactly that much and leaves everything else the evaluator scores alone.
    """
    n_cells = FIVE_SPOT_NX * FIVE_SPOT_NX
    times = np.array([0.0, SECONDS_PER_DAY], dtype=np.float64)
    so = np.vstack([np.full(n_cells, 0.8), _areal_so(FIVE_SPOT_NX, uniform=uniform)])
    so[-1, 0] += asymmetry
    sw = 1.0 - so
    pressure = np.full((2, n_cells), 1.5e7)
    cell_pv = (FIVE_SPOT_EXTENT_M / FIVE_SPOT_NX) ** 2 * FIVE_SPOT_THICKNESS_M * FIVE_SPOT_POROSITY
    return {
        "states": _write_states(
            paths,
            label,
            times_s=times,
            pressure_pa=pressure,
            sw=sw,
            so=so,
            pore_volume_m3=np.full((2, n_cells), cell_pv),
        ),
        "balances": _write_balances(paths, label),
        "connections": _write_connections(paths, label),
    }


def test_a_reflection_is_measured_about_each_axis_separately() -> None:
    field = _areal_so(4)
    assert mirror_symmetry_abs(field, (4, 4, 1)) == (0.0, 0.0)
    moved = field.copy()
    moved[0] += 0.125
    about_x, about_y = mirror_symmetry_abs(moved, (4, 4, 1))
    assert about_x == pytest.approx(0.125)
    assert about_y == pytest.approx(0.125)


def test_a_symmetric_five_spot_passes_and_records_both_reflections(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    check = evaluate_physics("five_spot", _five_spot_outputs(paths), tolerances)
    assert check.status == "PASS", check.reason
    assert check.metrics["five_spot_symmetry_x_abs"] == 0.0
    assert check.metrics["five_spot_symmetry_y_abs"] == 0.0
    # The field it is symmetric about really varies, so the verdict is a statement.
    assert check.metrics["five_spot_so_range"] > 0.2
    assert check.unrun_metrics == ()


def test_one_cell_that_breaks_the_reflection_turns_the_five_spot_red(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    # Ten times the frozen gate, on one cell of the final state.
    asymmetry = 10 * tolerances["five_spot_symmetry_abs_max"]
    check = evaluate_physics(
        "five_spot", _five_spot_outputs(paths, label="skew", asymmetry=asymmetry), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "five_spot_symmetry_abs" in check.reason
    assert check.metrics["five_spot_symmetry_abs"] == pytest.approx(asymmetry, rel=1e-9)
    # Nothing else moved: the balance and the saturation gates stay quiet.
    assert check.metrics["balance_cumulative_relative"] == 0.0
    assert check.metrics["saturation_sum_abs"] <= tolerances["saturation_sum_abs_max"]


def test_a_uniform_field_is_symmetric_and_still_fails_the_five_spot(
    paths: ProjectPaths, tolerances: dict[str, float]
) -> None:
    # A flat saturation is symmetric about every axis and demonstrates nothing at all. The
    # structural companion gate is what stops a case that never displaced anything passing
    # the symmetry claim by being empty.
    check = evaluate_physics(
        "five_spot", _five_spot_outputs(paths, label="flat", uniform=True), tolerances
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "five_spot_so_range" in check.reason
    assert check.metrics["five_spot_symmetry_abs"] == 0.0


# --------------------------------------------------------------------------------------
# 10.8 the refinement comparison
# --------------------------------------------------------------------------------------


def test_a_support_that_does_not_tile_a_grid_into_whole_cells_is_refused() -> None:
    with pytest.raises(ValueError, match="whole cells"):
        cartesian_zone_ids(16, 16, 5)


def test_nested_grids_get_the_same_zones_over_the_same_physical_square() -> None:
    coarse = cartesian_zone_ids(16, 16, SUPPORT_SIDE)
    fine = cartesian_zone_ids(32, 32, SUPPORT_SIDE)
    assert coarse.shape == (256,) and fine.shape == (1024,)
    # Every zone is 2x2 coarse cells and 4x4 fine ones, and every zone is occupied on both.
    assert np.bincount(coarse).tolist() == [4] * SUPPORT_SIDE**2
    assert np.bincount(fine).tolist() == [16] * SUPPORT_SIDE**2
    # The zone of a coarse cell is the zone of the fine cells that replaced it: cell (2,3) on
    # the coarse grid covers the same rock as (4..5, 6..7) on the fine one.
    assert coarse[2 + 16 * 3] == fine[4 + 32 * 6] == fine[5 + 32 * 7]


def test_a_zone_id_outside_the_support_is_refused_before_any_aggregation() -> None:
    good = cartesian_zone_ids(16, 16, SUPPORT_SIDE)
    bad = good.copy()
    bad[0] = SUPPORT_SIDE**2
    with pytest.raises(ValueError, match="outside"):
        CommonSupport(
            name="five_spot_refinement",
            n_zones=SUPPORT_SIDE**2,
            coarse_zone_id=bad,
            fine_zone_id=cartesian_zone_ids(32, 32, SUPPORT_SIDE),
        )


def test_a_support_zone_no_cell_falls_in_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        CommonSupport(
            name="five_spot_refinement",
            n_zones=SUPPORT_SIDE**2 + 1,
            coarse_zone_id=cartesian_zone_ids(16, 16, SUPPORT_SIDE),
            fine_zone_id=cartesian_zone_ids(32, 32, SUPPORT_SIDE),
        )


def _support() -> CommonSupport:
    return CommonSupport(
        name="five_spot_refinement",
        n_zones=SUPPORT_SIDE**2,
        coarse_zone_id=cartesian_zone_ids(16, 16, SUPPORT_SIDE),
        fine_zone_id=cartesian_zone_ids(32, 32, SUPPORT_SIDE),
    )


def _continuous_so(nx: int) -> NDArray[np.float64]:
    """The SAME continuous saturation field, sampled at the cell centres of an `nx` grid.

    Sampling one function on both grids is what makes the pair a refinement of one problem:
    a second field drawn independently on the fine grid would be a different case, and the
    difference between them would not be a discretisation error.
    """
    centres = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    field = 0.35 + 0.2 * np.sin(np.pi * centres)[None, :] * np.sin(np.pi * centres)[:, None]
    return np.asarray(field.ravel(), dtype=np.float64)


def _refinement_side(
    paths: ProjectPaths,
    *,
    label: str,
    nx: int,
    so_shift: float = 0.0,
    pore_volume_scale: float = 1.0,
    monthly_scale: float = 1.0,
    months: int = 3,
) -> dict[str, Path]:
    """One published side of a refinement pair: its final states and its monthly volumes."""
    n_cells = nx * nx
    so = np.vstack([np.full(n_cells, 0.8), _continuous_so(nx) + so_shift])
    sw = 1.0 - so
    cell_pv = (
        (FIVE_SPOT_EXTENT_M / nx) ** 2 * FIVE_SPOT_THICKNESS_M * FIVE_SPOT_POROSITY
    ) * pore_volume_scale
    # `_write_states` writes bw = bo = 1 everywhere, so the standard-volume inventory of a
    # side is `sum(S * PV)` and the only things that can move it are the two knobs above.
    states = _write_states(
        paths,
        label,
        times_s=np.array([0.0, SECONDS_PER_DAY], dtype=np.float64),
        pressure_pa=np.full((2, n_cells), 1.5e7),
        sw=sw,
        so=so,
        pore_volume_m3=np.full((2, n_cells), cell_pv),
    )
    rows = [
        {
            "well_id": "PRO1",
            "month_index": month,
            "start_s": float(month) * SECONDS_PER_DAY,
            "end_s": float(month + 1) * SECONDS_PER_DAY,
            "oil_prod_m3_sc": 1000.0 * monthly_scale,
            "water_prod_m3_sc": 10.0 * monthly_scale,
            "water_inj_m3_sc": 0.0,
            "liquid_prod_m3_sc": 1010.0 * monthly_scale,
            "flowing_s": SECONDS_PER_DAY,
            "fw": 10.0 / 1010.0,
            "fw_valid": True,
        }
        for month in range(months)
    ]
    monthly = paths.artifacts / label / "monthly.parquet"
    monthly.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MONTHLY_SCHEMA), monthly)
    return {"states": states, "monthly": monthly}


def test_two_grids_that_agree_on_the_same_support_pass(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    check = compare_refinement(
        _refinement_side(paths, label="ref-c", nx=16),
        _refinement_side(paths, label="ref-f", nx=32),
        _support(),
        tolerances,
    )
    assert check.status == "PASS", check.reason
    # The mapping really is pore-volume-conservative: 4 coarse cells of 1250 m3 and 16 fine
    # ones of 312.5 m3 both put 5000 m3 in each of the 64 zones.
    assert check.metrics["support_pore_volume_relative"] == 0.0
    assert check.metrics["n_zones"] == 64.0
    assert check.metrics["coarse_cells"] == 256.0 and check.metrics["fine_cells"] == 1024.0
    # Not vacuous: the two grids do NOT agree exactly, because sampling one continuous field
    # at two resolutions is not the same average.
    assert 0.0 < check.metrics["so_pv_mae"] < tolerances["refinement_so_pv_mae_max"]
    assert check.unrun_metrics == ()
    assert set(check.input_hashes) == {
        "coarse_states",
        "coarse_monthly",
        "fine_states",
        "fine_monthly",
    }


def test_a_saturation_that_moved_on_the_fine_grid_turns_the_support_mae_red(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    shift = 5 * tolerances["refinement_so_pv_mae_max"]
    check = compare_refinement(
        _refinement_side(paths, label="mae-c", nx=16),
        _refinement_side(paths, label="mae-f", nx=32, so_shift=shift),
        _support(),
        tolerances,
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "so_pv_mae" in check.reason
    # The shift, to within the discretisation difference the clean pair already carried.
    assert check.metrics["so_pv_mae"] == pytest.approx(shift, rel=0.01)
    # The pore-volume mapping and the monthly volumes are untouched and stay quiet.
    assert check.metrics["support_pore_volume_relative"] == 0.0
    assert check.metrics["monthly_volume_relative"] == 0.0


def test_a_monthly_volume_that_moved_turns_the_refinement_red(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    scale = 1.0 + 5 * tolerances["refinement_monthly_volume_relative_max"]
    check = compare_refinement(
        _refinement_side(paths, label="mv-c", nx=16),
        _refinement_side(paths, label="mv-f", nx=32, monthly_scale=scale),
        _support(),
        tolerances,
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "monthly_volume_relative" in check.reason
    assert check.metrics["so_pv_mae"] < tolerances["refinement_so_pv_mae_max"]


def test_a_mapping_that_is_not_pore_volume_conservative_turns_red_first(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    # The same saturations on both grids, but the fine grid carries 5% more pore volume in
    # every zone. Nothing about the saturation comparison would notice; the inventory and the
    # structural pore-volume gate both do, and the pore-volume one is the one that says why.
    check = compare_refinement(
        _refinement_side(paths, label="pv-c", nx=16),
        _refinement_side(paths, label="pv-f", nx=32, pore_volume_scale=1.05),
        _support(),
        tolerances,
    )
    assert check.status == "FAIL"
    assert check.reason is not None and "support_pore_volume_relative" in check.reason
    assert check.metrics["support_pore_volume_relative"] == pytest.approx(0.05, rel=1e-9)
    assert check.metrics["inventory_relative"] == pytest.approx(0.05, rel=1e-2)
    assert check.metrics["so_pv_mae"] < tolerances["refinement_so_pv_mae_max"]


def test_a_refinement_whose_artifacts_are_missing_is_reported_unrun(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    coarse = _refinement_side(paths, label="unrun-c", nx=16)
    fine = dict(_refinement_side(paths, label="unrun-f", nx=32))
    del fine["monthly"]
    check = compare_refinement(coarse, fine, _support(), tolerances)
    assert check.status == "NOT_RUN"
    assert check.reason is not None and "fine_monthly" in check.reason
    assert check.metrics == {}
    # The thresholds it WOULD have been scored against are still on the record.
    assert "refinement_so_pv_mae_max" in check.thresholds
    assert "so_pv_mae" in check.unrun_metrics


def test_a_refinement_fixture_is_not_scored_by_the_single_result_evaluator(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    with pytest.raises(ValueError, match="compare_refinement"):
        evaluate_physics("five_spot_refinement", _closed_outputs(paths), tolerances)


def test_a_fixture_that_is_not_a_refinement_pair_is_refused_by_compare_refinement(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    support = CommonSupport(
        name="five_spot",
        n_zones=SUPPORT_SIDE**2,
        coarse_zone_id=cartesian_zone_ids(16, 16, SUPPORT_SIDE),
        fine_zone_id=cartesian_zone_ids(32, 32, SUPPORT_SIDE),
    )
    with pytest.raises(ValueError, match="registered refinement fixture"):
        compare_refinement(
            _refinement_side(paths, label="wrong-c", nx=16),
            _refinement_side(paths, label="wrong-f", nx=32),
            support,
            tolerances,
        )


def test_minor_phase_disagreement_is_not_diluted_by_total_throughput(
    tolerances: dict[str, float], paths: ProjectPaths
) -> None:
    coarse = _refinement_side(paths, label="phase-c", nx=16)
    fine = _refinement_side(paths, label="phase-f", nx=32)
    rows = pq.read_table(fine["monthly"]).to_pylist()
    for row in rows:
        row["water_prod_m3_sc"] = 30.0
        row["oil_prod_m3_sc"] = 980.0
    pq.write_table(pa.Table.from_pylist(rows, schema=MONTHLY_SCHEMA), fine["monthly"])
    check = compare_refinement(coarse, fine, _support(), tolerances)
    assert check.status == "FAIL"
    assert check.metrics["monthly_volume_relative"] == pytest.approx(2.0)


@pytest.mark.parametrize("fine_water, expected", [(5e-7, "PASS"), (2e-6, "FAIL")])
def test_near_zero_phase_uses_explicit_absolute_tolerance(
    tolerances: dict[str, float], paths: ProjectPaths, fine_water: float, expected: str
) -> None:
    coarse = _refinement_side(paths, label="zero-c", nx=16)
    fine = _refinement_side(paths, label="zero-f", nx=32)
    for side, water in ((coarse, 0.0), (fine, fine_water)):
        rows = pq.read_table(side["monthly"]).to_pylist()
        for row in rows:
            row["water_prod_m3_sc"] = water
        pq.write_table(pa.Table.from_pylist(rows, schema=MONTHLY_SCHEMA), side["monthly"])
    check = compare_refinement(coarse, fine, _support(), tolerances)
    assert check.status == expected
    assert check.metrics["monthly_volume_near_zero_absolute_m3_sc"] == fine_water


@pytest.mark.parametrize("value", [0.0, float("inf"), float("nan")])
@pytest.mark.parametrize("via_file", [False, True])
def test_tolerance_limits_must_be_finite_and_positive(
    tmp_path: Path, value: float, via_file: bool
) -> None:
    from so_recon.validation.physics import require_tolerances

    payload = _valid_payload()
    payload["refinement_monthly_volume_relative_max"] = value
    with pytest.raises(ValueError, match="refinement_monthly_volume_relative_max"):
        if via_file:
            load_tolerances(_write_tolerances(tmp_path / "invalid-limit.yml", payload))
        else:
            del payload["schema_version"]
            require_tolerances(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None])
def test_invalid_monthly_phase_volume_is_unreadable(
    tolerances: dict[str, float], paths: ProjectPaths, value: float | None
) -> None:
    coarse = _refinement_side(paths, label="bad-volume-c", nx=16)
    fine = _refinement_side(paths, label="bad-volume-f", nx=32)
    rows = pq.read_table(fine["monthly"]).to_pylist()
    rows[0]["water_prod_m3_sc"] = value
    pq.write_table(pa.Table.from_pylist(rows), fine["monthly"])
    check = compare_refinement(coarse, fine, _support(), tolerances)
    assert check.status == "NOT_RUN"
    assert check.reason is not None and "water_prod_m3_sc" in check.reason


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf"), None, 5])
def test_invalid_month_index_is_unreadable(
    tolerances: dict[str, float], paths: ProjectPaths, value: int | float | None
) -> None:
    coarse = _refinement_side(paths, label="bad-month-c", nx=16)
    fine = _refinement_side(paths, label="bad-month-f", nx=32)
    rows = pq.read_table(fine["monthly"]).to_pylist()
    rows[0]["month_index"] = value
    pq.write_table(pa.Table.from_pylist(rows), fine["monthly"])
    check = compare_refinement(coarse, fine, _support(), tolerances)
    assert check.status == "NOT_RUN"
    assert check.reason is not None and "month_index" in check.reason


def test_coarse_sensitivity_is_registered_with_unchanged_refinement_gates() -> None:
    tolerances = load_tolerances(Path(__file__).resolve().parents[2] / "configs/e01_tolerances.yml")
    checks = [
        compare_refinement(
            {},
            {},
            CommonSupport(
                name=name, n_zones=1, coarse_zone_id=np.array([0]), fine_zone_id=np.array([0])
            ),
            tolerances,
        )
        for name in ("five_spot_coarse_sensitivity", "five_spot_refinement")
    ]
    assert all(c.status == "NOT_RUN" for c in checks)
    assert checks[0].thresholds == checks[1].thresholds
