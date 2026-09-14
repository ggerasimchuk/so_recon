"""E01.7: accepted-substep integrals, the monthly aggregator and the balance evaluator.

The claim this file exists to hold is the one SPEC §9.3 makes: «Snapshot в конце месяца не
заменяет это интегрирование.» A monthly volume is the integral of the accepted substeps of
that month, and the test that proves it is the one where the two answers differ — a well
that produced 1 m³/day for ten days and 3 m³/day for twenty produced 70 m³, and the final
rate times the month says 90.

The rest of the file is the rest of that contract: public rates are non-negative and
production and injection never cancel, a substep never crosses a month boundary, a month
with no liquid keeps its row and reports no water cut at all, and the component balance is
evaluated against a source that was computed independently of the inventory it is compared
with — which is why it can fail, and is made to, by dropping one interval's injection.

Nothing here simulates anything. The native extraction is a fixture built by hand, with an
inventory and a source that agree by construction, so that every test below is about the
code that reads them rather than about a solver's tolerances. The real solver's numbers are
`tests/integration/test_e01_outputs.py`'s subject.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.contracts import (
    SECONDS_PER_DAY,
    CaseBundle,
    CostRecord,
    JobDescriptor,
    OutputRequest,
)
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    MONTHLY_FILENAME,
    RESULT_FILENAME,
    STATES_FILENAME,
    ExtractionError,
    ForwardResultIntegrityError,
    connection_step_table,
    control_infeasibility,
    integrate_connections,
    integrate_monthly,
    load_forward_result,
    publish_forward_result,
    well_step_table,
    write_forward_result,
)
from so_recon.simulator.schedule import compile_schedule
from so_recon.validation import (
    BALANCE_FLOOR_M3_SC,
    component_balance,
    relative_balance_errors,
)
from tests.forward_case import N_CELLS, build_case, write_case_arrays

DAY = SECONDS_PER_DAY


# --------------------------------------------------------------------------------------
# 7.1 / 7.5 the aggregator
# --------------------------------------------------------------------------------------


def _step(
    well_id: str,
    start_days: float,
    end_days: float,
    *,
    oil: float = 0.0,
    water: float = 0.0,
    injected: float = 0.0,
) -> dict[str, Any]:
    """One accepted substep, with its rates given as m³_sc/DAY for readability."""
    return {
        "well_id": well_id,
        "start_s": start_days * DAY,
        "end_s": end_days * DAY,
        "oil_prod_m3_s": oil / DAY,
        "water_prod_m3_s": water / DAY,
        "water_inj_m3_s": injected / DAY,
    }


def _steps(*rows: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist(list(rows))


def test_monthly_volume_is_not_final_rate_times_month() -> None:
    steps = pa.Table.from_pylist(
        [
            dict(
                well_id="P",
                start_s=0.0,
                end_s=864000.0,
                oil_prod_m3_s=1 / 86400,
                water_prod_m3_s=0.0,
                water_inj_m3_s=0.0,
            ),
            dict(
                well_id="P",
                start_s=864000.0,
                end_s=2592000.0,
                oil_prod_m3_s=3 / 86400,
                water_prod_m3_s=0.0,
                water_inj_m3_s=0.0,
            ),
        ]
    )
    rows = integrate_monthly(steps, (0.0, 2592000.0)).to_pylist()
    assert rows[0]["oil_prod_m3_sc"] == pytest.approx(70.0)
    # And the answer the snapshot would have given, so the difference is on the record.
    assert rows[0]["oil_prod_m3_sc"] != pytest.approx(3.0 * 30.0)


def test_injection_never_cancels_production_when_a_well_changes_role() -> None:
    """A well that injected and then produced inside one month keeps both groups."""
    rows = integrate_monthly(
        _steps(
            _step("W", 0.0, 10.0, injected=6.0),
            _step("W", 10.0, 30.0, oil=2.0, water=1.0),
        ),
        (0.0, 30.0 * DAY),
    ).to_pylist()
    assert rows[0]["water_inj_m3_sc"] == pytest.approx(60.0)
    assert rows[0]["water_prod_m3_sc"] == pytest.approx(20.0)
    assert rows[0]["oil_prod_m3_sc"] == pytest.approx(40.0)
    # The water cut is produced water over produced liquid: the 60 m³ that went back down
    # the injector is not in it, and a signed total would have made it negative.
    assert rows[0]["fw"] == pytest.approx(20.0 / 60.0)
    assert rows[0]["fw_valid"] is True


def test_a_month_with_no_liquid_keeps_its_row_and_reports_no_water_cut() -> None:
    rows = integrate_monthly(
        _steps(
            _step("W", 0.0, 31.0, oil=2.0, water=2.0),
            _step("W", 31.0, 60.0),
        ),
        (0.0, 31.0 * DAY, 60.0 * DAY),
    ).to_pylist()
    assert [row["month_index"] for row in rows] == [0, 1]
    assert rows[1]["oil_prod_m3_sc"] == 0.0
    assert rows[1]["liquid_prod_m3_sc"] == 0.0
    assert rows[1]["fw"] is None
    assert rows[1]["fw_valid"] is False
    assert rows[1]["flowing_s"] == 0.0
    # And the month that DID flow reports one, so the null above is not a blanket refusal.
    assert rows[0]["fw"] == pytest.approx(0.5)


def test_a_shut_well_still_gets_a_row_in_every_month() -> None:
    """An absent row is indistinguishable from a month nobody computed."""
    rows = integrate_monthly(
        _steps(
            _step("PRO1", 0.0, 31.0, oil=1.0),
            _step("PRO1", 31.0, 60.0, oil=1.0),
            _step("INJ1", 0.0, 31.0, injected=5.0),
            _step("INJ1", 31.0, 60.0),
        ),
        (0.0, 31.0 * DAY, 60.0 * DAY),
    ).to_pylist()
    assert [(row["well_id"], row["month_index"]) for row in rows] == [
        ("INJ1", 0),
        ("INJ1", 1),
        ("PRO1", 0),
        ("PRO1", 1),
    ]
    assert rows[1]["water_inj_m3_sc"] == 0.0
    assert rows[1]["fw_valid"] is False


def test_a_substep_that_crosses_a_month_boundary_is_refused() -> None:
    with pytest.raises(ValueError, match="crosses the month boundary"):
        integrate_monthly(
            _steps(_step("W", 20.0, 40.0, oil=1.0)),
            (0.0, 31.0 * DAY, 60.0 * DAY),
        )


def test_overlapping_substeps_are_refused() -> None:
    with pytest.raises(ValueError, match="substeps overlap"):
        integrate_monthly(
            _steps(_step("W", 0.0, 20.0, oil=1.0), _step("W", 10.0, 30.0, oil=1.0)),
            (0.0, 30.0 * DAY),
        )


def test_a_substep_outside_the_reported_horizon_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the reported horizon"):
        integrate_monthly(_steps(_step("W", 30.0, 40.0, oil=1.0)), (0.0, 30.0 * DAY))


def test_a_negative_public_volume_is_refused() -> None:
    with pytest.raises(ValueError, match="Public rates are non-negative"):
        integrate_monthly(_steps(_step("W", 0.0, 30.0, oil=-1.0)), (0.0, 30.0 * DAY))


def test_a_nonfinite_rate_is_refused_rather_than_integrated() -> None:
    with pytest.raises(ValueError, match="nonfinite rate is refused"):
        integrate_monthly(_steps(_step("W", 0.0, 30.0, oil=float("nan"))), (0.0, 30.0 * DAY))


def test_a_degenerate_substep_is_refused() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        integrate_monthly(_steps(_step("W", 10.0, 10.0, oil=1.0)), (0.0, 30.0 * DAY))


# --------------------------------------------------------------------------------------
# 7.6 the balance evaluator
# --------------------------------------------------------------------------------------


def test_a_closed_inventory_with_no_source_balances_exactly() -> None:
    inventory = [[200.0, 500.0]] * 4
    source = [[0.0, 0.0]] * 3
    per_step, cumulative = relative_balance_errors(inventory, source)
    assert per_step.shape == (3, 2)
    assert not per_step.any()
    assert not cumulative.any()


def test_dropping_one_intervals_injection_is_caught() -> None:
    """The evaluator can fail, and this is the proof.

    The inventory and the source below agree step by step. Removing ONE interval's injection
    from the source — exactly what a chunk that was lost, or an interval whose forces were
    never applied, would look like — leaves the inventory untouched and the residual has
    nowhere to hide.
    """
    inventory = [[1000.0, 2000.0]]
    source = []
    for _ in range(5):
        step = [50.0, -20.0]
        source.append(step)
        inventory.append([a + b for a, b in zip(inventory[-1], step, strict=True)])

    good = component_balance(inventory, source, components=("water", "oil"))
    assert good.cumulative_relative == pytest.approx((0.0, 0.0))
    assert good.within_spec_tolerance
    assert good.failures() == ()

    lost = copy.deepcopy(source)
    lost[2] = [0.0, lost[2][1]]  # this interval's injection never reached the source
    bad = component_balance(inventory, lost, components=("water", "oil"))
    assert bad.cumulative_relative[0] > 0.0
    assert bad.absolute_residual[0] == pytest.approx(50.0)
    assert bad.max_step_absolute[0] == pytest.approx(50.0)
    # Oil is untouched, so the failure names water and only water.
    assert bad.absolute_residual[1] == pytest.approx(0.0)
    assert not bad.within_spec_tolerance
    assert any("water" in message for message in bad.failures())
    assert not any("oil" in message for message in bad.failures())


def test_a_large_standing_inventory_cannot_hide_a_small_offtake_error() -> None:
    """Why `absolute_residual` and `throughput_relative` are reported beside the ratio.

    A million cubic metres in place and a hundred produced: losing a tenth of the month's
    offtake is 1e-5 against the inventory — inside SPEC 23.1 — and a tenth against what
    actually flowed. Both numbers are on the record so the second one can be looked at.
    """
    inventory = [[1_000_000.0], [999_900.0]]
    source = [[-90.0]]  # ten of the hundred produced cubic metres are missing
    metrics = component_balance(inventory, source, components=("water",))
    assert metrics.cumulative_relative[0] == pytest.approx(1e-5)
    assert metrics.within_spec_tolerance
    assert metrics.absolute_residual[0] == pytest.approx(10.0)
    assert metrics.throughput_relative[0] == pytest.approx(10.0 / 90.0)


def test_the_mass_to_standard_volume_conversion_is_a_number_a_person_can_check() -> None:
    """200 m³ of pore volume at Sw = 0.3, weighed at the §3.1 surface densities.

    Water: 200 * 0.3 = 60 m³_sc, which is 60 000 kg at 1000 kg/m³.
    Oil:   200 * 0.7 = 140 m³_sc, which is 112 000 kg at 800 kg/m³.

    The extraction divides the native component mass by the reference density to reach the
    inventory, so this is that arithmetic done by hand in the other direction.
    """
    masses_kg = np.array([[60_000.0, 112_000.0], [50_000.0, 112_000.0]])
    rho_sc = np.array([1000.0, 800.0])
    inventory = masses_kg / rho_sc
    assert inventory[0].tolist() == [60.0, 140.0]
    # Ten cubic metres of water left over the step, and the source says so.
    metrics = component_balance(inventory, [[-10.0, 0.0]], components=("water", "oil"))
    assert metrics.absolute_residual == pytest.approx((0.0, 0.0))
    assert metrics.net_source_m3_sc == pytest.approx((-10.0, 0.0))


def test_an_inventory_without_its_initial_state_is_refused() -> None:
    with pytest.raises(ValueError, match="inventory must include initial state"):
        relative_balance_errors([[1.0]], [[0.0]])


def test_the_balance_floor_stops_an_absent_component_dividing_by_zero() -> None:
    per_step, cumulative = relative_balance_errors([[0.0], [0.0]], [[0.0]])
    assert per_step.tolist() == [[0.0]]
    assert cumulative.tolist() == [0.0]
    metrics = component_balance([[0.0], [0.0]], [[0.0]], components=("water",))
    assert metrics.floor_m3_sc == BALANCE_FLOOR_M3_SC


def test_a_nonfinite_inventory_is_refused() -> None:
    with pytest.raises(ValueError, match="inventory holds nonfinite values"):
        relative_balance_errors([[float("inf")], [1.0]], [[0.0]])


# --------------------------------------------------------------------------------------
# 7.3 / 7.4 reading the native extraction
# --------------------------------------------------------------------------------------

#: The synthetic extraction below runs two 15-day months, split into four 7.5-day accepted
#: substeps, over the two wells of `tests.forward_case`. PRO1 produces 10 m³/day of oil and
#: 5 of water, INJ1 injects 20 m³/day of water, and the inventory moves by exactly the
#: surface source each step — so every number in it is one a reader can recompute.
_SUBSTEP_DAYS = 7.5
_N_SUBSTEPS = 4
_OIL_PROD_M3_DAY = 10.0
_WATER_PROD_M3_DAY = 5.0
_WATER_INJ_M3_DAY = 20.0
_RHO_SC = (1000.0, 800.0)

#: What the wellbores take up per substep, m³_sc. It is what makes the reservoir-only
#: balance a DIFFERENT statement from the whole-model one rather than a copy of it.
_WELL_STORAGE_STEP = (1.0, -0.5)


def _extraction() -> dict[str, Any]:
    dt = _SUBSTEP_DAYS * DAY
    starts = [index * dt for index in range(_N_SUBSTEPS)]
    ends = [(index + 1) * dt for index in range(_N_SUBSTEPS)]
    intervals = [0, 0, 1, 1]

    def native(rate_m3_day: float) -> float:
        return rate_m3_day / DAY

    wells = {
        "PRO1": {
            "cells": [3, 7],
            "surface_water_m3_s": [-native(_WATER_PROD_M3_DAY)] * _N_SUBSTEPS,
            "surface_oil_m3_s": [-native(_OIL_PROD_M3_DAY)] * _N_SUBSTEPS,
            "surface_component_mass_kg_s": [
                [-native(_WATER_PROD_M3_DAY) * _RHO_SC[0]] * _N_SUBSTEPS,
                [-native(_OIL_PROD_M3_DAY) * _RHO_SC[1]] * _N_SUBSTEPS,
            ],
            "bhp_pa": [1.8e7] * _N_SUBSTEPS,
            "operating_target": ["lrat"] * _N_SUBSTEPS,
        },
        "INJ1": {
            "cells": [0, 4],
            "surface_water_m3_s": [native(_WATER_INJ_M3_DAY)] * _N_SUBSTEPS,
            "surface_oil_m3_s": [0.0] * _N_SUBSTEPS,
            "surface_component_mass_kg_s": [
                [native(_WATER_INJ_M3_DAY) * _RHO_SC[0]] * _N_SUBSTEPS,
                [0.0] * _N_SUBSTEPS,
            ],
            "bhp_pa": [3.0e7] * _N_SUBSTEPS,
            "operating_target": ["wrat"] * _N_SUBSTEPS,
        },
    }
    source = [
        [
            (_WATER_INJ_M3_DAY - _WATER_PROD_M3_DAY) * _SUBSTEP_DAYS,
            -_OIL_PROD_M3_DAY * _SUBSTEP_DAYS,
        ]
    ] * _N_SUBSTEPS
    inventory = [[1000.0, 2000.0]]
    for step in source:
        inventory.append([a + b for a, b in zip(inventory[-1], step, strict=True)])

    # The wellbores are filling, so the reservoir's own inventory and the connection flux
    # are NOT the full-system pair: they differ by exactly what the wells are storing. The
    # two published balances must therefore carry different numbers and both close.
    storage = [[_WELL_STORAGE_STEP[c] * step for c in range(2)] for step in range(len(inventory))]
    reservoir_inventory = [
        [total - held for total, held in zip(row, held_row, strict=True)]
        for row, held_row in zip(inventory, storage, strict=True)
    ]
    connection_source = [
        [value - held for value, held in zip(row, _WELL_STORAGE_STEP, strict=True)]
        for row in source
    ]

    connections = []
    for step in range(_N_SUBSTEPS):
        for well_id, sign in (("PRO1", 1.0), ("INJ1", -1.0)):
            for connection, cell in enumerate(wells[well_id]["cells"]):
                connections.append(
                    {
                        "well_id": well_id,
                        "connection_id": connection,
                        "cell_id": cell,
                        "step": step,
                        "water_mass_kg_s": sign * 1.0,
                        "oil_mass_kg_s": sign * 2.0,
                        "total_mass_kg_s": sign * 3.0,
                        "connection_open": True,
                        "actual_target": wells[well_id]["operating_target"][step],
                        "bhp_pa": wells[well_id]["bhp_pa"][step],
                    }
                )

    times = [0.0, 15.0 * DAY, 30.0 * DAY]
    return {
        "schema_version": "forward-extract-1",
        "status": "COMPLETE",
        "reason": None,
        "components": ["water", "oil"],
        "reference_densities_kg_m3": list(_RHO_SC),
        "chunk": {
            "horizon_start_s": 0.0,
            "horizon_end_s": 30.0 * DAY,
            "edges_s": [0.0, 15.0 * DAY, 30.0 * DAY],
            "start_s": starts,
            "end_s": ends,
            "dt_s": [dt] * _N_SUBSTEPS,
            "interval_index": intervals,
        },
        "wells": wells,
        "connections": connections,
        "inventory_m3_sc": inventory,
        "reservoir_inventory_m3_sc": reservoir_inventory,
        "net_surface_source_m3_sc": source,
        "net_connection_source_m3_sc": connection_source,
        "states": {
            "times_s": times,
            "pressure_pa": [[2.5e7 - 1e5 * index] * N_CELLS for index in range(len(times))],
            "sw": [[0.3 + 0.01 * index] * N_CELLS for index in range(len(times))],
            "so": [[0.7 - 0.01 * index] * N_CELLS for index in range(len(times))],
            "pore_volume_m3": [[25000.0] * N_CELLS for _ in times],
            "bw": [[0.99] * N_CELLS for _ in times],
            "bo": [[0.98] * N_CELLS for _ in times],
        },
        "solver": {"accepted_steps": 4, "cut_steps": 1, "nonlinear_iterations": 17},
        "control_evidence": {
            well_id: [
                {"honoured": True, "operating_target": well["operating_target"][step]}
                for step in range(_N_SUBSTEPS)
            ]
            for well_id, well in wells.items()
        },
        "control_infeasible_reason": None,
    }


def test_the_native_sign_is_split_into_non_negative_public_columns() -> None:
    rows = well_step_table(_extraction()).to_pylist()
    assert len(rows) == 2 * _N_SUBSTEPS
    injector = [row for row in rows if row["well_id"] == "INJ1"]
    producer = [row for row in rows if row["well_id"] == "PRO1"]
    # The injector's native water rate is positive and lands in the injection column only.
    assert all(row["water_inj_m3_s"] > 0.0 for row in injector)
    assert all(row["water_prod_m3_s"] == 0.0 for row in injector)
    # The producer's is negative and lands in the production column only.
    assert all(row["water_prod_m3_s"] > 0.0 for row in producer)
    assert all(row["water_inj_m3_s"] == 0.0 for row in producer)
    assert all(
        row[column] >= 0.0
        for row in rows
        for column in ("oil_prod_m3_s", "water_prod_m3_s", "water_inj_m3_s")
    )


def test_oil_being_injected_is_refused_rather_than_dropped() -> None:
    payload = _extraction()
    payload["wells"]["PRO1"]["surface_oil_m3_s"][1] = 1e-3
    with pytest.raises(ExtractionError, match="oil being injected"):
        well_step_table(payload)


def _shorten_last_step(chunk: dict[str, Any]) -> None:
    """One accepted substep lost from the end: `sum(dt)` no longer spans the chunk."""
    chunk["start_s"].pop()
    chunk["end_s"].pop()
    chunk["dt_s"].pop()
    chunk["interval_index"].pop()


def _open_a_gap(chunk: dict[str, Any]) -> None:
    """A substep that starts after the previous one ended, with a consistent `dt`."""
    chunk["start_s"][2] += 1000.0
    chunk["dt_s"][2] -= 1000.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["dt_s"].__setitem__(1, 0.0), "positive duration"),
        (lambda c: c["dt_s"].__setitem__(1, c["dt_s"][1] * 2), "reports dt_s"),
        (_shorten_last_step, r"sum\(dt\)"),
        (_open_a_gap, "without gaps"),
    ],
)
def test_the_accepted_substeps_must_tile_the_chunk(mutate: Any, message: str) -> None:
    """Positive `dt`, no gaps and `sum(dt)` equal to the chunk duration (plan 7.3)."""
    payload = _extraction()
    mutate(payload["chunk"])
    with pytest.raises(ExtractionError, match=message):
        well_step_table(payload)


def test_a_substep_that_belongs_to_no_recorded_step_is_refused() -> None:
    payload = _extraction()
    payload["connections"][0]["step"] = _N_SUBSTEPS
    with pytest.raises(ExtractionError, match="outside the 4 accepted substeps"):
        connection_step_table(payload)


def test_the_connection_integrals_keep_the_mask_and_summarise_the_pressure() -> None:
    payload = _extraction()
    # The producer's lower completion is shut for the second half of the horizon.
    for row in payload["connections"]:
        if row["well_id"] == "PRO1" and row["connection_id"] == 1 and row["step"] >= 2:
            row["connection_open"] = False
            row["water_mass_kg_s"] = 0.0
            row["oil_mass_kg_s"] = 0.0
            row["total_mass_kg_s"] = 0.0
    rows = integrate_connections(
        connection_step_table(payload), (0.0, 15.0 * DAY, 30.0 * DAY)
    ).to_pylist()
    shut = [
        row
        for row in rows
        if row["well_id"] == "PRO1" and row["connection_id"] == 1 and row["month_index"] == 1
    ]
    assert len(shut) == 1
    assert shut[0]["open_s"] == 0.0
    assert shut[0]["total_mass_kg"] == 0.0
    # The open one in the same month carries the real integral, so the zero is the mask and
    # not an empty table.
    open_row = next(
        row
        for row in rows
        if row["well_id"] == "PRO1" and row["connection_id"] == 0 and row["month_index"] == 1
    )
    assert open_row["open_s"] == pytest.approx(15.0 * DAY)
    assert open_row["total_mass_kg"] == pytest.approx(3.0 * 15.0 * DAY)
    assert open_row["bhp_mean_pa"] == pytest.approx(1.8e7)
    assert open_row["bhp_min_pa"] == open_row["bhp_max_pa"] == pytest.approx(1.8e7)


def test_control_infeasibility_needs_the_evidence_and_the_reason_to_agree() -> None:
    payload = _extraction()
    assert control_infeasibility(payload) is None

    payload["control_evidence"]["PRO1"][2]["honoured"] = False
    with pytest.raises(ExtractionError, match="gives no CONTROL_INFEASIBLE reason"):
        control_infeasibility(payload)

    payload["control_infeasible_reason"] = "well PRO1 step 3: requested lrat=40.0 ..."
    assert control_infeasibility(payload) == payload["control_infeasible_reason"]

    payload["control_evidence"]["PRO1"][2]["honoured"] = True
    with pytest.raises(ExtractionError, match="every control in the evidence was honoured"):
        control_infeasibility(payload)


# --------------------------------------------------------------------------------------
# 7.7 publishing a result, and reading it back
# --------------------------------------------------------------------------------------


def _project(tmp_project: Path) -> tuple[ProjectPaths, CaseBundle, JobDescriptor]:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    refs = write_case_arrays(paths)
    case = build_case(refs)
    case_path = paths.artifacts / "case.json"
    case_path.write_text(json.dumps(case.model_dump(mode="json")), encoding="utf-8")
    solver_path = paths.artifacts / "solver.json"
    solver_path.write_text(json.dumps({"max_timestep_days": 30}), encoding="utf-8")
    job = JobDescriptor(
        job_id="job-e01-outputs",
        case_path=paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash=case.model_hash,
        solver_config_path=paths.relative(solver_path),
        solver_config_sha256=sha256_file(solver_path),
        output_request=OutputRequest(
            state_times_s=(0.0, 15.0 * DAY, 30.0 * DAY), keep_native_restart=False
        ),
        seed=20260913,
        result_dir="artifacts/results/job-e01-outputs",
        attempt=1,
    )
    return paths, case, job


def _cost() -> CostRecord:
    return CostRecord(
        wall_s=1.0,
        cpu_s=1.0,
        peak_rss_bytes=1,
        output_bytes=0,
        accepted_steps=4,
        cut_steps=1,
        nonlinear_iterations=17,
        retry_count=0,
        measurement_method="fixture",
    )


def _publish(tmp_project: Path, payload: dict[str, Any]) -> tuple[ProjectPaths, Path]:
    paths, case, job = _project(tmp_project)
    result = publish_forward_result(
        job, case, payload, paths, cost=_cost(), solver_metadata={"protocol_version": "worker-1"}
    )
    assert result.status == "COMPLETE", result.reason
    record = paths.resolve(job.result_dir) / RESULT_FILENAME
    write_forward_result(result, record)
    return paths, record


def test_a_published_result_is_complete_and_loads_back(tmp_project: Path) -> None:
    paths, record = _publish(tmp_project, _extraction())
    result = load_forward_result(record, paths)

    assert result.status == "COMPLETE"
    assert result.reason is None
    assert result.times_s == (0.0, 15.0 * DAY, 30.0 * DAY)
    assert result.completed_time_s == 30.0 * DAY
    assert sorted(result.states) == ["bo", "bw", "pore_volume_m3", "pressure_pa", "so", "sw"]
    for ref in result.states.values():
        assert ref.axis_order == ("time", "cell")
        assert ref.shape == (3, N_CELLS)
        assert ref.path.endswith(STATES_FILENAME)
    assert result.monthly_path is not None and result.monthly_path.endswith(MONTHLY_FILENAME)
    assert result.connections_path is not None
    assert result.balances_path is not None

    # The monthly table is the integral, not the last rate: 15 days at 10 m³/day of oil.
    monthly = pq.read_table(paths.resolve(result.monthly_path)).to_pylist()
    producer = [row for row in monthly if row["well_id"] == "PRO1"]
    assert [row["oil_prod_m3_sc"] for row in producer] == pytest.approx([150.0, 150.0])
    assert [row["water_prod_m3_sc"] for row in producer] == pytest.approx([75.0, 75.0])
    assert [row["fw"] for row in producer] == pytest.approx([1 / 3, 1 / 3])
    injector = [row for row in monthly if row["well_id"] == "INJ1"]
    assert [row["water_inj_m3_sc"] for row in injector] == pytest.approx([300.0, 300.0])
    assert [row["fw"] for row in injector] == [None, None]

    # BOTH balances are published, each saying which system it closed over and against which
    # source term — plan 12.9 has the stage validator read the published results rather than
    # re-run the suite, so a balance that lives only in a test cannot be cited.
    balances = pq.read_table(paths.resolve(result.balances_path)).to_pylist()
    assert [(row["balance"], row["component"]) for row in balances] == [
        ("full_system_surface", "water"),
        ("full_system_surface", "oil"),
        ("reservoir_connections", "water"),
        ("reservoir_connections", "oil"),
    ]
    assert {row["source_term"] for row in balances} == {
        "surface component flux (q_t * mix)",
        "reservoir-well connection flux (geometry/state/PVT)",
    }
    assert {row["system"] for row in balances} == {"reservoir + wellbores", "reservoir"}
    assert all(row["absolute_residual_m3_sc"] == pytest.approx(0.0) for row in balances)
    # The unprefixed metadata keys stay the whole-model statement, and the other one is
    # beside them under its own name rather than replacing it.
    assert result.solver_metadata["balance_headline"] == "full_system_surface"
    assert json.loads(result.solver_metadata["balance_systems"]) == [
        "full_system_surface",
        "reservoir_connections",
    ]
    assert result.solver_metadata["balance_within_spec_tolerance"] == "true"
    assert result.solver_metadata["balance.reservoir_connections_within_spec_tolerance"] == "true"
    assert json.loads(result.solver_metadata["balance_absolute_residual_m3_sc"]) == {
        "water": 0.0,
        "oil": 0.0,
    }
    assert json.loads(
        result.solver_metadata["balance.reservoir_connections_absolute_residual_m3_sc"]
    ) == {"water": 0.0, "oil": 0.0}
    # SPEC 17.1: HDF5 states are not a restart, and this result does not claim one.
    assert result.restart is None


def test_the_two_published_balances_are_two_different_statements(tmp_project: Path) -> None:
    """They differ by what the wellbores are storing, and both are on the record.

    The whole model against the surface flux, and the reservoir alone against the native
    connection flux. A single number covering both would hide the well storage, which is a
    real quantity; averaging them would report neither.
    """
    payload = _extraction()
    paths, record = _publish(tmp_project, payload)
    result = load_forward_result(record, paths)
    rows = pq.read_table(paths.resolve(str(result.balances_path))).to_pylist()
    by_balance = {(row["balance"], row["component"]): row for row in rows}
    full = by_balance[("full_system_surface", "water")]
    reservoir = by_balance[("reservoir_connections", "water")]

    # Both close exactly on this fixture...
    assert full["absolute_residual_m3_sc"] == pytest.approx(0.0)
    assert reservoir["absolute_residual_m3_sc"] == pytest.approx(0.0)
    # ...and they are nevertheless different numbers, by the storage the wells took up.
    held = _WELL_STORAGE_STEP[0] * _N_SUBSTEPS
    assert full["net_source_m3_sc"] - reservoir["net_source_m3_sc"] == pytest.approx(held)
    assert full["final_inventory_m3_sc"] - reservoir["final_inventory_m3_sc"] == pytest.approx(held)
    assert full["initial_inventory_m3_sc"] == pytest.approx(reservoir["initial_inventory_m3_sc"])
    assert held != 0.0


def test_a_broken_connection_balance_is_published_rather_than_hidden(
    tmp_project: Path,
) -> None:
    """The reservoir balance is a real check, and the record carries the failure.

    Losing one substep's connection flux leaves the whole-model balance untouched — it uses
    a different source term — so a result that published only the headline number would look
    perfect. Plan 12.9 has the stage validator read the published tables, so both have to be
    there for it to see this at all.
    """
    payload = _extraction()
    payload["net_connection_source_m3_sc"][2] = [0.0, 0.0]
    paths, record = _publish(tmp_project, payload)
    result = load_forward_result(record, paths)
    rows = pq.read_table(paths.resolve(str(result.balances_path))).to_pylist()
    full = [row for row in rows if row["balance"] == "full_system_surface"]
    reservoir = [row for row in rows if row["balance"] == "reservoir_connections"]

    assert all(row["absolute_residual_m3_sc"] == pytest.approx(0.0) for row in full)
    assert any(row["absolute_residual_m3_sc"] > 1.0 for row in reservoir)
    assert result.solver_metadata["balance_within_spec_tolerance"] == "true"
    assert result.solver_metadata["balance.reservoir_connections_within_spec_tolerance"] == "false"


def test_the_states_file_carries_its_own_axes_and_no_model_geometry(tmp_project: Path) -> None:
    import h5py

    paths, record = _publish(tmp_project, _extraction())
    result = load_forward_result(record, paths)
    with h5py.File(paths.resolve(result.solver_metadata["states_path"]), "r") as handle:
        assert sorted(handle) == [
            "bo",
            "bw",
            "cell_id",
            "pore_volume_m3",
            "pressure_pa",
            "so",
            "sw",
            "time_s",
        ]
        assert handle["time_s"][()].tolist() == [0.0, 15.0 * DAY, 30.0 * DAY]
        assert handle["cell_id"][()].tolist() == list(range(N_CELLS))
        assert handle["pressure_pa"].shape == (3, N_CELLS)


def test_a_corrupted_state_array_is_refused_on_read(tmp_project: Path) -> None:
    import h5py

    paths, record = _publish(tmp_project, _extraction())
    states = paths.resolve(
        json.loads(record.read_text(encoding="utf-8"))["solver_metadata"]["states_path"]
    )
    with h5py.File(states, "r+") as handle:
        handle["pressure_pa"][0, 0] = 1.0
    with pytest.raises(ForwardResultIntegrityError, match="hashes to"):
        load_forward_result(record, paths)


def test_a_corrupted_parquet_table_is_refused_on_read(tmp_project: Path) -> None:
    paths, record = _publish(tmp_project, _extraction())
    payload = json.loads(record.read_text(encoding="utf-8"))
    paths.resolve(payload["monthly_path"]).write_bytes(b"not a parquet file")
    with pytest.raises(ForwardResultIntegrityError, match=MONTHLY_FILENAME):
        load_forward_result(record, paths)


def test_a_record_whose_time_axis_drifted_from_its_states_is_refused(tmp_project: Path) -> None:
    paths, record = _publish(tmp_project, _extraction())
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["times_s"] = [0.0, 14.0 * DAY, 30.0 * DAY]
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForwardResultIntegrityError, match="not the time axis stored"):
        load_forward_result(record, paths)


def test_a_missing_output_digest_is_refused_rather_than_assumed(tmp_project: Path) -> None:
    paths, record = _publish(tmp_project, _extraction())
    payload = json.loads(record.read_text(encoding="utf-8"))
    del payload["solver_metadata"][f"{BALANCES_FILENAME}.sha256"]
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForwardResultIntegrityError, match="an absent hash"):
        load_forward_result(record, paths)


def test_a_published_state_outside_its_physical_range_is_refused(tmp_project: Path) -> None:
    payload = _extraction()
    payload["states"]["sw"][1] = [1.4] * N_CELLS
    paths, case, job = _project(tmp_project)
    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    record = paths.resolve(job.result_dir) / RESULT_FILENAME
    write_forward_result(result, record)
    with pytest.raises(ForwardResultIntegrityError, match=r"states\['sw'\] lies outside"):
        load_forward_result(record, paths)


def test_saturations_that_do_not_sum_to_one_are_refused_not_renormalised(
    tmp_project: Path,
) -> None:
    payload = _extraction()
    payload["states"]["so"][2] = [0.5] * N_CELLS
    paths, case, job = _project(tmp_project)
    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    record = paths.resolve(job.result_dir) / RESULT_FILENAME
    write_forward_result(result, record)
    with pytest.raises(ForwardResultIntegrityError, match="never renormalised"):
        load_forward_result(record, paths)


# --------------------------------------------------------------------------------------
# the classified refusals on the path that produces results (plan 3.3)
# --------------------------------------------------------------------------------------


def test_a_case_whose_schedule_cannot_compile_is_invalid_input(tmp_project: Path) -> None:
    """`compile_schedule`'s refusals are a classified status, not a bare exception."""
    paths, case, job = _project(tmp_project)
    # A hole in PRO1's schedule: its only segment stops on day 10, and the interval from
    # there to the first report edge has no control for it. `CaseBundle` accepts this — it
    # checks overlaps and the horizon, not coverage — so the schedule compiler is the only
    # thing standing between it and a well that silently stopped producing.
    truncated = case.controls[1].model_copy(update={"end_s": 10.0 * DAY})
    broken = case.model_copy(update={"controls": (case.controls[0], truncated)})
    with pytest.raises(ValueError, match="has no control on interval"):
        compile_schedule(broken.report_edges_s, broken.controls)

    result = publish_forward_result(
        job, broken, _extraction(), paths, cost=_cost(), solver_metadata={}
    )
    assert result.status == "INVALID_INPUT"
    assert result.reason is not None
    assert "schedule cannot be compiled" in result.reason
    assert "PRO1" in result.reason
    assert (result.monthly_path, result.connections_path, result.balances_path) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("truncate_at_the_end", [True, False])
def test_an_extraction_that_stops_short_of_the_horizon_is_refused(
    tmp_project: Path, truncate_at_the_end: bool
) -> None:
    """A month nobody simulated must not be published as a month in which nothing flowed.

    `_chunk` proves the substeps tile THEIR OWN chunk and cannot know what that chunk should
    have been; `integrate_monthly` emits a row for every month regardless; and
    `Schedule.reject_flow_without_uptime` refuses flow without uptime, never uptime without
    flow. So the coverage check is the only thing between a truncated forward and a
    `COMPLETE` result whose second month reads as a physical zero (SPEC 18.4).
    """
    paths, case, job = _project(tmp_project)
    payload = _extraction()
    chunk = payload["chunk"]
    if truncate_at_the_end:
        # Only the first month of a two-month case.
        keep = 2
        for key in ("start_s", "end_s", "dt_s", "interval_index"):
            chunk[key] = chunk[key][:keep]
        chunk["horizon_end_s"] = chunk["end_s"][-1]
        for well in payload["wells"].values():
            well["surface_water_m3_s"] = well["surface_water_m3_s"][:keep]
            well["surface_oil_m3_s"] = well["surface_oil_m3_s"][:keep]
        expected = "it ends at"
    else:
        # A chunk that begins a month late: the second half of the same case.
        drop = 2
        for key in ("start_s", "end_s", "dt_s", "interval_index"):
            chunk[key] = chunk[key][drop:]
        chunk["horizon_start_s"] = chunk["start_s"][0]
        for well in payload["wells"].values():
            well["surface_water_m3_s"] = well["surface_water_m3_s"][drop:]
            well["surface_oil_m3_s"] = well["surface_oil_m3_s"][drop:]
        expected = "it starts at"

    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    assert result.status == "INCOMPLETE_BUDGET"
    assert result.reason is not None
    assert "does not cover the case's reported horizon" in result.reason
    assert expected in result.reason
    assert (result.monthly_path, result.connections_path, result.balances_path) == (
        None,
        None,
        None,
    )
    # And nothing was written for it: a truncated forward leaves no outputs behind.
    assert not paths.resolve(job.result_dir).exists()


def test_a_full_horizon_publishes_so_the_coverage_check_is_not_vacuous(
    tmp_project: Path,
) -> None:
    paths, case, job = _project(tmp_project)
    payload = _extraction()
    assert payload["chunk"]["horizon_start_s"] == case.report_edges_s[0]
    assert payload["chunk"]["horizon_end_s"] == case.report_edges_s[-1]
    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    assert result.status == "COMPLETE", result.reason


def test_an_unhonoured_control_becomes_a_control_infeasible_result(tmp_project: Path) -> None:
    paths, case, job = _project(tmp_project)
    payload = _extraction()
    payload["control_evidence"]["PRO1"][1]["honoured"] = False
    payload["control_infeasible_reason"] = (
        "well PRO1 step 2: requested lrat=40.0 but operated on bhp at 3.1 m3_sc/day"
    )
    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    assert result.status == "CONTROL_INFEASIBLE"
    assert result.reason is not None
    assert "operated on bhp" in result.reason
    assert result.monthly_path is None
    # No outputs were written for it: the result directory stays empty.
    assert (
        not list(paths.resolve(job.result_dir).glob("*"))
        or not (paths.resolve(job.result_dir) / STATES_FILENAME).exists()
    )


def test_a_flow_over_a_month_the_schedule_left_shut_is_refused(tmp_project: Path) -> None:
    """SPEC 18.4 the other way round: E01 is given its uptime and never reconstructs it."""
    paths, case, job = _project(tmp_project)
    # PRO1 is shut for the whole of the second month, but the extraction still reports it
    # flowing there. The schedule and the flow are then not the same case.
    shut = case.controls[1].model_copy(
        update={
            "start_s": 15.0 * DAY,
            "role": "shut",
            "target": "disabled",
            "value": 0.0,
            "bhp_limit_pa": None,
        }
    )
    producing = case.controls[1].model_copy(update={"end_s": 15.0 * DAY})
    contradicted = case.model_copy(update={"controls": (case.controls[0], producing, shut)})
    with pytest.raises(ValueError, match="does not reconstruct an unknown uptime"):
        publish_forward_result(
            job, contradicted, _extraction(), paths, cost=_cost(), solver_metadata={}
        )


def test_a_well_the_schedule_does_not_control_is_refused(tmp_project: Path) -> None:
    paths, case, job = _project(tmp_project)
    payload = _extraction()
    payload["wells"]["GHOST"] = copy.deepcopy(payload["wells"]["PRO1"])
    with pytest.raises(ValueError, match="are not the same case"):
        publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})


def test_an_extraction_in_an_unknown_shape_is_a_protocol_failure(tmp_project: Path) -> None:
    paths, case, job = _project(tmp_project)
    payload = _extraction()
    payload["schema_version"] = "forward-extract-0"
    result = publish_forward_result(job, case, payload, paths, cost=_cost(), solver_metadata={})
    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None
    assert "forward-extract-0" in result.reason


def test_the_accepted_step_diagnostics_are_named_and_therefore_checked(
    tmp_project: Path,
) -> None:
    paths, record = _publish(tmp_project, _extraction())
    result = load_forward_result(record, paths)
    steps_path = result.solver_metadata["accepted_steps.parquet.path"]
    rows = pq.read_table(paths.resolve(steps_path)).to_pylist()
    assert [row["step_index"] for row in rows] == list(range(_N_SUBSTEPS))
    assert [row["month_index"] for row in rows] == [0, 0, 1, 1]
    assert all(row["dt_s"] == pytest.approx(_SUBSTEP_DAYS * DAY) for row in rows)
    # Named in the record, so a corrupted copy of it cannot pass unnoticed.
    paths.resolve(steps_path).write_bytes(b"not a parquet file")
    with pytest.raises(ForwardResultIntegrityError, match="hashes to"):
        load_forward_result(record, paths)


def test_a_native_refusal_keeps_its_own_status_and_reason(tmp_project: Path) -> None:
    paths, case, job = _project(tmp_project)
    result = publish_forward_result(
        job,
        case,
        {"status": "NUMERICAL_FAILURE", "reason": "the solver reported 1 report step for 2"},
        paths,
        cost=_cost(),
        solver_metadata={},
    )
    assert result.status == "NUMERICAL_FAILURE"
    assert result.reason == "the solver reported 1 report step for 2"


def test_a_status_the_adapter_may_not_reach_is_a_protocol_failure(tmp_project: Path) -> None:
    paths, case, job = _project(tmp_project)
    result = publish_forward_result(
        job,
        case,
        {"status": "RESOURCE_FAILURE", "reason": "out of memory"},
        paths,
        cost=_cost(),
        solver_metadata={},
    )
    # A resource verdict is the transport's to make from a measurement, never the
    # adapter's to claim (SPEC 18.4).
    assert result.status == "PROTOCOL_FAILURE"
