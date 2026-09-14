"""E01.11: the first P1 parent set, run on the pinned solver and recorded whatever it did.

Five worlds — seeds 41, 42 and 43 as `base`, 44 as `low_vertical`, 45 as `high_contrast` —
each 16x16x2 cells over three real calendar years, through the PRODUCTION forward path:
`simulate` on a persistent Julia worker, under a real `BudgetLedger` on the `P1_LOOP`
profile. That is the difference between this file and the verification suites of Tasks 9 and
10: those run diagnostics through a profile-bounded test launcher and are deliberately not
charged to a ledger, while these are worker-submitted jobs that spend a session's budget and
are booked for it.

Each world is run twice and for two different reasons.

**The equilibrium preflight**, first: one month with every well shut at the surface AND
every perforation masked, over a closed boundary — a system with no source at all. The
oil-connected hydrostatic column `p1.py` writes satisfies the CONTINUOUS balance by
construction; what this measures is whether it satisfies the solver's own discrete two-point
one, and it is scored against `hydrostatic_pressure_relative_drift_max`,
`hydrostatic_saturation_drift_max` and `closed_connection_mass_kg_s_max` from the frozen
tolerance block, exactly as Task 9's hydrostatic fixture is.

**The world itself**, then: thirty-six months of the modulated policy with P2's lower
completion shut for months 19 to 24. It is scored on the component balance (SPEC §23.1), on
the finiteness and the physical bounds of every published state, on the bottom-hole pressures
against the limits the case recorded, and on the completion event having actually happened.

Nothing here tunes anything to make a world pass. A world whose forward does not come back
`COMPLETE`, or whose metrics miss a threshold, keeps its row in the suite manifest with the
number it missed by; `fully_accepted` on that manifest is what says whether the first P1
suite was accepted, and it is written before any assertion is made.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P1_LOOP_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import read_array
from so_recon.simulator.contracts import (
    SECONDS_PER_DAY,
    CaseBundle,
    ForwardResult,
    OutputRequest,
)
from so_recon.simulator.forward import SolverConfig, check_requested_outputs, simulate
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.results import RESULT_FILENAME, load_forward_result, write_forward_result
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.p1 import (
    COMPLETION_REOPEN_EDGE,
    COMPLETION_SHUT_EDGE,
    INJECTOR_IDS,
    INJECTOR_MAX_BHP_PA,
    P1_PARENTS,
    PRODUCER_IDS,
    PRODUCER_MIN_BHP_PA,
    P1Design,
    RenderedWorld,
    closed_preflight_world,
    output_request,
    preflight_output_request,
    render_p1,
)
from so_recon.synthetic.world_io import (
    SUITE_MANIFEST_FILENAME,
    build_p1_case,
    world_locations,
    world_row,
    write_suite_manifest,
    write_world,
)
from so_recon.validation.physics import DEFAULT_TOLERANCES_RELPATH, load_tolerances

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES_PATH = ROOT / DEFAULT_TOLERANCES_RELPATH

#: The same driving configuration every E01 forward uses. Five days is well below a calendar
#: month, so a month is an integral over several accepted steps rather than one backward
#: Euler step (SPEC §9.3).
SOLVER = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)

#: How far a bottom-hole pressure may sit past the limit the case recorded before this suite
#: calls it a violation. It is float64 slack on a pressure of order 1e7 Pa, not a tolerance
#: anybody may widen: a well that really switched to its limit reports the limit, and one
#: that broke through it is a finding.
BHP_SLACK_PA = 1.0


def _skip_unless_julia_is_installed() -> Path:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        return find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")


def _bounded_probe(session_dir: Path) -> Callable[[], ResourceSnapshot]:
    """A real measurement with only the MACHINE's memory pinned, never this run's own.

    `total_bytes`, `available_bytes` and `swap_used_bytes` describe the host, and pinning
    them is what keeps this suite from passing or failing on how much RAM some other
    application happened to be holding — the same choice every E01 integration test makes.

    `process_rss_bytes` is deliberately NOT pinned, and that is the difference from the
    earlier suites: it is what `CostRecord.peak_rss_bytes` is the maximum of, so pinning it
    would turn the memory this task is asked to REPORT into a constant the test wrote down
    itself. With the host figures above, the effective hard cap is the profile's own 16 GiB,
    which a 512-cell model is three orders of magnitude below; a run that really approached
    it would be a resource failure worth seeing.
    """

    def probe() -> ResourceSnapshot:
        return probe_resources(None, session_dir).model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "swap_used_bytes": 0,
            }
        )

    return probe


def _states(result: ForwardResult, paths: ProjectPaths) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(read_array(ref, paths), dtype=np.float64)
        for name, ref in result.states.items()
    }


def _rows(path: str | None, paths: ProjectPaths) -> list[dict[str, Any]]:
    assert path is not None
    out: list[dict[str, Any]] = pq.read_table(paths.resolve(path)).to_pylist()
    return out


def _state_metrics(states: dict[str, NDArray[np.float64]]) -> dict[str, float]:
    """Finiteness, the saturation identity and the physical bounds of every published state."""
    for name, values in states.items():
        assert np.isfinite(values).all(), f"{name} holds nonfinite values"
    sw, so, pressure = states["sw"], states["so"], states["pressure_pa"]
    return {
        "saturation_sum_abs": float(np.abs(sw + so - 1.0).max()),
        "saturation_bound_violation": max(
            0.0, float(-sw.min()), float(sw.max() - 1.0), float(-so.min()), float(so.max() - 1.0)
        ),
        "pressure_min_pa": float(pressure.min()),
        "pressure_max_pa": float(pressure.max()),
        "sw_min": float(sw.min()),
        "sw_max": float(sw.max()),
    }


def _balance_metrics(result: ForwardResult, paths: ProjectPaths) -> dict[str, float]:
    """The worst of BOTH published balances, over both components (plan §7.6)."""
    rows = _rows(result.balances_path, paths)
    assert len({row["balance"] for row in rows}) == 2, "a forward publishes both balances"
    return {
        "balance_cumulative_relative": max(float(r["cumulative_relative"]) for r in rows),
        "balance_step_median_relative": max(float(r["median_step_relative"]) for r in rows),
        "balance_max_step_relative": max(float(r["max_step_relative"]) for r in rows),
        "balance_absolute_residual_m3_sc": max(float(r["absolute_residual_m3_sc"]) for r in rows),
    }


def _connection_mass_rate(result: ForwardResult, paths: ProjectPaths) -> float:
    """The largest mean connection mass rate any perforation carried, kg/s."""
    worst = 0.0
    for row in _rows(result.connections_path, paths):
        duration = float(row["end_s"]) - float(row["start_s"])
        assert duration > 0.0
        worst = max(worst, abs(float(row["total_mass_kg"])) / duration)
    return worst


def _bhp_metrics(result: ForwardResult, paths: ProjectPaths) -> dict[str, float]:
    """How close every well came to the limit its own control segment recorded."""
    rows = [row for row in _rows(result.connections_path, paths) if row["bhp_min_pa"] is not None]
    producers = [row for row in rows if str(row["well_id"]) in PRODUCER_IDS]
    injectors = [row for row in rows if str(row["well_id"]) in INJECTOR_IDS]
    assert producers and injectors
    return {
        "producer_bhp_min_pa": min(float(row["bhp_min_pa"]) for row in producers),
        "producer_bhp_max_pa": max(float(row["bhp_max_pa"]) for row in producers),
        "injector_bhp_min_pa": min(float(row["bhp_min_pa"]) for row in injectors),
        "injector_bhp_max_pa": max(float(row["bhp_max_pa"]) for row in injectors),
    }


def _rate_control_relative(
    world: RenderedWorld, result: ForwardResult, paths: ProjectPaths
) -> float:
    """How far any well's monthly volume fell from the rate its control asked for.

    The target is the segment's own day rate times the length of that calendar month, so a
    well that held its rate matches it to the solver's own zero and one that switched to a
    pressure limit does not. It is REPORTED whatever it is: `CONTROL_INFEASIBLE` is the
    verdict on a control that was not honoured, and this is the size of the miss.
    """
    edges = world.case_fields["report_edges_s"]
    targets: dict[tuple[str, int], float] = {}
    for segment in world.case_fields["controls"]:
        for month in range(len(edges) - 1):
            overlap = min(segment.end_s, edges[month + 1]) - max(segment.start_s, edges[month])
            if overlap > 0.0:
                targets[(segment.well_id, month)] = (
                    targets.get((segment.well_id, month), 0.0)
                    + segment.value * overlap / SECONDS_PER_DAY
                )
    worst = 0.0
    for row in _rows(result.monthly_path, paths):
        key = (str(row["well_id"]), int(row["month_index"]))
        target = targets.get(key, 0.0)
        actual = (
            float(row["water_inj_m3_sc"])
            if key[0] in INJECTOR_IDS
            else float(row["liquid_prod_m3_sc"])
        )
        worst = max(worst, abs(actual - target) / max(abs(target), 1e-6))
    return worst


def _completion_event_months(
    world: RenderedWorld, result: ForwardResult, paths: ProjectPaths
) -> dict[str, list[int]]:
    """Which months P2's LOWER perforation was open for, as the published table saw it."""
    wells = {well.well_id: well for well in world.case_fields["wells"]}
    lower_cell = wells["P2"].cells[-1]
    rows = [
        row
        for row in _rows(result.connections_path, paths)
        if str(row["well_id"]) == "P2" and int(row["cell_id"]) == lower_cell
    ]
    assert rows, "the published connections name no lower perforation for P2"
    return {
        "open": sorted(int(r["month_index"]) for r in rows if float(r["open_s"]) > 0.0),
        "shut": sorted(int(r["month_index"]) for r in rows if float(r["open_s"]) == 0.0),
    }


def _run(
    world: RenderedWorld,
    paths: ProjectPaths,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    request: OutputRequest,
) -> tuple[CaseBundle, ForwardResult, float]:
    """One case through the production path, re-read through every check a record owes."""
    ctx = RunContext.start(command="forward", argv=[], cfg=None, paths=paths)
    case = build_p1_case(world, paths, ctx)
    started = time.monotonic()
    result = simulate(case, request, worker=worker, ctx=ctx, ledger=ledger, solver_config=SOLVER)
    elapsed = time.monotonic() - started
    check_requested_outputs(result, request)
    if result.status == "COMPLETE":
        assert result.monthly_path is not None
        record_path = paths.resolve(result.monthly_path).parent / RESULT_FILENAME
        write_forward_result(result, record_path)
        result = load_forward_result(record_path, paths)
    return case, result, elapsed


@pytest.mark.julia
def test_the_first_p1_parent_set_runs_and_every_outcome_is_recorded(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    tolerances = load_tolerances(TOLERANCES_PATH)
    session = paths.artifacts / "p1-session"
    probe = _bounded_probe(session)
    ledger = BudgetLedger.start(
        profile=P1_LOOP_PROFILE,
        path=paths.artifacts / "ledger-p1.json",
        session_id="session-p1-first-parents",
        probe=probe,
    )

    rows: list[dict[str, Any]] = []
    with PersistentJuliaWorker(
        julia_exe, ROOT / "julia", session, P1_LOOP_PROFILE, paths=paths, probe=probe
    ) as worker:
        for seed, family in P1_PARENTS:
            world = render_p1(seed, P1Design(family=family))

            # ---- the equilibrium preflight ------------------------------------------
            preflight_case, preflight, preflight_s = _run(
                closed_preflight_world(world),
                paths,
                worker,
                ledger,
                preflight_output_request(world.design),
            )
            equilibrium: dict[str, Any] = {
                "status": preflight.status,
                "reason": preflight.reason,
                "wall_s": preflight_s,
                "case_id": preflight_case.case_id,
            }
            if preflight.status == "COMPLETE":
                states = _states(preflight, paths)
                pressure, sw, so = states["pressure_pa"], states["sw"], states["so"]
                equilibrium |= {
                    "pressure_relative_drift": float(
                        (np.abs(pressure - pressure[0]) / np.abs(pressure[0])).max()
                    ),
                    "state_saturation_drift": max(
                        float(np.abs(sw - sw[0]).max()), float(np.abs(so - so[0]).max())
                    ),
                    "connection_mass_kg_s": _connection_mass_rate(preflight, paths),
                    **_balance_metrics(preflight, paths),
                }

            # ---- the world itself -----------------------------------------------------
            request = output_request(world.design)
            _case, result, wall_s = _run(world, paths, worker, ledger, request)
            gates: dict[str, Any] = {
                "status": result.status,
                "reason": result.reason,
                "equilibrium_preflight": equilibrium,
                "measured_wall_s": wall_s,
            }
            if result.status == "COMPLETE":
                gates |= _balance_metrics(result, paths)
                gates |= _state_metrics(_states(result, paths))
                gates |= _bhp_metrics(result, paths)
                gates["rate_control_relative"] = _rate_control_relative(world, result, paths)
                gates["p2_lower_completion"] = _completion_event_months(world, result, paths)
                assert result.restart is not None
                gates["restart"] = {
                    "completed_report_step": result.restart.completed_report_step,
                    "completed_time_s": result.restart.completed_time_s,
                    "native_format": result.restart.native_format,
                }

            manifest_ref = write_world(
                world,
                result,
                paths,
                RunContext.start(command="forward", argv=[], cfg=None, paths=paths),
            )
            manifest = json.loads(paths.resolve(manifest_ref.path).read_text(encoding="utf-8"))
            rows.append(
                world_row(world, result, manifest, manifest_path=manifest_ref.path, gates=gates)
            )

    # The record is written BEFORE anything is asserted: a world that failed keeps its row,
    # and `fully_accepted` is what says whether the first P1 suite was accepted (plan 11.9).
    suite_ref = write_suite_manifest(
        rows,
        paths.reports / SUITE_MANIFEST_FILENAME,
        paths,
        RunContext.start(command="forward", argv=[], cfg=None, paths=paths),
    )
    suite = json.loads(paths.resolve(suite_ref.path).read_text(encoding="utf-8"))
    assert len(suite["rows"]) == 5
    assert [row["seed"] for row in suite["rows"]] == [41, 42, 43, 44, 45]
    assert [row["family"] for row in suite["rows"]] == [
        "base",
        "base",
        "base",
        "low_vertical",
        "high_contrast",
    ]
    assert len({row["model_hash"] for row in suite["rows"]}) == 5

    # ---- the preflight: the initial state does not drift ---------------------------------
    for row in suite["rows"]:
        equilibrium = row["gates"]["equilibrium_preflight"]
        assert equilibrium["status"] == "COMPLETE", (row["parent_world_id"], equilibrium["reason"])
        assert (
            equilibrium["pressure_relative_drift"]
            <= tolerances["hydrostatic_pressure_relative_drift_max"]
        ), row["parent_world_id"]
        assert (
            equilibrium["state_saturation_drift"] <= tolerances["hydrostatic_saturation_drift_max"]
        ), row["parent_world_id"]
        # A well that is shut at the surface AND masked at every perforation moves nothing.
        assert (
            equilibrium["connection_mass_kg_s"] <= tolerances["closed_connection_mass_kg_s_max"]
        ), row["parent_world_id"]
        assert (
            equilibrium["balance_cumulative_relative"]
            <= tolerances["balance_cumulative_relative_max"]
        )

    # ---- the worlds --------------------------------------------------------------------
    for row in suite["rows"]:
        name = row["parent_world_id"]
        gates = row["gates"]
        assert row["status"] == "COMPLETE", f"{name}: {row['reason']}"
        assert row["accepted"] is True, name
        # SPEC 23.1, both components and both published balances.
        assert (
            gates["balance_cumulative_relative"] <= tolerances["balance_cumulative_relative_max"]
        ), f"{name}: {gates['balance_cumulative_relative']:.3e}"
        assert (
            gates["balance_step_median_relative"] <= tolerances["balance_step_median_relative_max"]
        ), f"{name}: {gates['balance_step_median_relative']:.3e}"
        # Every published state is finite, sums to one and stays inside [0, 1].
        assert gates["saturation_sum_abs"] <= tolerances["saturation_sum_abs_max"], name
        assert gates["saturation_bound_violation"] <= tolerances["saturation_bound_slack"], name
        assert gates["pressure_min_pa"] > 0.0, name
        # The wells held the limits the case recorded, on both sides.
        assert gates["producer_bhp_min_pa"] >= PRODUCER_MIN_BHP_PA - BHP_SLACK_PA, name
        assert gates["injector_bhp_max_pa"] <= INJECTOR_MAX_BHP_PA + BHP_SLACK_PA, name
        # The completion event really happened, on exactly the months the policy named.
        completion = gates["p2_lower_completion"]
        assert completion["shut"] == list(range(COMPLETION_SHUT_EDGE, COMPLETION_REOPEN_EDGE))
        assert completion["open"] == [
            month
            for month in range(36)
            if not COMPLETION_SHUT_EDGE <= month < COMPLETION_REOPEN_EDGE
        ]
        # Every well held the rate its control asked for, month by month.
        assert gates["rate_control_relative"] <= tolerances["rate_control_relative_max"], (
            f"{name}: {gates['rate_control_relative']:.3e}"
        )
        # The water really moved: a world in which nothing happened would be no world.
        assert gates["sw_max"] > 0.2 + 1e-6, name
        assert gates["restart"]["completed_report_step"] == 36
        assert gates["restart"]["native_format"] == "Jutul-native"

    assert suite["fully_accepted"] is True, [
        (row["parent_world_id"], row["status"], row["reason"]) for row in suite["rows"]
    ]

    # ---- the boundary, on the artifacts a real run left behind --------------------------
    for row in suite["rows"]:
        context = json.loads(
            world_locations(paths, row["parent_world_id"]).context_file.read_text(encoding="utf-8")
        )
        assert "truth" not in json.dumps(context)
        assert set(context) == {
            "parent_world_id",
            "design_id",
            "derived_view_id",
            "split",
            "observation_ref",
            "control_ref",
            "observation_masks",
            "cutoff",
        }

    # ---- what the session cost -----------------------------------------------------------
    totals = ledger.session_totals()
    assert totals.forwards == 2 * len(P1_PARENTS)
    assert len(ledger.completed_job_ids) == 2 * len(P1_PARENTS)
    assert totals.output_bytes <= P1_LOOP_PROFILE.disk_budget_bytes
    assert totals.wall_s <= P1_LOOP_PROFILE.wall_budget_s
    on_disk = sum(f.stat().st_size for f in paths.artifacts.rglob("*") if f.is_file())
    print(
        "\nP1 suite cost: "
        f"{totals.forwards} forwards, wall {totals.wall_s:.1f} s, "
        f"cpu {totals.cpu_s:.1f} s, ledger output {totals.output_bytes / 1024**2:.2f} MiB, "
        f"artifacts tree {on_disk / 1024**2:.2f} MiB, against a session disk budget of "
        f"{P1_LOOP_PROFILE.disk_budget_bytes / 1024**3:.0f} GiB"
    )
    for row in suite["rows"]:
        cost, gates = row["cost"], row["gates"]
        pre = gates["equilibrium_preflight"]
        print(
            f"  {row['parent_world_id']}: {row['status']}\n"
            f"    cost      wall {cost['wall_s']:.1f} s, cpu {cost['cpu_s']:.1f} s, "
            f"peak rss {cost['peak_rss_bytes'] / 1024**2:.0f} MiB, "
            f"output {cost['output_bytes'] / 1024**2:.2f} MiB, "
            f"steps {cost['accepted_steps']} (+{cost['cut_steps']} cut), "
            f"newton {cost['nonlinear_iterations']}\n"
            f"    balance   cumulative {gates['balance_cumulative_relative']:.3e}, "
            f"median step {gates['balance_step_median_relative']:.3e}, "
            f"abs {gates['balance_absolute_residual_m3_sc']:.3e} m3_sc\n"
            f"    states    sat sum {gates['saturation_sum_abs']:.3e}, "
            f"bound violation {gates['saturation_bound_violation']:.3e}, "
            f"sw [{gates['sw_min']:.4f}, {gates['sw_max']:.4f}], "
            f"p [{gates['pressure_min_pa'] / 1e6:.3f}, {gates['pressure_max_pa'] / 1e6:.3f}] MPa\n"
            f"    wells     producer bhp min {gates['producer_bhp_min_pa'] / 1e6:.3f} MPa, "
            f"injector bhp max {gates['injector_bhp_max_pa'] / 1e6:.3f} MPa, "
            f"rate control {gates['rate_control_relative']:.2e}, "
            f"P2 lower shut months {gates['p2_lower_completion']['shut']}\n"
            f"    preflight {pre['status']} in {pre['wall_s']:.1f} s: "
            f"pressure drift {pre['pressure_relative_drift']:.2e}, "
            f"saturation drift {pre['state_saturation_drift']:.2e}, "
            f"connection {pre['connection_mass_kg_s']:.2e} kg/s, "
            f"balance {pre['balance_cumulative_relative']:.2e}"
        )
