"""Published P1 physics evidence and a frozen educational observability screen.

The screen excludes startup (months 1–12), requires late watercut >= 0.05 and
at least 0.02 temporal range in one producer. These are fixed before drawing
worlds. This is a useful dynamic signal check, not an identifiability claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_bytes, sha256_file, sha256_json
from so_recon.simulator.case_io import compute_model_hash, load_case, read_array
from so_recon.simulator.contracts import SECONDS_PER_DAY, CaseBundle, ForwardResult
from so_recon.synthetic.p1 import (
    COMPLETION_REOPEN_EDGE,
    COMPLETION_SHUT_EDGE,
    INJECTOR_IDS,
    INJECTOR_MAX_BHP_PA,
    PRODUCER_IDS,
    PRODUCER_MIN_BHP_PA,
    RenderedWorld,
    closed_preflight_world,
)

SCREEN_VERSION = "p1-late-watercut-1"
LATE_MONTH_START = 12
LATE_WATERCUT_MIN = 0.05
LATE_WATERCUT_RANGE_MIN = 0.02


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
    assert {str(r["well_id"]) for r in rows} == set(PRODUCER_IDS + INJECTOR_IDS), (
        "BHP evidence missing a well"
    )
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
    monthly = _rows(result.monthly_path, paths)
    keys = [(str(r["well_id"]), int(r["month_index"])) for r in monthly]
    assert len(keys) == len(targets) and set(keys) == set(targets), (
        "missing/duplicate monthly rate rows"
    )
    for row in monthly:
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


def watercut_signal(result: ForwardResult, paths: ProjectPaths) -> dict[str, Any]:
    rows = _rows(result.monthly_path, paths)
    by_well: dict[str, Any] = {}
    for well in PRODUCER_IDS:
        selected = [
            r
            for r in rows
            if r["well_id"] == well and LATE_MONTH_START <= int(r["month_index"]) < 36
        ]
        values = [
            float(r["fw"])
            for r in selected
            if r.get("fw") is not None and r.get("fw_valid") is True
        ]
        valid = (
            len(values) == 24
            and len({r["month_index"] for r in selected}) == 24
            and all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        )
        by_well[well] = {
            "valid": valid,
            "maximum": max(values) if valid else None,
            "range": max(values) - min(values) if valid else None,
        }
    passed = all(v["valid"] for v in by_well.values()) and any(
        v["maximum"] >= LATE_WATERCUT_MIN and v["range"] >= LATE_WATERCUT_RANGE_MIN
        for v in by_well.values()
        if v["valid"]
    )
    return {
        "version": SCREEN_VERSION,
        "month_start": LATE_MONTH_START,
        "minimum_maximum": LATE_WATERCUT_MIN,
        "minimum_range": LATE_WATERCUT_RANGE_MIN,
        "by_producer": by_well,
        "passed": passed,
        "limitation": (
            "Educational observability screen; does not establish inverse identifiability."
        ),
    }


def _score_p1_world(
    world: RenderedWorld,
    result: ForwardResult,
    preflight: ForwardResult | None,
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Compute metrics from the published outputs, without discarding failed results."""
    equilibrium: dict[str, Any] = {"status": None if preflight is None else preflight.status}
    if preflight is not None and preflight.status == "COMPLETE":
        states = _states(preflight, paths)
        _state_metrics(states)  # fail before accepting nonfinite preflight states
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
    gates: dict[str, Any] = {"status": result.status, "equilibrium_preflight": equilibrium}
    if result.status == "COMPLETE":
        gates |= _balance_metrics(result, paths)
        gates |= _state_metrics(_states(result, paths))
        gates |= _bhp_metrics(result, paths)
        gates["rate_control_relative"] = _rate_control_relative(world, result, paths)
        gates["p2_lower_completion"] = _completion_event_months(world, result, paths)
        gates["watercut_signal"] = watercut_signal(result, paths)
        gates["restart"] = (
            {}
            if result.restart is None
            else {
                "completed_report_step": result.restart.completed_report_step,
                "native_format": result.restart.native_format,
            }
        )
    gates["evidence_identity"] = {
        "parent_world_id": world.parent_world_id,
        "forward": result_evidence_identity(result, paths),
        "preflight": None if preflight is None else result_evidence_identity(preflight, paths),
        "preflight_result": None if preflight is None else preflight.model_dump(mode="json"),
    }
    if not evidence_matches_world(world, result, gates, paths):
        raise ValueError("physical evidence is not bound to this world's forward and preflight")
    return gates


def evaluate_p1_acceptance(
    gates: Mapping[str, Any] | None, tolerances: Mapping[str, float] | None
) -> dict[str, Any]:
    """Fail closed on absent, nonfinite, malformed or failed required physical evidence."""
    checks: dict[str, bool] = {}
    if isinstance(gates, Mapping) and isinstance(tolerances, Mapping):

        def below(name: str, limit: str, source: Mapping[str, Any] = gates) -> bool:
            value = source.get(name)
            return (
                isinstance(value, (float, int))
                and math.isfinite(value)
                and 0 <= value <= tolerances[limit]
            )

        try:
            checks["complete"] = gates.get("status") == "COMPLETE"
            for name, limit in (
                ("balance_cumulative_relative", "balance_cumulative_relative_max"),
                ("balance_step_median_relative", "balance_step_median_relative_max"),
                ("saturation_sum_abs", "saturation_sum_abs_max"),
                ("saturation_bound_violation", "saturation_bound_slack"),
                ("rate_control_relative", "rate_control_relative_max"),
            ):
                checks[name] = below(name, limit)
            pre = gates["equilibrium_preflight"]
            checks["preflight"] = pre.get("status") == "COMPLETE" and all(
                below(name, limit, pre)
                for name, limit in (
                    ("pressure_relative_drift", "hydrostatic_pressure_relative_drift_max"),
                    ("state_saturation_drift", "hydrostatic_saturation_drift_max"),
                    ("connection_mass_kg_s", "closed_connection_mass_kg_s_max"),
                    ("balance_cumulative_relative", "balance_cumulative_relative_max"),
                    ("balance_step_median_relative", "balance_step_median_relative_max"),
                )
            )
            checks["pressure"] = (
                math.isfinite(gates["pressure_min_pa"]) and gates["pressure_min_pa"] > 0
            )
            checks["bhp"] = (
                math.isfinite(gates["producer_bhp_min_pa"])
                and math.isfinite(gates["injector_bhp_max_pa"])
                and gates["producer_bhp_min_pa"] >= PRODUCER_MIN_BHP_PA - 1.0
                and gates["injector_bhp_max_pa"] <= INJECTOR_MAX_BHP_PA + 1.0
            )
            shut = list(range(COMPLETION_SHUT_EDGE, COMPLETION_REOPEN_EDGE))
            checks["completion"] = gates["p2_lower_completion"] == {
                "shut": shut,
                "open": [m for m in range(36) if m not in shut],
            }
            checks["restart"] = (
                gates["restart"]["completed_report_step"] == 36
                and gates["restart"]["native_format"] == "Jutul-native"
            )
            signal = gates["watercut_signal"]
            checks["signal"] = (
                signal["version"] == SCREEN_VERSION
                and all(
                    v["valid"] is True
                    and math.isfinite(v["maximum"])
                    and math.isfinite(v["range"])
                    and 0 <= v["maximum"] <= 1
                    and 0 <= v["range"] <= v["maximum"]
                    for v in signal["by_producer"].values()
                )
                and set(signal["by_producer"]) == set(PRODUCER_IDS)
                and any(
                    math.isfinite(v["maximum"])
                    and math.isfinite(v["range"])
                    and LATE_WATERCUT_MIN <= v["maximum"] <= 1.0
                    and LATE_WATERCUT_RANGE_MIN <= v["range"] <= 1.0
                    for v in signal["by_producer"].values()
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            checks["complete_evidence"] = False
    return {
        "version": "p1-physical-acceptance-1",
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "metrics": dict(gates) if isinstance(gates, Mapping) else None,
        "tolerances": dict(tolerances) if isinstance(tolerances, Mapping) else None,
    }


def score_p1_world(
    world: RenderedWorld,
    result: ForwardResult,
    preflight: ForwardResult | None,
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Retain malformed/missing outputs as explicit failed evidence for the suite row."""
    try:
        return _score_p1_world(world, result, preflight, paths)
    except (AssertionError, OSError, KeyError, ValueError, TypeError) as error:
        return {"status": result.status, "evidence_error": f"{type(error).__name__}: {error}"}


def result_evidence_identity(result: ForwardResult, paths: ProjectPaths) -> dict[str, Any]:
    """Bind a result record and the bytes actually scored, including state arrays."""
    output_paths = {
        p
        for p in (result.monthly_path, result.connections_path, result.balances_path)
        if p is not None
    }
    output_paths.update(ref.path for ref in result.states.values())
    outputs = {path: sha256_file(paths.resolve(path)) for path in sorted(output_paths)}
    return {
        "job_id": result.job_id,
        "model_hash": result.model_hash,
        "case_sha256": result.case_sha256,
        "result_sha256": sha256_json(result.model_dump(mode="json")),
        "outputs": outputs,
        "outputs_sha256": sha256_json(outputs),
    }


def evidence_matches_world(
    world: RenderedWorld,
    result: ForwardResult | None,
    gates: Mapping[str, Any] | None,
    paths: ProjectPaths,
    case: CaseBundle | None = None,
) -> bool:
    """Verify provenance against the current result and independently rendered preflight.

    Numeric metrics are not a signed attestation. This binding prevents accidental reuse
    of another world's metrics or a previous attempt's outputs, and checks the bytes have
    not changed since scoring. It does not defend against intentional metric forgery.
    """
    if result is None or not isinstance(gates, Mapping):
        return False
    try:
        identity = gates["evidence_identity"]
        if identity["parent_world_id"] != world.parent_world_id:
            return False
        if identity["forward"] != result_evidence_identity(result, paths):
            return False
        preflight = ForwardResult.model_validate(identity["preflight_result"])
        if identity["preflight"] != result_evidence_identity(preflight, paths):
            return False
        if case is None:
            if result.monthly_path is None:
                return False
            case = load_case(paths.resolve(result.monthly_path).parent.parent / "case.json", paths)

        def digest(case: CaseBundle) -> str:
            return sha256_bytes(
                (
                    json.dumps(
                        case.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )

        if (
            case.case_id != world.case_fields["case_id"]
            or case.model_hash != result.model_hash
            or digest(case) != result.case_sha256
        ):
            return False
        fields = closed_preflight_world(world).case_fields
        expected = case.model_copy(
            update={
                name: fields[name] for name in ("case_id", "cutoff", "report_edges_s", "controls")
            }
        )
        expected = expected.model_copy(update={"model_hash": compute_model_hash(expected)})
        expected_bytes = (
            json.dumps(
                expected.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        return (
            preflight.model_hash == expected.model_hash
            and preflight.case_sha256 == sha256_bytes(expected_bytes)
        )
    except (OSError, KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return False
