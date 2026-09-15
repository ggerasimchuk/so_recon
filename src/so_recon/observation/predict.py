"""Extract E02 model observations from one complete E01 physical result."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so_recon.inference.contracts import (
    ModelObservations,
    ObservationBundle,
    ObservationSupportMismatch,
)
from so_recon.paths import ProjectPaths
from so_recon.simulator.case_io import read_array
from so_recon.simulator.contracts import ForwardResult
from so_recon.simulator.results import MONTHLY_SCHEMA

WATERCUT_PARITY_ATOL = 1e-12


def extract_monthly_watercut(table: pa.Table) -> dict[tuple[str, int], float | None]:
    """Compute produced watercut from monthly integrals and parity-check stored ``fw``.

    ``water_inj_m3_sc`` is intentionally never read into the ratio.  Injection belongs to
    another side of the balance and adding it to produced liquid would make an injector
    look like a producer with a high watercut.
    """
    missing = [field.name for field in MONTHLY_SCHEMA if field.name not in table.column_names]
    if missing:
        raise ObservationSupportMismatch(f"monthly output is missing columns {missing}")
    out: dict[tuple[str, int], float | None] = {}
    for payload in table.select(MONTHLY_SCHEMA.names).to_pylist():
        key = (str(payload["well_id"]), int(payload["month_index"]))
        if key in out:
            raise ObservationSupportMismatch(f"monthly output repeats well-month {key}")
        oil = float(payload["oil_prod_m3_sc"])
        water = float(payload["water_prod_m3_sc"])
        liquid = float(payload["liquid_prod_m3_sc"])
        if any(not math.isfinite(value) or value < 0.0 for value in (oil, water, liquid)):
            raise ObservationSupportMismatch(
                f"monthly output {key} has invalid produced volumes oil={oil}, water={water}, "
                f"liquid={liquid}"
            )
        produced = oil + water
        if not math.isclose(liquid, produced, rel_tol=0.0, abs_tol=WATERCUT_PARITY_ATOL):
            raise ObservationSupportMismatch(
                f"monthly output {key} liquid parity failed: oil+water={produced}, "
                f"liquid_prod_m3_sc={liquid}"
            )
        valid = bool(payload["fw_valid"])
        stored = payload["fw"]
        if not valid:
            if stored is not None:
                raise ObservationSupportMismatch(
                    f"monthly output {key} marks fw invalid but stores {stored!r}"
                )
            out[key] = None
            continue
        if produced <= 0.0 or stored is None:
            raise ObservationSupportMismatch(
                f"monthly output {key} marks fw valid without positive produced liquid"
            )
        computed = water / produced
        published = float(stored)
        if not math.isfinite(published) or not math.isclose(
            computed, published, rel_tol=0.0, abs_tol=WATERCUT_PARITY_ATOL
        ):
            raise ObservationSupportMismatch(
                f"monthly output {key} fw parity failed: produced ratio={computed}, "
                f"stored fw={published}"
            )
        out[key] = computed
    return out


def _state_so(result: ForwardResult, paths: ProjectPaths) -> np.ndarray:
    if "so" in result.states:
        return np.asarray(read_array(result.states["so"], paths), dtype=np.float64)
    if "sw" in result.states and result.physics_class == "OW":
        return 1.0 - np.asarray(read_array(result.states["sw"], paths), dtype=np.float64)
    raise ObservationSupportMismatch(
        f"forward result {result.job_id} publishes neither so nor an OW sw state"
    )


def predict_observations(
    result: ForwardResult, observations: ObservationBundle, paths: ProjectPaths
) -> ModelObservations:
    """Project exact monthly integrals and exact saved states onto one observation bundle."""
    if result.status != "COMPLETE" or result.monthly_path is None:
        raise ObservationSupportMismatch(
            f"forward result {result.job_id} is {result.status}, not a complete prediction"
        )
    monthly_path = paths.resolve(result.monthly_path)
    if not monthly_path.is_file():
        raise ObservationSupportMismatch(f"monthly output {result.monthly_path} is missing")
    monthly = extract_monthly_watercut(pq.read_table(monthly_path))
    fw = {row.key: monthly.get(row.key) for row in observations.history}

    so_support: dict[tuple[str, float], float] = {}
    if observations.logs:
        states = _state_so(result, paths)
        if states.ndim != 2 or states.shape[0] != len(result.times_s):
            raise ObservationSupportMismatch(
                f"forward state shape {states.shape} does not match {len(result.times_s)} times"
            )
        time_index = {float(time): index for index, time in enumerate(result.times_s)}
        for row in observations.logs:
            cells = np.asarray(row.support_cell_ids, dtype=np.int64)
            if np.any(cells >= states.shape[1]):
                raise ObservationSupportMismatch(
                    f"log row {row.observation_id!r} support exceeds {states.shape[1]} cells"
                )
            weights = np.asarray(row.support_weights, dtype=np.float64)
            for time_s in row.date_times_s:
                index = time_index.get(float(time_s))
                if index is None:
                    raise ObservationSupportMismatch(
                        f"log row {row.observation_id!r} requested state time {time_s}, but "
                        f"forward {result.job_id} saved {result.times_s}; E02 does not interpolate"
                    )
                so_support[(row.observation_id, time_s)] = float(weights @ states[index, cells])

    return ModelObservations(fw=fw, so_support=so_support, model_hash=result.model_hash)


__all__ = [
    "WATERCUT_PARITY_ATOL",
    "extract_monthly_watercut",
    "predict_observations",
]
