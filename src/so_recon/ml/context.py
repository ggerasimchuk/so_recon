"""The E03 context builder: canonical corpus payloads to leakage-safe tensors.

The builder reads EXACTLY the canonical six-key inference input the corpus publishes —
`context` (with its design geometry), `G`, `U`, `observations` — plus the per-world
identity the manifest carries, and produces a `ContextBatch`. What it refuses matters as
much as what it builds:

* BHP setpoints are encoded as `control_kind` + `control_value` with a unit-derived
  normalization — a BHP protocol never masquerades as a measured rate (plan §4.1);
* observed zero, missing, shut, padding and newly-opened completions are DIFFERENT
  channels: `observed_valid`, `reset`, `padding`, `upper/lower_connection_open`;
* a prefix view truncates the month axis, so mutating anything past the prefix cannot
  change the tensor of that prefix (plan §9 «Mask/prefix leakage»);
* well identity never enters the tensors — order is the manifest's sorted well order,
  and permuting wells permutes rows and edges together, which the encoders are invariant
  to by construction and by test.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from so_recon.config.learning import SplitName
from so_recon.inference.contracts import HistoryRow, ObservationBundle
from so_recon.ml.contracts import F64, ContextBatch, ContextSpec
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import WELL_TIME_FEATURES, default_context_spec
from so_recon.synthetic.p1 import (
    BASE_RATE_M3_SC_DAY,
    INJECTOR_COLUMNS,
    INJECTOR_MAX_BHP_PA,
    PRODUCER_COLUMNS,
    PRODUCER_MIN_BHP_PA,
)

#: Control value normalization constants, derived from the declared units. Rates are
#: scaled by the design's declared base rate; BHP by the declared producer/injector
#: bounds, so a `control_value_normalized` of one kind never means something else of
#: another. Both scales are the SAME constants every corpus design modulates.
BHP_MIN_PA = PRODUCER_MIN_BHP_PA
BHP_MAX_PA = INJECTOR_MAX_BHP_PA


class ForbiddenContextInput(ValueError):
    """An input the encoder is not allowed to see was offered anyway."""


def _control_fields(segment: Mapping[str, Any]) -> tuple[str, float, bool, bool]:
    target = str(segment["target"])
    value = float(segment["value"])
    open_flags = segment.get("connection_open") or (True, True)
    upper = bool(open_flags[0]) if len(open_flags) > 0 else True
    lower = bool(open_flags[1]) if len(open_flags) > 1 else upper
    return target, value, upper, lower


def _design_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The declared physical design the context was conditioned with.

    Every corpus design publishes `start_date`, `n_months` and `shape`; the T2/T3/T4
    designs also publish their `well_columns`, while the P1 design declares them as the
    frozen constants of `so_recon.synthetic.p1`. Either way this is typed design data
    the prior already used for conditioning — unlike the true connectivity, which never
    passes through here.
    """
    context = payload["inference_input"]["context"]
    design = context.get("design", {})
    for key in ("p1_design", "inverse_design", "loop_design"):
        rows = design.get(key)
        if rows is not None:
            return rows.model_dump() if hasattr(rows, "model_dump") else dict(rows)
    raise ForbiddenContextInput("the context carries no design the builder knows")


def _well_geometry(
    design_payload: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float, int]], tuple[float, float]]:
    """(i, j, is_producer) per well plus the areal (nx, ny) the coordinates divide by."""
    columns: dict[str, tuple[float, float, int]] = {}
    declared = design_payload.get("well_columns")
    if declared is None:
        # The P1 design: its wells are frozen constants of the design itself, not rows
        # the payload repeats. Producers and injectors carry their role by membership.
        for name, (i, j) in PRODUCER_COLUMNS:
            columns[str(name)] = (float(i), float(j), 1)
        for name, (i, j) in INJECTOR_COLUMNS:
            columns[str(name)] = (float(i), float(j), 0)
    else:
        for row in declared:
            columns[str(row["well_id"])] = (
                float(row["column"][0]),
                float(row["column"][1]),
                1 if row["role"] == "producer" else 0,
            )
    if not columns:
        raise ForbiddenContextInput("the design declares no well columns")
    shape = design_payload.get("shape")
    if not shape or len(shape) < 2:
        raise ForbiddenContextInput("the design declares no areal shape to scale by")
    return columns, (float(shape[0]), float(shape[1]))


def build_context_batch(
    payload: Mapping[str, Any],
    *,
    spec: ContextSpec | None = None,
    prefix_months: int | None = None,
) -> ContextBatch:
    """Build one world's tensors from its canonical corpus payload.

    `prefix_months` truncates the month axis to a registered prefix view (12/24/36);
    nothing past the prefix is read, so a future-tail mutation cannot leak.
    """
    inference_input = payload["inference_input"]
    for forbidden in ("truth", "theta", "seed"):
        if forbidden in json_keys(inference_input):
            raise ForbiddenContextInput(
                f"the inference input carries {forbidden!r}: the encoder never sees it"
            )
    # The payload's row order carries no meaning: the builder sorts before the bundle
    # validates, so a reordered history is the same history (no fixed well order).
    observations_payload = dict(inference_input["observations"])
    observations_payload["history"] = sorted(
        observations_payload["history"],
        key=lambda row: (str(row["well_id"]), int(row["month_index"])),
    )
    observations = ObservationBundle.model_validate(observations_payload)
    cutoff_month = _cutoff_month(observations)
    months = prefix_months if prefix_months is not None else cutoff_month
    if months < 1 or months > cutoff_month:
        raise ValueError(
            f"prefix_months={months} outside 1..{cutoff_month} of this world's history"
        )
    if spec is None:
        spec = default_context_spec(cutoff_s=observations.cutoff_s, max_wells=8, max_months=36)
    if months > spec.max_months:
        raise ForbiddenContextInput(
            f"{months} months exceed the {spec.max_months} the spec's calendar declares"
        )
    if "control_value_normalized" not in spec.well_time_features:
        raise ValueError("the spec must order control_value_normalized for the builder")

    # Every well of the world: wells that REPORT history and wells that only OPERATE
    # controls (injectors report nothing) are both wells of the tensor.
    control_wells = {str(segment["well_id"]) for segment in inference_input["U"]}
    history_wells = {row.well_id for row in observations.history}
    well_ids = sorted(history_wells | control_wells)
    if len(well_ids) > spec.max_wells:
        raise ForbiddenContextInput(
            f"{len(well_ids)} wells exceed the {spec.max_wells} the spec declares"
        )
    design_payload = _design_payload(payload)
    geometry, grid_shape = _well_geometry(design_payload)
    unknown = [well for well in well_ids if well not in geometry]
    if unknown:
        raise ForbiddenContextInput(
            f"wells {unknown} appear in the history but not in the declared geometry"
        )
    n_wells = len(well_ids)
    n_features = len(spec.well_time_features)
    well_time = np.zeros((n_wells, months, n_features), dtype=np.float64)
    well_mask = np.zeros((n_wells, months), dtype=bool)

    month_edges = _month_edges(design_payload, months, observations.cutoff_s)
    controls = [
        (
            str(segment["well_id"]),
            float(segment["start_s"]),
            float(segment["end_s"]),
            *_control_fields(segment),
        )
        for segment in inference_input["U"]
    ]
    edges_by_group = {
        str(group): np.asarray(edges, dtype=np.float64)
        for group, edges in observations.bin_edges_by_group.items()
    }

    by_key: dict[tuple[str, int], HistoryRow] = {row.key: row for row in observations.history}
    calendar_scale = max(spec.max_months - 1, 1)
    for w, well_id in enumerate(well_ids):
        for month in range(months):
            well_mask[w, month] = True
            edge = month_edges[month]
            active = [
                control
                for control in controls
                if control[0] == well_id and control[1] <= edge < control[2]
            ]
            target, value, upper, lower = (
                (active[0][3], active[0][4], active[0][5], active[0][6])
                if active
                else ("disabled", 0.0, True, True)
            )
            row = by_key.get((well_id, month))
            observed = row is not None and row.observed_valid
            if observed and row is not None and row.bin_index is not None:
                grid_edges = edges_by_group[row.quality_group]
                bin_width = float(grid_edges[row.bin_index + 1] - grid_edges[row.bin_index])
                # The published raw value IS the bin's reported center (the noise law
                # publishes `grid.centers[chosen]`); an observed zero stays exactly 0.
                bin_value = (
                    float(row.raw_value)
                    if row.raw_value is not None
                    else float((grid_edges[row.bin_index] + grid_edges[row.bin_index + 1]) / 2.0)
                )
            else:
                bin_value = 0.0
                bin_width = 0.0
            features = {
                "control_kind_bhp": 1.0 if target == "bhp" else 0.0,
                "control_kind_liquid_rate": 1.0 if target == "liquid_rate" else 0.0,
                "control_kind_water_rate": 1.0 if target == "water_rate" else 0.0,
                "control_kind_disabled": 1.0 if target == "disabled" else 0.0,
                "control_value_normalized": _normalized_control(target, value),
                "observed_valid": 1.0 if observed else 0.0,
                "observed_bin_center": bin_value,
                "observed_bin_width": bin_width,
                "reset": 1.0 if (row is not None and row.reset) else 0.0,
                # The calendar scale is the spec's declared horizon, not this view's
                # length: a prefix view must reproduce the full tensor's prefix exactly.
                "month_index_scaled": float(month) / calendar_scale,
                "elapsed_month_scaled": float(month) / max(cutoff_month, 1),
                "upper_connection_open": 1.0 if upper else 0.0,
                "lower_connection_open": 1.0 if lower else 0.0,
                "padding": 0.0,
            }
            for f, name in enumerate(spec.well_time_features):
                well_time[w, month, f] = features[name]

    static = _static_features(payload, well_ids, geometry, grid_shape, spec)
    edge_index, edge_attr = _graph_geometry(well_ids, geometry, grid_shape, spec)
    return ContextBatch(
        parent_id=str(payload["parent_id"]),
        design_id=str(payload["design_id"]),
        split=cast(SplitName, payload["split"]),
        spec_hash=spec.spec_hash,
        well_time=well_time,
        well_mask=well_mask,
        static=static,
        edge_index=edge_index,
        edge_attr=edge_attr,
        well_ids=tuple(well_ids),
    )


def json_keys(mapping: Mapping[str, Any]) -> set[str]:
    return set(mapping.keys())


def _cutoff_month(observations: ObservationBundle) -> int:
    if not observations.history:
        raise ValueError("an empty history carries no month axis")
    return max(row.month_index for row in observations.history) + 1


def _month_edges(design_payload: Mapping[str, Any], months: int, cutoff_s: float) -> np.ndarray:
    """The month boundaries of the DECLARED calendar, never inferred from controls.

    Controls come in blocks (the P1 rate modulation changes three times in 36 months),
    so control starts are not month edges. The design publishes its real calendar
    (`start_date` + `n_months`) and the history publishes the cutoff both must agree on.
    """
    start = date.fromisoformat(str(design_payload["start_date"]))
    declared_months = int(design_payload["n_months"])
    if months > declared_months:
        raise ForbiddenContextInput(
            f"{months} months exceed the {declared_months} the design's calendar declares"
        )
    edges = np.asarray(month_edges_s(start, declared_months), dtype=np.float64)
    if abs(float(edges[-1]) - float(cutoff_s)) > 1.0:
        raise ForbiddenContextInput(
            f"the design calendar ends at {edges[-1]:.0f}s but the history's cutoff is "
            f"{cutoff_s:.0f}s: the two disagree about what a month is"
        )
    return edges[:months]


def _normalized_control(target: str, value: float) -> float:
    if target in ("liquid_rate", "water_rate"):
        return float(np.clip(value / BASE_RATE_M3_SC_DAY, -2.0, 2.0))
    if target == "bhp":
        return float((value - BHP_MIN_PA) / (BHP_MAX_PA - BHP_MIN_PA))
    return 0.0


#: The uninformative sigma of a well the available G never covered: a wide sigma says
#: "no support", a fabricated tight value would assert knowledge nobody measured.
UNSUPPORTED_LOG_K_SIGMA = 1.0


def _static_features(
    payload: Mapping[str, Any],
    well_ids: list[str],
    geometry: Mapping[str, tuple[float, float, int]],
    grid_shape: tuple[float, float],
    spec: ContextSpec,
) -> F64:
    g_rows = payload["inference_input"]["G"]
    by_well: dict[str, tuple[float, float]] = {}
    for row in g_rows:
        name = str(row["observation_id"])
        well = name.split("-L")[0] if "-L" in name else name.split("-layer")[0]
        value = float(row["value"])
        sigma = float(row.get("sigma", 0.2))
        current = by_well.get(well)
        if current is None:
            by_well[well] = (value, sigma)
        else:
            by_well[well] = (
                0.5 * (current[0] + value),
                0.5 * (current[1] + sigma),
            )
    rows = []
    for well in well_ids:
        i, j, is_producer = geometry[well]
        value, sigma = by_well.get(well, (0.0, UNSUPPORTED_LOG_K_SIGMA))
        rows.append(
            [
                i / grid_shape[0],
                j / grid_shape[1],
                float(is_producer),
                value,
                sigma,
            ]
        )
    static = np.asarray(rows, dtype=np.float64).reshape(len(well_ids), -1)
    if static.shape[1] != len(spec.static_features):
        raise ValueError(
            f"static features produced {static.shape[1]} columns for the "
            f"{len(spec.static_features)} the spec declares"
        )
    return static


def _graph_geometry(
    well_ids: list[str],
    geometry: Mapping[str, tuple[float, float, int]],
    grid_shape: tuple[float, float],
    spec: ContextSpec,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    index = np.asarray(
        [[a, b] for a in range(len(well_ids)) for b in range(len(well_ids)) if a != b],
        dtype=np.int64,
    ).T.reshape(2, -1)
    attributes: list[list[float]] = []
    for a, b in index.T:
        ia, ja, _ = geometry[well_ids[a]]
        ib, jb, _ = geometry[well_ids[b]]
        dx = (ia - ib) / grid_shape[0]
        dy = (ja - jb) / grid_shape[1]
        distance = float(np.hypot(dx, dy))
        attributes.append([dx, dy, distance, 1.0, 0.0])
    edge_attr = np.asarray(attributes, dtype=np.float64)
    if edge_attr.shape[1] != len(spec.edge_features):
        raise ValueError(
            f"edge features produced {edge_attr.shape[1]} columns for the "
            f"{len(spec.edge_features)} the spec declares"
        )
    return index.astype(np.int64), edge_attr


def permute_batch(batch: ContextBatch, order: npt.NDArray[np.int64]) -> ContextBatch:
    """Reorder wells — rows, mask, static, well ids — remapping edges to match."""
    order = np.asarray(order, dtype=np.int64)
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    edge_index = remap[batch.edge_index]
    return ContextBatch(
        parent_id=batch.parent_id,
        design_id=batch.design_id,
        split=batch.split,
        spec_hash=batch.spec_hash,
        well_time=np.ascontiguousarray(batch.well_time[order]),
        well_mask=np.ascontiguousarray(batch.well_mask[order]),
        static=np.ascontiguousarray(batch.static[order]),
        edge_index=np.ascontiguousarray(edge_index),
        edge_attr=batch.edge_attr,
        well_ids=tuple(batch.well_ids[int(i)] for i in order),
    )


__all__ = [
    "BASE_RATE_M3_SC_DAY",
    "BHP_MIN_PA",
    "BHP_MAX_PA",
    "ForbiddenContextInput",
    "WELL_TIME_FEATURES",
    "build_context_batch",
    "default_context_spec",
    "permute_batch",
]
