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

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from so_recon.inference.contracts import HistoryRow, ObservationBundle
from so_recon.ml.contracts import ContextBatch, ContextSpec
from so_recon.synthetic.inverse_corpus import WELL_TIME_FEATURES, default_context_spec

#: Control value normalization constants, derived from the declared units. Rates are
#: scaled by the E01 base rate; BHP by the declared producer/injector bounds, so a
#: `control_value_normalized` of one kind never means something else of another.
BASE_RATE_M3_SC_DAY = 60.0
BHP_MIN_PA = 5.0e6
BHP_MAX_PA = 3.0e7


class ForbiddenContextInput(ValueError):
    """An input the encoder is not allowed to see was offered anyway."""


def _control_fields(segment: Mapping[str, Any]) -> tuple[str, float, bool, bool]:
    target = str(segment["target"])
    value = float(segment["value"])
    open_flags = segment.get("connection_open") or (True, True)
    upper = bool(open_flags[0]) if len(open_flags) > 0 else True
    lower = bool(open_flags[1]) if len(open_flags) > 1 else upper
    return target, value, upper, lower


def _month_of(seconds: float, month_edges: np.ndarray) -> int:
    return int(np.searchsorted(month_edges, seconds, side="right") - 1)


def _well_geometry(payload: Mapping[str, Any]) -> dict[str, tuple[float, float, int]]:
    """(i, j, is_producer) per well from the design the context was conditioned with.

    The design is typed data the prior already used for conditioning — geometry is a
    DECLARED quantity, unlike the true connectivity, which never passes through here.
    """
    context = payload["inference_input"]["context"]
    design = context.get("design", {})
    columns: dict[str, tuple[float, float, int]] = {}
    rows = design.get("p1_design")
    if rows is None:
        rows = design.get("inverse_design") or design.get("loop_design")
    if rows is None:
        raise ForbiddenContextInput("the context carries no design the builder knows")
    payload_design = rows.model_dump() if hasattr(rows, "model_dump") else dict(rows)
    for row in payload_design.get("well_columns", []):
        columns[str(row["well_id"])] = (
            float(row["column"][0]),
            float(row["column"][1]),
            1 if row["role"] == "producer" else 0,
        )
    if not columns:
        raise ForbiddenContextInput("the design declares no well columns")
    return columns


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
    observations = ObservationBundle.model_validate(inference_input["observations"])
    cutoff_month = _cutoff_month(observations)
    months = prefix_months if prefix_months is not None else cutoff_month
    if months < 1 or months > cutoff_month:
        raise ValueError(
            f"prefix_months={months} outside 1..{cutoff_month} of this world's history"
        )
    if spec is None:
        spec = default_context_spec(
            cutoff_s=observations.cutoff_s, max_wells=8, max_months=36
        )
    if "control_value_normalized" not in spec.well_time_features:
        raise ValueError("the spec must order control_value_normalized for the builder")

    well_ids = sorted({row.well_id for row in observations.history})
    geometry = _well_geometry(payload)
    unknown = [well for well in well_ids if well not in geometry]
    if unknown:
        raise ForbiddenContextInput(
            f"wells {unknown} appear in the history but not in the declared geometry"
        )
    n_wells = len(well_ids)
    n_features = len(spec.well_time_features)
    well_time = np.zeros((n_wells, months, n_features), dtype=np.float64)
    well_mask = np.zeros((n_wells, months), dtype=bool)

    month_edges = _month_edges(inference_input, months)
    controls = [
        (
            str(segment["well_id"]),
            float(segment["start_s"]),
            float(segment["end_s"]),
            *_control_fields(segment),
        )
        for segment in inference_input["U"]
    ]
    grid_centers, grid_width = _grid_geometry(observations)

    by_key: dict[tuple[str, int], HistoryRow] = {row.key: row for row in observations.history}
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
                _control_fields_of(active) if active else ("disabled", 0.0, True, True)
            )
            row = by_key.get((well_id, month))
            observed = row is not None and row.observed_valid
            features = {
                "control_kind_bhp": 1.0 if target == "bhp" else 0.0,
                "control_kind_liquid_rate": 1.0 if target == "liquid_rate" else 0.0,
                "control_kind_water_rate": 1.0 if target == "water_rate" else 0.0,
                "control_kind_disabled": 1.0 if target == "disabled" else 0.0,
                "control_value_normalized": _normalized_control(target, value),
                "observed_valid": 1.0 if observed else 0.0,
                "observed_bin_center": (
                    float(grid_centers[row.bin_index]) if observed and row and row.bin_index is not None else 0.0
                ),
                "observed_bin_width": (
                    float(grid_width) if observed else 0.0
                ),
                "reset": 1.0 if (row is not None and row.reset) else 0.0,
                "month_index_scaled": float(month) / max(months - 1, 1),
                "elapsed_month_scaled": float(month) / max(cutoff_month, 1),
                "upper_connection_open": 1.0 if upper else 0.0,
                "lower_connection_open": 1.0 if lower else 0.0,
                "padding": 0.0,
            }
            for f, name in enumerate(spec.well_time_features):
                well_time[w, month, f] = features[name]

    static = _static_features(payload, well_ids, geometry, spec)
    edge_index, edge_attr = _graph_geometry(well_ids, geometry, spec)
    return ContextBatch(
        parent_id=str(payload["parent_id"]),
        design_id=str(payload["design_id"]),
        split=payload["split"],  # type: ignore[arg-type]
        spec_hash=spec.spec_hash,
        well_time=well_time,
        well_mask=well_mask,
        static=static,
        edge_index=edge_index,
        edge_attr=edge_attr,
        well_ids=tuple(well_ids),
    )


def _control_fields_of(active: list[tuple[str, float, float, str, float, bool, bool]]):
    target, value, upper, lower = active[0][3], active[0][4], active[0][5], active[0][6]
    return target, value, upper, lower


def json_keys(mapping: Mapping[str, Any]) -> set[str]:
    return set(mapping.keys())


def _cutoff_month(observations: ObservationBundle) -> int:
    if not observations.history:
        raise ValueError("an empty history carries no month axis")
    return max(row.month_index for row in observations.history) + 1


def _month_edges(inference_input: Mapping[str, Any], months: int) -> np.ndarray:
    controls = inference_input["U"]
    starts = sorted({float(segment["start_s"]) for segment in controls})
    if len(starts) < months:
        raise ValueError("the control calendar cannot cover the requested months")
    return np.asarray(starts[:months], dtype=np.float64)


def _normalized_control(target: str, value: float) -> float:
    if target in ("liquid_rate", "water_rate"):
        return float(np.clip(value / BASE_RATE_M3_SC_DAY, -2.0, 2.0))
    if target == "bhp":
        return float((value - BHP_MIN_PA) / (BHP_MAX_PA - BHP_MIN_PA))
    return 0.0


def _grid_geometry(observations: ObservationBundle) -> tuple[np.ndarray, float]:
    edges = next(iter(observations.bin_edges_by_group.values()))
    centers = np.asarray(
        [(lo + hi) / 2.0 for lo, hi in zip(edges[:-1], edges[1:], strict=False)],
        dtype=np.float64,
    )
    return centers, float(np.mean(np.diff(edges)))


def _static_features(
    payload: Mapping[str, Any],
    well_ids: list[str],
    geometry: Mapping[str, tuple[float, float, int]],
    spec: ContextSpec,
) -> F64:
    shape = (16.0, 16.0)
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
        value, sigma = by_well.get(well, (0.0, 0.2))
        rows.append([i / shape[0], j / shape[1], float(is_producer), value, sigma])
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
    spec: ContextSpec,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    index = np.asarray(
        [
            [a, b]
            for a in range(len(well_ids))
            for b in range(len(well_ids))
            if a != b
        ],
        dtype=np.int64,
    ).T.reshape(2, -1)
    attributes: list[list[float]] = []
    for a, b in index.T:
        ia, ja, _ = geometry[well_ids[a]]
        ib, jb, _ = geometry[well_ids[b]]
        dx = (ia - ib) / 16.0
        dy = (ja - jb) / 16.0
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
