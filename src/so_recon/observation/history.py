"""Calendar-aware recursive watercut likelihood for E02.

The residual carried from one measured month to the next is defined in bounded
``g(f) = asin(sqrt(f))`` space.  It is carried from the centre of the *reported bin*, not
from an unobservable continuous draw, and decays over the actual calendar gap.  Missing
months therefore do not become observations, while a reset on a missing row still clears
the well's memory.

This module owns the conditional mean used by both the scorer and the generator in
``noise.py``.  Keeping that recursion in one function is part of the likelihood contract:
sampling from one AR convention and scoring with another would make calibration tests
meaningless.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from so_recon.inference.contracts import (
    HistoryRow,
    LoglikResult,
    NoiseTheta,
    ObservationSupportMismatch,
)
from so_recon.observation.bins import BinGrid, bin_log_probs, to_g_space
from so_recon.registry.hashing import sha256_json

HistoryPrediction = Mapping[tuple[str, int], float | None]
HistoryGrids = Mapping[str, BinGrid]
PreviousObservation = tuple[int, float, float]


def _fraction(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite fraction in [0, 1], got {number!r}")
    return number


def conditional_mean(
    predicted: float,
    previous: PreviousObservation | None,
    month: int,
    rho: float,
    reset: bool,
) -> float:
    """Conditional location in g-space for one observed month.

    ``previous`` is ``(calendar_month, published_bin_center, model_prediction)`` for the
    last valid observation of the same well.  A reset discards it before this row.  The
    function validates the tuple itself because it is a public mathematical kernel used
    directly by analytic tests as well as by the row traversal below.
    """
    current = _fraction(predicted, label="predicted watercut")
    if not math.isfinite(rho) or not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be finite and in [0, 1), got {rho!r}")
    current_g = float(to_g_space(np.asarray(current, dtype=np.float64)))
    if previous is None or reset:
        return current_g

    old_month, old_center, old_prediction = previous
    if month <= old_month:
        raise ValueError(
            f"calendar months for one well must be strictly increasing, got previous "
            f"{old_month} and current {month}"
        )
    published = _fraction(old_center, label="previous published bin center")
    modelled = _fraction(old_prediction, label="previous model prediction")
    published_g, modelled_g = to_g_space(np.asarray([published, modelled], dtype=np.float64))
    return float(current_g + rho ** (month - old_month) * (published_g - modelled_g))


def _check_row_order(rows: Sequence[HistoryRow]) -> None:
    last_month: dict[str, int] = {}
    for row in rows:
        previous = last_month.get(row.well_id)
        if previous is not None and row.month_index <= previous:
            raise ValueError(
                f"calendar months for well {row.well_id!r} must be strictly increasing, "
                f"got {previous} then {row.month_index}"
            )
        last_month[row.well_id] = row.month_index


def _grid_for(row: HistoryRow, grids: HistoryGrids) -> BinGrid:
    try:
        return grids[row.quality_group]
    except KeyError as exc:
        raise ValueError(
            f"history row {row.key} names quality group {row.quality_group!r}, but the "
            f"available grids are {sorted(grids)}"
        ) from exc


def _prediction_for(row: HistoryRow, prediction: HistoryPrediction) -> float:
    value = prediction.get(row.key)
    if value is None:
        raise ObservationSupportMismatch(
            f"observed history row {row.key} has no physical watercut prediction; an "
            "observed positive-liquid month cannot be removed by the model's dry mask"
        )
    try:
        return _fraction(value, label=f"prediction for history row {row.key}")
    except ValueError as exc:
        raise ObservationSupportMismatch(str(exc)) from exc


def _history_semantic_hash(rows: Sequence[HistoryRow], grids: HistoryGrids) -> str:
    return sha256_json(
        {
            "rows": [row.model_dump(mode="json") for row in rows],
            "grids": {
                group: {
                    "centers": grid.centers.tolist(),
                    "edges": grid.edges.tolist(),
                }
                for group, grid in sorted(grids.items())
            },
            "version": "recursive-history-1",
        }
    )


def history_loglik(
    rows: Sequence[HistoryRow],
    prediction: HistoryPrediction,
    noise: NoiseTheta,
    grids: HistoryGrids,
    *,
    observation_hash: str | None = None,
) -> LoglikResult:
    """Score the observed rows without renormalising over the rows that happened to exist.

    Invalid/missing rows contribute no term, but are still traversed so their reset flag
    can clear the corresponding well.  A valid row must have a valid physical prediction;
    absence is a support mismatch, not permission to mask inconvenient evidence.
    """
    row_tuple = tuple(rows)
    _check_row_order(row_tuple)
    previous_by_well: dict[str, PreviousObservation] = {}
    terms: list[float] = []

    for row in row_tuple:
        if row.reset:
            previous_by_well.pop(row.well_id, None)
        if not row.observed_valid:
            continue

        grid = _grid_for(row, grids)
        if row.bin_index is None or row.bin_index >= grid.n_bins:
            raise ObservationSupportMismatch(
                f"observed history row {row.key} names bin {row.bin_index!r}, outside "
                f"the {grid.n_bins} bins of group {row.quality_group!r}"
            )
        predicted = _prediction_for(row, prediction)
        mu = conditional_mean(
            predicted,
            previous_by_well.get(row.well_id),
            row.month_index,
            noise.rho,
            False,
        )
        probabilities = bin_log_probs(
            grid,
            mu=mu,
            sigma=noise.sigma * row.sigma_multiplier,
            nu=noise.nu,
        )
        terms.append(float(probabilities[row.bin_index]))
        previous_by_well[row.well_id] = (
            row.month_index,
            float(grid.centers[row.bin_index]),
            predicted,
        )

    value = math.fsum(terms) if terms else 0.0
    return LoglikResult(
        value=value,
        value_in_support=True,
        terms=tuple(terms),
        terms_in_support=tuple(True for _ in terms),
        n_used=len(terms),
        observation_hash=observation_hash or _history_semantic_hash(row_tuple, grids),
    )


__all__ = [
    "HistoryGrids",
    "HistoryPrediction",
    "PreviousObservation",
    "conditional_mean",
    "history_loglik",
]
