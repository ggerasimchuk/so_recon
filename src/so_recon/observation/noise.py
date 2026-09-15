"""Draw rounded history rows from the exact recursive law used by the scorer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from so_recon.inference.contracts import HistoryRow, NoiseTheta
from so_recon.observation.bins import bin_log_probs
from so_recon.observation.history import (
    HistoryGrids,
    HistoryPrediction,
    PreviousObservation,
    _check_row_order,
    _grid_for,
    _prediction_for,
    conditional_mean,
)


def draw_history(
    rows: Sequence[HistoryRow],
    prediction: HistoryPrediction,
    noise: NoiseTheta,
    grids: HistoryGrids,
    rng: np.random.Generator,
) -> tuple[HistoryRow, ...]:
    """Publish one discrete rounded history while preserving the declared masks.

    ``rng.choice`` samples the normalised bin masses themselves.  There is no continuous
    Student-t draw followed by clipping: clipping would pile out-of-support mass into the
    endpoint bins and would not be the bounded law that ``history_loglik`` evaluates.
    """
    row_tuple = tuple(rows)
    _check_row_order(row_tuple)
    previous_by_well: dict[str, PreviousObservation] = {}
    drawn: list[HistoryRow] = []

    for row in row_tuple:
        if row.reset:
            previous_by_well.pop(row.well_id, None)
        if not row.observed_valid:
            # Preserve the measured-data mask even if a caller supplied stale payload in
            # a model constructed without validation.
            payload = row.model_dump()
            payload.update(raw_value=None, bin_index=None)
            drawn.append(HistoryRow.model_validate(payload))
            continue

        grid = _grid_for(row, grids)
        predicted = _prediction_for(row, prediction)
        mu = conditional_mean(
            predicted,
            previous_by_well.get(row.well_id),
            row.month_index,
            noise.rho,
            False,
        )
        log_probabilities = bin_log_probs(
            grid,
            mu=mu,
            sigma=noise.sigma * row.sigma_multiplier,
            nu=noise.nu,
        )
        probabilities = np.exp(log_probabilities - logsumexp(log_probabilities))
        chosen = int(rng.choice(grid.n_bins, p=probabilities))
        published = float(grid.centers[chosen])

        payload = row.model_dump()
        payload.update(raw_value=published, bin_index=chosen)
        drawn.append(HistoryRow.model_validate(payload))
        previous_by_well[row.well_id] = (row.month_index, published, predicted)

    return tuple(drawn)


__all__ = ["draw_history"]
