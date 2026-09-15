"""Dated saturation-log likelihood with joint date marginalisation and shared bias.

An uncertain log date is a latent choice shared by every interval in its ``date_group``.
For a candidate date we first multiply (sum in log space) all interval likelihoods, and
only then integrate over the date.  Marginalising every interval independently would let
one physical logging run happen on several dates at once and would invent information.

The log interpretation bias is the single ``NoiseTheta.log_bias`` coordinate carried by
the bundle.  It shifts every saturation interval in g-space.  Its Gaussian prior belongs
to the latent density; this module evaluates the conditional likelihood at the supplied
bias and does not integrate or count that prior a second time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from so_recon.inference.contracts import (
    F64,
    PROBABILITY_SUM_TOLERANCE,
    LoglikResult,
    LogRow,
    ModelObservations,
    NoiseTheta,
    ObservationSupportMismatch,
)
from so_recon.observation.bins import BinGrid, bin_log_probs, to_g_space
from so_recon.registry.hashing import sha256_json

# So-log noise is fixed independently of the history's inferred sigma/rho.  Bias remains
# inferred through NoiseTheta.log_bias.  All quantities are in g-space.
LOG_SIGMA = 0.03
LogGrids = Mapping[str, BinGrid]


def log_date_mixture(log_terms: F64, probabilities: F64) -> float:
    """Stable ``log(sum_d p_d exp(log_terms_d))`` with no silent renormalisation."""
    terms = np.asarray(log_terms, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    if terms.ndim != 1 or weights.ndim != 1 or terms.shape != weights.shape or terms.size == 0:
        raise ValueError(
            f"date mixture needs two non-empty one-dimensional arrays of one shape, got "
            f"{terms.shape} and {weights.shape}"
        )
    if np.isnan(terms).any() or np.isposinf(terms).any():
        raise ValueError("date log terms may be finite or -inf, never NaN or +inf")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError(f"date probabilities must be finite and non-negative, got {weights!r}")
    total = math.fsum(float(value) for value in weights)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"date probabilities must sum to one within {PROBABILITY_SUM_TOLERANCE}, got {total!r}"
        )
    keep = weights > 0.0
    if not bool(np.any(keep)):
        raise ValueError("a date mixture must give positive mass to at least one date")
    return float(logsumexp(np.log(weights[keep]) + terms[keep]))


def _active_groups(
    rows: Sequence[LogRow], *, likelihood_only: bool
) -> dict[str, tuple[LogRow, ...]]:
    seen: set[str] = set()
    groups: defaultdict[str, list[LogRow]] = defaultdict(list)
    bias_groups: set[str] = set()
    for row in rows:
        if row.observation_id in seen:
            raise ValueError(f"duplicate log observation_id {row.observation_id!r}")
        seen.add(row.observation_id)
        if not row.valid or (likelihood_only and row.use != "likelihood"):
            continue
        groups[row.date_group].append(row)
        bias_groups.add(row.bias_group)
    if len(bias_groups) > 1:
        raise ValueError(
            f"one bias_group per log bundle, got {sorted(bias_groups)}: another group "
            "requires another anchored latent, not reuse of the same bias"
        )

    out: dict[str, tuple[LogRow, ...]] = {}
    for group, members in groups.items():
        reference = members[0]
        for member in members[1:]:
            if (
                member.date_times_s != reference.date_times_s
                or member.date_weights != reference.date_weights
            ):
                raise ValueError(
                    f"rows in date_group {group!r} must name the same candidate dates and "
                    "probabilities: the group is one latent logging date"
                )
        out[group] = tuple(members)
    return out


def _grid_for(row: LogRow, grids: LogGrids) -> BinGrid:
    try:
        return grids[row.bias_group]
    except KeyError as exc:
        raise ValueError(
            f"log row {row.observation_id!r} names bias group {row.bias_group!r}, but the "
            f"available grids are {sorted(grids)}"
        ) from exc


def _so_at(row: LogRow, time_s: float, prediction: ModelObservations) -> float:
    key = (row.observation_id, time_s)
    try:
        return float(prediction.so_support[key])
    except KeyError as exc:
        raise ObservationSupportMismatch(
            f"log row {row.observation_id!r} requests date {time_s!r}, but model "
            f"{prediction.model_hash} has no saved state on that observation support; "
            "E02 does not interpolate unrequested physical states"
        ) from exc


def _row_log_probability(
    row: LogRow,
    time_s: float,
    prediction: ModelObservations,
    noise: NoiseTheta,
    grids: LogGrids,
) -> float:
    grid = _grid_for(row, grids)
    if row.bin_index is None or row.bin_index >= grid.n_bins:
        raise ObservationSupportMismatch(
            f"valid log row {row.observation_id!r} names bin {row.bin_index!r}, outside "
            f"the {grid.n_bins} bins of group {row.bias_group!r}"
        )
    so = _so_at(row, time_s, prediction)
    mu = float(to_g_space(np.asarray(so, dtype=np.float64))) + noise.log_bias
    return float(bin_log_probs(grid, mu=mu, sigma=LOG_SIGMA, nu=noise.nu)[row.bin_index])


def _logs_semantic_hash(rows: Sequence[LogRow], grids: LogGrids) -> str:
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
            "version": "dated-so-log-1",
        }
    )


def logs_loglik(
    rows: Sequence[LogRow],
    prediction: ModelObservations,
    noise: NoiseTheta,
    grids: LogGrids,
    *,
    observation_hash: str | None = None,
) -> LoglikResult:
    """Score each date group as one coupled likelihood factor.

    Rows used to condition the prior or reserved as diagnostics are retained in the input
    but excluded here, preventing the same information from being multiplied twice.
    """
    row_tuple = tuple(rows)
    groups = _active_groups(row_tuple, likelihood_only=True)
    factors: list[float] = []
    for members in groups.values():
        reference = members[0]
        joint_by_date = np.asarray(
            [
                math.fsum(
                    _row_log_probability(member, time_s, prediction, noise, grids)
                    for member in members
                )
                for time_s in reference.date_times_s
            ],
            dtype=np.float64,
        )
        factors.append(
            log_date_mixture(joint_by_date, np.asarray(reference.date_weights, dtype=np.float64))
        )

    value = -math.inf if any(term == -math.inf for term in factors) else math.fsum(factors)
    return LoglikResult(
        value=value,
        value_in_support=value != -math.inf,
        terms=tuple(factors),
        terms_in_support=tuple(term != -math.inf for term in factors),
        n_used=len(factors),
        observation_hash=observation_hash or _logs_semantic_hash(row_tuple, grids),
    )


def draw_logs(
    rows: Sequence[LogRow],
    prediction: ModelObservations,
    noise: NoiseTheta,
    grids: LogGrids,
    rng: np.random.Generator,
) -> tuple[LogRow, ...]:
    """Draw bins after one shared date draw per group, preserving invalid rows."""
    row_tuple = tuple(rows)
    groups = _active_groups(row_tuple, likelihood_only=False)
    updates: dict[str, int] = {}
    for members in groups.values():
        reference = members[0]
        chosen_date_index = int(rng.choice(len(reference.date_times_s), p=reference.date_weights))
        time_s = reference.date_times_s[chosen_date_index]
        for member in members:
            grid = _grid_for(member, grids)
            so = _so_at(member, time_s, prediction)
            mu = float(to_g_space(np.asarray(so, dtype=np.float64))) + noise.log_bias
            log_probabilities = bin_log_probs(grid, mu=mu, sigma=LOG_SIGMA, nu=noise.nu)
            probabilities = np.exp(log_probabilities - logsumexp(log_probabilities))
            updates[member.observation_id] = int(rng.choice(grid.n_bins, p=probabilities))

    drawn: list[LogRow] = []
    for row in row_tuple:
        payload = row.model_dump()
        payload["bin_index"] = updates.get(row.observation_id) if row.valid else None
        drawn.append(LogRow.model_validate(payload))
    return tuple(drawn)


__all__ = [
    "LOG_SIGMA",
    "LogGrids",
    "draw_logs",
    "log_date_mixture",
    "logs_loglik",
]
