"""The E03 comparison matrix as a declared object (plan §10).

This module ASSEMBLES; it computes no new science. Zone products and truth-conditional
scores are `validation.ensemble_states`, the per-run rows and the world-averaging refusal
are `validation.learned_comparison`, and the residual diagnostic is
`validation.proposal_diagnostics`. What is fixed here is the shape of the evidence:

* **The matrix is enumerated from the preregistration, not from the results.**
  `planned_main_cells` takes the `ComparisonProtocol` (`ml.contracts`) and forms the §10.1
  main evaluation — parents × {B1, M} × particle counts × inference seeds, the 64 SMC runs
  of the frozen design. B0/Q, truth, T5, the second training seed and the ablations are
  counted ADDITIONALLY and never inside those cells.

* **No cell can vanish.** Every planned cell leaves an outcome: `RESULT`, `INCOMPLETE`,
  `FAILURE` or `NOT_RUN`, each of the last three with its reason. A row that matches no
  planned cell is published as `unplanned_rows` rather than dropped, and it makes the
  matrix partial — a campaign that ran something else than it registered has not completed
  the registered matrix. `completeness` is derived from the counts alone, so a partial
  matrix cannot present itself as complete (§10.1).

* **A beta<1 run is a diagnostic.** It occupies its cell as `INCOMPLETE` — visible, with
  its beta and algorithm status in the reason — and is excluded from `results_for`, so it
  cannot enter the accuracy curve (§10.4). Removing incomplete or failed cells before
  averaging and calling the remainder a win is exactly what `WorldAverage.evidence` makes
  impossible to do silently.

* **Aggregation is §5.5.** Seeds are averaged WITHIN a world, then worlds equally. The
  value is produced by `learned_comparison.average_over_worlds`, so its cross-method
  pooling refusal stands behind every number here.

* **Comparison needs one target.** Every result row of one parent must carry the SAME
  `scientific_target_identity` (`validation.e03_protocol`, §4.3) or the matrix is refused
  before any B1-against-M number exists.

* **T5 is a paired view.** A fine child belongs to its parent's world and never adds one
  (§3.3); a fine run that did not happen is NOT_RUN/resource-limited, never a PASS.

**A note on `posterior_claim`.** `inference.learned_loop` publishes that field as PROSE
(`"POSTERIOR"` / `"REFUSED: ..."`) while `ml.contracts` types it `bool`; both strings are
truthy, so reading it as a boolean reads a refusal as a claim. Nothing in this module ever
reads it. `cell_status_from_learned_smc_payload` derives the claim from `beta`,
`algorithm_status` and `ensemble_kind` through `e03_protocol.completed_posterior`, and
refuses a payload whose label disagrees with its own beta. The field itself is routed to
Task 12.

Nothing here starts a native job, reads the registry or touches the network (§12): every
function takes payloads and records the caller already loaded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from so_recon.ml.contracts import METHOD_IDS, ComparisonProtocol, MethodId
from so_recon.validation.e03_protocol import (
    DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
    POSTERIOR_ENSEMBLE_KIND,
    completed_posterior,
    ensemble_kind,
)
from so_recon.validation.learned_comparison import average_over_worlds

E03_COMPARISON_REPORT_SCHEMA = "e03-comparison-report-1"

#: §10.1: the main evaluation is B1 against M. Every other method is additional evidence.
MAIN_METHOD_IDS: tuple[MethodId, ...] = ("B1", "M")

#: §10.2: the frozen primary is PV-weighted MAE on the posterior mean over the eight
#: non-overlapping quadrants by layer at the final month; RMSE is secondary. Months 12/24
#: are temporal diagnostics, and layer aggregates and T4 remote zones are separate
#: diagnostics that never re-enter the primary score.
PRIMARY_METRIC = "mae"
SECONDARY_METRIC = "rmse"
PRIMARY_MONTH = 36
PRIMARY_SUPPORT = "eight_quadrants"
PRIMARY_ESTIMATOR = "posterior_mean"
DIAGNOSTIC_MONTHS: tuple[int, ...] = (12, 24)
DIAGNOSTIC_ONLY_SUPPORTS: tuple[str, ...] = ("layer_aggregates", "t4_remote_zones")

#: §11 PROMISING_STATE: the frozen main comparison is the N64 slice; N32 stays a
#: convergence/cost diagnostic and is never a way to pick the smaller error.
FROZEN_MAIN_PARTICLE_COUNT = 64

CellStatus = Literal["RESULT", "INCOMPLETE", "FAILURE", "NOT_RUN"]
CELL_STATUSES: tuple[CellStatus, ...] = ("RESULT", "INCOMPLETE", "FAILURE", "NOT_RUN")
Completeness = Literal["COMPLETE", "PARTIAL", "NOT_RUN"]
Evidence = Literal["COMPLETE", "PARTIAL"]


# --------------------------------------------------------------------------------------
# the planned matrix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class MatrixCell:
    """One planned cell of the §10.1 matrix: world × method × N × inference seed."""

    parent_id: str
    method_id: str
    n_particles: int
    inference_seed: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "method_id": self.method_id,
            "n_particles": self.n_particles,
            "inference_seed": self.inference_seed,
        }


def planned_main_cells(
    protocol: ComparisonProtocol,
    *,
    main_methods: Sequence[str] = MAIN_METHOD_IDS,
) -> tuple[MatrixCell, ...]:
    """Enumerate the main evaluation cells declared by the preregistration (§10.1).

    The cells come from the protocol, never from the results: this is what «all planned
    cells» means when the report later has to say which of them are missing.
    """
    if not main_methods:
        raise ValueError("the main evaluation names at least one method")
    registered = {method.method_id for method in protocol.methods}
    unknown = [method_id for method_id in main_methods if method_id not in registered]
    if unknown:
        raise ValueError(
            f"main method(s) {sorted(unknown)} are not registered by protocol "
            f"{protocol.protocol_id!r} (it declares {sorted(registered)}): the main matrix "
            "cannot contain a method the preregistration never named"
        )
    return tuple(
        MatrixCell(
            parent_id=parent.parent_id,
            method_id=str(method_id),
            n_particles=int(n_particles),
            inference_seed=int(seed),
        )
        for parent in protocol.parents
        for method_id in main_methods
        for n_particles in protocol.particle_counts
        for seed in protocol.inference_seeds
    )


# --------------------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellOutcome:
    """What the campaign has to say about ONE planned cell — including nothing at all."""

    cell: MatrixCell
    status: CellStatus
    reason: str
    row: Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            **self.cell.as_payload(),
            "status": self.status,
            "reason": self.reason,
            "has_row": self.row is not None,
        }


def _row_cell(row: Mapping[str, Any]) -> MatrixCell:
    missing = [
        key for key in ("parent_id", "method_id", "n_particles", "inference_seed") if key not in row
    ]
    if missing:
        raise ValueError(
            f"a comparison row omits {missing}: a row that cannot name its cell cannot be "
            "matched against the planned matrix"
        )
    return MatrixCell(
        parent_id=str(row["parent_id"]),
        method_id=str(row["method_id"]),
        n_particles=int(row["n_particles"]),
        inference_seed=int(row["inference_seed"]),
    )


def _row_status(row: Mapping[str, Any]) -> tuple[CellStatus, str]:
    kind = str(row.get("ensemble_kind", ""))
    if kind == POSTERIOR_ENSEMBLE_KIND:
        return "RESULT", "beta=1 COMPLETE posterior ensemble"
    if kind == DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND:
        beta = row.get("beta")
        status = row.get("algorithm_status")
        return (
            "INCOMPLETE",
            f"diagnostic partial ensemble (beta={beta!r}, algorithm_status={status!r}): a "
            "beta<1 or incomplete run is a diagnostic and never enters the accuracy curve",
        )
    raise ValueError(
        f"a main-matrix cell carries ensemble_kind {kind!r}: the main evaluation compares "
        "completed SMC posteriors, and an uncorrected prior or raw proposal ensemble is "
        "additional evidence reported under its own method, not a main cell"
    )


def _check_one_scientific_target(outcomes: Sequence[CellOutcome]) -> dict[str, str]:
    """§4.3: B1 and M are compared only on an EQUAL scientific target identity."""
    by_parent: dict[str, dict[str, MatrixCell]] = {}
    for outcome in outcomes:
        if outcome.status != "RESULT" or outcome.row is None:
            continue
        identity = outcome.row.get("scientific_target_identity")
        if identity is None:
            raise ValueError(
                f"result row for cell {outcome.cell} carries no scientific target identity: "
                "§4.3 requires the identity to be shown EQUAL before B1 and M are compared, "
                "and an unstated identity cannot be shown equal"
            )
        seen = by_parent.setdefault(outcome.cell.parent_id, {})
        seen.setdefault(str(identity), outcome.cell)
    identities: dict[str, str] = {}
    for parent_id, seen in by_parent.items():
        if len(seen) > 1:
            named = ", ".join(
                f"{value!r} ({cell.method_id})" for value, cell in sorted(seen.items())
            )
            raise ValueError(
                f"world {parent_id!r} carries {len(seen)} scientific target identities "
                f"({named}): the runs infer different posteriors and §4.3 refuses the "
                "comparison before any number is produced"
            )
        identities[parent_id] = next(iter(seen))
    return identities


@dataclass(frozen=True)
class ComparisonMatrix:
    """The assembled §10 matrix: every planned cell, with or without a result."""

    protocol_id: str
    preregistration_hash: str
    outcomes: tuple[CellOutcome, ...]
    unplanned_rows: tuple[Mapping[str, Any], ...] = ()
    scientific_target_identities: Mapping[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = dict.fromkeys(CELL_STATUSES, 0)
        for outcome in self.outcomes:
            tally[outcome.status] += 1
        return tally

    @property
    def n_planned(self) -> int:
        return len(self.outcomes)

    @property
    def complete(self) -> bool:
        """True only when EVERY planned cell has a result and nothing unplanned ran."""
        return not self.unplanned_rows and self.counts()["RESULT"] == self.n_planned

    @property
    def completeness(self) -> Completeness:
        """`NOT_RUN` only when NOTHING was attempted: a campaign that ran and failed, or
        that produced an incomplete run, is PARTIAL evidence and says so."""
        if self.complete:
            return "COMPLETE"
        if self.counts()["NOT_RUN"] == self.n_planned and not self.unplanned_rows:
            return "NOT_RUN"
        return "PARTIAL"

    @property
    def parent_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for outcome in self.outcomes:
            if outcome.cell.parent_id not in seen:
                seen.append(outcome.cell.parent_id)
        return tuple(seen)

    def cells_for(
        self, method_id: str, *, n_particles: int | None = None
    ) -> tuple[MatrixCell, ...]:
        return tuple(
            outcome.cell
            for outcome in self.outcomes
            if outcome.cell.method_id == method_id
            and (n_particles is None or outcome.cell.n_particles == n_particles)
        )

    def results_for(
        self, method_id: str, *, n_particles: int | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        """The rows that may enter an accuracy statement: completed posteriors only."""
        return tuple(
            outcome.row
            for outcome in self.outcomes
            if outcome.status == "RESULT"
            and outcome.row is not None
            and outcome.cell.method_id == method_id
            and (n_particles is None or outcome.cell.n_particles == n_particles)
        )

    def unfinished(self) -> tuple[CellOutcome, ...]:
        """Every cell that is not a result, with the reason it is not — never filtered."""
        return tuple(outcome for outcome in self.outcomes if outcome.status != "RESULT")

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": E03_COMPARISON_REPORT_SCHEMA,
            "protocol_id": self.protocol_id,
            "preregistration_hash": self.preregistration_hash,
            "primary": {
                "metric": PRIMARY_METRIC,
                "secondary_metric": SECONDARY_METRIC,
                "month": PRIMARY_MONTH,
                "support": PRIMARY_SUPPORT,
                "estimator": PRIMARY_ESTIMATOR,
                "diagnostic_months": list(DIAGNOSTIC_MONTHS),
                "diagnostic_only_supports": list(DIAGNOSTIC_ONLY_SUPPORTS),
            },
            "completeness": self.completeness,
            "complete": self.complete,
            "n_planned": self.n_planned,
            "counts": self.counts(),
            "cells": [outcome.as_payload() for outcome in self.outcomes],
            "unplanned_rows": [dict(row) for row in self.unplanned_rows],
            "scientific_target_identities": dict(self.scientific_target_identities),
        }


def assemble_comparison_matrix(
    protocol: ComparisonProtocol,
    rows: Sequence[Mapping[str, Any]],
    *,
    failures: Mapping[MatrixCell, str] | None = None,
    main_methods: Sequence[str] = MAIN_METHOD_IDS,
) -> ComparisonMatrix:
    """Match published rows to the planned cells and account for every one of them.

    Outcomes are not filtered by how they turned out: an incomplete run keeps its cell as
    `INCOMPLETE`, a declared failure keeps it as `FAILURE`, and a cell nobody ran is
    `NOT_RUN`. A row that matches no planned cell is published separately and makes the
    matrix partial rather than disappearing.
    """
    planned = planned_main_cells(protocol, main_methods=main_methods)
    planned_set = set(planned)
    declared_failures = dict(failures or {})
    unknown_failures = sorted(str(cell) for cell in declared_failures if cell not in planned_set)
    if unknown_failures:
        raise ValueError(
            f"declared failure(s) name cells outside the planned matrix: {unknown_failures}"
        )
    by_cell: dict[MatrixCell, Mapping[str, Any]] = {}
    unplanned: list[Mapping[str, Any]] = []
    for row in rows:
        cell = _row_cell(row)
        if cell not in planned_set:
            unplanned.append(row)
            continue
        if cell in by_cell:
            raise ValueError(
                f"cell {cell} is filled twice: one world/method/N/seed has one run, and two "
                "results for it mean the rows describe something other than this matrix"
            )
        by_cell[cell] = row
    outcomes: list[CellOutcome] = []
    for cell in planned:
        published = by_cell.get(cell)
        if published is not None:
            status, reason = _row_status(published)
            if cell in declared_failures:
                raise ValueError(
                    f"cell {cell} is declared failed and also carries a row: the campaign "
                    "record and the results disagree about what happened"
                )
            outcomes.append(CellOutcome(cell=cell, status=status, reason=reason, row=published))
        elif cell in declared_failures:
            reason = str(declared_failures[cell]).strip()
            if not reason:
                raise ValueError(f"the failure declared for cell {cell} carries no reason")
            outcomes.append(CellOutcome(cell=cell, status="FAILURE", reason=reason))
        else:
            outcomes.append(
                CellOutcome(
                    cell=cell,
                    status="NOT_RUN",
                    reason="no result was published for this planned cell",
                )
            )
    identities = _check_one_scientific_target(outcomes)
    return ComparisonMatrix(
        protocol_id=protocol.protocol_id,
        preregistration_hash=protocol.preregistration_hash,
        outcomes=tuple(outcomes),
        unplanned_rows=tuple(unplanned),
        scientific_target_identities=identities,
    )


def cell_status_from_learned_smc_payload(payload: Mapping[str, Any]) -> tuple[CellStatus, str]:
    """Derive one run's cell status from what `learned_smc.json` actually publishes.

    Reads `beta`, `algorithm_status` and `ensemble_kind` — the keys the producer writes
    with the types it writes them in — and NEVER `posterior_claim`, which
    `inference.learned_loop` publishes as prose while `ml.contracts` types it `bool`: both
    the claim and the refusal are truthy strings, so a boolean read of that field turns a
    refusal into a claim. The label is cross-checked against the beta it claims to
    summarise, so a payload that calls a beta<1 run a posterior is refused here.
    """
    beta = float(payload["beta"])
    algorithm_status = str(payload["algorithm_status"])
    label = str(payload["ensemble_kind"])
    expected = ensemble_kind(beta, algorithm_status)
    if label != expected:
        raise ValueError(
            f"the run publishes ensemble_kind {label!r} for beta={beta!r} and "
            f"algorithm_status={algorithm_status!r}, where §4.4 gives {expected!r}: the "
            "label and the run disagree, and the label is not the evidence"
        )
    if completed_posterior(beta, algorithm_status):
        return "RESULT", "beta=1 COMPLETE posterior ensemble"
    return (
        "INCOMPLETE",
        f"beta={beta:g}, algorithm_status={algorithm_status}: a beta<1 or incomplete run "
        "is a diagnostic and never enters the accuracy curve (plan §10.4)",
    )


# --------------------------------------------------------------------------------------
# aggregation (§5.5, §10.2, §11)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldAverage:
    """A metric averaged seeds-within-world then equally over worlds, with its evidence."""

    method_id: str
    metric: str
    n_particles: int | None
    value: float
    world_values: Mapping[str, float]
    n_worlds: int
    missing_cells: tuple[MatrixCell, ...]
    evidence: Evidence

    def as_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "metric": self.metric,
            "n_particles": self.n_particles,
            "value": self.value,
            "world_values": dict(self.world_values),
            "n_worlds": self.n_worlds,
            "evidence": self.evidence,
            "missing_cells": [cell.as_payload() for cell in self.missing_cells],
        }


def world_average(
    matrix: ComparisonMatrix,
    *,
    method_id: str,
    metric: str = SECONDARY_METRIC,
    n_particles: int | None = FROZEN_MAIN_PARTICLE_COUNT,
) -> WorldAverage:
    """Average ONE method's completed results: seeds within a world, then equal worlds.

    The number itself is `learned_comparison.average_over_worlds`, so its refusal to pool
    methods stands behind this call too. Cells without a result are not quietly dropped:
    they are named in `missing_cells`, and `evidence` becomes `PARTIAL` — «partial
    execution остаётся partial evidence» (§10.1), and §10.4 forbids removing incomplete or
    failed cases before averaging and calling the remainder a win.
    """
    if not isinstance(method_id, str):
        raise ValueError(
            f"world_average takes exactly one method, got {method_id!r}: averaging across "
            "methods pools them, which plan §5.5 forbids"
        )
    rows = matrix.results_for(method_id, n_particles=n_particles)
    if not rows:
        raise ValueError(
            f"method {method_id!r} has no completed result at "
            f"n_particles={n_particles!r}: there is nothing to average, and an average "
            "over nothing is not a zero"
        )
    planned = matrix.cells_for(method_id, n_particles=n_particles)
    filled = {
        MatrixCell(
            parent_id=str(row["parent_id"]),
            method_id=method_id,
            n_particles=int(row["n_particles"]),
            inference_seed=int(row["inference_seed"]),
        )
        for row in rows
    }
    missing = tuple(cell for cell in planned if cell not in filled)
    by_world: dict[str, list[float]] = {}
    for row in rows:
        if metric not in row or row[metric] is None:
            raise ValueError(
                f"result row for world {row.get('parent_id')!r} carries no metric {metric!r}"
            )
        by_world.setdefault(str(row["parent_id"]), []).append(float(row[metric]))
    world_values = {parent_id: sum(values) / len(values) for parent_id, values in by_world.items()}
    return WorldAverage(
        method_id=method_id,
        metric=metric,
        n_particles=n_particles,
        value=average_over_worlds(rows, metric),
        world_values=world_values,
        n_worlds=len(world_values),
        missing_cells=missing,
        evidence="COMPLETE" if not missing else "PARTIAL",
    )


@dataclass(frozen=True)
class RelativeImprovement:
    """`1 - method/baseline` on the SAME worlds, with the weaker of the two evidences."""

    method_id: str
    baseline_id: str
    metric: str
    value: float
    method_value: float
    baseline_value: float
    n_worlds: int
    evidence: Evidence

    def as_payload(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "baseline_id": self.baseline_id,
            "metric": self.metric,
            "value": self.value,
            "method_value": self.method_value,
            "baseline_value": self.baseline_value,
            "n_worlds": self.n_worlds,
            "evidence": self.evidence,
        }


def relative_improvement(method: WorldAverage, baseline: WorldAverage) -> RelativeImprovement:
    """The §11 PROMISING_STATE ratio, refused unless both averages cover ONE world set."""
    if method.metric != baseline.metric:
        raise ValueError(
            f"a relative improvement compares one metric: {method.metric!r} against "
            f"{baseline.metric!r}"
        )
    if method.method_id == baseline.method_id:
        raise ValueError(f"method and baseline are both {method.method_id!r}")
    if method.n_particles != baseline.n_particles:
        raise ValueError(
            f"the two averages use N={method.n_particles!r} and N={baseline.n_particles!r}: "
            "the frozen main comparison is at one particle count, and a ratio across two is "
            "an accuracy claim built on a convergence diagnostic (plan §11)"
        )
    if set(method.world_values) != set(baseline.world_values):
        raise ValueError(
            "the two averages cover different worlds "
            f"({sorted(method.world_values)} against {sorted(baseline.world_values)}): a "
            "ratio of averages over different world sets is not a relative improvement"
        )
    if baseline.value <= 0.0:
        raise ValueError(
            f"baseline {baseline.method_id!r} averages {baseline.value!r}: a relative "
            "improvement against a non-positive baseline error is uninterpretable, and "
            "§11 asks for an absolute-error diagnostic instead"
        )
    evidence: Evidence = (
        "COMPLETE"
        if method.evidence == "COMPLETE" and baseline.evidence == "COMPLETE"
        else "PARTIAL"
    )
    return RelativeImprovement(
        method_id=method.method_id,
        baseline_id=baseline.method_id,
        metric=method.metric,
        value=1.0 - method.value / baseline.value,
        method_value=method.value,
        baseline_value=baseline.value,
        n_worlds=method.n_worlds,
        evidence=evidence,
    )


# --------------------------------------------------------------------------------------
# cost (§10.4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColdStartCost:
    """§10.4: `C_cold = C_context/corpus + C_preprocess + C_training + C_online + C_verif`."""

    context_corpus_s: float
    preprocess_s: float
    all_training_trials_s: float
    online_s: float
    verification_s: float

    @property
    def shared_s(self) -> float:
        """Everything that is NOT the per-object online run — the amortizable part."""
        return (
            self.context_corpus_s
            + self.preprocess_s
            + self.all_training_trials_s
            + self.verification_s
        )

    @property
    def cold_start_s(self) -> float:
        """The headline for a single object: nothing is amortized away."""
        return self.shared_s + self.online_s

    def amortized_s(self, k: int) -> float:
        """`C_shared / K + C_online`, valid only for genuinely compatible later objects."""
        if k < 1:
            raise ValueError(f"amortization needs at least one object, got K={k!r}")
        return self.shared_s / k + self.online_s

    def as_payload(self) -> dict[str, Any]:
        return {
            "context_corpus_s": self.context_corpus_s,
            "preprocess_s": self.preprocess_s,
            "all_training_trials_s": self.all_training_trials_s,
            "online_s": self.online_s,
            "verification_s": self.verification_s,
            "shared_s": self.shared_s,
            "cold_start_s": self.cold_start_s,
        }


def break_even_runs(*, shared_s: float, b1_online_s: float, m_online_s: float) -> int | None:
    """`ceil(C_shared / (C_B1_online - C_M_online))`, or None when there is no saving.

    The break-even EXISTS only when the online saving is positive (§10.4); without one no
    number of future objects repays the shared cost, and returning a large integer instead
    of None would present that as a threshold. It is a computed threshold under the stated
    assumptions, never a count of future applications.
    """
    saving = b1_online_s - m_online_s
    if saving <= 0.0:
        return None
    return math.ceil(shared_s / saving)


@dataclass(frozen=True)
class MethodCampaignCost:
    """What one method cost STANDALONE and what it actually cost in this campaign."""

    method_id: str
    actual_campaign_s: float
    standalone_replay_s: float | None
    cache_hits_from_other_methods: int = 0


def campaign_cost_report(costs: Sequence[MethodCampaignCost]) -> dict[str, Any]:
    """Publish both numbers per method (§10.4): a shared cache makes nothing free.

    A method that consumed forwards another method paid for must state its standalone
    replay cost; otherwise the campaign number alone would report the second method run as
    nearly free, which is an artefact of the disk cache and not a property of the method.
    """
    if not costs:
        raise ValueError("a campaign cost report names at least one method")
    methods: dict[str, Any] = {}
    for cost in costs:
        if cost.cache_hits_from_other_methods > 0 and cost.standalone_replay_s is None:
            raise ValueError(
                f"method {cost.method_id!r} reused {cost.cache_hits_from_other_methods} "
                "cached evaluations from another method but reports no standalone replay "
                "cost: the shared cache would make it look free (plan §10.4)"
            )
        methods[cost.method_id] = {
            "actual_campaign_s": cost.actual_campaign_s,
            "standalone_replay_s": cost.standalone_replay_s,
            "cache_hits_from_other_methods": cost.cache_hits_from_other_methods,
        }
    return {
        "schema_version": E03_COMPARISON_REPORT_SCHEMA,
        "methods": methods,
        "note": (
            "actual_campaign_s is what this campaign spent with its shared cache; "
            "standalone_replay_s is what the method costs on its own. Neither replaces "
            "the other (plan §10.4)"
        ),
    }


# --------------------------------------------------------------------------------------
# T5 (§3.3)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class T5Pair:
    """A paired coarse/fine view of ONE pre-designated evaluation parent (§3.3)."""

    parent_id: str
    fine_status: Literal["RESULT", "NOT_RUN"]
    reason: str | None = None


def t5_outcomes(pairs: Sequence[T5Pair]) -> tuple[dict[str, Any], ...]:
    """Report each pair; a fine child that did not run is NOT_RUN, never a PASS."""
    outcomes: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.fine_status == "RESULT":
            outcomes.append(
                {
                    "parent_id": pair.parent_id,
                    "status": "PAIRED",
                    "reason": pair.reason or "coarse and fine truth both available",
                    "certifies_grid_accuracy": True,
                    "adds_independent_world": False,
                }
            )
            continue
        if not (pair.reason or "").strip():
            raise ValueError(
                f"T5 parent {pair.parent_id!r} has no fine run and states no reason: an "
                "unrun fine child is NOT_RUN/resource-limited and must say which (§3.3)"
            )
        outcomes.append(
            {
                "parent_id": pair.parent_id,
                "status": "NOT_RUN",
                "reason": pair.reason,
                "certifies_grid_accuracy": False,
                "adds_independent_world": False,
            }
        )
    return tuple(outcomes)


def independent_world_count(matrix: ComparisonMatrix, t5_pairs: Sequence[T5Pair] = ()) -> int:
    """The number of INDEPENDENT worlds: the matrix's parents, and T5 adds none (§3.3).

    A fine child is a paired view of a parent already counted. Naming a T5 parent that the
    matrix does not evaluate is refused rather than counted, because such a pair could only
    enter the world count as a ninth observation.
    """
    parents = set(matrix.parent_ids)
    outside = sorted({pair.parent_id for pair in t5_pairs} - parents)
    if outside:
        raise ValueError(
            f"T5 pair(s) name {outside}, which is not an evaluation parent of protocol "
            f"{matrix.protocol_id!r}: a fine child belongs to a parent of this matrix and "
            "never adds an independent world (plan §3.3)"
        )
    return len(parents)


# --------------------------------------------------------------------------------------
# the frozen configuration
# --------------------------------------------------------------------------------------


def verify_comparison_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check the frozen `configs/e03_experiments.json` comparison block (§10).

    The declared `main_matrix_smc_runs` must EQUAL the product of its own factors —
    evaluation parents × main methods × particle counts × inference seeds — so the 64 of
    §10.1 cannot be restated as a smaller number under the same name. Pure config
    arithmetic: nothing is started here (§12).
    """
    comparison = payload.get("comparison")
    corpus = payload.get("corpus")
    if not isinstance(comparison, Mapping) or not isinstance(corpus, Mapping):
        raise ValueError("the experiments config carries no comparison/corpus blocks")
    splits = dict(dict(corpus["scientific"])["splits"])
    n_parents = int(splits["evaluation"])
    methods = tuple(str(name) for name in comparison["methods"])
    main_methods = tuple(str(name) for name in comparison["main_methods"])
    unknown = sorted(set(methods) - set(METHOD_IDS))
    if unknown:
        raise ValueError(f"the config declares method(s) {unknown} the contracts do not know")
    absent = sorted(set(main_methods) - set(methods))
    if absent:
        raise ValueError(
            f"main method(s) {absent} are not among the declared methods {list(methods)}"
        )
    additional = tuple(str(name) for name in comparison["additional_methods"])
    if set(additional) | set(main_methods) != set(methods) or set(additional) & set(main_methods):
        raise ValueError(
            f"the declared methods {list(methods)} are not partitioned by main "
            f"{list(main_methods)} and additional {list(additional)}: §10.1 counts B0/Q and "
            "the ablations ADDITIONALLY, so every method is one or the other, never both "
            "and never neither"
        )
    counts = tuple(int(value) for value in comparison["particle_counts"])
    seeds = tuple(int(value) for value in comparison["inference_seeds"])
    expected = n_parents * len(main_methods) * len(counts) * len(seeds)
    declared = int(comparison["main_matrix_smc_runs"])
    if declared != expected:
        raise ValueError(
            f"the config declares main_matrix_smc_runs={declared} but its own factors give "
            f"{expected} ({n_parents} evaluation parents x {len(main_methods)} main methods "
            f"x {len(counts)} particle counts x {len(seeds)} inference seeds): a matrix "
            "cannot be renamed smaller than the design it registers (plan §10.1)"
        )
    if int(comparison["primary_month"]) != PRIMARY_MONTH:
        raise ValueError(
            f"the frozen primary month is {PRIMARY_MONTH}, the config declares "
            f"{comparison['primary_month']!r}"
        )
    if str(comparison["primary_support"]) != PRIMARY_SUPPORT:
        raise ValueError(f"the frozen primary support is {PRIMARY_SUPPORT!r}")
    for key, frozen in (("primary_metric", PRIMARY_METRIC), ("secondary_metric", SECONDARY_METRIC)):
        if str(comparison[key]) != frozen:
            raise ValueError(
                f"the frozen {key} is {frozen!r}, the config declares {comparison[key]!r}"
            )
    if int(comparison["frozen_main_particle_count"]) != FROZEN_MAIN_PARTICLE_COUNT:
        raise ValueError(
            f"the frozen main comparison is N={FROZEN_MAIN_PARTICLE_COUNT}, the config "
            f"declares {comparison['frozen_main_particle_count']!r}"
        )
    if FROZEN_MAIN_PARTICLE_COUNT not in counts:
        raise ValueError(
            f"the frozen main comparison is N={FROZEN_MAIN_PARTICLE_COUNT}, which the "
            f"config's particle counts {list(counts)} do not contain"
        )
    curve = dict(comparison["learning_curve"])
    expansion = tuple(int(value) for value in curve["registered_expansion"])
    if len(expansion) != 2 or expansion[0] >= expansion[1]:
        raise ValueError(
            f"the registered corpus expansion must be one increasing step, got {list(expansion)}"
        )
    if int(splits["train"]) != expansion[0]:
        raise ValueError(
            f"the registered expansion starts at {expansion[0]}, but the train split is "
            f"{splits['train']}: the learning-curve step must start from the corpus that exists"
        )
    for size in (int(value) for value in curve["not_registered"]):
        if size <= expansion[1]:
            raise ValueError(
                f"corpus size {size} is listed as not registered but is not larger than the "
                f"registered expansion {expansion[1]}"
            )
    return {
        "protocol_id": str(comparison["protocol_id"]),
        "methods": methods,
        "main_methods": main_methods,
        "additional_methods": additional,
        "particle_counts": counts,
        "inference_seeds": seeds,
        "evaluation_parents": n_parents,
        "main_matrix_smc_runs": declared,
        "primary_metric": str(comparison["primary_metric"]),
        "secondary_metric": str(comparison["secondary_metric"]),
        "primary_month": int(comparison["primary_month"]),
        "primary_support": str(comparison["primary_support"]),
        "diagnostic_months": tuple(int(value) for value in comparison["diagnostic_months"]),
        "registered_expansion": expansion,
        "not_registered": tuple(int(value) for value in curve["not_registered"]),
    }


__all__ = [
    "CELL_STATUSES",
    "CellOutcome",
    "CellStatus",
    "ColdStartCost",
    "ComparisonMatrix",
    "DIAGNOSTIC_MONTHS",
    "DIAGNOSTIC_ONLY_SUPPORTS",
    "E03_COMPARISON_REPORT_SCHEMA",
    "FROZEN_MAIN_PARTICLE_COUNT",
    "MAIN_METHOD_IDS",
    "MatrixCell",
    "MethodCampaignCost",
    "PRIMARY_ESTIMATOR",
    "PRIMARY_METRIC",
    "PRIMARY_MONTH",
    "PRIMARY_SUPPORT",
    "RelativeImprovement",
    "SECONDARY_METRIC",
    "T5Pair",
    "WorldAverage",
    "assemble_comparison_matrix",
    "break_even_runs",
    "campaign_cost_report",
    "cell_status_from_learned_smc_payload",
    "independent_world_count",
    "planned_main_cells",
    "relative_improvement",
    "t5_outcomes",
    "verify_comparison_config",
    "world_average",
]
