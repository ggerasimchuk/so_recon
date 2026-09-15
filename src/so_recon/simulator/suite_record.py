"""E01.12 — the schemas of a suite: what was planned, what one attempt cost, what was published.

This module holds the RECORD and nothing that produces one. `simulator/suites.py` runs the
groups, `simulator/commands.py` holds the command bodies and `validation/e01_report.py`
reads what both of them left behind — and all three need the same `PlannedJob`,
`JobOutcome` and `SuiteReport`. Keeping those here is what lets the report module read a
published suite without importing the command module: before this split, `e01_report`
imported `commands` at module level and `commands.run_e01_report` had to import
`e01_report` *inside the function* to dodge the resulting cycle. A deferred import like
that is not a workaround, it is the proof that the seam is in the wrong place.

Two rules the fields themselves carry.

**An unmeasured quantity is `None`, never `0`.** A forward that ran inside a Julia
verification diagnostic has no peak RSS of its own and no measured CPU time; publishing a
zero for either makes the stage page read as if somebody had measured it and found nothing.
`measurement_method` says how each row was arrived at, and the renderer prints it.

**A failed attempt keeps its row and its spent time.** Plan 12.5 requires failures to stay
in the denominator of the failure rate, so nothing here deletes an outcome to make a rate
look better.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from so_recon.config.resources import ResourceProfileId
from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.validation.physics import PhysicsCheck

#: Where a suite leaves the record the stage validator reads.
SUITE_REPORT_FILENAME = "e01_suite.json"
SUITE_REPORT_SCHEMA_VERSION: Literal["e01-suite-1"] = "e01-suite-1"
JOB_PLAN_SCHEMA_VERSION: Literal["e01-jobs-1"] = "e01-jobs-1"

#: Where the planned matrix lives, relative to the repository root.
DEFAULT_JOB_PLAN_RELPATH = "configs/e01_jobs.json"

#: Where `benchmark-forward` leaves its attempt table, inside its own run directory.
BENCHMARK_FILENAME = "e01_benchmark.json"

SUITES: tuple[str, ...] = ("p0", "p1", "bo")

#: 12.5: fewer than five completed warm timings is not a p50/p90, it is a guess.
MIN_WARM_RUNS = 5

#: 12.5: above this coefficient of variation the report RECOMMENDS 10-20 repeats. It never
#: runs them: a larger sample is a larger budget and a budget is granted, not assumed.
WARM_CV_RECOMMENDATION_THRESHOLD = 0.25

#: Which checks the stage gate is allowed to be green without, and which it is not. An
#: exploratory check that did not run is a stated limitation; a mandatory one that did not
#: run is an unrun gate, and plan 12.9 requires zero of those in the stage matrix.
MANDATORY_CHECKS: frozenset[str] = frozenset(
    {
        "closed_cell_pvt",
        "closed_box_pvt",
        "hydrostatic",
        "segregation",
        "bl",
        "operations",
        "controls_calendar",
        "restart_round_trip",
        "worker_isolation",
        "five_spot",
        "five_spot_refinement",
        "bl_refinement",
        "p1_worlds",
    }
)


#: Task 13: the checks the BLACK-OIL suite is gated on, inside its own session.
#:
#: They are deliberately NOT in `MANDATORY_CHECKS`. That set is the oil-water stage matrix,
#: and `validation/e01_report.py` fails the stage for anything in it that is missing — so a
#: black-oil name there would make a stage with no black-oil session a FAILED stage, which is
#: exactly the coupling plan 13.5 forbids. `evaluate_suite` scores these as mandatory for the
#: session that produces them, which is how a failing capability exits non-zero and reaches
#: the stage report as `BO status: FAIL` instead of being green by omission.
BO_CHECKS: frozenset[str] = frozenset({"black_oil", "black_oil_restart"})

#: What `evaluate_suite` treats as a gate: the oil-water stage matrix, plus the black-oil
#: capability's own checks when a session produced them. A p0 or p1 session never produces a
#: `black_oil` check and is unaffected.
GATED_CHECKS: frozenset[str] = MANDATORY_CHECKS | BO_CHECKS


class CommandError(RuntimeError):
    """A command was asked for something it must refuse. Recorded as FAIL by `execute_run`."""


# --------------------------------------------------------------------------------------
# 12.1 the exit codes
# --------------------------------------------------------------------------------------


def forward_exit_code(status: str) -> int:
    """0 only when what was asked for happened; 2 for every refusal, partial or failure."""
    return 0 if status == "COMPLETE" else 2


def suite_exit_code(run_dir: Path) -> int:
    """The exit code a finished suite recorded, read back from its own report.

    A run directory with no suite report is not a zero: a suite that did not get far enough
    to write its record did not pass.
    """
    path = Path(run_dir) / SUITE_REPORT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    code = payload.get("exit_code")
    return int(code) if isinstance(code, int) else 1


# --------------------------------------------------------------------------------------
# 12.4 the planned matrix
# --------------------------------------------------------------------------------------


class PlannedJob(StrictModel):
    """One planned forward evaluation, exactly as `configs/e01_jobs.json` declares it."""

    job_id: str
    group: str
    kind: Literal["fixture", "forward", "resume", "world", "benchmark"]
    profile: ResourceProfileId
    expected_outcome: str
    output_request: str
    julia_threads: int
    cache_bypass: bool
    accounting: Literal["launcher", "ledger"]
    native_source: str | None
    scored_as: str
    declared_by_plan: bool
    parent_job_id: str | None = None
    note: str | None = None
    deferred_to: str | None = None


class SuitePlan(StrictModel):
    profile: ResourceProfileId
    wall_budget_s: int
    declared_count_without_retry: int
    groups: tuple[str, ...]
    jobs: tuple[PlannedJob, ...]
    status: Literal["RUN", "NOT_RUN"] = "RUN"
    not_run_reason: str | None = None

    def group_jobs(self, group: str) -> tuple[PlannedJob, ...]:
        """The jobs of one group that this suite really runs (a deferred one runs elsewhere)."""
        return tuple(job for job in self.jobs if job.group == group and job.deferred_to is None)

    def group_checks(self, group: str) -> frozenset[str]:
        """The check names the jobs of one group are scored as.

        Derived from the plan rather than from a second table, so a group that gains a job
        gains its check here too and a resumed session cannot carry forward a verdict that
        no longer belongs to the group it came from.
        """
        return frozenset(job.scored_as for job in self.group_jobs(group))


class JobPlan(StrictModel):
    schema_version: Literal["e01-jobs-1"]
    note: str
    tolerances_path: str
    solver_tolerance_hash: str
    counting: dict[str, Any]
    suites: dict[str, SuitePlan]


def load_job_plan(path: Path, paths: ProjectPaths) -> JobPlan:
    """Read the planned matrix and prove it is about the tolerances in force.

    The plan carries the digest of `configs/e01_tolerances.yml`. If the two disagree the
    matrix was written against a different gate, and running it would score today's results
    against yesterday's agreement without anybody noticing. That is a refusal.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommandError(f"cannot read the job plan {path}: {exc}") from exc
    plan = JobPlan.model_validate(payload)
    tolerance_path = paths.resolve(plan.tolerances_path)
    digest = sha256_file(tolerance_path)
    if digest != plan.solver_tolerance_hash:
        raise CommandError(
            f"{paths.relative(path)} declares solver_tolerance_hash "
            f"{plan.solver_tolerance_hash}, and {plan.tolerances_path} hashes to {digest}; a "
            "matrix planned against a different tolerance block is refused rather than run"
        )
    return plan


# --------------------------------------------------------------------------------------
# 12.5 what one attempt cost
# --------------------------------------------------------------------------------------


class JobOutcome(StrictModel):
    """One attempt's row of the benchmark table (plan 12.5).

    Every field is measured or explicitly absent. `cpu_s`, `peak_rss_bytes` and
    `output_bytes` are `None` on a row nobody measured them for — a launcher forward has no
    process of its own to weigh — and the renderer prints an em dash rather than a zero that
    would read as a measurement. A failed attempt keeps its row and its spent time: 12.5
    requires failures to stay in the denominator of the failure rate.
    """

    job_id: str
    group: str
    kind: str
    profile: str
    accounting: Literal["launcher", "ledger"]
    expected_outcome: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    wall_s: float
    cpu_s: float | None
    peak_rss_bytes: int | None
    output_bytes: int | None
    native_chunk_calls: int | None
    accepted_steps: int | None
    cut_steps: int | None
    nonlinear_iterations: int | None
    retry_count: int
    reason: str | None = None
    parent_job_id: str | None = None
    cold_import_s: float | None = None
    first_specialisation_s: float | None = None
    restart_read_s: float | None = None
    restart_write_s: float | None = None
    julia_threads: int | None = None
    cache_bypass: bool | None = None
    total_ram_bytes: int | None = None
    available_bytes: int | None = None
    swap_used_bytes: int | None = None
    memory_pressure: str | None = None
    measurement_method: str | None = None
    #: Where this attempt's published `result.json` is, project-relative, when it wrote one.
    #: It is what makes an attempt REUSABLE by a later session: the record reads back through
    #: `load_forward_result`, which re-proves its bytes, so a resumed session can skip the
    #: solver without taking the previous session's word for the numbers.
    result_record_path: str | None = None
    #: Set when a resumed session reused this attempt instead of re-entering the solver.
    reused_from_run_id: str | None = None

    @property
    def as_expected(self) -> bool:
        return self.status == self.expected_outcome


def warm_summary(seconds: Sequence[float]) -> dict[str, float | int | str]:
    """p50/p90 over the warm repeats, and the tail status they are honestly worth.

    `tail_status` is `exploratory` and stays `exploratory`: five samples fix a median well
    and a ninetieth percentile barely at all, and calling it anything else would invite a
    reader to plan a session on it.
    """
    if len(seconds) < MIN_WARM_RUNS:
        raise CommandError(
            f"at least {MIN_WARM_RUNS} completed warm timings are required for a p50/p90, "
            f"got {len(seconds)}"
        )
    x = np.asarray(seconds, dtype=np.float64)
    mean = float(x.mean())
    cv = float(x.std(ddof=1) / mean) if mean > 0.0 else 0.0
    return {
        "n": len(x),
        "p50_s": float(np.quantile(x, 0.5)),
        "p90_s": float(np.quantile(x, 0.9)),
        "min_s": float(x.min()),
        "max_s": float(x.max()),
        "mean_s": mean,
        "coefficient_of_variation": cv,
        "tail_status": "exploratory",
        "recommendation": (
            "10-20 repeats would be needed for a usable tail; this report does not run them "
            "without a separately accepted budget"
            if cv > WARM_CV_RECOMMENDATION_THRESHOLD
            else "five repeats are consistent; no further repeats recommended"
        ),
    }


# --------------------------------------------------------------------------------------
# the suite record
# --------------------------------------------------------------------------------------


class SuiteOutcome(StrictModel):
    """What a suite's jobs produced. The exit code is computed from it, never inside it."""

    jobs: tuple[JobOutcome, ...] = ()
    checks: tuple[PhysicsCheck, ...] = ()
    remaining_job_ids: tuple[str, ...] = ()
    stopped_reason: str | None = None
    benchmark: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    resumed_from_run_id: str | None = None
    reused_group_ids: tuple[str, ...] = ()


class SuiteReport(StrictModel):
    """The published record of one `verify-physics` session."""

    schema_version: Literal["e01-suite-1"] = SUITE_REPORT_SCHEMA_VERSION
    suite: str
    profile: str
    exit_code: int
    run_id: str
    command: str
    started_at: str
    finished_at: str
    git_commit: str | None
    git_dirty: bool | None
    environment_lock_hash: str
    tolerances_path: str
    tolerances_sha256: str
    job_plan_path: str
    job_plan_sha256: str
    planned_job_ids: tuple[str, ...]
    deferred_job_ids: tuple[str, ...]
    declared_count_without_retry: int
    jobs: tuple[JobOutcome, ...]
    checks: tuple[PhysicsCheck, ...]
    mandatory_checks: tuple[str, ...]
    exploratory_checks: tuple[str, ...]
    remaining_job_ids: tuple[str, ...]
    stopped_reason: str | None
    limitations: tuple[str, ...]
    benchmark: dict[str, Any]
    artifacts: dict[str, str]
    launcher_forwards: int
    ledger_forwards: int
    resumed_from_run_id: str | None = None
    reused_group_ids: tuple[str, ...] = ()


def relative_evidence(check: PhysicsCheck, paths: ProjectPaths) -> PhysicsCheck:
    """Turn a check's evidence paths into repository paths before it is persisted.

    `evaluate_physics` and `compare_refinement` record whatever path they were handed, and
    every caller hands them a path resolved under `ProjectPaths` — which is absolute. A
    record that reaches `reports/` carrying a home-directory prefix would be a machine
    fingerprint in a tracked file, and `tests/test_no_absolute_paths.py` would (rightly) fail
    on it. Fixed here, at the boundary the command owns, rather than in the evaluator: the
    evaluator has to OPEN those files, and a relative path is not openable without a root.
    """
    relative: list[str] = []
    for raw in check.evidence_paths:
        candidate = Path(raw)
        try:
            relative.append(paths.relative(candidate) if candidate.is_absolute() else raw)
        except ValueError:
            relative.append(candidate.name)
    return check.model_copy(update={"evidence_paths": tuple(relative)})


def evaluate_suite(outcome: SuiteOutcome, plan: SuitePlan) -> tuple[int, tuple[str, ...]]:
    """The exit code of a finished suite, and the limitations it is green in spite of.

    One mandatory check that FAILED is a failure of the stage and exits 1. A mandatory check
    that never ran, a declared job that was not reached, or an attempt whose status is not the
    outcome the matrix expected, is an incomplete session and exits 2 — plan 12.10's partial
    report. Everything else that did not happen is named as a limitation and does not move the
    code, because an exploratory benchmark is not a gate.
    """
    limitations: list[str] = []
    failed = [c.name for c in outcome.checks if c.status == "FAIL" and c.name in GATED_CHECKS]
    unrun = [c.name for c in outcome.checks if c.status == "NOT_RUN" and c.name in GATED_CHECKS]
    present = {c.name for c in outcome.checks}
    expected_here = {
        job.scored_as
        for job in plan.jobs
        if job.deferred_to is None and job.scored_as in GATED_CHECKS
    }
    missing = sorted(expected_here - present)
    surprised = [job.job_id for job in outcome.jobs if not job.as_expected]
    for check in outcome.checks:
        if check.status == "NOT_RUN" and check.name not in GATED_CHECKS:
            limitations.append(f"{check.name}: NOT_RUN — {check.reason}")
    for job in plan.jobs:
        if job.deferred_to is not None:
            limitations.append(
                f"{job.job_id}: declared in this suite and executed in suite "
                f"{job.deferred_to} — {job.note}"
            )
    if outcome.reused_group_ids:
        limitations.append(
            f"resumed from run {outcome.resumed_from_run_id}: groups "
            f"{list(outcome.reused_group_ids)} were not re-run in this session; their job rows "
            "and their check verdicts are carried forward from that session's published record "
            "(plan 12.6). `--replay` re-runs them."
        )
    if failed:
        return 1, tuple(limitations)
    if unrun or missing or outcome.remaining_job_ids or surprised:
        return 2, tuple(limitations)
    return 0, tuple(limitations)


def measured(values: Sequence[int | float | None]) -> list[int | float]:
    """Only the numbers somebody measured. An absent one is not a zero to sum."""
    return [v for v in values if v is not None]


def summed(values: Sequence[int | None]) -> int | None:
    """The total of the measured entries, or None when nothing was measured at all."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def summed_f(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(sum(present)) if present else None


def group_by(jobs: Sequence[JobOutcome]) -> dict[str, list[JobOutcome]]:
    out: dict[str, list[JobOutcome]] = {}
    for job in jobs:
        out.setdefault(job.group, []).append(job)
    return out


def read_suite_report(run_dir: Path) -> SuiteReport | None:
    """The published record of one session, or None when there is not one to read."""
    path = Path(run_dir) / SUITE_REPORT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        return SuiteReport.model_validate(payload)
    except ValueError:
        return None


__all__ = [
    "BENCHMARK_FILENAME",
    "BO_CHECKS",
    "GATED_CHECKS",
    "DEFAULT_JOB_PLAN_RELPATH",
    "JOB_PLAN_SCHEMA_VERSION",
    "MANDATORY_CHECKS",
    "MIN_WARM_RUNS",
    "SUITES",
    "SUITE_REPORT_FILENAME",
    "SUITE_REPORT_SCHEMA_VERSION",
    "WARM_CV_RECOMMENDATION_THRESHOLD",
    "CommandError",
    "JobOutcome",
    "JobPlan",
    "PlannedJob",
    "SuiteOutcome",
    "SuitePlan",
    "SuiteReport",
    "evaluate_suite",
    "forward_exit_code",
    "group_by",
    "load_job_plan",
    "measured",
    "read_suite_report",
    "relative_evidence",
    "suite_exit_code",
    "summed",
    "summed_f",
    "warm_summary",
]
