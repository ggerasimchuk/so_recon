"""E01.12 — the registered suites: one worker, one session, the groups in plan order.

`commands.py` holds the six command bodies; everything that RUNS a group of E01's matrix
lives here. The split is the seam `validation/e01_report.py` needed: the report reads the
schemas in `simulator/suite_record.py` and never the module that drives a solver.

Three things this module owns that a library call does not.

**A session that outlives one job.** `simulate` runs ONE case: it has no view across jobs
and so no place to hold the memory drift Task 8 measures between them, nor the session disk
budget COMPUTE §7 puts at 5 GiB. `Session` below owns exactly one `MemoryDriftMonitor` for
the whole suite, takes its baseline after the first job has warmed the worker, ACTS on the
decision — a warning and a recycle on the first drift, a `RESOURCE_FAILURE` when a fresh
process drifts again — and carries the session's own written bytes, because the per-job
ledgers each start at zero and none of them can see the total. At most one worker is ever
alive (COMPUTE §5).

**A resume that resumes.** Plan 12.6: read the ledger, read what the previous session
finished, and run the jobs that are missing in the order the plan fixes. `load_resume_state`
reads the parent session's own published `e01_suite.json`; a group all of whose planned jobs
came back as the matrix expected and whose checks all PASSED is carried forward rather than
re-run, and a forward whose exact model and case bytes already have a COMPLETE attempt on
the parent's ledger is reused rather than re-solved. `--replay` is the one thing that turns
both off. A FAILED check is never carried: a new session's budget is not permission to skip
a failed case.

**A measured number, or none at all.** A forward that ran inside a Julia verification
diagnostic has no process of its own to weigh, so its `cpu_s` and `output_bytes` are `None`
rather than `0`; what CAN be measured around it — the peak RSS of this process tree while
the diagnostic runs — is sampled and recorded. `measurement_method` says, on every row,
which of the two kinds of number it is.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from so_recon.config.resources import ResourceProfile
from so_recon.config.schema import ProjectConfig
from so_recon.environment.resources import ResourceSnapshot
from so_recon.environment.resources import probe_resources as probe_machine
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RUN_RECORD_SCHEMA_VERSION, RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.simulator.budget import BudgetLedger, BudgetStop, disk_stop, load_ledger
from so_recon.simulator.case_io import (
    case_manifest_sha256,
    compute_model_hash,
    compute_static_hash,
    load_case,
    read_array,
)
from so_recon.simulator.contracts import CaseBundle, ForwardResult, OutputRequest
from so_recon.simulator.forward import (
    DriftDecision,
    MemoryDriftMonitor,
    SolverConfig,
    check_requested_outputs,
    predicted_output_bytes,
    resume,
    simulate,
)
from so_recon.simulator.julia_bridge import JuliaRunError, SubprocessJuliaLauncher, find_julia
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    CONNECTIONS_FILENAME,
    HEADLINE_BALANCE,
    MONTHLY_FILENAME,
    RESERVOIR_BALANCE,
    RESULT_FILENAME,
    STATES_FILENAME,
    load_forward_result,
    write_forward_result,
)
from so_recon.simulator.suite_record import (
    MIN_WARM_RUNS,
    CommandError,
    JobOutcome,
    PlannedJob,
    SuiteOutcome,
    SuitePlan,
    read_suite_report,
    warm_summary,
)
from so_recon.simulator.worker import PersistentJuliaWorker, request_stop_after_chunk
from so_recon.synthetic.acceptance import score_p1_world
from so_recon.synthetic.p1 import (
    P1_PARENTS,
    P1Design,
    closed_preflight_world,
    output_request,
    preflight_output_request,
    render_p1,
)
from so_recon.synthetic.world_io import (
    SUITE_MANIFEST_FILENAME,
    build_p1_case,
    world_row,
    write_suite_manifest,
    write_world,
)
from so_recon.validation.fixtures import publish_fixture
from so_recon.validation.physics import (
    BLACKOIL_GATES,
    DEFAULT_BLACKOIL_TOLERANCES_RELPATH,
    CommonSupport,
    PhysicsCheck,
    bl_cell_average,
    cartesian_zone_ids,
    common_metrics,
    compare_refinement,
    evaluate_physics,
    load_blackoil_tolerances,
)

log = logging.getLogger(__name__)

#: The five-spot refinement support of plan 10.8: an 8x8 partition both meshes tile exactly.
SUPPORT_SIDE = 8

#: The solver configuration every E01 verification job runs under. Five days is the P1
#: generator's own step and fifteen iterations is the base limit SPEC 3.3 fixes before a
#: registered retry may raise it.
E01_SOLVER = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)

#: Which published verification case each P0 worker group runs on. A P1 world is 36 months
#: and 512 cells — outside P0_VERIFY's ceiling of twelve intervals — and the P1 policy is
#: written for exactly 36 months, so a shortened one does not exist. What does exist by the
#: time these groups run is the case the analytic diagnostic already published: a real
#: `CaseBundle`, validated and written by the production writer, with three report steps.
#: `segregation` is a closed column that really moves over thirty days, which is what makes a
#: restart round trip a comparison of something rather than of two static fields.
P0_RESTART_CASE = "gravity_segregation"
P0_ISOLATION_A = "gravity_segregation"
P0_ISOLATION_B = "closed_box"


# --------------------------------------------------------------------------------------
# the session: ONE worker, ONE ledger, ONE memory drift monitor
# --------------------------------------------------------------------------------------


class ResourceFailure(RuntimeError):
    """The session refused to continue: a drift a fresh process still shows (plan 8.7)."""


def _bounded_probe(session_dir: Path, pid: int | None = None) -> Callable[[], ResourceSnapshot]:
    def probe() -> ResourceSnapshot:
        return probe_machine(pid, session_dir)

    return probe


class Session:
    """One live Julia worker at a time, with the cross-job memory drift decision (12.3).

    `simulate` runs one case and cannot see between jobs; the drift Task 8 measures is
    between them, so the monitor belongs to the session loop and is owned here. The baseline
    is taken AFTER the first job, which is the one that pays for the specialisation a warm
    worker legitimately keeps.
    """

    def __init__(
        self,
        *,
        julia: Path,
        project: Path,
        session_dir: Path,
        profile: ResourceProfile,
        paths: ProjectPaths,
        ledger_dir: Path,
        session_id: str,
        resume_from: Path | None = None,
    ) -> None:
        self._julia = julia
        self._project = project
        self._session_dir = session_dir
        self._profile = profile
        self._paths = paths
        self._ledger_dir = ledger_dir
        self._session_id = session_id
        self._resume_from = resume_from
        self.ledgers: dict[str, BudgetLedger] = {}
        self._worker: PersistentJuliaWorker | None = None
        self._monitor: MemoryDriftMonitor | None = None
        self.decisions: list[DriftDecision] = []
        self.recycles = 0
        self.cold_start_s: float | None = None
        self._output_bytes = 0

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    # ------------------------------------------------------------------ the session disk

    @property
    def output_bytes(self) -> int:
        """Everything this SESSION has written, across every per-job ledger it opened."""
        return self._output_bytes

    def record_output(self, output_bytes: int | None) -> None:
        """Add one finished job's published bytes to the session's running total."""
        self._output_bytes += int(output_bytes or 0)

    def admit_output(self, *, predicted_bytes: int) -> None:
        """Refuse the next job when the SESSION's disk budget cannot hold what it predicts.

        `BudgetLedger.reserve` enforces three cumulative caps, and the 5 GiB disk budget of
        COMPUTE §7 is one of them — through `disk_stop(session_output_bytes=...)`. This
        suite deliberately opens ONE ACCOUNT PER RUN OF A MODEL (see `Session.ledger`: the
        isolation quadruple, the restart triple and five warm repeats each need more than
        the two attempts per model hash a single ledger allows), and the price of that is
        that every one of those accounts sees a single entry and none of them can see the
        session's total. The wall bound and the forward count the suite already keeps
        itself; this is the third cap, kept here, with the same `disk_stop` and the same
        `DISK_FAILURE_RESERVE_BYTES` left free so that whatever happens next can still be
        written down.
        """
        stop = disk_stop(
            self._profile,
            session_output_bytes=self._output_bytes,
            predicted_bytes=predicted_bytes,
            free_bytes=probe_machine(None, self._session_dir).disk_free_bytes,
        )
        if stop is not None:
            raise stop

    def ledger(self, job_id: str) -> BudgetLedger:
        """The budget account of ONE run of one model.

        `BudgetLedger.reserve` allows SPEC 3.3's two attempts per physical model and no
        more, which is the whole point of it: a third reservation against the same model
        hash is a protocol failure, not a third try. E01 deliberately runs some models more
        than twice — an A -> B -> A -> A isolation quadruple, a restart's continuous run
        against its prefix and its continuation, five warm repeats of `world41` — and none
        of those is a retry. So each of them is its OWN session, exactly as Task 8's
        restart suite decided (`tests/integration/test_e01_restart.py:_ledger`), and every
        one of those sessions is written down.

        What that gives up is the ledger's own session-level wall and forward accumulation,
        because a fresh session starts a fresh clock. The suite keeps both itself:
        `SuiteRun.deadline_s` is the session's real wall bound and the planned matrix is its
        real forward count. Nothing is unaccounted — the per-job ledgers are all published
        under the run directory and the stage report sums them.
        """
        existing = self.ledgers.get(job_id)
        if existing is not None:
            return existing
        path = self._ledger_dir / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = _bounded_probe(self._session_dir)
        if self._resume_from is not None and not self.ledgers:
            # 12.6: the FIRST account of a resumed session continues the one the previous
            # command printed, so the chain is traceable and its cost is not reset.
            ledger = BudgetLedger.resume(
                parent_path=self._resume_from,
                path=path,
                session_id=f"{self._session_id}-{job_id}",
                probe=probe,
                profile=self._profile,
            )
        else:
            ledger = BudgetLedger.start(
                profile=self._profile,
                path=path,
                session_id=f"{self._session_id}-{job_id}",
                probe=probe,
            )
        self.ledgers[job_id] = ledger
        return ledger

    def ledger_paths(self, paths: ProjectPaths) -> dict[str, str]:
        return {
            f"ledger.{job_id}": paths.relative(ledger.path)
            for job_id, ledger in sorted(self.ledgers.items())
        }

    def worker(self) -> PersistentJuliaWorker:
        if self._worker is None:
            started = time.monotonic()
            self._worker = PersistentJuliaWorker(
                self._julia,
                self._project,
                self._session_dir,
                self._profile,
                paths=self._paths,
                probe=_bounded_probe(self._session_dir),
            )
            elapsed = time.monotonic() - started
            if self.cold_start_s is None:
                self.cold_start_s = elapsed
        return self._worker

    def _rss(self) -> int | None:
        if self._worker is None:
            return None
        snapshot = probe_machine(self._worker.pid, self._session_dir)
        return snapshot.process_rss_bytes

    def after_job(self) -> DriftDecision | None:
        """Observe the worker's retained memory and act on what the monitor decided."""
        rss = self._rss()
        if rss is None:
            return None
        if self._monitor is None:
            # The post-warm baseline: the first job paid for the specialisation.
            self._monitor = MemoryDriftMonitor(baseline_rss_bytes=rss)
            return None
        decision = self._monitor.observe(rss)
        self.decisions.append(decision)
        if decision.action == "recycle":
            log.warning("recycling the worker: %s", decision.reason)
            self.recycle()
            fresh = self._rss()
            if fresh is not None:
                self._monitor.rebaseline(fresh)
            self.recycles += 1
        elif decision.action == "stop":
            raise ResourceFailure(str(decision.reason))
        return decision

    def recycle(self) -> None:
        """Close the worker and start a fresh one. Never two alive at once (COMPUTE §5)."""
        self.close()
        self.worker()

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None


# --------------------------------------------------------------------------------------
# small measurement helpers
# --------------------------------------------------------------------------------------


def _states(result: ForwardResult, paths: ProjectPaths) -> dict[str, Any]:
    return {
        name: np.asarray(read_array(ref, paths), dtype=np.float64)
        for name, ref in result.states.items()
    }


def chunk_calls(result: ForwardResult) -> int:
    """How many native chunk calls one trajectory took. Never a count of worlds (12.4)."""
    raw = result.solver_metadata.get("chunk_diagnostics")
    if not raw:
        return 0
    try:
        payload = json.loads(raw)
    except ValueError:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        value = payload.get("n_chunks")
        return int(value) if isinstance(value, int | float) else 0
    return 0


def _outcome_from_result(
    job: PlannedJob,
    result: ForwardResult,
    *,
    wall_s: float,
    started_at: str,
    snapshot: ResourceSnapshot | None,
    restart_read_s: float | None = None,
    restart_write_s: float | None = None,
    cold_import_s: float | None = None,
) -> JobOutcome:
    cost = result.cost
    record = (
        str(Path(result.monthly_path).parent / RESULT_FILENAME)
        if result.status == "COMPLETE" and result.monthly_path is not None
        else None
    )
    return JobOutcome(
        job_id=job.job_id,
        group=job.group,
        kind=job.kind,
        profile=job.profile,
        accounting=job.accounting,
        expected_outcome=job.expected_outcome,
        status=result.status,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        wall_s=wall_s,
        cpu_s=cost.cpu_s,
        peak_rss_bytes=cost.peak_rss_bytes,
        output_bytes=cost.output_bytes,
        native_chunk_calls=chunk_calls(result),
        accepted_steps=cost.accepted_steps,
        cut_steps=cost.cut_steps,
        nonlinear_iterations=cost.nonlinear_iterations,
        retry_count=cost.retry_count,
        reason=result.reason,
        parent_job_id=job.parent_job_id,
        cold_import_s=cold_import_s,
        first_specialisation_s=None,
        restart_read_s=restart_read_s,
        restart_write_s=restart_write_s,
        julia_threads=job.julia_threads,
        cache_bypass=job.cache_bypass,
        total_ram_bytes=None if snapshot is None else snapshot.total_bytes,
        available_bytes=None if snapshot is None else snapshot.available_bytes,
        swap_used_bytes=None if snapshot is None else snapshot.swap_used_bytes,
        memory_pressure=None if snapshot is None else snapshot.pressure.status,
        measurement_method=cost.measurement_method,
        result_record_path=record,
    )


#: How the wall of a launcher row was arrived at, printed under the stage report's job
#: table. It is the caveat that makes the divided number readable rather than misleading.
LAUNCHER_MEASUREMENT_METHOD = (
    "verification diagnostic through the test launcher: the wall time is the WHOLE "
    "diagnostic's, divided evenly across the forwards it ran, the peak RSS is this process "
    "tree's measured maximum while that one diagnostic was running (shared by every forward "
    "in it, not attributed to one), and the solver counters are this fixture's own. CPU time "
    "and published bytes are not measured per forward here and are recorded absent rather "
    "than as zero (plan 12.4: accounted by the launcher, not the ledger)."
)

LAUNCHER_NO_FIXTURE_METHOD = (
    "verification diagnostic through the test launcher, with no per-fixture extraction in "
    "its report: the wall time is the whole diagnostic's divided evenly and the peak RSS is "
    "this process tree's measured maximum while it ran. This diagnostic asserts its claims "
    "natively and publishes no per-forward solver counters, so they are recorded absent "
    "rather than as zero."
)


def _launcher_outcome(
    job: PlannedJob,
    *,
    status: str,
    wall_s: float,
    started_at: str,
    fixture: Mapping[str, Any] | None,
    peak_rss_bytes: int | None = None,
    reason: str | None = None,
) -> JobOutcome:
    """The row of a forward that ran inside a verification diagnostic, not on the ledger.

    Obligation of plan 12.4 and of Tasks 6, 9 and 10: these forwards are accounted by the
    test launcher, separately, and they are NOT charged to a `BudgetLedger` — inventing a
    `JobDescriptor` and a case digest for them to charge would add fiction. Their solver
    counters are the diagnostic's own, so the row says what really happened.

    What it does NOT say is a number nobody measured. `cpu_s` and `output_bytes` are per
    forward and this route measures neither, so they are `None` and the report prints an em
    dash; `peak_rss_bytes` is the sampled maximum of this process tree while the diagnostic
    ran, which IS measured — shared across its forwards, and said so in
    `measurement_method`.
    """
    solver: Mapping[str, Any] | None = None
    chunks: int | None = None
    if fixture is not None:
        extraction = fixture.get("extraction")
        if isinstance(extraction, Mapping):
            raw = extraction.get("solver")
            if isinstance(raw, Mapping):
                solver = raw
            chunk = extraction.get("chunk")
            if isinstance(chunk, Mapping):
                chunks = len(chunk.get("dt_s", []))
    return JobOutcome(
        job_id=job.job_id,
        group=job.group,
        kind=job.kind,
        profile=job.profile,
        accounting="launcher",
        expected_outcome=job.expected_outcome,
        status=status,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        wall_s=wall_s,
        cpu_s=None,
        peak_rss_bytes=peak_rss_bytes,
        output_bytes=None,
        native_chunk_calls=chunks,
        accepted_steps=None if solver is None else int(solver.get("accepted_steps", 0)),
        cut_steps=None if solver is None else int(solver.get("cut_steps", 0)),
        nonlinear_iterations=(
            None if solver is None else int(solver.get("nonlinear_iterations", 0))
        ),
        retry_count=0,
        reason=reason,
        parent_job_id=job.parent_job_id,
        julia_threads=job.julia_threads,
        cache_bypass=job.cache_bypass,
        measurement_method=(
            LAUNCHER_MEASUREMENT_METHOD if fixture is not None else LAUNCHER_NO_FIXTURE_METHOD
        ),
    )


def not_run_outcome(job: PlannedJob, reason: str) -> JobOutcome:
    """A job that never started. Its zeros are real: nothing ran, so nothing was spent."""
    return JobOutcome(
        job_id=job.job_id,
        group=job.group,
        kind=job.kind,
        profile=job.profile,
        accounting=job.accounting,
        expected_outcome=job.expected_outcome,
        status="NOT_RUN",
        wall_s=0.0,
        cpu_s=0.0,
        peak_rss_bytes=0,
        output_bytes=0,
        native_chunk_calls=0,
        accepted_steps=0,
        cut_steps=0,
        nonlinear_iterations=0,
        retry_count=0,
        reason=reason,
        parent_job_id=job.parent_job_id,
        julia_threads=job.julia_threads,
        cache_bypass=job.cache_bypass,
        measurement_method="nothing ran, so every count is a measured zero",
    )


def _gate(
    name: str,
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
    gates: Sequence[tuple[str, str]],
    *,
    evidence: Sequence[str] = (),
    hashes: Mapping[str, str] | None = None,
) -> PhysicsCheck:
    """Score a set of measured numbers against the frozen tolerance block.

    Used for the checks that have no entry in the evaluator's fixture registry — the
    operational suite, the restart round trip, the isolation triple, the P1 parent set. The
    THRESHOLDS are the same frozen file every registered fixture is scored against; nothing
    here invents one.
    """
    failures = [
        f"{metric}={metrics[metric]:.6g} exceeds {threshold}={thresholds[threshold]:g}"
        for metric, threshold in gates
        if metric in metrics and metrics[metric] > thresholds[threshold]
    ]
    unmeasured = [metric for metric, _ in gates if metric not in metrics]
    if unmeasured:
        return PhysicsCheck(
            name=name,
            status="NOT_RUN",
            metrics={},
            thresholds={t: float(thresholds[t]) for _, t in gates},
            input_hashes=dict(hashes or {}),
            evidence_paths=tuple(evidence),
            reason=f"{name}: {sorted(unmeasured)} were not measured",
        )
    return PhysicsCheck(
        name=name,
        status="FAIL" if failures else "PASS",
        metrics={k: float(v) for k, v in metrics.items()},
        thresholds={t: float(thresholds[t]) for _, t in gates},
        input_hashes=dict(hashes or {}),
        evidence_paths=tuple(evidence),
        reason=f"{name}: " + "; ".join(failures) if failures else None,
    )


# --------------------------------------------------------------------------------------
# 12.6 what a previous session already finished
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeState:
    """What the session behind `--resume-ledger` finished, read from its own artifacts.

    Plan 12.6's sequence is `ledger / hash lock / config -> check completed outputs ->
    missing jobs in the previous order`. The ledger the operator names lives inside a run
    directory, and that directory holds the session's published `e01_suite.json`: the job
    rows, the check verdicts and the artifact paths. This reads all three.

    A group is COMPLETE here only when every planned job of it came back as the matrix
    EXPECTED and every check it is scored by PASSED. A failed check is never complete: 12.6
    says in as many words that a new session's budget is not permission to skip a failed
    case, and a resumed session that carried a FAIL forward would be laundering it.
    """

    run_dir: Path
    run_id: str
    suite: str
    completed_groups: frozenset[str]
    outcomes: tuple[JobOutcome, ...]
    checks: tuple[PhysicsCheck, ...]
    artifacts: Mapping[str, str]
    ledgers: tuple[BudgetLedger, ...]

    def jobs_of(self, group: str) -> tuple[JobOutcome, ...]:
        return tuple(job for job in self.outcomes if job.group == group)

    def checks_of(self, names: frozenset[str]) -> tuple[PhysicsCheck, ...]:
        return tuple(check for check in self.checks if check.name in names)

    def outcome(self, job_id: str) -> JobOutcome | None:
        return next((job for job in self.outcomes if job.job_id == job_id), None)

    def already_complete(self, *, model_hash: str, case_sha256: str) -> bool:
        """Whether the parent session already ran THIS physical model on THESE bytes.

        `BudgetLedger.is_already_complete` is the authority and this is its production
        caller: the key is the model hash together with the digest of the published case
        manifest, so a regenerated case is different work however familiar its model hash
        looks, and a pending or failed attempt is never a skip.
        """
        return any(
            ledger.is_already_complete(model_hash=model_hash, case_sha256=case_sha256)
            for ledger in self.ledgers
        )


def load_resume_state(
    ledger_path: Path, *, suite: str, plan: SuitePlan, paths: ProjectPaths
) -> ResumeState | None:
    """Read the session that printed `ledger_path`, or None when it published no record.

    `--resume-ledger` takes the path a previous command printed, which is
    `<run>/ledgers/<job>.json`; the session's record is its grandparent's
    `e01_suite.json`. A ledger that does not sit inside a session that published one is not
    an error here — `open_suite_run` has already refused a file that is not a ledger — it
    simply means there is nothing completed to carry, and every group runs.
    """
    run_dir = Path(ledger_path).resolve().parent.parent
    report = read_suite_report(run_dir)
    if report is None or report.suite != suite:
        return None
    ledgers: list[BudgetLedger] = []
    probe = _bounded_probe(run_dir)
    for path in sorted((run_dir / "ledgers").glob("*.json")):
        try:
            ledgers.append(BudgetLedger.read(path, probe=probe))
        except (OSError, ValueError):  # a half-written ledger is not evidence of completion
            continue
    by_id = {job.job_id: job for job in report.jobs}
    verdicts = {check.name: check for check in report.checks}
    completed: set[str] = set()
    for group in plan.groups:
        planned = plan.group_jobs(group)
        if not planned:
            continue
        if not all(
            (found := by_id.get(job.job_id)) is not None and found.status == job.expected_outcome
            for job in planned
        ):
            continue
        names = plan.group_checks(group)
        scored = [verdicts.get(name) for name in names if name in verdicts]
        if not scored or any(check is None or check.status != "PASS" for check in scored):
            continue
        completed.add(group)
    return ResumeState(
        run_dir=run_dir,
        run_id=report.run_id,
        suite=report.suite,
        completed_groups=frozenset(completed),
        outcomes=report.jobs,
        checks=report.checks,
        artifacts=dict(report.artifacts),
        ledgers=tuple(ledgers),
    )


# --------------------------------------------------------------------------------------
# what a group runner is given
# --------------------------------------------------------------------------------------


@dataclass
class SuiteRun:
    """Everything a group needs, and the accumulator every group writes into."""

    cfg: ProjectConfig
    paths: ProjectPaths
    ctx: RunContext
    log: logging.Logger
    suite: str
    plan: SuitePlan
    profile: ResourceProfile
    tolerances: dict[str, float]
    julia: Path
    session: Session
    deadline_s: float
    jobs: list[JobOutcome] = field(default_factory=list)
    checks: list[PhysicsCheck] = field(default_factory=list)
    benchmark: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    #: What the session behind `--resume-ledger` already finished, or None for a fresh one.
    resume: ResumeState | None = None
    #: The explicit replay of plan 12.6. With it nothing is reused and every group re-runs.
    replay: bool = False
    reused_groups: list[str] = field(default_factory=list)

    def planned(self, job_id: str) -> PlannedJob:
        for job in self.plan.jobs:
            if job.job_id == job_id:
                return job
        raise CommandError(f"{job_id!r} is not a planned job of suite {self.suite!r}")

    def group_jobs(self, group: str) -> tuple[PlannedJob, ...]:
        return tuple(job for job in self.plan.jobs if job.group == group)

    def out_of_time(self) -> bool:
        return time.monotonic() >= self.deadline_s

    def remaining(self) -> tuple[str, ...]:
        done = {job.job_id for job in self.jobs}
        return tuple(
            job.job_id
            for job in self.plan.jobs
            if job.job_id not in done and job.deferred_to is None
        )

    # ------------------------------------------------------------------------- 12.6 resume

    def reuses(self) -> ResumeState | None:
        """The resumed session's completed work, unless an explicit replay forbids reuse."""
        return None if self.replay else self.resume

    def skips(self, group: str) -> bool:
        state = self.reuses()
        return state is not None and group in state.completed_groups

    def carry(self, group: str) -> None:
        """Take one completed group's rows, verdicts and artifacts from the parent session.

        Nothing is re-scored here and nothing is invented: the rows and the `PhysicsCheck`s
        are the ones that session published, and the artifacts it named (the cases the P0
        worker groups run on, among them) come with them so the groups that DO run can find
        what the skipped one produced.
        """
        state = self.reuses()
        if state is None:
            return
        parent = state.run_id
        self.jobs.extend(
            job.model_copy(update={"reused_from_run_id": parent}) for job in state.jobs_of(group)
        )
        self.checks.extend(state.checks_of(self.plan.group_checks(group)))
        for key, value in state.artifacts.items():
            self.artifacts.setdefault(key, value)
        self.reused_groups.append(group)

    def reusable_outcome(self, job: PlannedJob, case: CaseBundle) -> JobOutcome | None:
        """The parent session's row for this job, when its exact result is still valid.

        12.6: a production repeat does NOT run the solver when a valid immutable matching
        result already exists. "Matching" is checked against the parent's LEDGER — the model
        hash and the digest of the published case manifest, through
        `BudgetLedger.is_already_complete` — rather than against the job id alone, so a case
        that was regenerated, or a job whose attempt failed or never resolved, runs again.
        """
        state = self.reuses()
        if state is None:
            return None
        if job.cache_bypass:
            # 12.4: the warm repeats and the thread comparison bypass the result cache by
            # design — their whole subject is the cost of running the solver again.
            return None
        previous = state.outcome(job.job_id)
        if previous is None or previous.status != job.expected_outcome:
            return None
        if not state.already_complete(
            model_hash=case.model_hash, case_sha256=case_manifest_sha256(case)
        ):
            return None
        return previous.model_copy(
            update={
                "reused_from_run_id": state.run_id,
                "measurement_method": (
                    f"reused from run {state.run_id}: that session's ledger holds a COMPLETE "
                    "attempt on this exact model hash and case manifest digest, so the solver "
                    "was not entered again (plan 12.6). Every cost below is that attempt's."
                ),
            }
        )


SuiteRunner = Callable[[SuiteRun], SuiteOutcome]


# --------------------------------------------------------------------------------------
# the launcher groups: a Julia verification diagnostic, published and scored
# --------------------------------------------------------------------------------------


class _TreePeakRss:
    """The maximum RSS of this process tree while a blocking diagnostic runs.

    `launcher.launch` blocks until the Julia process exits, so the only moment at which its
    memory can be weighed is WHILE it runs — which is why the launcher rows used to publish
    a peak RSS of zero. `probe_resources(None, ...)` sums this process and every descendant,
    and the diagnostic is a descendant, so the maximum over the samples is a real,
    conservative peak for the tree over the diagnostic's life. It is the TREE's number,
    shared by every forward that diagnostic ran, and `measurement_method` says exactly that
    rather than letting it read as one forward's own high-water mark.

    A sample that cannot be read leaves the peak unchanged; a peak nothing could be read for
    stays `None`, because "not measured" and "zero" are different facts (SPEC 18.4).
    """

    def __init__(self, session_dir: Path, *, interval_s: float) -> None:
        self._session_dir = session_dir
        self._interval_s = max(interval_s, 0.05)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="e01-tree-rss", daemon=True)
        self.peak_bytes: int | None = None

    def _sample(self) -> None:
        try:
            rss = probe_machine(None, self._session_dir).process_rss_bytes
        except Exception:  # a probe that failed is not a measurement of zero
            return
        if rss is not None and (self.peak_bytes is None or rss > self.peak_bytes):
            self.peak_bytes = rss

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def __enter__(self) -> _TreePeakRss:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._sample()


@dataclass(frozen=True)
class LaunchMeasurement:
    """What one run of a verification diagnostic produced, and what it cost the tree."""

    payload: dict[str, Any]
    wall_s: float
    peak_rss_bytes: int | None


def _launch(
    run: SuiteRun, script: str, flag: str, label: str, *, extra: Sequence[str] = ()
) -> LaunchMeasurement:
    out_path = run.ctx.run_dir / "verification" / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    launcher = SubprocessJuliaLauncher(
        run.julia, run.paths.julia, timeout_s=run.profile.job_timeout_s
    )
    started = time.monotonic()
    with _TreePeakRss(run.session.session_dir, interval_s=run.profile.poll_interval_s) as sampler:
        launcher.launch(run.paths.root / script, [flag, *extra], out_path)
    elapsed = time.monotonic() - started
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "ok"):
        raise JuliaRunError(f"{label}: the diagnostic reported {payload.get('status')!r}")
    run.artifacts[f"verification.{label}"] = run.paths.relative(out_path)
    return LaunchMeasurement(payload=payload, wall_s=elapsed, peak_rss_bytes=sampler.peak_bytes)


def _publish(run: SuiteRun, report: Mapping[str, Any], name: str, label: str) -> Path:
    fixture = report["fixtures"][name]
    _result, result_dir = publish_fixture(
        dict(fixture), run.paths, label, dict(report), world=run.suite
    )
    return result_dir


def _outputs(result_dir: Path) -> dict[str, Path]:
    return {
        "states": result_dir / STATES_FILENAME,
        "balances": result_dir / BALANCES_FILENAME,
        "connections": result_dir / CONNECTIONS_FILENAME,
        "monthly": result_dir / MONTHLY_FILENAME,
    }


def _run_analytic(run: SuiteRun) -> None:
    """9.5-9.7: the closed cell, the closed box, the column, segregation and Buckley-Leverett."""
    started_at = datetime.now(UTC).isoformat()
    launched = _launch(run, "julia/verification/analytic.jl", "--test-analytic", "analytic")
    report = launched.payload
    jobs = run.group_jobs("analytic")
    per_job = launched.wall_s / max(len(jobs), 1)
    directories: dict[str, Path] = {}
    for job in jobs:
        native = str(job.native_source).split("#")[-1]
        label = f"{run.suite}-{job.job_id}"
        directories[native] = _publish(run, report, native, label)
        run.artifacts[f"case.{job.job_id}"] = run.paths.relative(
            run.paths.artifacts / f"case-{label}.json"
        )
        run.jobs.append(
            _launcher_outcome(
                job,
                status=str(report["fixtures"][native]["status"]),
                wall_s=per_job,
                started_at=started_at,
                fixture=report["fixtures"][native],
                peak_rss_bytes=launched.peak_rss_bytes,
            )
        )
    for name in ("closed_cell_pvt", "closed_box_pvt", "hydrostatic", "segregation"):
        run.checks.append(evaluate_physics(name, _outputs(directories[name]), run.tolerances))
    bl_outputs = {f"states_{n}": directories[f"bl_{n}"] / STATES_FILENAME for n in (32, 64, 128)}
    bl_outputs["balances_128"] = directories["bl_128"] / BALANCES_FILENAME
    run.checks.append(evaluate_physics("bl", bl_outputs, run.tolerances))


def _run_operations(run: SuiteRun) -> None:
    """10.4-10.6: mixing, crossflow, roles, the bottom-hole probes and the three boundaries."""
    started_at = datetime.now(UTC).isoformat()
    launched = _launch(run, "julia/verification/operations.jl", "--test-operations", "operations")
    report = launched.payload
    jobs = run.group_jobs("operations")
    per_job = launched.wall_s / max(len(jobs), 1)
    metrics: dict[str, float] = {}
    evidence: list[str] = []
    for job in jobs:
        native = str(job.native_source).split("#")[-1]
        result_dir = _publish(run, report, native, f"{run.suite}-{job.job_id}")
        outputs = _outputs(result_dir)
        published = load_forward_result(result_dir / RESULT_FILENAME, run.paths)
        # A classified result publishes NOTHING — `bhp_infeasible` is meant to come back
        # CONTROL_INFEASIBLE, and a well that could not hold its demanded control has no
        # states to score. Scoring the absence would be scoring a case that did not run;
        # what the refusal itself is checked for is its CLASSIFICATION, below.
        if published.status == "COMPLETE":
            evidence.append(run.paths.relative(outputs["balances"]))
            # The subset every forward owes. The drift and connection gates of a CLOSED
            # fixture do not apply to a sector that is producing, and scoring one against
            # them would be scoring it on the opposite of what it demonstrates.
            scored = common_metrics(outputs["states"], outputs["balances"], run.tolerances)
            for key, value in scored.items():
                metrics[key] = max(metrics.get(key, 0.0), float(value))
        run.jobs.append(
            _launcher_outcome(
                job,
                status=published.status,
                wall_s=per_job,
                started_at=started_at,
                fixture=report["fixtures"][native],
                peak_rss_bytes=launched.peak_rss_bytes,
                reason=published.reason,
            )
        )
    # The one structural claim of 10.5: the infeasible probe really was classified as such,
    # and the feasible one was not. A zero here is a requirement, not a tolerance.
    statuses = {job.job_id: job.status for job in run.jobs if job.group == "operations"}
    metrics["bhp_classification_errors"] = float(
        int(statuses.get("bhp_infeasible") != "CONTROL_INFEASIBLE")
        + int(statuses.get("bhp_feasible") != "COMPLETE")
    )
    thresholds = dict(run.tolerances)
    thresholds["bhp_classification_errors_max"] = 0.0
    run.checks.append(
        _gate(
            "operations",
            metrics,
            thresholds,
            (
                ("balance_cumulative_relative", "balance_cumulative_relative_max"),
                ("balance_step_median_relative", "balance_step_median_relative_max"),
                ("saturation_sum_abs", "saturation_sum_abs_max"),
                ("saturation_bound_violation", "saturation_bound_slack"),
                ("bhp_classification_errors", "bhp_classification_errors_max"),
            ),
            evidence=evidence,
        )
    )


#: The structural claims the controls diagnostic makes and asserts natively, restated here
#: so the gate is bound to a number this side can read rather than to Julia's own @test
#: alone. They are STRUCTURAL — a masked completion isolates a well, a closed completion
#: takes the crossflow away — and not tolerances anybody may widen; they mirror the
#: diagnostic's own constants exactly as `refinement.jl`'s symmetry gate is mirrored.
CONTROLS_ISOLATION_RATE_RATIO_MAX = 0.01
CONTROLS_CROSSFLOW_FLUX_RATIO_MAX = 0.02


def _run_controls(run: SuiteRun) -> None:
    """The calendar and control diagnostic: seven small forwards in one launcher process.

    The diagnostic asserts its own claims natively — the real month lengths, the compiled
    intervals, the uptime and the monthly volumes the rate controls imply — and exits nonzero
    when one of them fails, which the launcher turns into a refusal here. What this check adds
    is the two STRUCTURAL ratios read back out of its report: a masked completion really does
    isolate a well, and closing both completions really does take the crossflow away. Both are
    scored here against the diagnostic's own constants, so deleting an assertion inside Julia
    would not quietly make this check green.
    """
    started_at = datetime.now(UTC).isoformat()
    launched = _launch(run, "julia/verification/fixtures.jl", "--test-controls", "controls")
    report = launched.payload
    jobs = run.group_jobs("controls")
    per_job = launched.wall_s / max(len(jobs), 1)
    for job in jobs:
        run.jobs.append(
            _launcher_outcome(
                job,
                status="COMPLETE",
                wall_s=per_job,
                started_at=started_at,
                fixture=None,
                peak_rss_bytes=launched.peak_rss_bytes,
            )
        )
    metrics: dict[str, float] = {"controls_forwards": float(len(jobs))}
    isolation = report.get("isolation")
    if isinstance(isolation, Mapping):
        open_rate = abs(float(isolation["open"]["liquid_rate_m3_day"]))
        shut_rate = abs(float(isolation["isolated"]["liquid_rate_m3_day"]))
        metrics["controls_isolation_rate_ratio"] = shut_rate / max(open_rate, 1e-12)
    crossflow = report.get("crossflow")
    if isinstance(crossflow, Mapping):
        coupled = abs(float(crossflow["coupled"]["well_segment_mass_flux_kg_s"][1]))
        isolated = abs(float(crossflow["isolated"]["well_segment_mass_flux_kg_s"][1]))
        metrics["controls_crossflow_flux_ratio"] = isolated / max(coupled, 1e-12)
    thresholds = dict(run.tolerances)
    thresholds["controls_isolation_rate_ratio_max"] = CONTROLS_ISOLATION_RATE_RATIO_MAX
    thresholds["controls_crossflow_flux_ratio_max"] = CONTROLS_CROSSFLOW_FLUX_RATIO_MAX
    run.checks.append(
        _gate(
            "controls_calendar",
            metrics,
            thresholds,
            (
                ("controls_isolation_rate_ratio", "controls_isolation_rate_ratio_max"),
                ("controls_crossflow_flux_ratio", "controls_crossflow_flux_ratio_max"),
            ),
            evidence=[run.artifacts["verification.controls"]],
        )
    )


def _run_refinement(run: SuiteRun) -> None:
    """10.7-10.8: the five-spot on two meshes, and Buckley-Leverett at two timesteps."""
    started_at = datetime.now(UTC).isoformat()
    launched = _launch(run, "julia/verification/refinement.jl", "--test-refinement", "refinement")
    report = launched.payload
    jobs = run.group_jobs("refinement")
    per_job = launched.wall_s / max(len(jobs), 1)
    directories: dict[str, Path] = {}
    for job in jobs:
        native = str(job.native_source).split("#")[-1]
        directories[native] = _publish(run, report, native, f"{run.suite}-{job.job_id}")
        run.jobs.append(
            _launcher_outcome(
                job,
                status=str(report["fixtures"][native]["status"]),
                wall_s=per_job,
                started_at=started_at,
                fixture=report["fixtures"][native],
                peak_rss_bytes=launched.peak_rss_bytes,
            )
        )
    coarse, fine = directories["five_spot_16"], directories["five_spot_48"]
    run.checks.append(evaluate_physics("five_spot", _outputs(coarse), run.tolerances))
    support = CommonSupport(
        name="five_spot_refinement",
        n_zones=SUPPORT_SIDE**2,
        coarse_zone_id=cartesian_zone_ids(16, 16, SUPPORT_SIDE),
        fine_zone_id=cartesian_zone_ids(48, 48, SUPPORT_SIDE),
    )
    run.checks.append(
        compare_refinement(
            {k: v for k, v in _outputs(coarse).items() if k in ("states", "monthly")},
            {k: v for k, v in _outputs(fine).items() if k in ("states", "monthly")},
            support,
            run.tolerances,
        )
    )
    run.checks.append(_bl_refinement_check(run, report))


def _bl_refinement_check(run: SuiteRun, report: Mapping[str, Any]) -> PhysicsCheck:
    """The timestep half of 10.8: does the analytic error grow when dt is halved?"""
    errors: dict[int, float] = {}
    for n_cells in (64, 128):
        fixture = report["fixtures"][f"bl_{n_cells}"]
        states = fixture["extraction"]["states"]
        sw = np.asarray(states["sw"][-1], dtype=np.float64)
        pore = np.asarray(states["pore_volume_m3"][-1], dtype=np.float64)
        total = float(pore.sum())
        edges = np.concatenate([[0.0], np.cumsum(pore) / total])
        reference = bl_cell_average(edges, 0.2)
        errors[n_cells] = float(np.sum(np.abs(sw - reference) * (pore / total)))
    ratio = errors[128] / max(errors[64], 1e-12)
    return _gate(
        "bl_refinement",
        {
            "bl_pv_l1_at_64_halfdt_pair": errors[64],
            "bl_pv_l1_at_128_halfdt": errors[128],
            "bl_refinement_ratio_64_to_128": ratio,
        },
        run.tolerances,
        (
            ("bl_pv_l1_at_128_halfdt", "bl_pv_l1_max_at_128"),
            ("bl_refinement_ratio_64_to_128", "bl_refinement_ratio_max"),
        ),
        evidence=[run.artifacts["verification.refinement"]],
    )


# --------------------------------------------------------------------------------------
# the ledger groups: real forwards through the production driver
# --------------------------------------------------------------------------------------


def _child_run(
    run: SuiteRun, command: str, work: Callable[[RunContext, logging.Logger], ForwardResult]
) -> tuple[RunContext, ForwardResult | None]:
    """Run one forward as its OWN recorded run, under the suite's run as its parent.

    `execute_run` is the only thing in this project that opens a run record and the only
    thing that maps an exception to FAIL, so a job that crashes leaves a FAIL record rather
    than a RUNNING one. Each job gets its own run directory, which is also what keeps a
    continuation from ever writing into its parent's result: `simulate` and `resume` derive
    the result directory from the run that asked for them.
    """
    holder: list[ForwardResult] = []

    def body(ctx: RunContext, child_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        holder.append(work(ctx, child_log))
        return "PASS", []

    ctx = execute_run(
        command=command,
        argv=("so-recon", command, f"--suite={run.suite}"),
        cfg=run.cfg,
        paths=run.paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION, "forward_result": "forward-1"},
        parent_run_ids=(run.ctx.run_id,),
    )
    return ctx, (holder[0] if holder else None)


def persist_result(result: ForwardResult, paths: ProjectPaths) -> ForwardResult:
    """Write the record beside its outputs and read it back through every check it owes."""
    if result.status != "COMPLETE" or result.monthly_path is None:
        return result
    record_path = paths.resolve(result.monthly_path).parent / RESULT_FILENAME
    write_forward_result(result, record_path)
    return load_forward_result(record_path, paths)


def _reusable_result(run: SuiteRun, outcome: JobOutcome) -> ForwardResult | None:
    """The published result of a reusable attempt, re-read through its own record.

    A record that no longer reads back is not a result: `load_forward_result` re-proves the
    bytes against the digests the record names, so a reuse either cites something that still
    verifies or does not happen.
    """
    if outcome.result_record_path is None:
        return None
    try:
        return load_forward_result(run.paths.resolve(outcome.result_record_path), run.paths)
    except Exception:
        return None


def _forward(
    run: SuiteRun, job: PlannedJob, case: CaseBundle, request: OutputRequest
) -> tuple[JobOutcome, ForwardResult | None]:
    reused = run.reusable_outcome(job, case)
    if reused is not None:
        # 12.6: a production repeat does not enter the solver when a valid immutable
        # matching result is already on the parent session's ledger. Before the worker,
        # deliberately: starting one would pay the import for work that will not happen.
        # "Valid" is checked and not assumed — the published record is read back through
        # `load_forward_result`, which re-proves its bytes against the digests it names.
        previous = _reusable_result(run, reused)
        if previous is not None or reused.status != "COMPLETE":
            run.log.info(
                "%s: reusing the %s attempt run %s published; the solver is not entered",
                job.job_id,
                reused.status,
                reused.reused_from_run_id,
            )
            run.session.record_output(reused.output_bytes)
            return reused, previous
        run.log.warning(
            "%s: run %s recorded a COMPLETE attempt but its published result no longer reads "
            "back; running it again rather than citing bytes that do not verify",
            job.job_id,
            reused.reused_from_run_id,
        )
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    # The SESSION's disk budget, which no single per-job ledger can see (see
    # `Session.admit_output`). A `BudgetStop` here stops the group and is recorded.
    run.session.admit_output(
        predicted_bytes=predicted_output_bytes(
            run.session.ledger(job.job_id),
            run.profile,
            n_cells=case.grid.n_cells,
            n_times=len(request.state_times_s),
            keep_native_restart=request.keep_native_restart,
            n_report_steps=len(case.report_edges_s) - 1,
        )
    )
    worker = run.session.worker()

    def work(ctx: RunContext, _log: logging.Logger) -> ForwardResult:
        return simulate(
            case,
            request,
            worker=worker,
            ctx=ctx,
            ledger=run.session.ledger(job.job_id),
            solver_config=E01_SOLVER,
        )

    child, result = _child_run(run, "forward", work)
    wall_s = time.monotonic() - started
    if result is None:
        # A worker that was terminated mid-job cannot be resynchronised, and every later
        # job would fail with the same protocol error rather than with its own outcome.
        # Close it here: the next job starts a fresh process and reports what IT found.
        run.session.close()
        outcome = not_run_outcome(
            job, f"the attempt did not produce a result: {'; '.join(child.record.notes)}"
        )
        return outcome.model_copy(update={"status": "RESOURCE_FAILURE", "wall_s": wall_s}), None
    result = persist_result(result, run.paths)
    check_requested_outputs(result, request)
    run.session.record_output(result.cost.output_bytes)
    snapshot = probe_machine(None, run.session.session_dir)
    return (
        _outcome_from_result(
            job,
            result,
            wall_s=wall_s,
            started_at=started_at,
            snapshot=snapshot,
            cold_import_s=run.session.cold_start_s,
        ),
        result,
    )


def _p0_case(run: SuiteRun, job_id: str) -> CaseBundle:
    """The published verification case a P0 worker group runs on.

    It is loaded from the manifest the analytic diagnostic published earlier in this same
    session, through `load_case`, which re-proves that the bytes still hash to the model they
    name. A group that runs before its case exists is a configuration error in the suite's
    group order and says so.
    """
    key = f"case.{job_id}"
    relative = run.artifacts.get(key)
    if relative is None:
        raise CommandError(
            f"suite {run.suite!r} asked for the published case of {job_id!r}, which no group "
            "has published yet; the analytic group runs before the worker groups"
        )
    return load_case(run.paths.resolve(relative), run.paths)


def _run_restart(run: SuiteRun) -> None:
    """The restart round trip: one continuous run against a prefix continued in a NEW worker.

    The parent's result is never written into. `resume` builds its own job in its own run
    directory and re-extracts from the native bytes the prefix left behind, which is why the
    stopped prefix survives as the immutable checkpoint (Task 8, SPEC §3.3).
    """
    case = _p0_case(run, P0_RESTART_CASE)
    request = OutputRequest(
        state_times_s=tuple(case.report_edges_s), keep_native_restart=True, chunk_months=1
    )

    outcome, continuous = _forward(run, run.planned("restart_continuous"), case, request)
    run.jobs.append(outcome)
    run.session.after_job()

    # The prefix: the same case, stopped after the first completed report step.
    stop_path = request_stop_after_chunk(
        run.session.session_dir,
        reason="E01.12 restart round trip: stop after the first completed report step",
        after_completed_time_s=float(case.report_edges_s[1]),
    )
    prefix_started = time.monotonic()
    prefix_outcome, prefix = _forward(run, run.planned("restart_prefix"), case, request)
    run.jobs.append(prefix_outcome)
    restart_write_s = time.monotonic() - prefix_started
    stop_path.unlink(missing_ok=True)

    suffix_job = run.planned("restart_suffix_new_worker")
    checkpoint = None if prefix is None else prefix.restart
    if checkpoint is None:
        run.jobs.append(
            not_run_outcome(
                suffix_job,
                "the prefix published no native checkpoint, so there is nothing to continue",
            )
        )
        run.checks.append(_gate("restart_round_trip", {}, run.tolerances, _RESTART_GATES))
        return

    # A NEW worker: the continuation must not depend on anything the first process held.
    run.session.recycle()
    suffix_started_at = datetime.now(UTC).isoformat()
    suffix_started = time.monotonic()
    future = tuple(
        segment
        for segment in case.controls
        if segment.start_s >= checkpoint.completed_time_s - 1e-6
    )

    def work(child_ctx: RunContext, _log: logging.Logger) -> ForwardResult:
        return resume(
            case,
            checkpoint,
            future,
            worker=run.session.worker(),
            ctx=child_ctx,
            ledger=run.session.ledger(suffix_job.job_id),
            solver_config=E01_SOLVER,
            output_request=request,
        )

    _child, suffix = _child_run(run, "forward-resume", work)
    suffix_wall = time.monotonic() - suffix_started
    if suffix is None:
        run.jobs.append(not_run_outcome(suffix_job, "the continuation produced no result"))
        run.checks.append(_gate("restart_round_trip", {}, run.tolerances, _RESTART_GATES))
        return
    suffix = persist_result(suffix, run.paths)
    run.jobs.append(
        _outcome_from_result(
            suffix_job,
            suffix,
            wall_s=suffix_wall,
            started_at=suffix_started_at,
            snapshot=probe_machine(None, run.session.session_dir),
            restart_read_s=suffix_wall,
            restart_write_s=restart_write_s,
        )
    )
    run.session.after_job()
    run.checks.append(_restart_check(run, continuous, suffix))


_RESTART_GATES: tuple[tuple[str, str], ...] = (
    ("restart_saturation_abs", "restart_saturation_abs_max"),
    ("restart_pressure_relative", "restart_pressure_relative_max"),
    ("restart_volume_relative", "restart_volume_relative_max"),
)


def _state_difference(
    left: Mapping[str, Any], right: Mapping[str, Any], prefix: str
) -> dict[str, float]:
    """The last published state of one result against another's, field by field."""
    return {
        f"{prefix}_saturation_abs": max(
            float(np.abs(left["sw"][-1] - right["sw"][-1]).max()),
            float(np.abs(left["so"][-1] - right["so"][-1]).max()),
        ),
        f"{prefix}_pressure_relative": float(
            (
                np.abs(left["pressure_pa"][-1] - right["pressure_pa"][-1])
                / np.maximum(np.abs(left["pressure_pa"][-1]), 1.0)
            ).max()
        ),
        f"{prefix}_volume_relative": float(
            (
                np.abs(left["pore_volume_m3"][-1] - right["pore_volume_m3"][-1])
                / np.maximum(np.abs(left["pore_volume_m3"][-1]), 1e-6)
            ).max()
        ),
    }


def _restart_check(
    run: SuiteRun, continuous: ForwardResult | None, suffix: ForwardResult
) -> PhysicsCheck:
    if continuous is None or continuous.status != "COMPLETE" or suffix.status != "COMPLETE":
        return _gate("restart_round_trip", {}, run.tolerances, _RESTART_GATES)
    metrics = _state_difference(
        _states(continuous, run.paths), _states(suffix, run.paths), "restart"
    )
    metrics["restart_completed_report_step"] = float(
        0 if suffix.restart is None else suffix.restart.completed_report_step
    )
    return _gate(
        "restart_round_trip",
        metrics,
        run.tolerances,
        _RESTART_GATES,
        evidence=[
            run.paths.relative(run.paths.resolve(str(continuous.monthly_path)).parent),
            run.paths.relative(run.paths.resolve(str(suffix.monthly_path)).parent),
        ],
    )


def _run_isolation(run: SuiteRun) -> None:
    """A -> B -> A in one worker, then A again in a fresh one. Nothing of B may survive."""
    case_a = _p0_case(run, P0_ISOLATION_A)
    case_b = _p0_case(run, P0_ISOLATION_B)
    results: dict[str, ForwardResult | None] = {}
    for job_id, case in (
        ("isolation_a1", case_a),
        ("isolation_b", case_b),
        ("isolation_a2", case_a),
        ("isolation_a_fresh", case_a),
    ):
        if job_id == "isolation_a_fresh":
            # A SECOND process, started only after the first was closed (COMPUTE §5).
            run.session.recycle()
        job = run.planned(job_id)
        request = OutputRequest(
            state_times_s=tuple(case.report_edges_s),
            keep_native_restart=False,
            chunk_months=1,
        )
        outcome, result = _forward(run, job, case, request)
        run.jobs.append(outcome)
        results[job_id] = result
        run.session.after_job()

    run.checks.append(_isolation_check(run, results))


def _isolation_check(run: SuiteRun, results: Mapping[str, ForwardResult | None]) -> PhysicsCheck:
    base = results["isolation_a1"]
    metrics: dict[str, float] = {}
    others = [results["isolation_a2"], results["isolation_a_fresh"]]
    complete = base is not None and base.status == "COMPLETE"
    complete = complete and all(o is not None and o.status == "COMPLETE" for o in others)
    if complete and base is not None:
        left = _states(base, run.paths)
        worst = {"isolation_saturation_abs": 0.0, "isolation_pressure_relative": 0.0}
        for other in others:
            assert other is not None
            difference = _state_difference(left, _states(other, run.paths), "isolation")
            for key in worst:
                worst[key] = max(worst[key], difference[key])
        metrics = worst
    return _gate(
        "worker_isolation",
        metrics,
        run.tolerances,
        (
            ("isolation_saturation_abs", "restart_saturation_abs_max"),
            ("isolation_pressure_relative", "restart_pressure_relative_max"),
        ),
    )


def _run_worlds(run: SuiteRun) -> None:
    """11.4-11.9: the five P1 parents, each with its closed equilibrium preflight."""
    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for (seed, family), job_id in zip(
        P1_PARENTS, ("world41", "world42", "world43", "world44", "world45"), strict=True
    ):
        if run.out_of_time():
            run.log.warning("the session's wall budget is spent; %s was not started", job_id)
            break
        job = run.planned(job_id)
        world = render_p1(seed, P1Design(family=family))

        preflight_ctx = RunContext.start(command="forward", argv=[], cfg=run.cfg, paths=run.paths)
        preflight_case = build_p1_case(closed_preflight_world(world), run.paths, preflight_ctx)
        preflight_outcome, preflight = _forward(
            run,
            job.model_copy(update={"job_id": f"{job_id}_preflight", "declared_by_plan": False}),
            preflight_case,
            preflight_output_request(world.design),
        )
        run.jobs.append(preflight_outcome)
        run.session.after_job()

        ctx = RunContext.start(command="forward", argv=[], cfg=run.cfg, paths=run.paths)
        case = build_p1_case(world, run.paths, ctx)
        request = output_request(world.design)
        outcome, result = _forward(run, job, case, request)
        run.jobs.append(outcome)
        run.session.after_job()

        gates: dict[str, Any] = (
            {"status": outcome.status, "reason": outcome.reason}
            if result is None
            else score_p1_world(world, result, preflight, run.paths)
        )
        gates["measured_wall_s"] = outcome.wall_s
        for key in (
            "balance_cumulative_relative",
            "balance_step_median_relative",
            "saturation_sum_abs",
            "saturation_bound_violation",
        ):
            value = gates.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = max(metrics.get(key, 0.0), float(value))
        world_ctx = RunContext.start(command="forward", argv=[], cfg=run.cfg, paths=run.paths)
        manifest_ref = write_world(
            world, result, run.paths, world_ctx, gates=gates, tolerances=run.tolerances
        )
        manifest = json.loads(run.paths.resolve(manifest_ref.path).read_text(encoding="utf-8"))
        if result is not None:
            check_requested_outputs(result, request)
            if manifest["accepted"] and result.status != "COMPLETE":
                raise CommandError(
                    f"{world.parent_world_id}: the world manifest accepted a {result.status} "
                    "forward; physical acceptance requires completed output and passing checks"
                )
        rows.append(
            world_row(world, result, manifest, manifest_path=manifest_ref.path, gates=gates)
        )

    suite_ctx = RunContext.start(command="forward", argv=[], cfg=run.cfg, paths=run.paths)
    suite_ref = write_suite_manifest(
        rows, run.paths.reports / SUITE_MANIFEST_FILENAME, run.paths, suite_ctx
    )
    run.artifacts["p1_suite_manifest"] = suite_ref.path
    metrics["worlds_not_accepted"] = float(sum(1 for row in rows if not row["accepted"]))
    metrics["n_parents"] = float(len(rows))
    thresholds = dict(run.tolerances)
    thresholds["worlds_not_accepted_max"] = 0.0
    run.checks.append(
        _gate(
            "p1_worlds",
            metrics,
            thresholds,
            (
                ("balance_cumulative_relative", "balance_cumulative_relative_max"),
                ("balance_step_median_relative", "balance_step_median_relative_max"),
                ("saturation_sum_abs", "saturation_sum_abs_max"),
                ("saturation_bound_violation", "saturation_bound_slack"),
                ("worlds_not_accepted", "worlds_not_accepted_max"),
            ),
            evidence=[suite_ref.path],
        )
    )


#: Why every warm repeat in this build really re-ran the solver.
STRUCTURAL_CACHE_BYPASS = (
    "Structural rather than a flag. A production repeat in a RESUMED session does consult "
    "`BudgetLedger.is_already_complete` and is reused instead of re-solved (plan 12.6), but "
    "these five warm attempts run inside ONE session with no parent ledger to match against, "
    "so every one of them entered the solver by construction."
)


def first_worker_job(jobs: Sequence[JobOutcome]) -> JobOutcome | None:
    """The first job of this session that went through the persistent worker.

    Not `jobs[0]`: a P1 session opens with the refinement diagnostic, which is a SUBPROCESS
    and pays no import of the worker's. The job that paid the worker's import and the model
    specialisation is the first one that came back carrying a `cold_import_s`, and naming
    any other job as the one that paid it is the same mistake finding 6 is about.
    """
    return next((job for job in jobs if job.cold_import_s is not None), None)


def first_cold_import_s(jobs: Sequence[JobOutcome]) -> float | None:
    """The import/compile cost this session paid, from the FIRST job that recorded one.

    Every job that reaches the worker carries the same `cold_import_s` — the session's
    worker start — because it is a property of the session and not of the job. Reading it
    from the first such job is reading it from the job that actually paid it.
    """
    first = first_worker_job(jobs)
    return None if first is None else first.cold_import_s


def warm_block(
    *,
    warm_s: Sequence[float],
    failures: Sequence[str],
    cold_import_s: float | None,
    first_run_of_model_s: float | None,
    first_job_id: str | None,
    first_job_wall_s: float | None,
) -> dict[str, Any]:
    """The warm-repeat summary, with the cold and the warm costs kept apart (plan 12.4).

    There is no `cold_s` here, and that absence is the point. `world41` is not a cold run:
    `world41_preflight` ran before it in the same worker and paid the import and the model
    specialisation, so publishing `world41`'s wall as `cold_s` told a reader that
    specialisation costs about a second when the session's own first job shows it costing
    thirty. 12.4 is explicit that warm-up and cold costs are recorded separately and that a
    cold run of one regime is never compared with a warm of another, so the three numbers
    are named for what they are: the session's import cost, the wall of the job that paid
    the specialisation, and the first run of THIS model after it.
    """
    block: dict[str, Any] = {
        "cold_import_s": cold_import_s,
        "first_job_id": first_job_id,
        "first_job_wall_s": first_job_wall_s,
        "first_run_of_model_s": first_run_of_model_s,
        "failed_attempts": list(failures),
        "failure_rate": (
            len(failures) / (len(failures) + len(warm_s)) if (failures or warm_s) else None
        ),
        "cache_bypass": STRUCTURAL_CACHE_BYPASS,
        "note": (
            "`first_run_of_model_s` is a WARM-worker number: the session's first job "
            "(`first_job_id`, wall `first_job_wall_s`) had already paid the import and the "
            "model specialisation. It is not a cold cost and must not be read as one."
        ),
    }
    block.update(warm_summary(warm_s) if len(warm_s) >= MIN_WARM_RUNS else {"n": len(warm_s)})
    return block


def _run_benchmark(run: SuiteRun) -> None:
    """12.5: five warm repeats of world41, and the thread comparison the presets forbid."""
    world = render_p1(41, P1Design())
    request = output_request(world.design)
    warm: list[float] = []
    failures: list[str] = []
    for index in range(1, MIN_WARM_RUNS + 1):
        job = run.planned(f"warm41_{index}")
        if run.out_of_time():
            run.log.warning("the wall budget is spent; %s was not started", job.job_id)
            break
        ctx = RunContext.start(command="forward", argv=[], cfg=run.cfg, paths=run.paths)
        case = build_p1_case(world, run.paths, ctx)
        outcome, _result = _forward(run, job, case, request)
        run.jobs.append(outcome)
        run.session.after_job()
        if outcome.status == "COMPLETE":
            warm.append(outcome.wall_s)
        else:
            failures.append(f"{job.job_id}: {outcome.status}")

    first = first_worker_job(run.jobs)
    run.benchmark["warm41"] = warm_block(
        warm_s=warm,
        failures=failures,
        cold_import_s=first_cold_import_s(run.jobs),
        first_run_of_model_s=next(
            (job.wall_s for job in run.jobs if job.job_id == "world41"), None
        ),
        first_job_id=None if first is None else first.job_id,
        first_job_wall_s=None if first is None else first.wall_s,
    )
    if run.session.decisions:
        run.benchmark["memory_drift"] = {
            "observations": len(run.session.decisions),
            "recycles": run.session.recycles,
            "max_drift_bytes": max(d.drift_bytes for d in run.session.decisions),
            "limit_bytes": run.session.decisions[-1].limit_bytes,
        }
    _thread_comparison(run)


#: Why the planned thread comparison does not run under the approved presets.
THREAD_COMPARISON_REASON = (
    "COMPUTE §5 fixes four Julia threads for E01 and `require_resource_profile` refuses any "
    "resource block that names an approved preset without carrying its numbers, `julia_threads` "
    "included. A one- or two-thread worker therefore needs a NEW approved preset, which is a "
    "specification change and not an implementer's choice; plan 12.4 itself says threads4 stays "
    "the default until there is a measured basis for moving it, and producing that basis is what "
    "these four jobs were for. Recorded NOT_RUN rather than run under a relabelled profile."
)


def _thread_comparison(run: SuiteRun) -> None:
    for job in run.group_jobs("benchmark"):
        if job.julia_threads == run.profile.julia_threads:
            continue
        run.jobs.append(not_run_outcome(job, THREAD_COMPARISON_REASON))
    run.benchmark["threads"] = {
        "configured": run.profile.julia_threads,
        "compared": [],
        "status": "NOT_RUN",
        "reason": THREAD_COMPARISON_REASON,
    }
    run.checks.append(
        PhysicsCheck(
            name="benchmark_threads",
            status="NOT_RUN",
            metrics={},
            thresholds={},
            input_hashes={},
            evidence_paths=(),
            reason=THREAD_COMPARISON_REASON,
        )
    )


#: 13.4: which report step the black-oil restart pair is split after. It is the boundary
#: `julia/verification/blackoil.jl` names — the end of the fourth interval, with the
#: producer's bottom-hole pressure about to cross the bubble point — so the continuation is
#: the part of the run in which gas comes out of solution rather than a quiet tail.
BO_RESTART_AFTER_STEP = 4

#: What the depletion fixture must reach for the capability to have shown anything: a free
#: gas saturation somewhere in the model, and a pressure below the bubble point of the
#: declared initial dissolved ratio. They are STRUCTURAL — a fixture in which nothing
#: happened has to fail rather than pass vacuously — so they are stated here rather than in
#: the tolerance block, and what the gate scores is the SHORTFALL against them.
BO_MIN_FREE_GAS_SATURATION = 1e-3

#: The gate pairs of the two black-oil checks, read from the one place that owns them.
_BO_GATES = BLACKOIL_GATES["black_oil"]
_BO_RESTART_GATES = BLACKOIL_GATES["black_oil_restart"]


def _relative(measured: float, reference: float, floor: float) -> float:
    return abs(measured - reference) / max(abs(reference), floor)


def _gas_split(states: Mapping[str, Any], index: int) -> tuple[float, float]:
    """Free and dissolved gas of one published state, m3_sc: `Sg*PV/Bg` and `Rs*So*PV/Bo`."""
    pv = states["pore_volume_m3"][index]
    free = float(np.sum(states["sg"][index] * pv / states["bg"][index]))
    dissolved = float(np.sum(states["rs"][index] * states["so"][index] * pv / states["bo"][index]))
    return free, dissolved


def _surface_volumes(result: ForwardResult, paths: ProjectPaths) -> dict[str, float]:
    """Cumulative surface volumes of all three components over a published result, m3_sc.

    Oil and water come from `monthly.parquet` and gas from the black-oil result's OWN gas
    table. The gas is never read out of a liquid column and never added into one: a standard
    cubic metre of gas is a different physical quantity, and 13.4 is explicit that gas units
    and source scales are not borrowed from the oil-water case.
    """
    monthly = pq.read_table(paths.resolve(str(result.monthly_path)))
    out = {
        "oil": float(sum(monthly.column("oil_prod_m3_sc").to_pylist())),
        "water": float(sum(monthly.column("water_prod_m3_sc").to_pylist())),
        "gas": 0.0,
    }
    if result.black_oil is not None:
        out["gas"] = float(result.black_oil.surface_gas_m3_sc)
    return out


def _blackoil_capability_metrics(
    run: SuiteRun, report: Mapping[str, Any], results: Mapping[str, ForwardResult]
) -> dict[str, float]:
    """13.1/13.4 measured on the PUBLISHED results, not on the diagnostic's own dictionary."""
    metrics: dict[str, float] = {}
    closed = results.get("bo_closed")
    depletion = results.get("bo_depletion")
    if closed is None or depletion is None:
        return metrics

    drift = 0.0
    for result in (closed, depletion):
        states = _states(result, run.paths)
        total = states["sw"] + states["so"] + states["sg"]
        drift = max(drift, float(np.abs(total - 1.0).max()))
    metrics["blackoil_saturation_sum_drift"] = drift

    # The closed cell: nothing moves, and the component gas inventory is what it started as.
    closed_states = _states(closed, run.paths)
    last = len(closed.times_s) - 1
    metrics["blackoil_closed_saturation_drift"] = max(
        float(np.abs(closed_states[name][last] - closed_states[name][0]).max())
        for name in ("sw", "so", "sg")
    )
    assert closed.black_oil is not None
    started = closed.black_oil.free_gas_m3_sc[0] + closed.black_oil.dissolved_gas_m3_sc[0]
    ended = closed.black_oil.free_gas_m3_sc[-1] + closed.black_oil.dissolved_gas_m3_sc[-1]
    metrics["blackoil_closed_gas_inventory_relative"] = _relative(ended, started, 1e-6)

    # The depletion: gas really comes out of solution, below a bubble point this build
    # computed from the exported saturation table rather than from a remembered number.
    states = _states(depletion, run.paths)
    final = len(depletion.times_s) - 1
    metrics["blackoil_final_max_sg"] = float(states["sg"][final].max())
    metrics["blackoil_free_gas_shortfall"] = max(
        0.0, BO_MIN_FREE_GAS_SATURATION - metrics["blackoil_final_max_sg"]
    )
    bubble = float(report["bubble_point_pa"])
    metrics["blackoil_bubble_point_pa"] = bubble
    metrics["blackoil_final_min_pressure_pa"] = float(states["pressure_pa"][final].min())
    metrics["blackoil_bubble_point_shortfall"] = max(
        0.0, (metrics["blackoil_final_min_pressure_pa"] - bubble) / bubble
    )

    # The dissolved term is a CLAIM, not a definition: the native component gas inventory is
    # `TotalMasses / rho_g_sc` and knows nothing of the split, so free + dissolved recomputed
    # from the published states agreeing with it is evidence that the dissolved gas is really
    # inside the gas component.
    assert depletion.black_oil is not None
    balances = pq.read_table(run.paths.resolve(str(depletion.balances_path))).to_pylist()
    gas_rows = {str(row["balance"]): row for row in balances if row["component"] == "gas"}
    headline = gas_rows.get(HEADLINE_BALANCE)
    if headline is not None:
        metrics["blackoil_gas_balance_cumulative_relative"] = float(headline["cumulative_relative"])
        metrics["blackoil_initial_gas_inventory_m3_sc"] = float(headline["initial_inventory_m3_sc"])
    reservoir = gas_rows.get(RESERVOIR_BALANCE)
    if reservoir is not None:
        # Against the RESERVOIR inventory, because the split below is a sum over reservoir
        # cells. The whole-model statement also holds the gas standing in the wellbores,
        # which is a real quantity and not an error; it is reported beside this rather than
        # folded into it.
        native_initial = float(reservoir["initial_inventory_m3_sc"])
        split = depletion.black_oil.free_gas_m3_sc[0] + depletion.black_oil.dissolved_gas_m3_sc[0]
        metrics["blackoil_gas_inventory_closure_relative"] = _relative(split, native_initial, 1e-6)
        metrics["blackoil_initial_reservoir_gas_inventory_m3_sc"] = native_initial
        if headline is not None:
            metrics["blackoil_wellbore_gas_inventory_m3_sc"] = (
                float(headline["initial_inventory_m3_sc"]) - native_initial
            )
    metrics["blackoil_initial_free_gas_m3_sc"] = depletion.black_oil.free_gas_m3_sc[0]
    metrics["blackoil_initial_dissolved_gas_m3_sc"] = depletion.black_oil.dissolved_gas_m3_sc[0]
    metrics["blackoil_final_free_gas_m3_sc"] = depletion.black_oil.free_gas_m3_sc[-1]
    metrics["blackoil_final_dissolved_gas_m3_sc"] = depletion.black_oil.dissolved_gas_m3_sc[-1]
    metrics["blackoil_surface_gas_m3_sc"] = depletion.black_oil.surface_gas_m3_sc
    return metrics


def _blackoil_restart_metrics(
    run: SuiteRun, continuous: ForwardResult | None, suffix: ForwardResult | None
) -> dict[str, float]:
    """13.4: continuous against restart, on So/Sw/Sg/p, both gas inventories and surface V."""
    if continuous is None or suffix is None:
        return {}
    if continuous.status != "COMPLETE" or suffix.status != "COMPLETE":
        return {}
    left = _states(continuous, run.paths)
    right = _states(suffix, run.paths)
    metrics = {
        "blackoil_restart_saturation_abs": max(
            float(np.abs(left[name][-1] - right[name][-1]).max()) for name in ("sw", "so", "sg")
        ),
        "blackoil_restart_pressure_relative": float(
            (
                np.abs(left["pressure_pa"][-1] - right["pressure_pa"][-1])
                / np.maximum(np.abs(left["pressure_pa"][-1]), 1.0)
            ).max()
        ),
        "blackoil_restart_rs_relative": float(
            (
                np.abs(left["rs"][-1] - right["rs"][-1]) / np.maximum(np.abs(left["rs"][-1]), 1e-6)
            ).max()
        ),
    }
    if continuous.black_oil is not None and suffix.black_oil is not None:
        metrics["blackoil_restart_free_gas_relative"] = _relative(
            suffix.black_oil.free_gas_m3_sc[-1], continuous.black_oil.free_gas_m3_sc[-1], 1e-6
        )
        metrics["blackoil_restart_dissolved_gas_relative"] = _relative(
            suffix.black_oil.dissolved_gas_m3_sc[-1],
            continuous.black_oil.dissolved_gas_m3_sc[-1],
            1e-6,
        )
    reference = _surface_volumes(continuous, run.paths)
    measured = _surface_volumes(suffix, run.paths)
    metrics["blackoil_restart_surface_volume_relative"] = max(
        _relative(measured[name], reference[name], 1e-6) for name in ("oil", "water", "gas")
    )
    for name in ("oil", "water", "gas"):
        metrics[f"blackoil_continuous_surface_{name}_m3_sc"] = reference[name]
        metrics[f"blackoil_restart_surface_{name}_m3_sc"] = measured[name]
    return metrics


def _run_black_oil(run: SuiteRun) -> None:
    """13.4: the four registered black-oil jobs, in one session of their own.

    `bo_closed` and `bo_depletion` run inside the Julia capability diagnostic and are
    published through the same production publisher every verification fixture uses;
    `bo_restart_prefix` and `bo_restart_suffix_new_worker` run through the production worker,
    on the case the diagnostic published, with the continuation taken in a NEW process.

    The continuous trajectory the restart is measured against is `bo_depletion` itself, which
    is why the diagnostic drives it through `run_forward_native` with `chunk_months = 1` and
    the worker's own solver settings: the two are the same discretisation of the same case,
    so a difference between them is a restart defect and not a setting nobody matched.
    """
    thresholds = {
        **run.tolerances,
        **load_blackoil_tolerances(run.paths.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH)),
    }
    run.artifacts["blackoil.tolerances"] = DEFAULT_BLACKOIL_TOLERANCES_RELPATH
    started_at = datetime.now(UTC).isoformat()
    native_dir = run.ctx.run_dir / "blackoil-native"
    launched = _launch(
        run,
        "julia/verification/blackoil.jl",
        "--test-blackoil",
        "blackoil",
        extra=["--native-dir", str(native_dir)],
    )
    report = launched.payload

    # The PVT and the relative permeability this capability was built from, published as a
    # versioned fixture artifact of their own and hashed. A benchmark PVT named only by the
    # function that returned it would be identified by a name (plan 13.3).
    pvt_path = run.ctx.run_dir / "verification" / "blackoil_pvt_fixture.json"
    pvt_path.write_text(
        json.dumps(
            {
                "schema_version": "e01-blackoil-fixture-1",
                "pvt": report["pvt_export"],
                "relperm": report["relperm_export"],
                "academic_benchmark": True,
                "note": (
                    "The academic benchmark PVT that ships inside the pinned JutulDarcy, "
                    "exported as the numbers the model was really built from. It is NOT a "
                    "Romashka PVT and this capability does not choose the physics of the "
                    "field case (plan 13.5)."
                ),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    run.artifacts["blackoil.pvt_fixture"] = run.paths.relative(pvt_path)

    fixture_jobs = [job for job in run.group_jobs("black_oil") if job.kind == "fixture"]
    per_job = launched.wall_s / max(len(fixture_jobs), 1)
    results: dict[str, ForwardResult] = {}
    for job in fixture_jobs:
        fixture = report["fixtures"][job.job_id]
        label = f"{run.suite}-{job.job_id}"
        result, result_dir = publish_fixture(
            dict(fixture), run.paths, label, dict(report), world=run.suite
        )
        results[job.job_id] = result
        run.artifacts[f"case.{job.job_id}"] = run.paths.relative(
            run.paths.artifacts / f"case-{label}.json"
        )
        run.artifacts[f"result.{job.job_id}"] = run.paths.relative(result_dir)
        run.jobs.append(
            _launcher_outcome(
                job,
                status=str(fixture["status"]),
                wall_s=per_job,
                started_at=started_at,
                fixture=fixture,
                peak_rss_bytes=launched.peak_rss_bytes,
            )
        )
        run.session.record_output(result.cost.output_bytes)

    metrics = _blackoil_capability_metrics(run, report, results)
    run.checks.append(
        _gate(
            "black_oil",
            metrics,
            thresholds,
            _BO_GATES,
            evidence=[run.artifacts["verification.blackoil"], run.artifacts["blackoil.pvt_fixture"]]
            + [run.artifacts[f"result.{job.job_id}"] for job in fixture_jobs],
            hashes={
                "blackoil_tolerances": sha256_file(
                    run.paths.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH)
                ),
                "blackoil_pvt_fixture": sha256_file(pvt_path),
            },
        )
    )
    _run_black_oil_restart(run, results.get("bo_depletion"), thresholds)


def _run_black_oil_restart(
    run: SuiteRun, continuous: ForwardResult | None, thresholds: Mapping[str, float]
) -> None:
    """The prefix and its continuation in a NEW worker, on the published depletion case."""
    prefix_job = run.planned("bo_restart_prefix")
    suffix_job = run.planned("bo_restart_suffix_new_worker")
    relative = run.artifacts.get("case.bo_depletion")
    if relative is None or continuous is None or continuous.status != "COMPLETE":
        reason = (
            "the depletion fixture published no complete result, so there is no case to "
            "continue and nothing to continue it against"
        )
        run.jobs.append(not_run_outcome(prefix_job, reason))
        run.jobs.append(not_run_outcome(suffix_job, reason))
        run.checks.append(_gate("black_oil_restart", {}, thresholds, _BO_RESTART_GATES))
        return

    case = load_case(run.paths.resolve(relative), run.paths)
    request = OutputRequest(
        state_times_s=tuple(case.report_edges_s), keep_native_restart=True, chunk_months=1
    )

    # The prefix is the SAME depletion over its first `BO_RESTART_AFTER_STEP` report steps,
    # and it RUNS TO THE END OF THAT SCHEDULE: a complete forward that publishes a native
    # checkpoint, not a run somebody stopped. The distinction is the job's declared outcome.
    # A stopped run's honest status is `INCOMPLETE_BUDGET` — which is exactly what the
    # oil-water `restart_prefix` is declared as, because a stop leaves months nobody
    # simulated — and the black-oil matrix declares `COMPLETE`, so the prefix here is built
    # to be complete rather than the declaration bent to fit a stop.
    #
    # It is the same physical case by construction: only `report_edges_s` and `controls`
    # are cut, and `compute_static_hash` — everything that determines F EXCEPT the schedule —
    # is therefore identical, which is what the continuation's `verify_restart` compares. The
    # schedule half is proved separately by the checkpoint's own prefix digest, and
    # `schedule_prefix_hash` truncates both cases to the checkpoint, so the prefix case's
    # digest at that time is the full case's digest at that time.
    cut_s = float(case.report_edges_s[BO_RESTART_AFTER_STEP])
    prefix_case = case.model_copy(
        update={
            "case_id": f"{case.case_id}-prefix",
            "report_edges_s": tuple(case.report_edges_s[: BO_RESTART_AFTER_STEP + 1]),
            "controls": tuple(c for c in case.controls if c.end_s <= cut_s + 1e-6),
        }
    )
    prefix_case = prefix_case.model_copy(update={"model_hash": compute_model_hash(prefix_case)})
    if compute_static_hash(prefix_case) != compute_static_hash(case):
        raise CommandError(
            "the black-oil restart prefix is not the same physical model as the case it is a "
            "prefix of; only the schedule may be cut"
        )
    prefix_request = OutputRequest(
        state_times_s=tuple(prefix_case.report_edges_s), keep_native_restart=True, chunk_months=1
    )
    prefix_started = time.monotonic()
    prefix_outcome, prefix = _forward(run, prefix_job, prefix_case, prefix_request)
    run.jobs.append(prefix_outcome)
    restart_write_s = time.monotonic() - prefix_started

    checkpoint = None if prefix is None else prefix.restart
    if checkpoint is None:
        run.jobs.append(
            not_run_outcome(
                suffix_job,
                "the prefix published no native checkpoint; there is nothing to continue",
            )
        )
        run.checks.append(_gate("black_oil_restart", {}, thresholds, _BO_RESTART_GATES))
        return

    # A NEW worker. The continuation must not depend on anything the first process held —
    # and for black oil that includes the phase state, which is what the native checkpoint
    # carries and a saturation alone would not.
    run.session.recycle()
    suffix_started_at = datetime.now(UTC).isoformat()
    suffix_started = time.monotonic()
    future = tuple(
        segment
        for segment in case.controls
        if segment.start_s >= checkpoint.completed_time_s - 1e-6
    )

    def work(child_ctx: RunContext, _log: logging.Logger) -> ForwardResult:
        return resume(
            case,
            checkpoint,
            future,
            worker=run.session.worker(),
            ctx=child_ctx,
            ledger=run.session.ledger(suffix_job.job_id),
            solver_config=E01_SOLVER,
            output_request=request,
        )

    _child, suffix = _child_run(run, "forward-resume", work)
    suffix_wall = time.monotonic() - suffix_started
    if suffix is None:
        run.jobs.append(not_run_outcome(suffix_job, "the continuation produced no result"))
        run.checks.append(_gate("black_oil_restart", {}, thresholds, _BO_RESTART_GATES))
        return
    suffix = persist_result(suffix, run.paths)
    run.jobs.append(
        _outcome_from_result(
            suffix_job,
            suffix,
            wall_s=suffix_wall,
            started_at=suffix_started_at,
            snapshot=probe_machine(None, run.session.session_dir),
            restart_read_s=suffix_wall,
            restart_write_s=restart_write_s,
        )
    )
    run.session.after_job()
    metrics = _blackoil_restart_metrics(run, continuous, suffix)
    metrics["blackoil_restart_completed_report_step"] = float(
        0 if suffix.restart is None else suffix.restart.completed_report_step
    )
    evidence = [run.paths.relative(run.paths.resolve(str(continuous.monthly_path)).parent)]
    if suffix.monthly_path is not None:
        evidence.append(run.paths.relative(run.paths.resolve(str(suffix.monthly_path)).parent))
    run.checks.append(
        _gate("black_oil_restart", metrics, thresholds, _BO_RESTART_GATES, evidence=evidence)
    )


GROUPS: dict[str, Callable[[SuiteRun], None]] = {
    "analytic": _run_analytic,
    "operations": _run_operations,
    "controls": _run_controls,
    "restart": _run_restart,
    "isolation": _run_isolation,
    "refinement": _run_refinement,
    "worlds": _run_worlds,
    "benchmark": _run_benchmark,
    "black_oil": _run_black_oil,
}


def run_suite_jobs(run: SuiteRun) -> SuiteOutcome:
    """Execute the registered groups of one suite in order, inside its wall budget.

    "In order" is the whole of plan 12.6's resume rule: a resumed session walks the SAME
    group sequence the plan fixes, carries forward the groups the parent session finished
    and enters only the ones it did not. Nothing is reordered to put the missing work first,
    because the P0 worker groups run on cases the analytic group publishes and a session
    that ran them out of order would be running them on something else.
    """
    stopped: str | None = None
    for group in run.plan.groups:
        runner = GROUPS.get(group)
        if runner is None:
            raise CommandError(f"suite {run.suite!r} names an unregistered group {group!r}")
        if run.skips(group):
            run.log.info(
                "group %s of suite %s was completed by run %s; carrying its rows and verdicts "
                "forward rather than re-running it (--replay re-runs)",
                group,
                run.suite,
                None if run.resume is None else run.resume.run_id,
            )
            run.carry(group)
            continue
        if run.out_of_time():
            stopped = (
                f"the session's {run.plan.wall_budget_s} s wall budget was spent before "
                f"group {group!r} started"
            )
            break
        run.log.info("running group %s of suite %s", group, run.suite)
        try:
            runner(run)
        except (BudgetStop, ResourceFailure) as exc:
            stopped = f"group {group!r} stopped: {exc}"
            break
        # Deliberately broad. A group that raised has still produced rows for the jobs that
        # ran, and plan 12.10 wants the partial report with the remaining jobs named rather
        # than a bare traceback and no record at all. The traceback goes to run.log, the
        # message becomes the stop reason, and the exit code is never 0.
        except Exception as exc:
            run.log.exception("group %s of suite %s failed", group, run.suite)
            stopped = f"group {group!r} failed: {type(exc).__name__}: {exc}"
            break
    if stopped is None and run.out_of_time() and run.remaining():
        stopped = f"the session's {run.plan.wall_budget_s} s wall budget was spent"
    return SuiteOutcome(
        jobs=tuple(run.jobs),
        checks=tuple(run.checks),
        remaining_job_ids=run.remaining(),
        stopped_reason=stopped,
        benchmark=dict(run.benchmark),
        artifacts={**run.artifacts, **run.session.ledger_paths(run.paths)},
        resumed_from_run_id=None if run.resume is None else run.resume.run_id,
        reused_group_ids=tuple(run.reused_groups),
    )


def open_suite_run(
    *,
    cfg: ProjectConfig,
    paths: ProjectPaths,
    ctx: RunContext,
    log: logging.Logger,
    suite: str,
    plan: SuitePlan,
    profile: ResourceProfile,
    tolerances: dict[str, float],
    julia: str | None,
    resume_ledger: Path | None,
    replay: bool = False,
) -> SuiteRun:
    exe = find_julia(julia)
    session_dir = ctx.run_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = ctx.run_dir / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    state: ResumeState | None = None
    if resume_ledger is not None:
        # Refuse a path that is not a ledger BEFORE anything is launched: `--resume-ledger`
        # takes the path a previous command printed, and a file that is not one is an
        # operator error rather than something to discover halfway through a session.
        load_ledger(resume_ledger)
        state = load_resume_state(resume_ledger, suite=suite, plan=plan, paths=paths)
        if state is None:
            log.warning(
                "%s names no published suite record for %s; every group of this session will run",
                paths.relative(resume_ledger),
                suite,
            )
        else:
            log.info(
                "resuming after run %s: groups %s are complete there and will be carried forward%s",
                state.run_id,
                sorted(state.completed_groups) or "[]",
                "; --replay overrides and re-runs them"
                if not replay
                else ", except --replay was given so every group re-runs",
            )
    log.info("per-job ledgers for this session: %s", paths.relative(ledger_dir))
    return SuiteRun(
        cfg=cfg,
        paths=paths,
        ctx=ctx,
        log=log,
        suite=suite,
        plan=plan,
        profile=profile,
        tolerances=tolerances,
        julia=exe,
        session=Session(
            julia=exe,
            project=paths.julia,
            session_dir=session_dir,
            profile=profile,
            paths=paths,
            ledger_dir=ledger_dir,
            session_id=f"{ctx.run_id}-{suite}",
            resume_from=resume_ledger,
        ),
        deadline_s=time.monotonic() + plan.wall_budget_s,
        artifacts={"ledgers": paths.relative(ledger_dir)},
        resume=state,
        replay=replay,
    )
