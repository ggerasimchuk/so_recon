"""E01.12 — the bodies of the E01 commands. `cli.py` parses; everything below acts.

Six commands live here: `forward`, `forward-resume`, `verify-physics`, `synthetic-p1`,
`benchmark-forward` and `e01-report`. Each of them is a thin composition over work Tasks
1-11 already built — `simulate`/`resume`, the physics evaluator, the P1 generator and its
publisher — and none of them re-implements any of it. What this module adds is the three
things a command line needs and a library does not.

**An exit code that is not a lie.** `forward_exit_code` maps a `ForwardStatus` to a shell
status, and only `COMPLETE` is zero. A checkpoint written, a partial report rendered or a
world published is not, on its own, success: plan 12.1 is explicit that no command returns 0
merely because it wrote something. A suite that spent its wall budget exits 2 naming the jobs
it did not reach; a suite whose mandatory check failed exits 1.

**A session that outlives one job.** `simulate` runs ONE case: it has no view across jobs and
so no place to hold the memory drift Task 8 measures between them. A CLI session does, so
`_Session` below owns exactly one `MemoryDriftMonitor` for the whole suite, takes its
baseline after the first job has warmed the worker, and ACTS on the decision — a warning and
a recycle on the first drift, a `RESOURCE_FAILURE` when a fresh process drifts again. At most
one worker is ever alive (COMPUTE §5).

**A record the stage report can read.** Every suite writes `e01_suite.json` into its run
directory: the planned jobs, what each attempt cost, and the `PhysicsCheck` verdicts
themselves. Plan 12.9 has the stage validator read published artifacts rather than re-run the
suite, so a metric that exists only inside this process is a metric the gate cannot cite.

Paths persisted here are project-relative, always. `PhysicsCheck.evidence_paths` is filled by
its callers with whatever path they opened, which is an absolute one for anything under a
resolved `ProjectPaths`; `_relative_evidence` below is the boundary that turns them back into
repository paths before a record leaves this process, because `reports/` is tracked and
`tests/test_no_absolute_paths.py` guards the tracked set.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from so_recon.config.resources import (
    ResourceProfile,
    ResourceProfileId,
    require_resource_profile,
    resource_profile,
)
from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.environment.resources import HardwareProfile, ResourceSnapshot, probe_hardware
from so_recon.environment.resources import probe_resources as probe_machine
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RUN_RECORD_SCHEMA_VERSION, RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.simulator.budget import BudgetLedger, BudgetStop, load_ledger
from so_recon.simulator.case_io import load_case, read_array
from so_recon.simulator.contracts import (
    CaseBundle,
    ControlSegment,
    ForwardResult,
    OutputRequest,
    RestartRef,
)
from so_recon.simulator.forward import (
    DriftDecision,
    MemoryDriftMonitor,
    SolverConfig,
    check_requested_outputs,
    restart_ref_from_manifest,
    resume,
    simulate,
)
from so_recon.simulator.julia_bridge import (
    JuliaRunError,
    SubprocessJuliaLauncher,
    find_julia,
)
from so_recon.simulator.results import (
    BALANCES_FILENAME,
    CONNECTIONS_FILENAME,
    MONTHLY_FILENAME,
    RESULT_FILENAME,
    STATES_FILENAME,
    load_forward_result,
    write_forward_result,
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
    CommonSupport,
    PhysicsCheck,
    bl_cell_average,
    cartesian_zone_ids,
    common_metrics,
    compare_refinement,
    evaluate_physics,
    load_tolerances,
)

log = logging.getLogger(__name__)

#: Where a suite leaves the record the stage validator reads.
SUITE_REPORT_FILENAME = "e01_suite.json"
SUITE_REPORT_SCHEMA_VERSION: Literal["e01-suite-1"] = "e01-suite-1"
JOB_PLAN_SCHEMA_VERSION: Literal["e01-jobs-1"] = "e01-jobs-1"

#: Where the planned matrix lives, relative to the repository root.
DEFAULT_JOB_PLAN_RELPATH = "configs/e01_jobs.json"

#: The run-scoped environment record every physical E01 session leaves (plan 12.8). The
#: deterministic E00 report under `reports/` is NOT touched by any of this.
ENVIRONMENT_RECORD_FILENAME = "e01_environment.json"
ENVIRONMENT_RECORD_SCHEMA_VERSION: Literal["e01-environment-1"] = "e01-environment-1"

SUITES: tuple[str, ...] = ("p0", "p1", "bo")

#: 12.5: fewer than five completed warm timings is not a p50/p90, it is a guess.
MIN_WARM_RUNS = 5

#: 12.5: above this coefficient of variation the report RECOMMENDS 10-20 repeats. It never
#: runs them: a larger sample is a larger budget and a budget is granted, not assumed.
WARM_CV_RECOMMENDATION_THRESHOLD = 0.25

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

    Every field is measured or explicitly absent. A failed attempt keeps its row and its
    spent time: 12.5 requires failures to stay in the denominator of the failure rate.
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
    cpu_s: float
    peak_rss_bytes: int
    output_bytes: int
    native_chunk_calls: int
    accepted_steps: int
    cut_steps: int
    nonlinear_iterations: int
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


def _relative_evidence(check: PhysicsCheck, paths: ProjectPaths) -> PhysicsCheck:
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
    failed = [c.name for c in outcome.checks if c.status == "FAIL" and c.name in MANDATORY_CHECKS]
    unrun = [c.name for c in outcome.checks if c.status == "NOT_RUN" and c.name in MANDATORY_CHECKS]
    present = {c.name for c in outcome.checks}
    expected_here = {
        job.scored_as
        for job in plan.jobs
        if job.deferred_to is None and job.scored_as in MANDATORY_CHECKS
    }
    missing = sorted(expected_here - present)
    surprised = [job.job_id for job in outcome.jobs if not job.as_expected]
    for check in outcome.checks:
        if check.status == "NOT_RUN" and check.name not in MANDATORY_CHECKS:
            limitations.append(f"{check.name}: NOT_RUN — {check.reason}")
    for job in plan.jobs:
        if job.deferred_to is not None:
            limitations.append(
                f"{job.job_id}: declared in this suite and executed in suite "
                f"{job.deferred_to} — {job.note}"
            )
    if failed:
        return 1, tuple(limitations)
    if unrun or missing or outcome.remaining_job_ids or surprised:
        return 2, tuple(limitations)
    return 0, tuple(limitations)


# --------------------------------------------------------------------------------------
# 12.8 the run-scoped environment record
# --------------------------------------------------------------------------------------


class E01EnvironmentRecord(StrictModel):
    """Lock hashes and the host, recorded PER RUN and never committed (plan 12.8).

    `reports/environment_report.md` is E00's deterministic artifact and is not rewritten
    here: it is the file whose whole purpose is to be byte-identical across machines, and a
    physical E01 session has nothing to add to it that would not break that. What an E01
    session has that E00 did not is the hardware it really ran on — `probe_hardware`, built
    and tested in Task 3 and persisted nowhere until now — and the digests of the lock files
    as they stand at this run, which is what makes a measured cost attributable.
    """

    schema_version: Literal["e01-environment-1"] = ENVIRONMENT_RECORD_SCHEMA_VERSION
    run_id: str
    created_at: str
    environment_lock_hash: str
    lock_hashes: dict[str, str]
    hardware: HardwareProfile
    profile: ResourceProfile
    note: str


def write_environment_record(
    paths: ProjectPaths, ctx: RunContext, profile: ResourceProfile
) -> Path:
    """Persist the run-scoped environment stamp of plan 12.8 into the run directory."""
    lock_hashes = {
        name: (sha256_file(paths.root / name) if (paths.root / name).is_file() else "missing")
        for name in ("uv.lock", "julia/Manifest.toml", "julia/.julia-version")
    }
    hardware = probe_hardware()
    # `probe_hardware` records the julia executable it found, which is an absolute path on
    # the host. It stays in the RUN directory, which is untracked; nothing here copies it
    # into `reports/`.
    record = E01EnvironmentRecord(
        run_id=ctx.run_id,
        created_at=datetime.now(UTC).isoformat(),
        environment_lock_hash=ctx.record.environment_lock_hash,
        lock_hashes=lock_hashes,
        hardware=hardware,
        profile=profile,
        note=(
            "Run-scoped (plan 12.8). The deterministic E00 report at "
            "reports/environment_report.md is not rewritten by an E01 session: it carries "
            "only what the locks fix, so that a repeated gate leaves no diff on any machine."
        ),
    )
    path = ctx.run_dir / ENVIRONMENT_RECORD_FILENAME
    path.write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ctx.update(
        julia_version=hardware.julia_version,
        jutul_version=None,
        jutuldarcy_version=None,
    )
    return path


# --------------------------------------------------------------------------------------
# the session: ONE worker, ONE ledger, ONE memory drift monitor
# --------------------------------------------------------------------------------------


class ResourceFailure(RuntimeError):
    """The session refused to continue: a drift a fresh process still shows (plan 8.7)."""


def _bounded_probe(session_dir: Path, pid: int | None = None) -> Callable[[], ResourceSnapshot]:
    def probe() -> ResourceSnapshot:
        return probe_machine(pid, session_dir)

    return probe


class _Session:
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

    @property
    def session_dir(self) -> Path:
        return self._session_dir

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


def _chunk_calls(result: ForwardResult) -> int:
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
        native_chunk_calls=_chunk_calls(result),
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
    )


def _launcher_outcome(
    job: PlannedJob,
    *,
    status: str,
    wall_s: float,
    started_at: str,
    fixture: Mapping[str, Any] | None,
    reason: str | None = None,
) -> JobOutcome:
    """The row of a forward that ran inside a verification diagnostic, not on the ledger.

    Obligation of plan 12.4 and of Tasks 6, 9 and 10: these forwards are accounted by the
    test launcher, separately, and they are NOT charged to a `BudgetLedger` — inventing a
    `JobDescriptor` and a case digest for them to charge would add fiction. Their solver
    counters are the diagnostic's own, so the row says what really happened.
    """
    solver: Mapping[str, Any] = {}
    chunks = 0
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
        cpu_s=0.0,
        peak_rss_bytes=0,
        output_bytes=0,
        native_chunk_calls=chunks,
        accepted_steps=int(solver.get("accepted_steps", 0)),
        cut_steps=int(solver.get("cut_steps", 0)),
        nonlinear_iterations=int(solver.get("nonlinear_iterations", 0)),
        retry_count=0,
        reason=reason,
        parent_job_id=job.parent_job_id,
        julia_threads=job.julia_threads,
        cache_bypass=job.cache_bypass,
        measurement_method=(
            "verification diagnostic through the test launcher; the wall time is the whole "
            "diagnostic's, divided evenly across the forwards it ran, and the solver counters "
            "are this fixture's own (plan 12.4: accounted by the launcher, not the ledger)"
        ),
    )


def _not_run_outcome(job: PlannedJob, reason: str) -> JobOutcome:
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
        measurement_method="nothing ran, so every count is zero",
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
    session: _Session
    deadline_s: float
    jobs: list[JobOutcome] = field(default_factory=list)
    checks: list[PhysicsCheck] = field(default_factory=list)
    benchmark: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

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


SuiteRunner = Callable[[SuiteRun], SuiteOutcome]


# --------------------------------------------------------------------------------------
# the launcher groups: a Julia verification diagnostic, published and scored
# --------------------------------------------------------------------------------------


def _launch(run: SuiteRun, script: str, flag: str, label: str) -> tuple[dict[str, Any], float]:
    out_path = run.ctx.run_dir / "verification" / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    launcher = SubprocessJuliaLauncher(
        run.julia, run.paths.julia, timeout_s=run.profile.job_timeout_s
    )
    started = time.monotonic()
    launcher.launch(run.paths.root / script, [flag], out_path)
    elapsed = time.monotonic() - started
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "ok"):
        raise JuliaRunError(f"{label}: the diagnostic reported {payload.get('status')!r}")
    run.artifacts[f"verification.{label}"] = run.paths.relative(out_path)
    return payload, elapsed


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
    report, elapsed = _launch(run, "julia/verification/analytic.jl", "--test-analytic", "analytic")
    jobs = run.group_jobs("analytic")
    per_job = elapsed / max(len(jobs), 1)
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
    report, elapsed = _launch(
        run, "julia/verification/operations.jl", "--test-operations", "operations"
    )
    jobs = run.group_jobs("operations")
    per_job = elapsed / max(len(jobs), 1)
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
    report, elapsed = _launch(run, "julia/verification/fixtures.jl", "--test-controls", "controls")
    jobs = run.group_jobs("controls")
    per_job = elapsed / max(len(jobs), 1)
    for job in jobs:
        run.jobs.append(
            _launcher_outcome(
                job, status="COMPLETE", wall_s=per_job, started_at=started_at, fixture=None
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
    report, elapsed = _launch(
        run, "julia/verification/refinement.jl", "--test-refinement", "refinement"
    )
    jobs = run.group_jobs("refinement")
    per_job = elapsed / max(len(jobs), 1)
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


def _persist(result: ForwardResult, paths: ProjectPaths) -> ForwardResult:
    """Write the record beside its outputs and read it back through every check it owes."""
    if result.status != "COMPLETE" or result.monthly_path is None:
        return result
    record_path = paths.resolve(result.monthly_path).parent / RESULT_FILENAME
    write_forward_result(result, record_path)
    return load_forward_result(record_path, paths)


def _forward(
    run: SuiteRun, job: PlannedJob, case: CaseBundle, request: OutputRequest
) -> tuple[JobOutcome, ForwardResult | None]:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
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
        outcome = _not_run_outcome(
            job, f"the attempt did not produce a result: {'; '.join(child.record.notes)}"
        )
        return outcome.model_copy(update={"status": "RESOURCE_FAILURE", "wall_s": wall_s}), None
    result = _persist(result, run.paths)
    check_requested_outputs(result, request)
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
            _not_run_outcome(
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
        run.jobs.append(_not_run_outcome(suffix_job, "the continuation produced no result"))
        run.checks.append(_gate("restart_round_trip", {}, run.tolerances, _RESTART_GATES))
        return
    suffix = _persist(suffix, run.paths)
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

    cold = next(
        (job.wall_s for job in run.jobs if job.job_id == "world41"),
        None,
    )
    run.benchmark["warm41"] = {
        "cold_s": cold,
        "failed_attempts": failures,
        "failure_rate": (
            len(failures) / (len(failures) + len(warm)) if (failures or warm) else None
        ),
        "cache_bypass": (
            "This build has no production result cache to bypass: `BudgetLedger."
            "is_already_complete` defines the key but nothing in the forward path consults "
            "it, so every warm repeat re-ran the solver by construction (plan 12.6)."
        ),
        **(warm_summary(warm) if len(warm) >= MIN_WARM_RUNS else {"n": len(warm)}),
    }
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
        run.jobs.append(_not_run_outcome(job, THREAD_COMPARISON_REASON))
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


def _run_black_oil(run: SuiteRun) -> None:
    reason = run.plan.not_run_reason or "the black-oil capability is not built in this stage"
    for job in run.plan.jobs:
        run.jobs.append(_not_run_outcome(job, reason))
    run.checks.append(
        PhysicsCheck(
            name="black_oil",
            status="NOT_RUN",
            metrics={},
            thresholds={},
            input_hashes={},
            evidence_paths=(),
            reason=reason,
        )
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
    """Execute the registered groups of one suite in order, inside its wall budget."""
    stopped: str | None = None
    for group in run.plan.groups:
        runner = GROUPS.get(group)
        if runner is None:
            raise CommandError(f"suite {run.suite!r} names an unregistered group {group!r}")
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
    )


# --------------------------------------------------------------------------------------
# 12.3 the commands
# --------------------------------------------------------------------------------------


def _profile_for(cfg: ProjectConfig, plan: SuitePlan, command: str) -> ResourceProfile:
    """The approved preset this suite runs under, proved against the configuration.

    A suite whose plan names the configured profile uses the configured block, which is what
    `require_resource_profile` validates. A suite that names a DIFFERENT approved preset — P1
    under a P0 configuration — selects that preset by name, which is the explicit override
    `so_recon.config.resources.resource_profile` exists for.
    """
    configured = require_resource_profile(cfg.resources, command=command)
    if plan.profile == configured.profile:
        return configured
    return resource_profile(plan.profile)


def _suite_plan(paths: ProjectPaths, suite: str) -> tuple[JobPlan, SuitePlan, Path]:
    path = paths.resolve(DEFAULT_JOB_PLAN_RELPATH)
    plan = load_job_plan(path, paths)
    if suite not in plan.suites:
        raise CommandError(
            f"{suite!r} is not a planned suite; {paths.relative(path)} declares "
            f"{sorted(plan.suites)}"
        )
    return plan, plan.suites[suite], path


def run_e01_suite(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    suite: str,
    *,
    resume_ledger: Path | None = None,
    argv: Sequence[str] = (),
    julia: str | None = None,
    runner: SuiteRunner | None = None,
) -> RunContext:
    """`verify-physics --suite {p0,p1,bo}`: run a registered suite and publish its record."""

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        started_at = datetime.now(UTC).isoformat()
        plan, suite_plan, plan_path = _suite_plan(paths, suite)
        profile = _profile_for(cfg, suite_plan, "verify-physics")
        write_environment_record(paths, ctx, profile)
        tolerance_path = paths.resolve(plan.tolerances_path)
        tolerances = load_tolerances(tolerance_path)

        execute = runner or run_suite_jobs
        if suite_plan.status == "NOT_RUN" and runner is None:
            outcome = _not_run_suite(suite_plan)
        else:
            outcome = execute(
                _open_suite_run(
                    cfg=cfg,
                    paths=paths,
                    ctx=ctx,
                    log=command_log,
                    suite=suite,
                    plan=suite_plan,
                    profile=profile,
                    tolerances=tolerances,
                    julia=julia,
                    resume_ledger=resume_ledger,
                )
            )
        code, limitations = evaluate_suite(outcome, suite_plan)
        report = SuiteReport(
            suite=suite,
            profile=profile.profile,
            exit_code=code,
            run_id=ctx.run_id,
            command="verify-physics",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            git_commit=ctx.record.git_commit,
            git_dirty=ctx.record.git_dirty,
            environment_lock_hash=ctx.record.environment_lock_hash,
            tolerances_path=plan.tolerances_path,
            tolerances_sha256=sha256_file(tolerance_path),
            job_plan_path=paths.relative(plan_path),
            job_plan_sha256=sha256_file(plan_path),
            planned_job_ids=tuple(job.job_id for job in suite_plan.jobs if job.deferred_to is None),
            deferred_job_ids=tuple(
                job.job_id for job in suite_plan.jobs if job.deferred_to is not None
            ),
            declared_count_without_retry=suite_plan.declared_count_without_retry,
            jobs=outcome.jobs,
            checks=tuple(_relative_evidence(check, paths) for check in outcome.checks),
            mandatory_checks=tuple(
                sorted({c.name for c in outcome.checks if c.name in MANDATORY_CHECKS})
            ),
            exploratory_checks=tuple(
                sorted({c.name for c in outcome.checks if c.name not in MANDATORY_CHECKS})
            ),
            remaining_job_ids=outcome.remaining_job_ids,
            stopped_reason=outcome.stopped_reason,
            limitations=limitations,
            benchmark=outcome.benchmark,
            artifacts=outcome.artifacts,
            launcher_forwards=sum(
                1 for j in outcome.jobs if j.accounting == "launcher" and j.status != "NOT_RUN"
            ),
            ledger_forwards=sum(
                1 for j in outcome.jobs if j.accounting == "ledger" and j.status != "NOT_RUN"
            ),
        )
        path = ctx.run_dir / SUITE_REPORT_FILENAME
        path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        notes = [f"suite {suite} exit_code={code}"]
        if outcome.stopped_reason:
            notes.append(outcome.stopped_reason)
        notes.extend(limitations)
        return ("PASS" if code == 0 else "FAIL"), notes

    return execute_run(
        command="verify-physics",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={
            "run_record": RUN_RECORD_SCHEMA_VERSION,
            "e01_suite": SUITE_REPORT_SCHEMA_VERSION,
            "e01_jobs": JOB_PLAN_SCHEMA_VERSION,
        },
    )


def _not_run_suite(plan: SuitePlan) -> SuiteOutcome:
    reason = plan.not_run_reason or "this suite is not built in this stage"
    return SuiteOutcome(
        jobs=tuple(_not_run_outcome(job, reason) for job in plan.jobs),
        checks=(
            PhysicsCheck(
                name="black_oil",
                status="NOT_RUN",
                metrics={},
                thresholds={},
                input_hashes={},
                evidence_paths=(),
                reason=reason,
            ),
        ),
        remaining_job_ids=(),
        stopped_reason=reason,
    )


def _open_suite_run(
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
) -> SuiteRun:
    exe = find_julia(julia)
    session_dir = ctx.run_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = ctx.run_dir / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if resume_ledger is not None:
        # Refuse a path that is not a ledger BEFORE anything is launched: `--resume-ledger`
        # takes the path a previous command printed, and a file that is not one is an
        # operator error rather than something to discover halfway through a session.
        load_ledger(resume_ledger)
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
        session=_Session(
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
    )


# --------------------------------------------------------------------------------------
# forward / forward-resume / synthetic-p1 / benchmark-forward
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardOutcome:
    ctx: RunContext
    exit_code: int


def _single_session(
    cfg: ProjectConfig, paths: ProjectPaths, ctx: RunContext, julia: str | None
) -> _Session:
    profile = require_resource_profile(cfg.resources, command="forward")
    session_dir = ctx.run_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = ctx.run_dir / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    return _Session(
        julia=find_julia(julia),
        project=paths.julia,
        session_dir=session_dir,
        profile=profile,
        paths=paths,
        ledger_dir=ledger_dir,
        session_id=ctx.run_id,
    )


def _case_request(case: CaseBundle, *, keep_native_restart: bool) -> OutputRequest:
    return OutputRequest(
        state_times_s=tuple(case.report_edges_s),
        keep_native_restart=keep_native_restart,
        chunk_months=1,
    )


def run_forward(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    case_path: str,
    argv: Sequence[str] = (),
    julia: str | None = None,
) -> ForwardOutcome:
    """`forward --case PATH`: run one published case to a published, verified result."""
    holder: list[ForwardResult] = []

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        resolved = paths.resolve(case_path)
        case = load_case(resolved, paths)
        profile = require_resource_profile(cfg.resources, command="forward")
        write_environment_record(paths, ctx, profile)
        session = _single_session(cfg, paths, ctx, julia)
        try:
            request = _case_request(case, keep_native_restart=True)
            result = simulate(
                case,
                request,
                worker=session.worker(),
                ctx=ctx,
                ledger=session.ledger("forward"),
                solver_config=E01_SOLVER,
            )
            result = _persist(result, paths)
            check_requested_outputs(result, request)
        finally:
            session.close()
        holder.append(result)
        command_log.info("forward %s finished %s", case.case_id, result.status)
        notes = [f"status={result.status}"] + ([result.reason] if result.reason else [])
        return ("PASS" if result.status == "COMPLETE" else "FAIL"), notes

    ctx = execute_run(
        command="forward",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION, "forward_result": "forward-1"},
    )
    status = holder[0].status if holder else "RESOURCE_FAILURE"
    return ForwardOutcome(ctx=ctx, exit_code=forward_exit_code(status))


def run_forward_resume(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    case_path: str,
    restart_path: str,
    argv: Sequence[str] = (),
    julia: str | None = None,
) -> ForwardOutcome:
    """`forward-resume --case PATH --restart PATH`: continue from a published checkpoint.

    The parent's result directory is never written into. `resume` builds its own descriptor
    under THIS run's directory and re-extracts from the native bytes the checkpoint names, so
    the stopped prefix stays exactly as it was published (Task 8, SPEC §3.3).
    """
    holder: list[ForwardResult] = []

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        case = load_case(paths.resolve(case_path), paths)
        manifest = paths.resolve(restart_path)
        restart: RestartRef = restart_ref_from_manifest(manifest, paths, model_hash=case.model_hash)
        profile = require_resource_profile(cfg.resources, command="forward")
        write_environment_record(paths, ctx, profile)
        session = _single_session(cfg, paths, ctx, julia)
        future: tuple[ControlSegment, ...] = tuple(
            segment
            for segment in case.controls
            if segment.start_s >= restart.completed_time_s - 1e-6
        )
        try:
            request = _case_request(case, keep_native_restart=True)
            result = resume(
                case,
                restart,
                future,
                worker=session.worker(),
                ctx=ctx,
                ledger=session.ledger("forward-resume"),
                solver_config=E01_SOLVER,
                output_request=request,
            )
            result = _persist(result, paths)
            check_requested_outputs(result, request)
        finally:
            session.close()
        holder.append(result)
        command_log.info("continuation of %s finished %s", case.case_id, result.status)
        notes = [f"status={result.status}"] + ([result.reason] if result.reason else [])
        return ("PASS" if result.status == "COMPLETE" else "FAIL"), notes

    ctx = execute_run(
        command="forward-resume",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION, "forward_result": "forward-1"},
    )
    status = holder[0].status if holder else "RESOURCE_FAILURE"
    return ForwardOutcome(ctx=ctx, exit_code=forward_exit_code(status))


def run_synthetic_p1(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    seeds: Sequence[int],
    argv: Sequence[str] = (),
    julia: str | None = None,
) -> ForwardOutcome:
    """`synthetic-p1 --seeds ...`: render, run and publish the P1 parent worlds.

    Each world runs its closed preflight and is scored by the same artifact-based
    collector as the verification suite. Completion alone never grants acceptance.
    """
    accepted: list[bool] = []

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        for seed in seeds:
            if seed < 0:
                raise CommandError(f"a world seed is non-negative, got {seed}")
        families = dict(P1_PARENTS)
        tolerances = load_tolerances(paths.resolve("configs/e01_tolerances.yml"))
        profile = require_resource_profile(cfg.resources, command="synthetic-p1")
        write_environment_record(paths, ctx, profile)
        session = _single_session(cfg, paths, ctx, julia)
        rows: list[dict[str, Any]] = []
        try:
            for seed in seeds:
                world = render_p1(seed, P1Design(family=families.get(seed, "base")))
                preflight_ctx = RunContext.start(
                    command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
                )
                preflight_case = build_p1_case(closed_preflight_world(world), paths, preflight_ctx)
                preflight = simulate(
                    preflight_case,
                    preflight_output_request(world.design),
                    worker=session.worker(),
                    ctx=preflight_ctx,
                    ledger=session.ledger(f"preflight-{seed}"),
                    solver_config=E01_SOLVER,
                )
                preflight = _persist(preflight, paths)
                session.after_job()
                case_ctx = RunContext.start(
                    command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
                )
                case = build_p1_case(world, paths, case_ctx)
                request = output_request(world.design)
                result = simulate(
                    case,
                    request,
                    worker=session.worker(),
                    ctx=case_ctx,
                    ledger=session.ledger(f"world-{seed}"),
                    solver_config=E01_SOLVER,
                )
                result = _persist(result, paths)
                check_requested_outputs(result, request)
                session.after_job()
                world_ctx = RunContext.start(
                    command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
                )
                gates = score_p1_world(world, result, preflight, paths)
                manifest_ref = write_world(
                    world, result, paths, world_ctx, gates=gates, tolerances=tolerances
                )
                manifest = json.loads(paths.resolve(manifest_ref.path).read_text(encoding="utf-8"))
                rows.append(
                    world_row(
                        world,
                        result,
                        manifest,
                        manifest_path=manifest_ref.path,
                        gates=gates,
                    )
                )
                accepted.append(bool(manifest["accepted"]))
        finally:
            session.close()
        suite_ctx = RunContext.start(
            command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
        )
        ref = write_suite_manifest(rows, paths.reports / SUITE_MANIFEST_FILENAME, paths, suite_ctx)
        ctx.add_output("p1_suite_manifest", ref)
        command_log.info("published %d P1 worlds", len(rows))
        failed = [row["parent_world_id"] for row in rows if not row["accepted"]]
        return ("PASS" if rows and not failed else "FAIL"), [
            f"{len(rows)} worlds published, {len(failed)} not accepted: {failed}"
        ]

    ctx = execute_run(
        command="synthetic-p1",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION},
    )
    return ForwardOutcome(
        ctx=ctx, exit_code=0 if ctx.record.status == "PASS" and all(accepted) else 2
    )


BENCHMARK_FILENAME = "e01_benchmark.json"


def run_benchmark_forward(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    case_path: str,
    warm_runs: int,
    argv: Sequence[str] = (),
    julia: str | None = None,
) -> ForwardOutcome:
    """`benchmark-forward --case PATH --warm-runs N`: one cold run and N warm repeats.

    Every repeat is its own attempt with its own job id, reserved and resolved on the
    ledger, and this build has no production result cache for a repeat to hit: the bypass of
    plan 12.6 is structural here rather than a flag. Failed attempts keep their row and their
    spent time; they are not removed from the failure rate's denominator.
    """
    codes: list[int] = []

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        if warm_runs < MIN_WARM_RUNS:
            raise CommandError(
                f"--warm-runs is at least {MIN_WARM_RUNS}: a p50/p90 over fewer samples is a "
                f"guess with a percentile's name on it, got {warm_runs}"
            )
        case = load_case(paths.resolve(case_path), paths)
        profile = require_resource_profile(cfg.resources, command="benchmark-forward")
        write_environment_record(paths, ctx, profile)
        session = _single_session(cfg, paths, ctx, julia)
        request = _case_request(case, keep_native_restart=False)
        timings: list[float] = []
        failures: list[str] = []
        cold_s: float | None = None
        rows: list[dict[str, Any]] = []
        try:
            for index in range(warm_runs + 1):
                child_ctx = RunContext.start(
                    command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
                )
                started = time.monotonic()
                result = simulate(
                    case,
                    request,
                    worker=session.worker(),
                    ctx=child_ctx,
                    ledger=session.ledger(f"attempt-{index}"),
                    solver_config=E01_SOLVER,
                )
                elapsed = time.monotonic() - started
                session.after_job()
                rows.append(
                    {
                        "attempt": index,
                        "kind": "cold" if index == 0 else "warm",
                        "wall_s": elapsed,
                        "cpu_s": result.cost.cpu_s,
                        "peak_rss_bytes": result.cost.peak_rss_bytes,
                        "output_bytes": result.cost.output_bytes,
                        "accepted_steps": result.cost.accepted_steps,
                        "cut_steps": result.cost.cut_steps,
                        "nonlinear_iterations": result.cost.nonlinear_iterations,
                        "native_chunk_calls": _chunk_calls(result),
                        "status": result.status,
                    }
                )
                if index == 0:
                    cold_s = elapsed
                    continue
                if result.status == "COMPLETE":
                    timings.append(elapsed)
                else:
                    failures.append(f"attempt {index}: {result.status}")
        finally:
            session.close()
        payload: dict[str, Any] = {
            "case_id": case.case_id,
            "model_hash": case.model_hash,
            "cold_s": cold_s,
            "attempts": rows,
            "failed_attempts": failures,
            "failure_rate": len(failures) / max(len(rows) - 1, 1),
            "cache_bypass": (
                "structural: no caller in the forward path consults "
                "BudgetLedger.is_already_complete, so every repeat re-ran the solver"
            ),
        }
        if len(timings) >= MIN_WARM_RUNS:
            payload["warm"] = warm_summary(timings)
        (ctx.run_dir / BENCHMARK_FILENAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        command_log.info("benchmark wrote %d attempts", len(rows))
        codes.append(0 if not failures and len(timings) >= MIN_WARM_RUNS else 2)
        return ("PASS" if codes[-1] == 0 else "FAIL"), [
            f"{len(timings)} warm timings, {len(failures)} failed attempts"
        ]

    ctx = execute_run(
        command="benchmark-forward",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION},
    )
    return ForwardOutcome(ctx=ctx, exit_code=codes[0] if codes else 2)


def _figure_inputs(run_dirs: Sequence[Path], paths: ProjectPaths) -> dict[str, Path | None]:
    """Locate the published files the stage figures are drawn from.

    Only files a real session published, and preferably the ones the CITED runs published:
    `reports/p1_suite_manifest.json` is rewritten by every P1 session, so a report about an
    earlier session must not be illustrated with a later session's world. The manifest is
    used when it names an accepted world; otherwise the cited run directories themselves are
    searched for the largest COMPLETE forward they hold. A missing input is a figure that is
    not drawn, never one drawn from a default.
    """
    found: dict[str, Path | None] = {
        "states": None,
        "monthly": None,
        "balances": None,
        "benchmark": None,
    }
    suite_manifest = paths.reports / SUITE_MANIFEST_FILENAME
    if suite_manifest.is_file():
        payload = json.loads(suite_manifest.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if not row.get("accepted") or not row.get("manifest_path"):
                continue
            manifest = json.loads(
                paths.resolve(str(row["manifest_path"])).read_text(encoding="utf-8")
            )
            outputs = manifest.get("forward_outputs", {})
            state = outputs.get("state.so")
            if state:
                found["states"] = paths.resolve(str(state).split("#")[0])
            for role in ("monthly", "balances"):
                if outputs.get(role):
                    found[role] = paths.resolve(str(outputs[role]))
            break
    if found["states"] is None:
        _biggest_published_forward(run_dirs, paths, found)
    for run_dir in run_dirs:
        candidate = run_dir / BENCHMARK_FILENAME
        if candidate.is_file():
            found["benchmark"] = candidate
    return found


def _with_child_runs(run_dirs: Sequence[Path], paths: ProjectPaths) -> list[Path]:
    """The cited runs plus every run that names one of them as a parent.

    A suite's forwards are recorded as their OWN runs — that is what keeps a continuation
    out of its parent's result directory — so the artifacts a suite produced live beside it
    rather than under it, and a search that only looked inside the cited directories would
    find none of them.
    """
    cited = {run_dir.name for run_dir in run_dirs}
    out = list(run_dirs)
    if not paths.runs.is_dir():
        return out
    for candidate in sorted(paths.runs.iterdir()):
        if candidate.name in cited or not (candidate / "run.json").is_file():
            continue
        try:
            record = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cited.intersection(record.get("parent_run_ids", ())):
            out.append(candidate)
    return out


def _biggest_published_forward(
    run_dirs: Sequence[Path], paths: ProjectPaths, found: dict[str, Path | None]
) -> None:
    """The largest COMPLETE forward under the cited runs, as the figures' subject.

    Largest by published values — times times cells — because the figure that says most
    about the stage is the one drawn from the longest trajectory it really ran. The record
    is re-read through `load_forward_result`, so what is drawn is bytes that still verify.
    """
    best: tuple[int, ForwardResult] | None = None
    for run_dir in _with_child_runs(run_dirs, paths):
        for record_path in sorted(run_dir.rglob(RESULT_FILENAME)):
            try:
                result = load_forward_result(record_path, paths)
            except Exception:  # a record that no longer verifies is not drawn from
                continue
            state = result.states.get("so")
            if result.status != "COMPLETE" or state is None:
                continue
            size = int(state.shape[0]) * int(state.shape[1])
            if best is None or size > best[0]:
                best = (size, result)
    if best is None:
        return
    result = best[1]
    state = result.states["so"]
    found["states"] = paths.resolve(state.path)
    if result.monthly_path is not None:
        found["monthly"] = paths.resolve(result.monthly_path)
    if result.balances_path is not None:
        found["balances"] = paths.resolve(result.balances_path)


def run_e01_report(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    run_dirs: Sequence[str],
    argv: Sequence[str] = (),
) -> ForwardOutcome:
    """`e01-report --runs PATH ...`: build the stage report from real run directories."""
    from so_recon.validation.e01_report import (
        STAGE_REPORT_RELPATH,
        build_e01_report,
        render_e01_report,
    )
    from so_recon.validation.plots import FIGURES_RELDIR, write_e01_figures

    codes: list[int] = []

    def body(ctx: RunContext, command_log: logging.Logger) -> tuple[RunStatus, list[str]]:
        resolved: list[Path] = []
        for name in run_dirs:
            candidate = paths.resolve(name) if not Path(name).is_absolute() else Path(name)
            paths.relative(candidate)
            if not (candidate / "run.json").is_file():
                raise CommandError(
                    f"{name} is not a run directory: it carries no run.json. Plan 12.10 has "
                    "the report read the ACTUAL run directories a previous command printed, "
                    "never an invented run id"
                )
            resolved.append(candidate)
        if not resolved:
            raise CommandError("e01-report needs at least one run directory")
        inputs = _figure_inputs(resolved, paths)
        drawn = write_e01_figures(
            paths.root / FIGURES_RELDIR,
            states_path=inputs["states"],
            monthly_path=inputs["monthly"],
            balances_path=inputs["balances"],
            benchmark_path=inputs["benchmark"],
        )
        command_log.info("drew %d figure(s)", len(drawn))
        report = build_e01_report(tuple(resolved), paths)
        text = render_e01_report(report)
        published = paths.resolve(STAGE_REPORT_RELPATH)
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(text, encoding="utf-8")
        (ctx.run_dir / "E01.md").write_text(text, encoding="utf-8")
        (ctx.run_dir / "E01.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command_log.info("stage report written to %s", STAGE_REPORT_RELPATH)
        codes.append(0 if report.status in ("PASS", "PASS_WITH_LIMITATIONS") else 2)
        return ("PASS" if codes[-1] == 0 else "FAIL"), [f"stage status {report.status}"]

    ctx = execute_run(
        command="e01-report",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"run_record": RUN_RECORD_SCHEMA_VERSION},
    )
    return ForwardOutcome(ctx=ctx, exit_code=codes[0] if codes else 1)


__all__ = [
    "BENCHMARK_FILENAME",
    "DEFAULT_JOB_PLAN_RELPATH",
    "MANDATORY_CHECKS",
    "SUITES",
    "SUITE_REPORT_FILENAME",
    "CommandError",
    "ForwardOutcome",
    "JobOutcome",
    "JobPlan",
    "PlannedJob",
    "SuiteOutcome",
    "SuitePlan",
    "SuiteReport",
    "SuiteRun",
    "SuiteRunner",
    "evaluate_suite",
    "forward_exit_code",
    "load_job_plan",
    "run_benchmark_forward",
    "run_e01_report",
    "run_e01_suite",
    "run_forward",
    "run_forward_resume",
    "run_suite_jobs",
    "run_synthetic_p1",
    "suite_exit_code",
    "warm_summary",
]
