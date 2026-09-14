"""E01.12 — the bodies of the E01 commands. `cli.py` parses; everything below acts.

Six commands live here: `forward`, `forward-resume`, `verify-physics`, `synthetic-p1`,
`benchmark-forward` and `e01-report`. Each of them is a thin composition over work Tasks
1-11 already built — `simulate`/`resume`, the physics evaluator, the P1 generator and its
publisher — and none of them re-implements any of it.

The module is a composition and no longer a warehouse. What a suite RUNS lives in
`simulator/suites.py`; what a suite RECORDS lives in `simulator/suite_record.py`; which
published files the stage figures are drawn from lives in `validation/figure_inputs.py`.
That last split is what lets `run_e01_report` import `validation/e01_report.py` and
`validation/plots.py` at MODULE level: those two used to be imported inside the function
body to break a cycle this module had created by owning the schemas the report module
needed.

What remains here is the three things a command line needs and a library does not.

**An exit code that is not a lie.** `forward_exit_code` maps a `ForwardStatus` to a shell
status, and only `COMPLETE` is zero. A checkpoint written, a partial report rendered or a
world published is not, on its own, success: plan 12.1 is explicit that no command returns 0
merely because it wrote something. A suite that spent its wall budget exits 2 naming the jobs
it did not reach; a suite whose mandatory check failed exits 1.

**A record the stage report can read.** Every suite writes `e01_suite.json` into its run
directory: the planned jobs, what each attempt cost, and the `PhysicsCheck` verdicts
themselves. Plan 12.9 has the stage validator read published artifacts rather than re-run the
suite, so a metric that exists only inside this process is a metric the gate cannot cite.

**A stamp of the machine that really ran.** `write_environment_record` puts the lock hashes
and the host into each session's own run directory. It never rewrites E00's deterministic
`reports/environment_report.md` (plan 12.8).

Paths persisted here are project-relative, always.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon.config.resources import (
    ResourceProfile,
    require_resource_profile,
    resource_profile,
)
from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.environment.resources import HardwareProfile, probe_hardware
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RUN_RECORD_SCHEMA_VERSION, RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.simulator.case_io import load_case
from so_recon.simulator.contracts import (
    CaseBundle,
    ControlSegment,
    ForwardResult,
    OutputRequest,
    RestartRef,
)
from so_recon.simulator.forward import (
    check_requested_outputs,
    restart_ref_from_manifest,
    resume,
    simulate,
)
from so_recon.simulator.julia_bridge import find_julia
from so_recon.simulator.suite_record import (
    BENCHMARK_FILENAME,
    DEFAULT_JOB_PLAN_RELPATH,
    JOB_PLAN_SCHEMA_VERSION,
    MANDATORY_CHECKS,
    MIN_WARM_RUNS,
    SUITE_REPORT_FILENAME,
    SUITE_REPORT_SCHEMA_VERSION,
    SUITES,
    WARM_CV_RECOMMENDATION_THRESHOLD,
    CommandError,
    JobOutcome,
    JobPlan,
    PlannedJob,
    SuiteOutcome,
    SuitePlan,
    SuiteReport,
    evaluate_suite,
    forward_exit_code,
    load_job_plan,
    relative_evidence,
    suite_exit_code,
    warm_summary,
)
from so_recon.simulator.suites import (
    E01_SOLVER,
    ResourceFailure,
    Session,
    SuiteRun,
    SuiteRunner,
    chunk_calls,
    not_run_outcome,
    open_suite_run,
    persist_result,
    run_suite_jobs,
)
from so_recon.synthetic.acceptance import score_p1_world
from so_recon.synthetic.p1 import (
    P1_PARENTS,
    P1Design,
    closed_preflight_world,
    preflight_output_request,
    render_p1,
)
from so_recon.synthetic.p1 import output_request as p1_output_request
from so_recon.synthetic.world_io import (
    SUITE_MANIFEST_FILENAME,
    build_p1_case,
    world_row,
    write_suite_manifest,
    write_world,
)
from so_recon.validation.e01_report import (
    STAGE_REPORT_RELPATH,
    build_e01_report,
    render_e01_report,
)
from so_recon.validation.figure_inputs import figure_inputs
from so_recon.validation.physics import PhysicsCheck, load_tolerances
from so_recon.validation.plots import FIGURES_RELDIR, write_e01_figures

log = logging.getLogger(__name__)

#: The run-scoped environment record every physical E01 session leaves (plan 12.8). The
#: deterministic E00 report under `reports/` is NOT touched by any of this.
ENVIRONMENT_RECORD_FILENAME = "e01_environment.json"
ENVIRONMENT_RECORD_SCHEMA_VERSION: Literal["e01-environment-1"] = "e01-environment-1"


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
    replay: bool = False,
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
                open_suite_run(
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
                    replay=replay,
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
            checks=tuple(relative_evidence(check, paths) for check in outcome.checks),
            mandatory_checks=tuple(
                sorted({c.name for c in outcome.checks if c.name in MANDATORY_CHECKS})
            ),
            exploratory_checks=tuple(
                sorted({c.name for c in outcome.checks if c.name not in MANDATORY_CHECKS})
            ),
            remaining_job_ids=outcome.remaining_job_ids,
            stopped_reason=outcome.stopped_reason,
            resumed_from_run_id=outcome.resumed_from_run_id,
            reused_group_ids=outcome.reused_group_ids,
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
        jobs=tuple(not_run_outcome(job, reason) for job in plan.jobs),
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


# --------------------------------------------------------------------------------------
# forward / forward-resume / synthetic-p1 / benchmark-forward
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardOutcome:
    ctx: RunContext
    exit_code: int


def _single_session(
    cfg: ProjectConfig, paths: ProjectPaths, ctx: RunContext, julia: str | None
) -> Session:
    profile = require_resource_profile(cfg.resources, command="forward")
    session_dir = ctx.run_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = ctx.run_dir / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    return Session(
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
            result = persist_result(result, paths)
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
            result = persist_result(result, paths)
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
                preflight = persist_result(preflight, paths)
                session.after_job()
                case_ctx = RunContext.start(
                    command="forward", argv=[], cfg=cfg, paths=paths, parent_run_ids=(ctx.run_id,)
                )
                case = build_p1_case(world, paths, case_ctx)
                request = p1_output_request(world.design)
                result = simulate(
                    case,
                    request,
                    worker=session.worker(),
                    ctx=case_ctx,
                    ledger=session.ledger(f"world-{seed}"),
                    solver_config=E01_SOLVER,
                )
                result = persist_result(result, paths)
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
                        "native_chunk_calls": chunk_calls(result),
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


def run_e01_report(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    run_dirs: Sequence[str],
    argv: Sequence[str] = (),
) -> ForwardOutcome:
    """`e01-report --runs PATH ...`: build the stage report from real run directories."""
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
        inputs = figure_inputs(resolved, paths)
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
    "ENVIRONMENT_RECORD_FILENAME",
    "MANDATORY_CHECKS",
    "MIN_WARM_RUNS",
    "SUITES",
    "SUITE_REPORT_FILENAME",
    "WARM_CV_RECOMMENDATION_THRESHOLD",
    "CommandError",
    "E01EnvironmentRecord",
    "ForwardOutcome",
    "JobOutcome",
    "JobPlan",
    "PlannedJob",
    "ResourceFailure",
    "Session",
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
    "write_environment_record",
]
