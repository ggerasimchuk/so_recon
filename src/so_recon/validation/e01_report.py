"""E01.12.11 — the stage report, built from evidence and from nothing else.

`build_e01_report` reads RUN DIRECTORIES: the run records `execute_run` wrote, the suite
reports `verify-physics` published and the ledgers its sessions kept. It runs nothing, scores
nothing and re-simulates nothing — plan 12.9 has the stage validator read the artifacts a
real session left behind, so a number that only ever existed inside a test process is a
number this report cannot cite and does not.

Three rules the file exists to hold.

**A status never runs ahead of its evidence.** `NOT_RUN` is the status of a stage nobody has
run; `PASS_WITH_LIMITATIONS` is what an accepted oil-water scope carries while black oil is
not run; `FAIL` is what a failed mandatory check produces, and no amount of everything-else
being green changes it. There is no path through `_status` that returns PASS without every
mandatory check present and passing.

**Every path is project-relative.** The rendered report is committed under `reports/`, and
`tests/test_no_absolute_paths.py` guards the tracked set. Anything read out of a run record
goes through `ProjectPaths.relative` or is dropped.

**Cost accounting includes the forwards nobody charged.** Tasks 6, 9 and 10 each ruled that
verification forwards run through the test launcher are not charged to a `BudgetLedger` —
manufacturing a `JobDescriptor` and a case digest for them would add fiction rather than
safety — and plan 12.4 accounts their time with the launcher separately. They are summed
here under `launcher_forwards`, beside the ledger's own totals, so that the stage's real cost
is visible in one place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.commands import (
    MANDATORY_CHECKS,
    SUITE_REPORT_FILENAME,
    JobOutcome,
    SuiteReport,
)
from so_recon.validation.physics import PhysicsCheck

STAGE_REPORT_SCHEMA_VERSION: Literal["e01-stage-1"] = "e01-stage-1"
STAGE_REPORT_RELPATH = "reports/stages/E01.md"

StageStatus = Literal["NOT_RUN", "FAIL", "PASS_WITH_LIMITATIONS", "PASS"]

#: The physical class E01 verifies and the fluids it verifies it with. Stated in the report
#: because a reader who cites a number from it has to know what it is a number about: these
#: are EDUCATIONAL two-phase oil-water fluids on synthetic truth, and nothing here is a field
#: model, a posterior or a decision.
PHYSICAL_CLASS = "oil_water"
FLUID_CLAIM = (
    "educational two-phase oil-water (SPEC §3.1): rho(p) = rho_sc*exp(c*(p - p_sc)), Corey "
    "n=2 relative permeabilities, zero capillary pressure. Synthetic truth, exploratory "
    "split. Not a field model and not calibrated to one."
)

#: What E02 is allowed to start on, and what it is not. Stated so that an accepted oil-water
#: scope cannot be read as a claim about anything else.
NEXT_STAGE_NOTE = (
    "E02 (probabilistic inverse) is permitted on an ACCEPTED oil-water forward scope only. "
    "No field data, no PhysicsDecision and no ML component is claimed ready by this report: "
    "E01 verified a forward operator on synthetic truth and published no posterior of any "
    "kind."
)

#: The corrected per-phase criterion, shared by the evaluator and this report.
MONTHLY_DENOMINATOR_NOTE = (
    "Monthly phase errors use their own reference volume, never production plus injection "
    "throughput. Under e01-tolerances-2 the criterion is abs(fine-coarse) <= "
    "max(1e-6 m3_sc, 0.02*abs(coarse)): references up to 5e-5 m3_sc use the "
    "absolute 1 millilitre criterion; larger references use the 2% phase-relative "
    "criterion. Raw phase-relative errors remain reported. The corrected five-spot "
    "study uses 16x16 and 48x48 grids with unchanged well coordinates and completed "
    "lengths. Old 16x16-to-32x32 results do not establish this criterion."
)

#: What a resource failure in this stage means, and what it does not. SPEC 18.4: a resource
#: failure is never a physical zero, so a check whose forwards were stopped by the memory
#: guard is NOT_RUN and never a pass.
RESOURCE_NOTE = (
    "Host memory and the ledger-accounted groups. `budget.effective_hard_bytes` caps a "
    "session at `min(hard_bytes, total - reserve, process_rss + available - reserve)`, and "
    "COMPUTE §5 fixes `reserve_bytes` at 6 GiB. A warm JutulDarcy worker holds about 1.3 GiB, "
    "so the groups that go through the persistent worker — the restart round trip, the "
    "isolation quadruple, the P1 worlds and the benchmark — need roughly 7.5 GiB free before "
    "the guard will let them run, while the LAUNCHER groups (the analytic, operational, "
    "controls and refinement diagnostics) are subprocesses the watchdog does not bound and "
    "run regardless. On a host whose other applications hold the rest of the memory, the same "
    "suite therefore passes or records RESOURCE_FAILURE depending on what else is running. "
    "That is the guard working, not a physics result: SPEC 18.4 makes a resource failure "
    "NOT_RUN and never a pass, and a stage accepted on such a session is not accepted."
)

#: The published-manifest distinction Task 1 introduced, closed here (plan 12.11 lineage).
MANIFEST_NAMING_NOTE = (
    "Published source manifests are one file per spec version "
    "(`so_recon.cli.PUBLISHED_MANIFEST_NAMES`): a 3.0 run writes "
    "`reports/manifests/source_manifest.json` and a 4.0 run writes "
    "`reports/manifests/source_manifest-4.0.json`. The two describe the same sources under "
    "different specifications and the historical one is frozen; no run record stated which "
    "of the two it wrote, so it is stated here."
)


class CommandRecord(StrictModel):
    """One command that ran, as its own run record describes it."""

    run_id: str
    run_dir: str
    command: str
    argv: tuple[str, ...]
    status: str
    exit_code: int | None
    created_at: str
    finished_at: str | None
    git_commit: str | None
    git_dirty: bool | None
    spec_version: str
    config_version: str
    resolved_config_hash: str
    environment_lock_hash: str
    notes: tuple[str, ...]


class CostAccount(StrictModel):
    """Everything the stage spent, with the two accounting routes kept apart."""

    ledger_forwards: int
    ledger_wall_s: float
    ledger_cpu_s: float
    ledger_output_bytes: int
    launcher_forwards: int
    launcher_wall_s: float
    peak_rss_bytes: int
    restart_write_s: float | None
    restart_read_s: float | None
    note: str


class BudgetForecast(StrictModel):
    """E01's measured unit costs and the headroom they leave. Never an E02 estimate."""

    measured_unit_cost_s: dict[str, float]
    jobs_this_matrix: dict[str, int]
    headroom: dict[str, Any]
    training_cost: Literal["not_applicable_in_E01"] = "not_applicable_in_E01"
    smc_cost: Literal["not_applicable_in_E01"] = "not_applicable_in_E01"
    adjoint_cost: Literal["not_applicable_in_E01"] = "not_applicable_in_E01"
    next_stage_total: Literal["not_estimated_without_E02_job_plan"] = (
        "not_estimated_without_E02_job_plan"
    )


class StageReport(StrictModel):
    """The whole of what E01 can say about itself, from artifacts alone."""

    schema_version: Literal["e01-stage-1"] = STAGE_REPORT_SCHEMA_VERSION
    stage: Literal["E01"] = "E01"
    status: StageStatus
    status_reason: str
    generated_at: str

    spec_version: str
    config_version: str
    resolved_config_hash: str
    environment_lock_hash: str
    lock_hashes: dict[str, str]
    tolerances_path: str
    tolerances_sha256: str
    job_plan_path: str
    job_plan_sha256: str
    source_manifest: dict[str, str]
    git_commit: str | None
    git_dirty: bool | None

    commands: tuple[CommandRecord, ...]
    suites: dict[str, str]
    checks: tuple[PhysicsCheck, ...]
    required_checks: tuple[str, ...]
    complete_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    unrun_checks: tuple[str, ...]
    jobs: tuple[JobOutcome, ...]
    remaining_job_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    ow_gate: str
    bo_status: Literal["NOT_RUN", "PASS", "FAIL"]
    physical_class: str
    fluid_claim: str
    restart_round_trip: dict[str, Any]
    benchmark: dict[str, Any]
    costs: CostAccount
    budget: BudgetForecast
    artifacts: dict[str, str]
    plots: tuple[str, ...]
    acceptance_notes: tuple[str, ...]
    next_stage: str


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _command_record(run_dir: Path, paths: ProjectPaths) -> CommandRecord | None:
    record = _read_json(run_dir / "run.json")
    if record is None:
        return None
    suite = _read_json(run_dir / SUITE_REPORT_FILENAME)
    return CommandRecord(
        run_id=str(record["run_id"]),
        run_dir=paths.relative(run_dir),
        command=str(record["command"]),
        argv=tuple(str(a) for a in record.get("argv", ())),
        status=str(record["status"]),
        exit_code=None if suite is None else int(suite["exit_code"]),
        created_at=str(record["created_at"]),
        finished_at=record.get("finished_at"),
        git_commit=record.get("git_commit"),
        git_dirty=record.get("git_dirty"),
        spec_version=str(record.get("spec_version", "unavailable")),
        config_version=str(record.get("config_version", "unavailable")),
        resolved_config_hash=str(record.get("resolved_config_hash", "unavailable")),
        environment_lock_hash=str(record.get("environment_lock_hash", "unavailable")),
        notes=tuple(str(n) for n in record.get("notes", ())),
    )


def _status(
    required: Sequence[str],
    failed: Sequence[str],
    unrun: Sequence[str],
    remaining: Sequence[str],
    bo_status: str,
    *,
    any_evidence: bool,
) -> tuple[StageStatus, str]:
    """The stage verdict. There is no branch here that outruns its evidence."""
    if not any_evidence:
        return "NOT_RUN", "no E01 suite has been run: there is nothing to accept or refuse"
    if failed:
        return "FAIL", f"mandatory checks failed: {sorted(failed)}"
    if unrun:
        return "FAIL", f"mandatory checks did not run: {sorted(unrun)}"
    missing = sorted(set(MANDATORY_CHECKS) - set(required))
    if missing:
        return (
            "FAIL",
            f"the stage matrix is missing mandatory checks {missing}; a matrix with a hole "
            "in it is not a passing matrix (plan 12.9)",
        )
    if remaining:
        return "FAIL", f"planned jobs were never reached: {sorted(remaining)}"
    if bo_status == "NOT_RUN":
        return (
            "PASS_WITH_LIMITATIONS",
            "every mandatory oil-water check passed and the black-oil capability is NOT_RUN",
        )
    return "PASS", "every mandatory check passed and black oil ran"


def _costs(jobs: Sequence[JobOutcome], ledgers: Sequence[Mapping[str, Any]]) -> CostAccount:
    ledger_jobs = [job for job in jobs if job.accounting == "ledger" and job.status != "NOT_RUN"]
    launcher_jobs = [
        job for job in jobs if job.accounting == "launcher" and job.status != "NOT_RUN"
    ]
    writes = [job.restart_write_s for job in jobs if job.restart_write_s is not None]
    reads = [job.restart_read_s for job in jobs if job.restart_read_s is not None]
    return CostAccount(
        ledger_forwards=len(ledger_jobs),
        ledger_wall_s=sum(job.wall_s for job in ledger_jobs),
        ledger_cpu_s=sum(job.cpu_s for job in ledger_jobs),
        ledger_output_bytes=sum(job.output_bytes for job in ledger_jobs),
        launcher_forwards=len(launcher_jobs),
        launcher_wall_s=sum(job.wall_s for job in launcher_jobs),
        peak_rss_bytes=max((job.peak_rss_bytes for job in jobs), default=0),
        restart_write_s=max(writes) if writes else None,
        restart_read_s=max(reads) if reads else None,
        note=(
            "Two accounting routes, kept apart on purpose. LEDGER forwards are worker-submitted "
            f"`JobDescriptor`s reserved and resolved on a `BudgetLedger` ({len(ledgers)} session "
            "ledger(s) read). LAUNCHER forwards ran inside a Julia verification diagnostic and "
            "are NOT charged to a ledger: Tasks 6, 9 and 10 each ruled that manufacturing "
            "synthetic `ForwardResult`s with invented case digests to charge them would add "
            "fiction rather than safety, and plan 12.4 accounts their time with the test "
            "launcher separately. Both are summed here so the stage's real cost is in one place."
        ),
    )


def _budget(jobs: Sequence[JobOutcome]) -> BudgetForecast:
    by_group: dict[str, list[float]] = {}
    for job in jobs:
        if job.status == "NOT_RUN":
            continue
        by_group.setdefault(job.group, []).append(job.wall_s)
    unit = {group: sum(v) / len(v) for group, v in by_group.items() if v}
    counts = {group: len(v) for group, v in by_group.items()}
    return BudgetForecast(
        measured_unit_cost_s=unit,
        jobs_this_matrix=counts,
        headroom={
            "p0_declared_jobs": 23,
            "p0_session_forward_ceiling": 64,
            "p1_declared_jobs": 17,
            "p1_session_forward_ceiling": 2000,
            "bo_declared_jobs": 4,
            "note": (
                "Counts for the NEXT experiment are not approved by this plan. These are E01's "
                "own measured unit costs and the exact matrix this stage ran; a forward-only "
                "measurement is not a full ML/SMC budget and is not offered as one."
            ),
        },
    )


def build_e01_report(run_dirs: tuple[Path, ...], paths: ProjectPaths) -> StageReport:
    """Read real run directories into the stage's verdict. Nothing here runs anything."""
    commands: list[CommandRecord] = []
    suites: dict[str, SuiteReport] = {}
    ledgers: list[Mapping[str, Any]] = []
    for run_dir in run_dirs:
        record = _command_record(run_dir, paths)
        if record is not None:
            commands.append(record)
        payload = _read_json(run_dir / SUITE_REPORT_FILENAME)
        if payload is not None:
            report = SuiteReport.model_validate(payload)
            suites[report.suite] = report
        # One account per run of a model (see `commands._Session.ledger`), so a run holds a
        # DIRECTORY of them rather than a single file.
        for ledger_path in sorted((run_dir / "ledgers").glob("*.json")):
            ledger = _read_json(ledger_path)
            if ledger is not None:
                ledgers.append(ledger)

    checks: list[PhysicsCheck] = []
    jobs: list[JobOutcome] = []
    remaining: list[str] = []
    limitations: list[str] = []
    artifacts: dict[str, str] = {}
    benchmark: dict[str, Any] = {}
    for name in sorted(suites):
        report = suites[name]
        checks.extend(report.checks)
        jobs.extend(report.jobs)
        remaining.extend(report.remaining_job_ids)
        limitations.extend(report.limitations)
        artifacts.update({f"{name}.{k}": v for k, v in report.artifacts.items()})
        limitations.extend(
            f"{name}/{job.job_id}: RESOURCE_FAILURE — {job.reason}"
            for job in report.jobs
            if job.status == "RESOURCE_FAILURE"
        )
        if report.benchmark:
            benchmark[name] = report.benchmark

    present = {check.name for check in checks}
    failed = sorted({c.name for c in checks if c.status == "FAIL" and c.name in MANDATORY_CHECKS})
    unrun = sorted({c.name for c in checks if c.status == "NOT_RUN" and c.name in MANDATORY_CHECKS})
    complete = sorted({c.name for c in checks if c.status == "PASS"})
    bo_report = suites.get("bo")
    bo_status: Literal["NOT_RUN", "PASS", "FAIL"] = "NOT_RUN"
    if bo_report is not None and bo_report.exit_code == 0:
        bo_status = "PASS"
    elif bo_report is not None and bo_report.exit_code == 1:
        bo_status = "FAIL"

    status, reason = _status(
        sorted(present & MANDATORY_CHECKS),
        failed,
        unrun,
        sorted(set(remaining)),
        bo_status,
        any_evidence=bool(suites),
    )
    anchor = suites.get("p0") or (next(iter(suites.values())) if suites else None)
    lock_hashes = {
        name: (sha256_file(paths.root / name) if (paths.root / name).is_file() else "missing")
        for name in ("uv.lock", "julia/Manifest.toml", "julia/.julia-version")
    }
    manifests = {
        name: sha256_file(paths.manifests / name)
        for name in ("source_manifest.json", "source_manifest-4.0.json")
        if (paths.manifests / name).is_file()
    }
    restart = next(
        (c.model_dump(mode="json") for c in checks if c.name == "restart_round_trip"), {}
    )
    plots = tuple(
        sorted(paths.relative(p) for p in (paths.reports / "figures").glob("*.png") if p.is_file())
    )
    return StageReport(
        status=status,
        status_reason=reason,
        generated_at=datetime.now(UTC).isoformat(),
        spec_version=commands[0].spec_version if commands else "unavailable",
        config_version=commands[0].config_version if commands else "unavailable",
        resolved_config_hash=commands[0].resolved_config_hash if commands else "unavailable",
        environment_lock_hash=commands[0].environment_lock_hash if commands else "unavailable",
        lock_hashes=lock_hashes,
        tolerances_path=anchor.tolerances_path if anchor else "configs/e01_tolerances.yml",
        tolerances_sha256=anchor.tolerances_sha256 if anchor else "unavailable",
        job_plan_path=anchor.job_plan_path if anchor else "configs/e01_jobs.json",
        job_plan_sha256=anchor.job_plan_sha256 if anchor else "unavailable",
        source_manifest=manifests,
        git_commit=commands[0].git_commit if commands else None,
        git_dirty=any(bool(c.git_dirty) for c in commands) if commands else None,
        commands=tuple(commands),
        suites={name: f"exit {report.exit_code}" for name, report in sorted(suites.items())},
        checks=tuple(checks),
        required_checks=tuple(sorted(MANDATORY_CHECKS)),
        complete_checks=tuple(complete),
        failed_checks=tuple(failed),
        unrun_checks=tuple(unrun),
        jobs=tuple(jobs),
        remaining_job_ids=tuple(sorted(set(remaining))),
        limitations=tuple(limitations),
        ow_gate=(
            "PASS" if not failed and not unrun and bool(suites) else "FAIL" if failed else "NOT_RUN"
        ),
        bo_status=bo_status,
        physical_class=PHYSICAL_CLASS,
        fluid_claim=FLUID_CLAIM,
        restart_round_trip=restart,
        benchmark=benchmark,
        costs=_costs(jobs, ledgers),
        budget=_budget(jobs),
        artifacts=artifacts,
        plots=plots,
        acceptance_notes=(
            MONTHLY_DENOMINATOR_NOTE,
            MANIFEST_NAMING_NOTE,
            RESOURCE_NOTE,
            "`reports/p1_suite_manifest.json` is written to one published path by every P1 "
            "session (Task 11's publication contract), so it always describes the MOST "
            "RECENT one — which is not necessarily a session this report cites. The run "
            "directories named under 'Commands and exit status' above are what this report "
            "is about; the per-suite records inside them are immutable.",
            "Legacy regression: `make smoke` and the E00 unit suite run unchanged under "
            "`scripts/e01_gate.sh`; the deterministic E00 environment report is not rewritten "
            "by any E01 command, and each E01 session records its own lock hashes and host in "
            "`e01_environment.json` inside its run directory (plan 12.8).",
            "Every forward listed here is on SYNTHETIC truth with educational fluids. No field "
            "data was read, no posterior was computed and no decision is claimed.",
        ),
        next_stage=NEXT_STAGE_NOTE,
    )


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def render_e01_report(report: StageReport) -> str:
    """The committed stage page. Every number in it came out of a published artifact."""
    lines: list[str] = [
        f"# Stage E01 — {report.status}",
        "",
        f"**{report.status_reason}**",
        "",
        f"Generated {report.generated_at} from published run artifacts only. This page runs "
        "nothing: plan 12.9 has the stage validator read what a real session left behind.",
        "",
        "## Identity",
        "",
    ]
    lines += _table(
        ("key", "value"),
        [
            ("stage", report.stage),
            ("status", report.status),
            ("spec_version", report.spec_version),
            ("config_version", report.config_version),
            ("resolved_config_hash", f"`{report.resolved_config_hash}`"),
            ("environment_lock_hash", f"`{report.environment_lock_hash}`"),
            ("git_commit", f"`{report.git_commit}`"),
            ("git_dirty", report.git_dirty),
            ("physical_class", report.physical_class),
            ("tolerances", f"`{report.tolerances_path}` sha256 `{report.tolerances_sha256}`"),
            ("job plan", f"`{report.job_plan_path}` sha256 `{report.job_plan_sha256}`"),
        ],
    )
    lines += ["### Lock and input hashes", ""]
    lines += _table(
        ("file", "sha256"),
        [(f"`{k}`", f"`{v}`") for k, v in sorted(report.lock_hashes.items())]
        + [
            (f"`reports/manifests/{k}`", f"`{v}`")
            for k, v in sorted(report.source_manifest.items())
        ],
    )
    lines += ["## Commands and exit status", ""]
    lines += _table(
        ("command", "run_id", "record", "exit", "argv"),
        [
            (
                f"`{c.command}`",
                f"`{c.run_id}`",
                c.status,
                "—" if c.exit_code is None else c.exit_code,
                "`" + " ".join(c.argv) + "`" if c.argv else "—",
            )
            for c in report.commands
        ],
    )
    lines += ["## Verification matrix", ""]
    lines += _table(
        ("check", "status", "mandatory", "worst metric vs threshold", "reason"),
        [
            (
                f"`{check.name}`",
                check.status,
                "yes" if check.name in MANDATORY_CHECKS else "no",
                _worst(check),
                (check.reason or "")[:160],
            )
            for check in sorted(report.checks, key=lambda c: c.name)
        ],
    )
    lines += [
        f"* required: `{', '.join(report.required_checks)}`",
        f"* passing: `{', '.join(report.complete_checks) or 'none'}`",
        f"* failed: `{', '.join(report.failed_checks) or 'none'}`",
        f"* unrun (mandatory): `{', '.join(report.unrun_checks) or 'none'}`",
        "",
        "### Metrics and thresholds, in full",
        "",
    ]
    for check in sorted(report.checks, key=lambda c: c.name):
        if not check.metrics and not check.thresholds:
            continue
        lines += [f"#### `{check.name}` — {check.status}", ""]
        lines += _table(
            ("metric", "value", "threshold", "limit"),
            [
                (
                    f"`{name}`",
                    f"{value:.6g}",
                    "",
                    "",
                )
                for name, value in sorted(check.metrics.items())
            ]
            + [
                ("", "", f"`{name}`", f"{value:.6g}")
                for name, value in sorted(check.thresholds.items())
            ],
        )
    lines += ["## Jobs, failures and retries", ""]
    lines += _table(
        (
            "job",
            "group",
            "accounting",
            "status",
            "expected",
            "wall s",
            "cpu s",
            "peak RSS",
            "out bytes",
            "chunks",
            "accepted",
            "cut",
            "newton",
            "retry",
        ),
        [
            (
                f"`{job.job_id}`",
                job.group,
                job.accounting,
                job.status,
                job.expected_outcome,
                _number(job.wall_s),
                _number(job.cpu_s),
                job.peak_rss_bytes,
                job.output_bytes,
                job.native_chunk_calls,
                job.accepted_steps,
                job.cut_steps,
                job.nonlinear_iterations,
                job.retry_count,
            )
            for job in report.jobs
        ],
    )
    if report.remaining_job_ids:
        lines += [
            "**Jobs that were planned and never reached:** "
            + ", ".join(f"`{j}`" for j in report.remaining_job_ids),
            "",
        ]
    lines += ["## Restart round trip", ""]
    if report.restart_round_trip:
        lines += _table(
            ("metric", "value"),
            [
                (f"`{k}`", f"{v:.6g}")
                for k, v in sorted(report.restart_round_trip.get("metrics", {}).items())
            ],
        )
    else:
        lines += ["_(not run)_", ""]
    lines += ["## Cost", ""]
    costs = report.costs
    lines += _table(
        ("quantity", "value"),
        [
            ("ledger forwards", costs.ledger_forwards),
            ("ledger wall s", _number(costs.ledger_wall_s)),
            ("ledger cpu s", _number(costs.ledger_cpu_s)),
            ("ledger output bytes", costs.ledger_output_bytes),
            ("launcher forwards", costs.launcher_forwards),
            ("launcher wall s", _number(costs.launcher_wall_s)),
            ("peak process-tree RSS bytes", costs.peak_rss_bytes),
            ("restart write s", _number(costs.restart_write_s)),
            ("restart read s", _number(costs.restart_read_s)),
        ],
    )
    lines += [costs.note, ""]
    lines += ["### Measured unit cost and headroom", ""]
    lines += _table(
        ("group", "jobs", "mean wall s"),
        [
            (group, report.budget.jobs_this_matrix.get(group, 0), _number(value))
            for group, value in sorted(report.budget.measured_unit_cost_s.items())
        ],
    )
    lines += [
        f"* training cost: `{report.budget.training_cost}`",
        f"* SMC cost: `{report.budget.smc_cost}`",
        f"* adjoint cost: `{report.budget.adjoint_cost}`",
        f"* next-stage total: `{report.budget.next_stage_total}`",
        "",
        str(report.budget.headroom["note"]),
        "",
    ]
    if report.benchmark:
        lines += ["## Benchmark (exploratory)", "", "```json"]
        lines += [json.dumps(report.benchmark, indent=2, sort_keys=True)]
        lines += ["```", ""]
    lines += ["## Black oil", ""]
    lines += [
        f"**BO status: {report.bo_status}.** The black-oil capability is a separate P0 "
        "session and is not part of this stage's evidence. `PASS_WITH_LIMITATIONS` is what "
        "an accepted oil-water scope carries while it is NOT_RUN; it is never what a failed "
        "mandatory oil-water, restart or balance check carries.",
        "",
    ]
    if report.limitations:
        lines += ["## Limitations", ""]
        lines += [f"* {item}" for item in report.limitations]
        lines += [""]
    lines += ["## Artifacts", ""]
    lines += _table(
        ("key", "path"),
        [(f"`{k}`", f"`{v}`") for k, v in sorted(report.artifacts.items())]
        + [("figure", f"`{p}`") for p in report.plots],
    )
    lines += ["## Physical class and fluids", "", report.fluid_claim, ""]
    lines += ["## Acceptance notes", ""]
    lines += [f"* {note}" for note in report.acceptance_notes]
    lines += ["", "## Next stage", "", report.next_stage, ""]
    return "\n".join(lines)


#: Metrics that are REPORTED beside a gate and never gated by it: a boolean that says
#: whether the stricter aim was met, a count, and the aim itself. Pairing one of them with a
#: threshold in the summary column would name a comparison nobody makes.
_UNGATED_METRIC_SUFFIXES = ("_met", "_target", "_forwards", "_cells", "_zones", "_months")


def _worst(check: PhysicsCheck) -> str:
    """The metric closest to (or furthest past) its own gate, for the summary row.

    Best effort, and it says so with an em dash when it cannot pair a metric with a
    threshold: `PhysicsCheck` records the two as separate maps and does not record which
    threshold gated which metric. The full metric and threshold tables below are the
    authority; this column is a reading aid.
    """
    if not check.metrics or not check.thresholds:
        return "—"
    best: tuple[float, str] | None = None
    for name, value in check.metrics.items():
        if name.startswith("n_") or name.endswith(_UNGATED_METRIC_SUFFIXES):
            continue
        for threshold, limit in check.thresholds.items():
            stem = threshold.removesuffix("_max").removesuffix("_min")
            if not name.startswith(stem) or limit <= 0:
                continue
            ratio = value / limit
            if best is None or ratio > best[0]:
                best = (ratio, f"`{name}`={value:.3g} vs `{threshold}`={limit:.3g}")
    return best[1] if best else "—"
