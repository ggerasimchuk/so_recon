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
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.suite_record import (
    BO_CHECKS,
    MANDATORY_CHECKS,
    SUITE_REPORT_FILENAME,
    JobOutcome,
    SuiteReport,
)
from so_recon.validation.physics import BLACKOIL_GATES, PhysicsCheck
from so_recon.validation.plots import FIGURES_RELDIR

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

#: Which of the caps a session declares are really imposed at SESSION scope, and which are
#: not. E01 opens one `BudgetLedger` per run of a model (`Session.ledger`), which is what lets
#: the isolation quadruple, the restart triple and the warm repeats be separate work rather
#: than retries — and the price is that no single account sees the session's totals. One of
#: the three cumulative caps is re-imposed above the ledgers and two are not. A load-bearing
#: comment that claimed otherwise is the defect class this stage has already hit; the page
#: states the position instead of leaving a reader to assume all of them bite.
SESSION_CAP_NOTE = (
    "Session-scope budget caps. `BudgetLedger.reserve` enforces three CUMULATIVE caps per "
    "account — the forward count against `max_new_forward`, the published bytes against "
    "`disk_budget_bytes`, and the elapsed wall against `wall_budget_s` — and only the "
    "COMPUTE §7 disk budget is re-imposed across the whole session (`Session.admit_output`, "
    "over the bytes every publishing route charges). The session FORWARD COUNT is not: "
    "`max_new_forward` is compared against ONE account's own entries, and E01 opens one "
    "account per run of a model, so each sees a handful and nothing sums a session's "
    "forwards. The caps are 64 under `P0_VERIFY` (`p0` and `bo`) and 2000 under `P1_LOOP` "
    "(`p1`); the declared matrices produce 30 job rows for `p0`, 23 for `p1` and 4 for `bo`, "
    "so no overrun is reachable through the plan's own job lists. The session WALL is "
    "checked at GROUP BOUNDARIES only — `run_suite_jobs` consults `SuiteRun.out_of_time()` "
    "before entering the next group, so a group already running is not interrupted; inside "
    "one, `job_timeout_s` bounds each job, the launcher subprocess and the persistent "
    "worker's per-job deadline alike. SPEC §3.3's two attempts per model hash IS enforced, "
    "and enforced per account: `reserve` counts the prior attempts on that ledger, and one "
    "run of a model is one account by design, so the rule does not bound a session's "
    "attempts across accounts. None of the three is a live overrun — the job lists are "
    "fixed and far under every forward cap, and the wall is bounded coarsely rather than "
    "not at all — and all three are described here rather than enforced, so that no reader "
    "takes an unenforced scope for a guarantee."
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


#: What the black-oil capability's own session came back as, as the stage page reports it.
#:
#: `NOT_RUN` means there is no published black-oil session at all. The other three are that
#: session's own exit code, and the third one matters: `evaluate_suite` returns 2 for a
#: session that did not reach every declared job — including one REFUSED outright, which is
#: what the black-oil group does when the two sides of the restart split disagree. Folding
#: that into `NOT_RUN` would let a capability that was attempted and refused read as one
#: nobody ever attempted, which is the same defect class as a failed one reading as a pass.
BlackOilStatus = Literal["NOT_RUN", "PASS", "FAIL", "INCOMPLETE"]

#: The exit codes `evaluate_suite` produces, and what each says about the capability.
_BO_STATUS_BY_EXIT: dict[int, BlackOilStatus] = {0: "PASS", 1: "FAIL"}

#: The name the black-oil capability's suite publishes itself under. It is the one suite whose
#: outcome is a SEPARATE gate (plan 13.5), so it is the one name this module has to be able to
#: tell apart when it merges what every session left behind.
BO_SUITE = "bo"


def _bo_status(report: SuiteReport | None) -> BlackOilStatus:
    """The black-oil verdict, from the session's own exit code and nothing else."""
    if report is None:
        return "NOT_RUN"
    return _BO_STATUS_BY_EXIT.get(report.exit_code, "INCOMPLETE")


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
    #: `None` where nothing measured it. A launcher forward has no process of its own to
    #: weigh, so summing its absence as a zero would publish a total nobody measured.
    ledger_cpu_s: float | None
    ledger_output_bytes: int | None
    launcher_forwards: int
    launcher_wall_s: float
    peak_rss_bytes: int | None
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
    #: Every DISTINCT commit the cited runs were produced at, in cited order.
    #:
    #: `git_commit` above is the first cited run's, and for most reports every cited run
    #: shares it — but a report that cites a session republished after a later commit beside
    #: sessions from an earlier one has two, and a single field cannot say so. Its sibling
    #: `git_dirty` already aggregates with `any(...)`, which is exactly what made the single
    #: `git_commit` read as a stage-wide claim it was not. 12.11 names «проверенный
    #: commit/dirty» an element of this page, so the split is recorded here, rendered in the
    #: Identity table and in the Commands table, and named as a limitation.
    git_commits: tuple[str, ...]
    git_dirty: bool | None

    commands: tuple[CommandRecord, ...]
    suites: dict[str, str]
    checks: tuple[PhysicsCheck, ...]
    required_checks: tuple[str, ...]
    complete_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    unrun_checks: tuple[str, ...]
    jobs: tuple[JobOutcome, ...]
    #: Every planned job no session reached, black-oil ones included, so the page names them
    #: all. The stage VERDICT is taken on the oil-water ones alone; which of these are the
    #: capability's is the field below.
    remaining_job_ids: tuple[str, ...]
    #: The black-oil capability's own unreached jobs, a subset of the field above. Its verdict
    #: is a separate gate (plan 13.5), so these never move the stage status or the OW gate.
    black_oil_remaining_job_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    ow_gate: str
    bo_status: BlackOilStatus
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
    #: Oil-water jobs that were planned and never reached. Black-oil ones are deliberately
    #: not in here; see `build_e01_report`.
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
    # OIL-WATER jobs only. A black-oil session's unreached jobs are kept out of this list by
    # `build_e01_report` and reported through `bo_status` instead; this rule is what stops a
    # half-run p0 or p1 from passing and is unchanged for them.
    if remaining:
        return "FAIL", f"planned jobs were never reached: {sorted(remaining)}"
    # A black-oil capability that FAILED, or that was REFUSED and left incomplete, is a
    # limitation on this stage exactly as one that never ran is. None of the three is a stage
    # FAIL — plan 13.5 is explicit that an oil-water deliverable is never removed or failed
    # because a black-oil benchmark is missing or failed — but none of them is a clean PASS
    # either, and the reason line has to NAME which one happened.
    #
    # Before Task 13 only `NOT_RUN` was reachable: every session there was had that status, so
    # a FAIL fell through to `return "PASS", "... and black oil ran"` and the stage read PASS
    # with no mention of the failure anywhere in that line. `INCOMPLETE` became reachable when
    # the black-oil group gained a refusal of its own — the split-step guard, which exits 2 —
    # and folding it into `NOT_RUN` would have made a refused capability read as an
    # unattempted one.
    if bo_status in ("NOT_RUN", "FAIL", "INCOMPLETE"):
        return (
            "PASS_WITH_LIMITATIONS",
            f"every mandatory oil-water check passed and the black-oil capability is {bo_status}",
        )
    return "PASS", "every mandatory check passed and black oil ran"


def _total(values: Sequence[float | int | None]) -> float | None:
    """The sum of what was measured, or None when nothing was.

    An absent measurement is not a zero contribution: it is an unknown one, and a total
    that quietly treated it as zero would understate the stage's cost without saying so.
    """
    present = [v for v in values if v is not None]
    return float(sum(present)) if present else None


def _ow_gate(status: StageStatus) -> str:
    """The oil-water gate, read off the stage verdict itself.

    `_status` is the only place that decides whether this stage's matrix is complete and
    green, and the OW gate is a name for exactly that decision about the oil-water scope. A
    second, looser computation of the same thing is how `ow_gate: PASS` came to sit inside
    the JSON of a report whose rendered status was FAIL.
    """
    if status == "NOT_RUN":
        return "NOT_RUN"
    return "PASS" if status in ("PASS", "PASS_WITH_LIMITATIONS") else "FAIL"


def _costs(jobs: Sequence[JobOutcome], ledgers: Sequence[Mapping[str, Any]]) -> CostAccount:
    ledger_jobs = [job for job in jobs if job.accounting == "ledger" and job.status != "NOT_RUN"]
    launcher_jobs = [
        job for job in jobs if job.accounting == "launcher" and job.status != "NOT_RUN"
    ]
    writes = [job.restart_write_s for job in jobs if job.restart_write_s is not None]
    reads = [job.restart_read_s for job in jobs if job.restart_read_s is not None]
    cpu = _total([job.cpu_s for job in ledger_jobs])
    written = _total([job.output_bytes for job in ledger_jobs])
    peaks = [job.peak_rss_bytes for job in jobs if job.peak_rss_bytes is not None]
    return CostAccount(
        ledger_forwards=len(ledger_jobs),
        ledger_wall_s=sum(job.wall_s for job in ledger_jobs),
        ledger_cpu_s=cpu,
        ledger_output_bytes=None if written is None else int(written),
        launcher_forwards=len(launcher_jobs),
        launcher_wall_s=sum(job.wall_s for job in launcher_jobs),
        peak_rss_bytes=max(peaks) if peaks else None,
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


P1_SUITE_MANIFEST_RELPATH = "reports/p1_suite_manifest.json"

#: Where `e01-report` records which session each committed figure was drawn from.
FIGURE_PROVENANCE_RELPATH = "reports/figures/e01_figures.json"


def write_figure_provenance(
    paths: ProjectPaths, *, run_id: str, cited: Sequence[Path], drawn: Sequence[Path]
) -> Path:
    """Record which session drew the committed figures, and from which runs.

    `reports/figures/*.png` are committed to one set of paths, so a report that could NOT
    draw them — because the session it cites published no COMPLETE forward to draw from —
    would otherwise list the PREVIOUS session's pictures among its own artifacts. The page
    has to be able to tell the difference, and a picture cannot say where it came from.
    """
    path = paths.root / FIGURE_PROVENANCE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "e01-figures-1",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "cited_runs": sorted(Path(d).name for d in cited),
        "figures": sorted(paths.relative(f) for f in drawn),
        "note": (
            "Written by `so-recon e01-report`. `figures` is what THIS command drew; any other "
            "PNG under reports/figures is an earlier session's and is named as a limitation."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def figure_provenance_limitations(
    paths: ProjectPaths, cited: Sequence[Path], plots: Sequence[str]
) -> tuple[str, ...]:
    """Say so when the committed figures were not drawn from the runs this report cites."""
    if not plots:
        return ()
    payload = _read_json(paths.root / FIGURE_PROVENANCE_RELPATH)
    cited_ids = sorted(Path(d).name for d in cited)
    if payload is None:
        return (
            f"The figures listed below are committed under `{FIGURES_RELDIR}` and carry no "
            f"`{FIGURE_PROVENANCE_RELPATH}` record, so which session drew them cannot be "
            "established from the repository. They are not evidence for this report.",
        )
    drew = [str(name) for name in payload.get("figures", [])]
    stale = [name for name in plots if name not in drew]
    if not stale and sorted(payload.get("cited_runs", [])) == cited_ids:
        return ()
    return (
        "Figures: this report cites runs "
        + ", ".join(f"`{name}`" for name in cited_ids)
        + f", and `{FIGURE_PROVENANCE_RELPATH}` records that the committed figures were drawn "
        "for runs "
        + (", ".join(f"`{n}`" for n in payload.get("cited_runs", ())) or "(none)")
        + ". "
        + (
            ", ".join(f"`{name}`" for name in stale)
            + " were NOT drawn by this report and are an earlier session's pictures"
            if stale
            else "Every figure below was drawn by this report"
        )
        + ". A figure is illustration, never a gate; the matrix above is the evidence.",
    )


def resource_failures_across_runs(paths: ProjectPaths, cited: Sequence[Path]) -> tuple[str, ...]:
    """Every RESOURCE_FAILURE this repository holds, named by run id, cited or not.

    Limitations used to be generated only from the suites the operator passed to `--runs`,
    which made a session the operator did not cite structurally invisible: the repository
    could hold four `RESOURCE_FAILURE` worlds in a published manifest while this page said
    the stage named none. An operator chooses which runs to CITE; an operator does not
    choose what happened on the host. So the whole of `artifacts/runs/` is swept here, and
    a refused session is named with its run id, its suite and its jobs whether or not it is
    part of the evidence — SPEC 18.4 and the standing ruling that every resource failure is
    a NAMED limitation.

    A cited run's failures are named by the per-suite pass above; this adds only the ones
    nobody cited, so nothing is listed twice.
    """
    cited_ids = {Path(run_dir).name for run_dir in cited}
    out: list[str] = []
    if not paths.runs.is_dir():
        return ()
    for run_dir in sorted(paths.runs.iterdir()):
        if run_dir.name in cited_ids or not run_dir.is_dir():
            continue
        payload = _read_json(run_dir / SUITE_REPORT_FILENAME)
        if payload is None:
            continue
        try:
            report = SuiteReport.model_validate(payload)
        except ValueError:
            continue
        refused = [job for job in report.jobs if job.status == "RESOURCE_FAILURE"]
        if not refused:
            continue
        out.append(
            f"uncited session {report.run_id} (suite {report.suite}, exit {report.exit_code}) "
            f"recorded {len(refused)} RESOURCE_FAILURE job(s): "
            + ", ".join(f"`{job.job_id}`" for job in refused)
            + ". It is not part of this report's evidence and it is named here because a "
            "resource failure is never a physical zero (SPEC 18.4) and every one of them is "
            "a named limitation."
        )
    return tuple(out)


def p1_manifest_reconciliation(
    paths: ProjectPaths, suites: Mapping[str, SuiteReport]
) -> tuple[str, ...]:
    """Whether the published P1 suite manifest agrees with the `p1_worlds` verdict above.

    `reports/p1_suite_manifest.json` is written to ONE published path by every P1 session,
    so the committed file always describes the most recent one — which need not be the
    session this report cites. When the two disagree, a reader sees a PASS beside a manifest
    whose rows are all refused, and nothing on the page explains it. This says it.
    """
    manifest = _read_json(paths.root / P1_SUITE_MANIFEST_RELPATH)
    if manifest is None:
        return ()
    rows = manifest.get("rows", [])
    refused = [
        str(row.get("parent_world_id"))
        for row in rows
        if isinstance(row, Mapping) and not row.get("accepted")
    ]
    if not refused:
        return ()
    p1 = suites.get("p1")
    scored = p1.checks if p1 is not None else ()
    verdict = next((c.status for c in scored if c.name == "p1_worlds"), None)
    cited = "no p1 suite is cited by this report" if p1 is None else f"run {p1.run_id}"
    return (
        f"`{P1_SUITE_MANIFEST_RELPATH}` is the COMMITTED manifest and it records "
        f"fully_accepted={manifest.get('fully_accepted')!r} with {len(refused)} of "
        f"{len(rows)} parent world(s) not accepted ("
        + ", ".join(f"`{name}`" for name in refused)
        + "). The `p1_worlds` verdict on this page is "
        + (f"{verdict}" if verdict is not None else "absent")
        + f" and is scored on {cited}. Task 11 publishes that manifest to a single path, so "
        "it describes the MOST RECENT P1 session rather than necessarily the cited one; the "
        "two are reconciled here rather than left to read as a contradiction. Where they "
        "disagree, the manifest is the state of the repository and the verdict is the state "
        "of the cited run.",
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
    # Jobs that were planned and never reached, kept in TWO lists and not one.
    #
    # `_status` fails the stage on `remaining` unconditionally, before it looks at anything
    # else, and that rule is what stops a half-run oil-water suite from passing — it must keep
    # biting exactly as it does. But a black-oil session that was REFUSED leaves its own
    # declared jobs unreached, and merging those into the same list made a refused capability
    # fail the whole stage and drag `ow_gate` down with it: `status="FAIL"`, reason
    # `planned jobs were never reached: ['bo_restart_prefix', ...]`, while `bo_status` was
    # correctly INCOMPLETE. That is precisely what plan 13.5 forbids — «OW deliverables не
    # удаляются и не объявляются failed только из-за отсутствующего BO benchmark».
    #
    # So the oil-water rule reads the oil-water list, unchanged; the capability's unreached
    # jobs are named in the limitation its own INCOMPLETE status already emits, and both lists
    # are published, so nothing is hidden — only re-attributed to the gate it belongs to.
    remaining: list[str] = []
    bo_remaining: list[str] = []
    limitations: list[str] = []
    artifacts: dict[str, str] = {}
    benchmark: dict[str, Any] = {}
    for name in sorted(suites):
        report = suites[name]
        checks.extend(report.checks)
        jobs.extend(report.jobs)
        (bo_remaining if name == BO_SUITE else remaining).extend(report.remaining_job_ids)
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
    bo_status: BlackOilStatus = _bo_status(bo_report)
    if bo_status == "INCOMPLETE" and bo_report is not None:
        limitations.append(
            f"bo: the black-oil session {bo_report.run_id} was REFUSED or left incomplete "
            f"(exit {bo_report.exit_code})"
            + (f" — {bo_report.stopped_reason}" if bo_report.stopped_reason else "")
            + ". A capability that was attempted and refused is not one that was never "
            "attempted; `BO status: INCOMPLETE` says which of the two happened."
            + (
                " Its declared jobs "
                + ", ".join(f"`{job_id}`" for job_id in sorted(set(bo_remaining)))
                + " were never reached. They are NOT counted against the oil-water "
                "job-completeness rule: a black-oil benchmark that did not finish does not "
                "fail an oil-water deliverable (plan 13.5)."
                if bo_remaining
                else ""
            )
        )

    # Which commit each cited run was produced at. `dict.fromkeys` keeps the cited order and
    # drops repeats, so a report whose runs all share a commit has exactly one entry.
    git_commits = tuple(dict.fromkeys(c.git_commit for c in commands if c.git_commit))
    if len(git_commits) > 1:
        limitations.append(
            "the cited runs were not all produced at one commit: "
            + "; ".join(
                f"{c.run_id} at {c.git_commit}" for c in commands if c.git_commit is not None
            )
            + ". The Identity table's `git_commit` is the first cited run's and says so; "
            "the commit column of the Commands table names each run's own."
        )

    status, reason = _status(
        sorted(present & MANDATORY_CHECKS),
        failed,
        unrun,
        sorted(set(remaining)),
        bo_status,
        any_evidence=bool(suites),
    )
    coarse_failed = any(
        c.name == "five_spot_coarse_sensitivity" and c.status == "FAIL" for c in checks
    )
    if coarse_failed:
        limitation = (
            "The designated reference comparison uses 112/144 grids; the 16/48 monthly "
            "phase-volume comparison failed and remains reported. This is not a resolution "
            "certificate for the 16-grid or heterogeneous P1/E02 models."
        )
        limitations.append(limitation)
        if status in ("PASS", "PASS_WITH_LIMITATIONS"):
            status = "PASS_WITH_LIMITATIONS"
            reason += "; " + limitation
    limitations.append(SESSION_CAP_NOTE)
    # Every RESOURCE_FAILURE in the repository, not only the ones inside a cited suite.
    limitations.extend(resource_failures_across_runs(paths, run_dirs))
    limitations.extend(p1_manifest_reconciliation(paths, suites))
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
    limitations.extend(figure_provenance_limitations(paths, run_dirs, plots))
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
        git_commits=git_commits,
        git_dirty=any(bool(c.git_dirty) for c in commands) if commands else None,
        commands=tuple(commands),
        suites={name: f"exit {report.exit_code}" for name, report in sorted(suites.items())},
        checks=tuple(checks),
        required_checks=tuple(sorted(MANDATORY_CHECKS)),
        complete_checks=tuple(complete),
        failed_checks=tuple(failed),
        unrun_checks=tuple(unrun),
        jobs=tuple(jobs),
        remaining_job_ids=tuple(sorted(set(remaining) | set(bo_remaining))),
        black_oil_remaining_job_ids=tuple(sorted(set(bo_remaining))),
        limitations=tuple(limitations),
        # The OW gate is the STAGE's own predicate, not a second one. It used to be computed
        # from failed/unrun alone, which ignored the `missing` and `remaining` branches
        # `_status` correctly fails on — so a p0-only report published `status: FAIL` beside
        # `ow_gate: PASS`. Derived from the same verdict now, so the two cannot disagree.
        ow_gate=_ow_gate(status),
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


#: What the capability establishes, and what it does not. Plan 13.5, in the report rather
#: than only in the plan, because this is the page a later stage reads before PhysicsDecision.
BO_SCOPE_NOTE = (
    "What a PASS here establishes: that this adapter builds, drives, balances and restarts a "
    "three-phase black-oil system with dissolved gas — support, component balance including "
    "the dissolved term, a phase transition, and a native restart that carries the phase "
    "state — on an EDUCATIONAL case. What it does not establish: it does not choose the "
    "physics of the Romashka case, its PVT is an academic benchmark that ships inside the "
    "pinned JutulDarcy and not a field sample, its three-phase relative permeability is an "
    "explicit approximation (independent Corey curves, no hysteresis) rather than a selected "
    "Stone or LET model, and it is not the paired oil-water/black-oil sensitivity study with "
    "matched oil inventory — that is E06. This dependency is due before PhysicsDecision and "
    "its verdict is a gate of its own: an oil-water deliverable is never removed or failed "
    "because a black-oil benchmark is missing or failed."
)


def _blackoil_section(report: StageReport) -> list[str]:
    """The measured black-oil evidence, or the reason there is none.

    It is rendered from the same `jobs` and `checks` the rest of the page is rendered from,
    filtered to the capability's own group and its own check names, so the BO block cannot
    drift from the session it claims to describe.
    """
    jobs = [job for job in report.jobs if job.group == "black_oil"]
    checks = [check for check in report.checks if check.name in BO_CHECKS]
    lines = [BO_SCOPE_NOTE, ""]
    if not jobs and not checks:
        return lines
    lines += ["### Black-oil jobs", ""]
    lines += _table(
        ("job", "kind", "accounting", "status", "expected", "wall s", "peak RSS"),
        [
            (
                f"`{job.job_id}`",
                job.kind,
                job.accounting,
                job.status,
                job.expected_outcome,
                _number(job.wall_s, 4),
                "—" if job.peak_rss_bytes is None else f"{job.peak_rss_bytes / 2**30:.2f} GiB",
            )
            for job in jobs
        ],
    )
    for check in checks:
        lines += [f"### `{check.name}` — {check.status}", ""]
        if check.reason:
            lines += [check.reason, ""]
        lines += _table(
            ("metric", "measured", "threshold"),
            [
                (
                    f"`{metric}`",
                    _number(value, 6),
                    next(
                        (
                            f"`{name}` = {_number(check.thresholds[name], 3)}"
                            for gated, name in BLACKOIL_GATES.get(check.name, ())
                            if gated == metric and name in check.thresholds
                        ),
                        "reported, not gated",
                    ),
                )
                for metric, value in sorted(check.metrics.items())
            ],
        )
        if check.evidence_paths:
            lines += ["Evidence: " + ", ".join(f"`{p}`" for p in check.evidence_paths), ""]
    return lines


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not rows:
        return ["_(none)_", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def _commit_cell(report: StageReport) -> str:
    """The Identity table's commit cell: one commit, or every one the cited runs span.

    A single value here reads as a claim about the whole page, because the field beside it
    aggregates. When the cited runs really do share a commit that claim is true and the cell
    is the bare digest it always was; when they do not, the cell says so and points at the
    per-run column rather than picking one of them and staying silent.
    """
    if len(report.git_commits) <= 1:
        return f"`{report.git_commit}`"
    return (
        "mixed — the cited runs span "
        + ", ".join(f"`{commit[:12]}`" for commit in report.git_commits)
        + "; see the commit column of the Commands table"
    )


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _count(value: int | None) -> str:
    """An em dash where nobody measured, a number where somebody did.

    A `0` in this column used to mean both "measured, and it was nothing" and "never
    measured" — 26 of 48 rows published `peak RSS 0`, `out bytes 0` and `cpu s 0` on a page
    whose own header says every number came out of a published artifact. They are different
    facts and they now look different.
    """
    return "—" if value is None else str(value)


def _measurement_methods(jobs: Sequence[JobOutcome]) -> list[str]:
    """How the rows above were arrived at, one paragraph per distinct method.

    `JobOutcome.measurement_method` was recorded from the beginning and never printed, so a
    reader of the jobs table had no way to know that a launcher row's `wall s` is the whole
    diagnostic's wall divided evenly across the forwards it ran, or that a reused row's
    costs belong to another session. Printed here, under the table it explains.
    """
    seen: dict[str, list[str]] = {}
    for job in jobs:
        if job.measurement_method:
            seen.setdefault(job.measurement_method, []).append(job.job_id)
    if not seen:
        return []
    out = ["**How these numbers were measured.** An em dash is a quantity nobody measured.", ""]
    for method, job_ids in sorted(seen.items(), key=lambda item: item[1][0]):
        shown = ", ".join(f"`{j}`" for j in job_ids[:6])
        more = f" (+{len(job_ids) - 6} more)" if len(job_ids) > 6 else ""
        out.append(f"* {shown}{more}: {method}")
    out.append("")
    return out


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
            ("git_commit", _commit_cell(report)),
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
        ("command", "run_id", "commit", "record", "exit", "argv"),
        [
            (
                f"`{c.command}`",
                f"`{c.run_id}`",
                "—" if c.git_commit is None else f"`{c.git_commit[:12]}`",
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
                _count(job.peak_rss_bytes),
                _count(job.output_bytes),
                _count(job.native_chunk_calls),
                _count(job.accepted_steps),
                _count(job.cut_steps),
                _count(job.nonlinear_iterations),
                job.retry_count,
            )
            for job in report.jobs
        ],
    )
    lines += _measurement_methods(report.jobs)
    if report.remaining_job_ids:
        lines += [
            "**Jobs that were planned and never reached:** "
            + ", ".join(f"`{j}`" for j in report.remaining_job_ids)
            + (
                " — of which "
                + ", ".join(f"`{j}`" for j in report.black_oil_remaining_job_ids)
                + " belong to the black-oil capability. Its verdict is a separate gate "
                "(plan 13.5), so they are reported under `BO status` and are not counted "
                "against the oil-water job-completeness rule."
                if report.black_oil_remaining_job_ids
                else ""
            ),
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
            ("ledger output bytes", _count(costs.ledger_output_bytes)),
            ("launcher forwards", costs.launcher_forwards),
            ("launcher wall s", _number(costs.launcher_wall_s)),
            ("peak process-tree RSS bytes", _count(costs.peak_rss_bytes)),
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
    lines += ["## Oil-water gate", ""]
    lines += [
        f"**OW gate: {report.ow_gate}.** This is the stage verdict itself, not a second "
        "reading of it: it is PASS only where `status` is PASS or PASS_WITH_LIMITATIONS, "
        "which means every mandatory oil-water check is present, passing, and joined by "
        "every planned job the matrix declares. A matrix with a hole in it is not a passing "
        "gate (plan 12.9).",
        "",
    ]
    lines += ["## Black oil", ""]
    lines += [
        f"**BO status: {report.bo_status}.** The black-oil capability is a separate P0 "
        "session and is not part of this stage's evidence. The four values are distinct on "
        "purpose: `NOT_RUN` is no session at all, `PASS` and `FAIL` are its verdict, and "
        "`INCOMPLETE` is a session that was attempted and REFUSED or left unfinished — a "
        "capability nobody ran and one that refused to run are not the same fact. "
        "`PASS_WITH_LIMITATIONS` is what an accepted oil-water scope carries under any of the "
        "three that are not `PASS`; none of them fails this stage (plan 13.5), and none of "
        "them is what a failed mandatory oil-water, restart or balance check carries.",
        "",
    ]
    lines += _blackoil_section(report)
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


#: One failed comparison, as the two writers of a FAIL reason spell it: `_failed_gates` in
#: `validation/physics.py` and `_gate` in `simulator/suites.py` both write
#: `<metric>=<value> exceeds|does not exceed <threshold>=<limit>`, joined by "; ". This is the
#: ONLY place a check records which threshold gated which metric, which is why a FAILING row
#: is read out of the reason instead of re-derived from the two separate maps.
_FAILED_GATE = re.compile(
    r"(?P<metric>\w+)=(?P<value>[-+0-9.eE]+) (?P<comparison>exceeds|does not exceed) "
    r"(?P<threshold>\w+)=(?P<limit>[-+0-9.eE]+)"
)


def _failed_comparison(check: PhysicsCheck) -> str:
    """The comparison a FAILED check's own reason names, furthest past its gate first.

    `_worst` pairs a metric to a threshold by name prefix, and on the E01 matrix 37 of 88
    thresholds pair with nothing — among them every threshold of `five_spot_refinement`
    except one. On a FAILING row that mattered: the single incidental match was
    `support_pore_volume_relative`, inside its gate by seven orders of magnitude, so the row
    that failed advertised a measurement that passed. A FAIL is therefore read out of the
    reason, which is the only record of which threshold actually gated which metric, and an
    unparseable reason gets the em dash rather than a guess.
    """
    best: tuple[float, str] | None = None
    for match in _FAILED_GATE.finditer(check.reason or ""):
        value, limit = float(match["value"]), float(match["limit"])
        # `exceeds` failed above its limit and `does not exceed` failed below it. Both are
        # ranked by how far past their own gate they are, so a row with several failures
        # shows the worst of them and never a ratio below 1.
        if match["comparison"] == "exceeds":
            severity = value / limit if limit else float("inf")
        else:
            severity = limit / value if value else float("inf")
        if best is None or severity > best[0]:
            best = (
                severity,
                f"`{match['metric']}`={value:.3g} vs `{match['threshold']}`={limit:.3g}",
            )
    return best[1] if best else "—"


def _worst(check: PhysicsCheck) -> str:
    """The metric closest to (or furthest past) its own gate, for the summary row.

    Best effort, and it says so with an em dash when it cannot pair a metric with a
    threshold: `PhysicsCheck` records the two as separate maps and does not record which
    threshold gated which metric. The full metric and threshold tables below are the
    authority; this column is a reading aid.

    A FAILED check is NOT best effort. There the column would be read as the evidence of the
    failure, so it is taken from the comparison the check's own reason names and from nothing
    else — see `_failed_comparison`.
    """
    if check.status == "FAIL":
        return _failed_comparison(check)
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
