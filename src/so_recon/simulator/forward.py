"""E01.8 — the forward driver: policy, orchestration, and the checks a result owes its request.

Three things live here and nothing else does.

**Policy.** Whether a failed attempt may be tried again (`should_retry`), what the one
permitted retry is allowed to change (`retry_descriptor`), what a session predicts an
attempt will cost before it books it (`predicted_wall_s`, `predicted_output_bytes`), and
what a worker whose memory kept growing is answered with (`MemoryDriftMonitor`). SPEC §3.3
allows two attempts per physical model — the original and ONE registered numerical retry —
and that retry may move exactly two numbers: the maximum timestep, halved, and the
nonlinear-iteration limit, 15 → 25. It may not move a convergence tolerance, a
permeability, a saturation or anything else that would buy convergence by changing the
question; `SolverConfig` forbids the extra fields that would be needed to try.

**Orchestration.** `simulate` runs one case to a published `ForwardResult` through the
persistent worker, retrying once if — and only if — the failure was numerical; `resume`
continues a case from a native checkpoint. Both count every attempt against the forward
budget, failed ones included, and neither ever overwrites the first attempt's result.

**The cross-record checks.** `load_forward_result` re-proves what a result alone can prove.
Whether the axis is the one an `OutputRequest` ASKED for, and whether a restart is present
because that request set `keep_native_restart`, needs BOTH records — so it is checked here,
where both are in hand, by `check_requested_outputs`. The other direction — whether this
build can deliver what the request ASKS for at all — is `check_request_is_deliverable`,
which runs before any work is booked.

What is deliberately NOT here: extraction and publication (that is `results.py`), native
continuation and the checkpoint manifest (that is `julia/adapter/restart.jl`), and the
process and line protocol (that is `worker.py`).

SPEC §18.4 governs the whole file — «Resource failure и numerical failure не дают физический
нулевой likelihood без анализа». Nothing here turns a failure into a zero, a partial horizon
into a success, or a retry into a licence to change the physics.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from so_recon.config.resources import ResourceProfile
from so_recon.config.schema import StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import (
    CASE_MANIFEST_FILENAME,
    compute_model_hash,
    compute_static_hash,
    load_case,
    write_case,
)
from so_recon.simulator.contracts import (
    MAX_ATTEMPTS,
    CaseBundle,
    ControlSegment,
    CostRecord,
    ForwardResult,
    ForwardStatus,
    JobDescriptor,
    OutputRequest,
    RestartRef,
)
from so_recon.simulator.results import publish_forward_result
from so_recon.simulator.worker import (
    ForwardHandoff,
    PersistentJuliaWorker,
    read_stop_request,
)

log = logging.getLogger(__name__)

#: JutulDarcy 0.3.11's own default nonlinear-iteration limit, which is also the one SPEC
#: §3.3 names as the value a numerical retry raises FROM.
BASE_MAX_NONLINEAR_ITERATIONS = 15

#: And the value it raises TO. Neither number is a tolerance: they bound how hard the solver
#: tries, not how close it has to get.
RETRY_MAX_NONLINEAR_ITERATIONS = 25

#: The only status a retry is ever taken for (SPEC §3.3).
RETRYABLE_STATUS: ForwardStatus = "NUMERICAL_FAILURE"

#: What a run drives the solver with unless it is told otherwise. Five days is well below a
#: calendar month, so every chunk takes several accepted steps and the monthly integral is an
#: integral rather than a single backward-Euler step; it is a driving choice, not a tolerance.
DEFAULT_MAX_TIMESTEP_DAYS = 5.0

#: The solver configuration a run publishes beside its case.
SOLVER_CONFIG_FILENAME = "solver_config.json"

#: How much longer than the longest measured attempt the next one is allowed to be predicted
#: at. A prediction that is exactly the last measurement books no room for a case that is a
#: little harder than the last one; one that is the whole job timeout books a session's
#: entire wall budget for two jobs.
WALL_SAFETY_FACTOR = 1.5

#: The same margin for disk. The states file dominates and is computed exactly below, so the
#: factor only has to cover the Parquet tables and the record.
BYTES_SAFETY_FACTOR = 1.5

#: Bytes per published state value, and how many per-cell fields a state carries
#: (`pressure_pa`, `sw`, `so`, `pore_volume_m3`, `bw`, `bo`).
_BYTES_PER_VALUE = 8
_STATE_FIELDS = 6

#: What a result costs beyond its states: four Parquet tables, the JSON record and the HDF5
#: framing. Measured against the fixtures of Task 7, rounded up to a round number.
_RESULT_OVERHEAD_BYTES = 256 * 1024

#: Plan 8.7's diagnostic threshold for retained memory between jobs: a worker whose RSS is
#: more than this above its post-warm baseline after a GC is recycled before the next job.
#: It is a decision of the plan and not a measured property of the machine; the memory PEAK
#: guards in `budget.py` apply regardless and are not relaxed by it.
MEMORY_DRIFT_FLOOR_BYTES = 256 * 1024 * 1024
MEMORY_DRIFT_FRACTION = 0.20

#: How many times one session recycles a drifting worker before it stops calling the drift a
#: transient. A rise that repeats after a fresh process is not something a further recycle
#: can fix, and an endless recycle loop is how a leak becomes a session that never finishes.
MAX_WORKER_RECYCLES = 1

#: A requested state time and a delivered one may differ by this much and still be the same
#: instant. The axis is built from day counts times 86400 on both sides, so anything above
#: float64 round-off at reservoir time scales is a real disagreement.
TIME_MATCH_TOLERANCE_S = 1e-6


class ForwardRequestError(RuntimeError):
    """A request and a record disagree, or a descriptor cannot legally be built."""


def check_request_is_deliverable(request: OutputRequest) -> None:
    """Refuse an `OutputRequest` field this build would otherwise silently ignore.

    `diagnostic_substeps` is the one such field. Plan 7.7 lists the accepted-step
    diagnostics table among E01's MINIMAL outputs, so `results.write_forward_outputs`
    writes one row per accepted substep for every job whatever the flag says; and E01
    publishes no full per-substep STATES for it to gate instead — `states.h5` carries the
    requested dates and nothing else, which is what keeps a result's size a function of the
    request rather than of the solver's step selection.

    So in this build the flag can only mean one of two things, and both are worse than a
    refusal: "give me the table I am already getting", or "give me per-substep states",
    which nothing here writes. The driver holds the request, so the driver says so. When a
    later epic publishes per-substep states, this refusal is what it deletes.
    """
    if request.diagnostic_substeps:
        raise ForwardRequestError(
            "output_request.diagnostic_substeps is set, and this build has nothing to gate "
            "with it: the accepted-substep diagnostics table is written for every job (plan "
            "7.7) and full per-substep states are not among E01's published outputs. A "
            "request nobody honours is refused rather than quietly ignored"
        )


# --------------------------------------------------------------------------- 8.3 policy


def should_retry(status: str, attempt: int) -> bool:
    """Whether a failed attempt gets the one registered numerical retry (SPEC §3.3).

    `attempt` is the ZERO-BASED index of the attempt that just failed, so the original
    attempt is 0 and the retry is 1. `JobDescriptor.attempt` counts from one, which makes
    the call site `should_retry(result.status, job.attempt - 1)`.

    Only `NUMERICAL_FAILURE` is retried. An invalid input, an unphysical case, a control
    nobody could hold, a timeout and a resource failure are all decisions about the QUESTION
    rather than about the arithmetic, and running the same question again cannot change any
    of them.
    """
    if attempt < 0:
        raise ValueError(f"attempt is a zero-based attempt index, got {attempt}")
    return status == RETRYABLE_STATUS and attempt + 1 < MAX_ATTEMPTS


class SolverConfig(StrictModel):
    """The whole of what a job may say about how the solver is driven.

    Two numbers, and `extra='forbid'`. That is the point: SPEC §3.3 lets a numerical retry
    halve the timestep and raise the iteration limit with «неизменными convergence
    tolerances и физическими входами», and a record with nowhere to put a tolerance cannot
    relax one. Both fields bound EFFORT — how small a step the solver may take and how many
    Newton iterations it may spend — never how close it has to get.
    """

    max_timestep_days: float = Field(gt=0.0)
    max_nonlinear_iterations: int = Field(ge=1)

    def retried(self) -> SolverConfig:
        """The one registered numerical retry of this configuration."""
        return SolverConfig(
            max_timestep_days=self.max_timestep_days / 2,
            max_nonlinear_iterations=RETRY_MAX_NONLINEAR_ITERATIONS,
        )


def read_solver_config(path: Path) -> SolverConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForwardRequestError(f"cannot read the solver configuration {path}: {exc}") from exc
    try:
        return SolverConfig.model_validate(payload)
    except ValidationError as exc:
        raise ForwardRequestError(f"{path} is not a solver configuration: {exc}") from exc


def write_solver_config(config: SolverConfig, path: Path, paths: ProjectPaths) -> tuple[str, str]:
    """Publish a solver configuration and return `(relative path, sha256)`."""
    write_json_atomic(path, config.model_dump(mode="json"))
    return paths.relative(path), sha256_file(path)


def retry_descriptor(parent: JobDescriptor, *, paths: ProjectPaths) -> JobDescriptor:
    """The ONE registered numerical retry of `parent` (SPEC §3.3).

    It carries the original model hash and the original case bytes, because it is the same
    physical model — a retry that re-rendered its case would be different work wearing the
    first attempt's name. It carries its OWN solver-config path and digest, because what it
    changed has to be on the record rather than inferred. And it names the attempt it
    descends from, because an unregistered retry is indistinguishable from a second first
    attempt.

    The parent's configuration is re-hashed before it is halved: a file that moved under the
    descriptor is a refusal, not a base to compute a retry from.
    """
    if parent.attempt >= MAX_ATTEMPTS:
        raise ForwardRequestError(
            f"job {parent.job_id!r} is already attempt {parent.attempt} of at most "
            f"{MAX_ATTEMPTS}; SPEC 3.3 allows one registered numerical retry per physical model"
        )
    config_path = paths.resolve(parent.solver_config_path)
    digest = sha256_file(config_path)
    if digest != parent.solver_config_sha256:
        raise ForwardRequestError(
            f"the solver configuration {parent.solver_config_path} hashes to sha256 {digest}, "
            f"but the descriptor of job {parent.job_id!r} declares "
            f"{parent.solver_config_sha256}; a retry is not computed from bytes nobody vouched for"
        )
    attempt = parent.attempt + 1
    job_id = f"{parent.job_id}-r{attempt}"
    retry_config_path = config_path.parent / f"{job_id}.solver.json"
    relative, retry_digest = write_solver_config(
        read_solver_config(config_path).retried(), retry_config_path, paths
    )
    return JobDescriptor(
        job_id=job_id,
        case_path=parent.case_path,
        case_sha256=parent.case_sha256,
        model_hash=parent.model_hash,
        solver_config_path=relative,
        solver_config_sha256=retry_digest,
        output_request=parent.output_request,
        seed=parent.seed,
        result_dir=f"{parent.result_dir}-r{attempt}",
        attempt=attempt,
        resume_from=parent.resume_from,
        parent_job_id=parent.job_id,
    )


# ------------------------------------------------------------------ measured predictions


def _measured_costs(ledger: BudgetLedger) -> list[CostRecord]:
    record = ledger.record
    return [
        entry.cost
        for entry in (*record.inherited_entries, *record.entries)
        if entry.cost is not None
    ]


def predicted_wall_s(ledger: BudgetLedger, profile: ResourceProfile) -> float:
    """How long the next attempt is predicted to take, from what this chain has measured.

    `BudgetLedger.reserve` refuses a job whose estimate does not fit in what is left of the
    session's wall budget, so the estimate decides how many forwards a session can book.
    Reserving the whole `job_timeout_s` makes a P0 session — 600 s of budget, a 300 s job
    timeout — refuse its third job however short the first two were, which is a budget
    failure invented by the estimate rather than measured by anything.

    With nothing measured yet the prediction is the session's own fair share,
    `wall_budget_s / max_new_forward`, the same shape as the disk fair share the transport
    already uses. Once attempts have been measured it is the longest of them with a margin,
    capped by the job timeout: a session whose forwards really are slow will run out of wall
    budget, and should.
    """
    measured = _measured_costs(ledger)
    if not measured:
        estimate = profile.wall_budget_s / profile.max_new_forward
    else:
        estimate = WALL_SAFETY_FACTOR * max(cost.wall_s for cost in measured)
    return float(min(profile.job_timeout_s, max(estimate, profile.poll_interval_s)))


def predicted_output_bytes(
    ledger: BudgetLedger,
    profile: ResourceProfile,
    *,
    n_cells: int,
    n_times: int,
    keep_native_restart: bool = False,
    n_report_steps: int = 0,
) -> int:
    """How much disk the next attempt is predicted to write.

    The states file dominates and is computable exactly — `n_times * n_cells` float64 values
    for each of the six published fields — so this is arithmetic on the case rather than a
    share of the budget. A native checkpoint adds the report steps it keeps, each of which
    holds one state of the same width plus its substates; the factor below is deliberately
    generous, because a reservation that is too small is a disk guard that fires in the
    middle of a job instead of before it.
    """
    measured = [cost.output_bytes for cost in _measured_costs(ledger) if cost.output_bytes > 0]
    states = n_times * n_cells * _STATE_FIELDS * _BYTES_PER_VALUE
    native = 0
    if keep_native_restart:
        native = max(n_report_steps, 1) * n_cells * _STATE_FIELDS * _BYTES_PER_VALUE * 4
    estimate = BYTES_SAFETY_FACTOR * (states + native) + _RESULT_OVERHEAD_BYTES
    if measured:
        estimate = max(estimate, BYTES_SAFETY_FACTOR * max(measured))
    return int(min(profile.disk_budget_bytes, math.ceil(estimate)))


# ------------------------------------------------------------- the memory drift monitor


DriftAction = Literal["continue", "recycle", "stop"]


@dataclass(frozen=True)
class DriftDecision:
    """What a post-job memory measurement concluded. A value; the caller does the acting."""

    action: DriftAction
    status: ForwardStatus | None
    reason: str | None
    baseline_rss_bytes: int
    observed_rss_bytes: int
    limit_bytes: int

    @property
    def drift_bytes(self) -> int:
        return self.observed_rss_bytes - self.baseline_rss_bytes


def memory_drift_limit_bytes(baseline_rss_bytes: int) -> int:
    """`max(256 MiB, 20% of the post-warm baseline)` — plan 8.7's diagnostic threshold."""
    return max(MEMORY_DRIFT_FLOOR_BYTES, int(MEMORY_DRIFT_FRACTION * baseline_rss_bytes))


class MemoryDriftMonitor:
    """Retained memory between jobs, and the escalation a rise is answered with.

    A warm Julia worker legitimately holds specialised code, so the baseline is taken AFTER
    the warm-up rather than at process start, and what is measured is the rise above it once
    a job's own allocations have been released. A rise above the threshold gives a warning
    and a recycle before the next job; a rise that repeats after that recycle is a
    `RESOURCE_FAILURE`, because a leak a fresh process still shows is not transient and an
    endless recycle loop is how it becomes a session that never finishes.

    This is a diagnostic decision of the plan, not a guard: the memory peak guards in
    `budget.ResourceWatchdog` apply throughout and nothing here relaxes them.
    """

    def __init__(self, *, baseline_rss_bytes: int) -> None:
        if baseline_rss_bytes < 0:
            raise ValueError(f"baseline_rss_bytes must be non-negative, got {baseline_rss_bytes}")
        self._baseline = baseline_rss_bytes
        self.recycles = 0
        self._terminal: DriftDecision | None = None

    @property
    def baseline_rss_bytes(self) -> int:
        return self._baseline

    @property
    def limit_bytes(self) -> int:
        return memory_drift_limit_bytes(self._baseline)

    def rebaseline(self, rss_bytes: int) -> None:
        """Take a fresh post-warm baseline, after a recycle produced a new process."""
        if self._terminal is not None:
            raise RuntimeError("this monitor already refused the session; start a new one")
        self._baseline = rss_bytes

    def observe(self, rss_bytes: int) -> DriftDecision:
        if self._terminal is not None:
            # A refusal does not un-happen because the next sample looks calm.
            return self._terminal
        limit = self.limit_bytes
        drift = rss_bytes - self._baseline
        if drift <= limit:
            return DriftDecision(
                action="continue",
                status=None,
                reason=None,
                baseline_rss_bytes=self._baseline,
                observed_rss_bytes=rss_bytes,
                limit_bytes=limit,
            )
        detail = (
            f"the worker retained {drift} bytes above its {self._baseline} byte post-warm "
            f"baseline after a collection, past the {limit} byte drift threshold"
        )
        if self.recycles < MAX_WORKER_RECYCLES:
            self.recycles += 1
            log.warning("%s; recycling the worker before the next job", detail)
            return DriftDecision(
                action="recycle",
                status=None,
                reason=f"{detail}; recycle the worker before the next job",
                baseline_rss_bytes=self._baseline,
                observed_rss_bytes=rss_bytes,
                limit_bytes=limit,
            )
        self._terminal = DriftDecision(
            action="stop",
            status="RESOURCE_FAILURE",
            reason=(
                f"{detail}, and it did so again after {self.recycles} recycle(s); a "
                "repeatable rise is a resource failure rather than another recycle"
            ),
            baseline_rss_bytes=self._baseline,
            observed_rss_bytes=rss_bytes,
            limit_bytes=limit,
        )
        return self._terminal


# ------------------------------------------------------- the cross-record COMPLETE check


def check_requested_outputs(result: ForwardResult, request: OutputRequest) -> None:
    """Prove a `COMPLETE` result is the result this request asked for.

    `load_forward_result` re-proves everything a result can prove ALONE: its digests, its
    shapes, its finiteness, and that its own time axis is the one its states file holds.
    Two things need the request as well, and are therefore checked here, where both records
    are in hand:

    * every requested state time is covered — a forward that stopped early and published the
      months it reached would otherwise read as a case in which nothing happened afterwards,
      which is exactly the physical zero SPEC §18.4 forbids;
    * a native restart is present when `keep_native_restart` asked for one, and belongs to
      this model.

    Only a `COMPLETE` is measured against the request. A refusal claims nothing, and judging
    it against an axis it never said it delivered would report the wrong failure.
    """
    if result.status != "COMPLETE":
        return None
    delivered = list(result.times_s)
    missing = [
        requested
        for requested in request.state_times_s
        if not any(abs(requested - have) <= TIME_MATCH_TOLERANCE_S for have in delivered)
    ]
    if missing:
        raise ForwardRequestError(
            f"job {result.job_id!r} reports COMPLETE but its time axis {delivered} does not "
            f"carry the requested states at {missing} s; a month nobody simulated would be "
            "published as a month in which nothing flowed"
        )
    if request.keep_native_restart and result.restart is None:
        raise ForwardRequestError(
            f"job {result.job_id!r} reports COMPLETE but carries no native restart, and the "
            "request set keep_native_restart; an HDF5 summary is not a restart (SPEC 17.1)"
        )
    if result.restart is not None and result.restart.model_hash != result.model_hash:
        raise ForwardRequestError(
            f"job {result.job_id!r} carries a restart for model_hash "
            f"{result.restart.model_hash}, but the result is of model {result.model_hash}; a "
            "checkpoint of a different model is INVALID_INPUT, never a continuation"
        )
    return None


# ------------------------------------------------------------------- the schedule prefix


def schedule_prefix_hash(
    report_edges_s: Sequence[float],
    controls: Sequence[ControlSegment],
    *,
    completed_time_s: float,
) -> str:
    """Canonical digest of the schedule up to `completed_time_s`.

    A continuation is only the same run if the part already simulated is the same part. The
    digest covers the report edges at or before the checkpoint and every control segment
    that touches them, in a canonical order, so a case whose PREFIX was edited cannot be
    resumed from a checkpoint of the original — while a case that changed only its future
    policy still can, which is what `resume`'s `future_policy` is for.
    """
    edges = [float(e) for e in report_edges_s if e <= completed_time_s + TIME_MATCH_TOLERANCE_S]
    prefix = sorted(
        (
            segment.model_dump(mode="json")
            for segment in controls
            if segment.start_s < completed_time_s - TIME_MATCH_TOLERANCE_S
        ),
        key=lambda s: (s["start_s"], s["well_id"], s["end_s"]),
    )
    for segment in prefix:
        # A segment that straddles the checkpoint is included only up to it: what was
        # simulated is what the digest is about.
        segment["end_s"] = min(float(segment["end_s"]), completed_time_s)
    return sha256_json({"report_edges_s": edges, "controls": prefix})


def restart_ref_from_manifest(
    manifest_path: Path, paths: ProjectPaths, *, model_hash: str
) -> RestartRef:
    """Read a published checkpoint manifest into the `RestartRef` that names it."""
    try:
        payload: Mapping[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForwardRequestError(
            f"cannot read the checkpoint manifest {manifest_path}: {exc}"
        ) from exc
    declared = str(payload.get("model_hash", ""))
    if declared != model_hash:
        raise ForwardRequestError(
            f"{manifest_path} is a checkpoint of model {declared}, not of {model_hash}"
        )
    return RestartRef(
        manifest_path=paths.relative(manifest_path),
        sha256=sha256_file(manifest_path),
        completed_report_step=int(payload["completed_report_step"]),
        completed_time_s=float(payload["completed_time_s"]),
        model_hash=model_hash,
        schedule_prefix_hash=str(payload["schedule_prefix_hash"]),
        environment_lock_hash=str(payload["environment_lock_hash"]),
    )


# ---------------------------------------------------------------------- orchestration


def forward_handoff(
    job: JobDescriptor,
    case: CaseBundle,
    paths: ProjectPaths,
    *,
    parent_attempt_ids: tuple[str, ...] = (),
) -> ForwardHandoff:
    """What the transport needs from the physics side to run ONE job.

    The prefix digests are computed for every report edge, because which one a checkpoint
    lands on is decided by how far the solver got and by whether an operator stopped it —
    and only this side canonicalises a case. The publisher turns the worker's verified
    record into a `ForwardResult` through `results.publish_forward_result`, which is where
    integration, the output files and their digests belong.
    """
    hashes = tuple(
        (
            float(edge),
            schedule_prefix_hash(case.report_edges_s, case.controls, completed_time_s=float(edge)),
        )
        for edge in case.report_edges_s[1:]
    )

    def publish(
        record: Mapping[str, Any], cost: CostRecord, solver_metadata: Mapping[str, str]
    ) -> ForwardResult:
        extraction = record.get("extraction")
        if not isinstance(extraction, Mapping):
            raise ValueError("the worker record carries no native extraction to publish")
        metadata = dict(solver_metadata)
        diagnostics = extraction.get("chunk_diagnostics")
        if diagnostics is not None:
            metadata["chunk_diagnostics"] = json.dumps(diagnostics)
        restart = _restart_of(record, paths, model_hash=job.model_hash)
        if restart is not None:
            metadata["restart_manifest_path"] = restart.manifest_path
            metadata["restart_manifest_sha256"] = restart.sha256
        return publish_forward_result(
            job,
            case,
            extraction,
            paths,
            cost=cost,
            solver_metadata=metadata,
            parent_attempt_ids=parent_attempt_ids,
            restart=restart,
        )

    return ForwardHandoff(
        static_hash=compute_static_hash(case), schedule_prefix_hashes=hashes, publish=publish
    )


def _restart_of(
    record: Mapping[str, Any], paths: ProjectPaths, *, model_hash: str
) -> RestartRef | None:
    """The checkpoint a worker record names, hashed here rather than taken on trust."""
    declared = record.get("restart")
    if not isinstance(declared, Mapping):
        return None
    manifest_path = paths.resolve(str(declared["manifest_path"]))
    ref = restart_ref_from_manifest(manifest_path, paths, model_hash=model_hash)
    for field in ("completed_report_step", "completed_time_s", "schedule_prefix_hash"):
        if getattr(ref, field) != declared[field]:
            raise ForwardRequestError(
                f"the worker record says the checkpoint's {field} is {declared[field]!r}, but "
                f"the published manifest says {getattr(ref, field)!r}"
            )
    return ref


#: What a job that never started costs and was measured by. Zeros here are not an unmeasured
#: guess: no process was launched, so every one of them is the truth, and
#: `measurement_method` says which of the two kinds of zero this is.
NOTHING_RAN_COST = CostRecord(
    wall_s=0.0,
    cpu_s=0.0,
    peak_rss_bytes=0,
    output_bytes=0,
    accepted_steps=0,
    cut_steps=0,
    nonlinear_iterations=0,
    retry_count=0,
    measurement_method=(
        "no job was started: a stop had been requested for this session, so there is nothing "
        "to measure and every count is zero because nothing ran"
    ),
)

NOTHING_RAN_METADATA: dict[str, str] = {"stopped_before_booking": "true"}


def stopped_before_booking(
    job: JobDescriptor,
    case: CaseBundle,
    worker: PersistentJuliaWorker,
    *,
    completed_time_s: float,
) -> ForwardResult | None:
    """The result a job gets when an operator asked this session to stop before it started.

    SPEC §3.3 makes a continuation «отдельная команда»: once a stop has been asked for, the
    session finishes the work in hand and starts nothing else. Without this check every
    later `simulate` call still books a job, still spends a forward and still spends the
    wall time of one chunk, only to be stopped inside Julia after the first month — which
    is the opposite of what was asked for, and it burns the budget the continuing command
    will need.

    The question asked is exactly the one Julia asks at a chunk boundary
    (`StopRequest.is_due`), asked at the boundary before the first chunk: an unconditional
    stop is due there, and a stop naming a month the run can still reach is not, so a job
    that was asked to run up to March and stop still runs.

    `completed_time_s` is how far the RUN has already got, and it is what the stop's own
    question is asked about: the start of the horizon for a fresh case, and the
    checkpoint's time for a continuation. Asking a continuation's question about time zero
    would book a job for a "finish March and stop" that March has already reached, and
    Julia would stop it one month later having published nothing new.

    The refusal is `INCOMPLETE_BUDGET` with no outputs — SPEC §18.4, a month nobody
    simulated is not published as a month in which nothing flowed — and it charges the
    ledger nothing, because nothing ran.
    """
    # The cheap question first, and it is the worker's own: a session with no stop file
    # pays one `is_file()` for this check and never opens anything.
    if not worker.stop_requested:
        return None
    stop = read_stop_request(worker.session_dir)
    if stop is None or not stop.is_due(completed_time_s):
        return None
    unsimulated = sum(
        1 for edge in case.report_edges_s if edge > completed_time_s + TIME_MATCH_TOLERANCE_S
    )
    asked = "" if stop.reason is None else f" ({stop.reason})"
    return ForwardResult(
        job_id=job.job_id,
        case_sha256=job.case_sha256,
        model_hash=case.model_hash,
        physics_class=case.fluids.kind,
        status="INCOMPLETE_BUDGET",
        reason=(
            f"a stop was requested for this session{asked} before job {job.job_id!r} "
            f"started, so none of its {unsimulated} report step(s) were simulated and none "
            "are published; continuing is a separate command (SPEC 3.3)"
        ),
        # Nothing, exactly as `results._classified` records every unsuccessful result: what
        # a continuation's PARENT completed is the parent's entry to carry, and the
        # checkpoint it resumes from is the authority on where the run stands.
        completed_time_s=float(case.report_edges_s[0]),
        times_s=(),
        states={},
        solver_metadata=dict(NOTHING_RAN_METADATA),
        cost=NOTHING_RAN_COST,
        parent_attempt_ids=(),
    )


def _run_attempt(
    job: JobDescriptor,
    case: CaseBundle,
    *,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    parent_attempt_ids: tuple[str, ...],
) -> ForwardResult:
    """One attempt: predict, book, run, publish. Every attempt is paid for either way."""
    paths = worker.paths
    profile = worker.profile
    n_cells = case.grid.n_cells
    request = job.output_request
    return worker.submit(
        job,
        ledger,
        handoff=forward_handoff(job, case, paths, parent_attempt_ids=parent_attempt_ids),
        # Measured, not the whole job timeout: see `predicted_wall_s`.
        estimated_s=predicted_wall_s(ledger, profile),
        estimated_bytes=predicted_output_bytes(
            ledger,
            profile,
            n_cells=n_cells,
            n_times=len(request.state_times_s),
            keep_native_restart=request.keep_native_restart,
            n_report_steps=len(case.report_edges_s) - 1,
        ),
    )


def simulate(
    case: CaseBundle,
    output_request: OutputRequest,
    *,
    worker: PersistentJuliaWorker,
    ctx: RunContext,
    ledger: BudgetLedger,
    solver_config: SolverConfig | None = None,
) -> ForwardResult:
    """Run one case to a published, verified `ForwardResult`.

    The sequence is the whole of SPEC §3.3 for a forward: publish the case, build the first
    attempt, run it, and — if and only if it failed numerically — build the ONE registered
    retry and run that. Both attempts are reserved and resolved on the ledger, so a failure
    is paid for; the first attempt's result is written under its own job id and is never
    replaced by the retry's.

    A `COMPLETE` is then checked against the request that asked for it: the whole axis, and
    a native restart if one was asked for. That check needs both records and so cannot live
    in `load_forward_result`, which sees only one.
    """
    check_request_is_deliverable(output_request)
    case_path = _publish_case(case, worker.paths, ctx)
    config_path, config_sha = _publish_solver_config(
        solver_config
        or SolverConfig(
            max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
            max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
        ),
        worker.paths,
        ctx,
    )
    job = JobDescriptor(
        job_id=_job_id(ctx, attempt=1),
        case_path=worker.paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash=case.model_hash,
        solver_config_path=config_path,
        solver_config_sha256=config_sha,
        output_request=output_request,
        seed=int(case.seeds.get("fixture", 0)),
        result_dir=f"{worker.paths.relative(ctx.run_dir)}/{_job_id(ctx, attempt=1)}",
        attempt=1,
    )
    stopped = stopped_before_booking(
        job, case, worker, completed_time_s=float(case.report_edges_s[0])
    )
    if stopped is not None:
        return stopped
    result = _run_attempt(job, case, worker=worker, ledger=ledger, parent_attempt_ids=())
    if should_retry(result.status, job.attempt - 1):
        log.warning(
            "job %s failed numerically (%s); taking the one registered retry with half the "
            "maximum timestep and a %d iteration limit",
            job.job_id,
            result.reason,
            RETRY_MAX_NONLINEAR_ITERATIONS,
        )
        retry = retry_descriptor(job, paths=worker.paths)
        # The retry is new work and passes the same guards: `submit` reserves it, and a
        # machine or a budget that cannot take it refuses it with `BudgetStop`.
        result = _run_attempt(
            retry, case, worker=worker, ledger=ledger, parent_attempt_ids=(job.job_id,)
        )
        check_requested_outputs(result, output_request)
        return result
    check_requested_outputs(result, output_request)
    return result


def resume(
    case: CaseBundle,
    restart: RestartRef,
    future_policy: tuple[ControlSegment, ...],
    *,
    worker: PersistentJuliaWorker,
    ctx: RunContext,
    ledger: BudgetLedger,
    solver_config: SolverConfig | None = None,
    output_request: OutputRequest | None = None,
) -> ForwardResult:
    """Continue `case` from a native checkpoint, with the policy AFTER it replaced.

    `future_policy` may repeat the original suffix or change the intervals after the
    checkpoint; the prefix must be identical, and that is proved here before a single native
    byte is read. The digest is recomputed from the case being resumed and compared with the
    one the checkpoint recorded, so a case whose history was edited is refused as
    `INVALID_INPUT` rather than continued into a run that never happened.

    The continuation runs as its OWN job, in its own result directory and its own native
    working directory. The parent's checkpoint is read and copied, never written to.
    """
    # Re-VALIDATED rather than copied: the merged schedule has to pass every rule a case's
    # schedule passes — no gap, no overlap, nothing past the horizon — and it has to come out
    # in the canonical control order, or an unchanged future policy would produce a different
    # model hash from the case it continues purely because of the order it was assembled in.
    merged = _merged_controls(case, restart, future_policy)
    resumed = CaseBundle.model_validate(
        {
            **case.model_dump(mode="json"),
            "controls": [segment.model_dump(mode="json") for segment in merged],
        }
    )
    resumed = resumed.model_copy(update={"model_hash": compute_model_hash(resumed)})
    prefix = schedule_prefix_hash(
        resumed.report_edges_s, resumed.controls, completed_time_s=restart.completed_time_s
    )
    if prefix != restart.schedule_prefix_hash:
        raise ForwardRequestError(
            f"the checkpoint at {restart.manifest_path} was taken after a schedule prefix that "
            f"hashes to {restart.schedule_prefix_hash}, and this case's prefix up to "
            f"{restart.completed_time_s} s hashes to {prefix}; a continuation of a different "
            "history is INVALID_INPUT, never a numerical retry"
        )
    if restart.environment_lock_hash != worker.environment_lock_hash:
        raise ForwardRequestError(
            f"the checkpoint at {restart.manifest_path} was written under environment lock "
            f"{restart.environment_lock_hash}, and this session runs "
            f"{worker.environment_lock_hash}; a restart from a different version is INVALID_INPUT"
        )

    case_path = _publish_case(resumed, worker.paths, ctx)
    config_path, config_sha = _publish_solver_config(
        solver_config
        or SolverConfig(
            max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
            max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
        ),
        worker.paths,
        ctx,
    )
    request = output_request or OutputRequest(
        state_times_s=tuple(resumed.report_edges_s), keep_native_restart=True
    )
    check_request_is_deliverable(request)
    job_id = _job_id(ctx, attempt=1)
    job = JobDescriptor(
        job_id=job_id,
        case_path=worker.paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash=resumed.model_hash,
        solver_config_path=config_path,
        solver_config_sha256=config_sha,
        output_request=request,
        seed=int(resumed.seeds.get("fixture", 0)),
        result_dir=f"{worker.paths.relative(ctx.run_dir)}/{job_id}",
        attempt=1,
        # The checkpoint travels WITH the descriptor, exactly as it was published, so the
        # worker re-hashes the manifest and every native file it names before it reads any
        # of them. Its `model_hash` is the PARENT's and stays the parent's: a continuation
        # that changed the policy after the checkpoint is a different model by construction,
        # and what has to match is the static half and the prefix.
        resume_from=restart,
    )
    stopped = stopped_before_booking(
        job, resumed, worker, completed_time_s=float(restart.completed_time_s)
    )
    if stopped is not None:
        return stopped
    result = _run_attempt(job, resumed, worker=worker, ledger=ledger, parent_attempt_ids=())
    check_requested_outputs(result, request)
    return result


def _merged_controls(
    case: CaseBundle, restart: RestartRef, future_policy: tuple[ControlSegment, ...]
) -> tuple[ControlSegment, ...]:
    """The case's own prefix plus the policy that replaces everything after the checkpoint.

    A segment that starts before the checkpoint belongs to the prefix and is kept exactly as
    it was — it has already been simulated, and rewriting it would make the checkpoint a
    continuation of something else. `future_policy` supplies every segment from the
    checkpoint onwards, and a policy that reaches back past it is refused rather than
    truncated.
    """
    boundary = restart.completed_time_s
    prefix = tuple(
        segment for segment in case.controls if segment.start_s < boundary - TIME_MATCH_TOLERANCE_S
    )
    for segment in prefix:
        if segment.end_s > boundary + TIME_MATCH_TOLERANCE_S:
            raise ForwardRequestError(
                f"well {segment.well_id!r} has a control segment [{segment.start_s}, "
                f"{segment.end_s}) that straddles the checkpoint at {boundary} s; a restart is "
                "taken at a report edge and every control boundary there is a report edge too"
            )
    for segment in future_policy:
        if segment.start_s < boundary - TIME_MATCH_TOLERANCE_S:
            raise ForwardRequestError(
                f"the future policy sets well {segment.well_id!r} from {segment.start_s} s, "
                f"before the checkpoint at {boundary} s; a continuation may change what happens "
                "after a checkpoint and nothing before it"
            )
    return (*prefix, *future_policy)


def _job_id(ctx: RunContext, *, attempt: int) -> str:
    """A job id unique per attempt (plan 3.1), derived from the run that asked for it."""
    return f"job-{ctx.run_id}-a{attempt}"


def _publish_case(case: CaseBundle, paths: ProjectPaths, ctx: RunContext) -> Path:
    """Publish the case into the run, unless this run already published the same bytes."""
    case_path = ctx.run_dir / CASE_MANIFEST_FILENAME
    if case_path.is_file():
        existing = load_case(case_path, paths)
        if existing.model_hash != case.model_hash:
            raise ForwardRequestError(
                f"run {ctx.run_id} already published case {existing.case_id} of model "
                f"{existing.model_hash}; a run publishes one case"
            )
        return case_path
    write_case(case, paths, ctx)
    return case_path


def _publish_solver_config(
    config: SolverConfig, paths: ProjectPaths, ctx: RunContext
) -> tuple[str, str]:
    path = ctx.run_dir / SOLVER_CONFIG_FILENAME
    return write_solver_config(config, path, paths)
