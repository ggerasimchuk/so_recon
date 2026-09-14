"""The policy half of the resource control plane: caps, watchdog and the session ledger.

Nothing here talks to the operating system. Every decision is a pure function of a
`ResourceSnapshot` and a `ResourceProfile`, so an 8 GiB laptop, a machine with 16 GiB
already resident in other applications, a full disk, a session past its wall budget and a
job past its timeout are all reachable in a test by describing them — never by allocating,
filling or sleeping.

Three things live here:

* **The measured cap.** `effective_hard_bytes` is the whole idea of the module in one
  expression: the limit is the smallest of what was configured, what the machine has
  after the OS reserve, and what this run could actually reach. It is clamped at zero,
  because a negative limit would read as "no limit" to any later comparison, and a cap of
  zero means nothing may start at all.
* **The watchdog.** `ResourceWatchdog` decides; it never acts. It returns the escalation
  — warn, drain, terminate — as a value, and the worker task that owns the Julia process
  group is what carries a termination out. Keeping the decision separate from the kill is
  what makes the escalation testable at all.
* **The ledger.** `BudgetLedger` is the session's account: what was attempted, what
  completed, what it cost, which inputs it ran on and which checkpoints it left. It is
  written atomically on every change, so a session that dies mid-job leaves a record that
  says the job was pending — not one that says nothing happened.

Two rules from SPEC 3.3 and 18.4 shape the accounting. Every attempt counts against the
forward budget, successful or not, so a run cannot buy extra work by failing. And a
resumed session gets its own wall budget but inherits the parent's entries and their
cost: the total is never reset by resuming, and a completed job is skipped only after its
recorded input hash matches the case being planned.

A stop recorded here is not a physical result. `BudgetStop` carries a `ForwardStatus` of
RESOURCE_FAILURE, TIMEOUT, INCOMPLETE_BUDGET or PROTOCOL_FAILURE precisely so that the
caller records why work stopped instead of publishing an unexplained empty result; SPEC
18.4 forbids turning a resource failure into a physical zero likelihood. Failing to write
this ledger, like failing to write run.json, must surface rather than pass silently —
`RunRecordUnavailableError` remains the named boundary for the run record itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from so_recon.config.resources import MIB, ResourceProfile
from so_recon.config.schema import StrictModel
from so_recon.environment.resources import ResourceSnapshot
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.contracts import (
    MAX_ATTEMPTS,
    CostRecord,
    ForwardResult,
    ForwardStatus,
    RestartRef,
    Sha256,
)

LEDGER_SCHEMA_VERSION: Literal["ledger-1"] = "ledger-1"

#: Space kept free so that the failure record and the ledger itself can still be written
#: when a job has just filled the disk. A budget nobody could record is not a budget.
DISK_FAILURE_RESERVE_BYTES = 64 * MIB

#: The soft threshold follows a measured hard cap down. A machine that can only be trusted
#: with 8 GiB should start warning well below 8, not at the 12 the profile asked for.
SOFT_FRACTION_OF_HARD = 0.75

EntryState = Literal["PENDING", "COMPLETE", "FAILED"]
WatchdogAction = Literal["continue", "warn", "drain", "terminate"]


class BudgetStop(RuntimeError):
    """A policy refusal, carrying the `ForwardStatus` the caller must record."""

    def __init__(self, status: ForwardStatus, reason: str) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


# ----------------------------------------------------------------------- measured caps


def effective_hard_bytes(
    *,
    configured: int,
    total: int,
    available: int,
    project_rss: int,
    reserve: int,
) -> int:
    """The memory this run may occupy, given what the rest of the machine is doing.

    `total - reserve` keeps the OS and other processes their share whatever they are
    currently using; `project_rss + available - reserve` is what this run could actually
    grow to right now, given memory already taken by other applications. `max(0, ...)`
    is load-bearing: on an 8 GiB laptop both terms go negative, and a negative cap would
    pass every `rss < cap` test ever written against it.
    """
    return max(0, min(configured, total - reserve, project_rss + available - reserve))


def effective_soft_bytes(*, configured: int, effective_hard: int) -> int:
    return min(configured, int(SOFT_FRACTION_OF_HARD * effective_hard))


class ResourceCaps(StrictModel):
    """What this profile is allowed on this machine, at this instant."""

    hard_bytes: int = Field(ge=0)
    soft_bytes: int = Field(ge=0)

    @property
    def allows_start(self) -> bool:
        """A cap of zero forbids starting: there is no room to run in at all."""
        return self.hard_bytes > 0


def effective_caps(profile: ResourceProfile, snapshot: ResourceSnapshot) -> ResourceCaps:
    """The caps this snapshot implies.

    An unmeasurable process tree contributes 0 to the reachable-memory term, which is the
    smallest assumption available and so the one that yields the tightest cap. It is not a
    claim that the run occupies nothing: `memory_stop` refuses outright on an unknown RSS,
    so these caps are never used to start work on a tree nobody could read.
    """
    hard = effective_hard_bytes(
        configured=profile.hard_bytes,
        total=snapshot.total_bytes,
        available=snapshot.available_bytes,
        project_rss=0 if snapshot.process_rss_bytes is None else snapshot.process_rss_bytes,
        reserve=profile.reserve_bytes,
    )
    return ResourceCaps(
        hard_bytes=hard,
        soft_bytes=effective_soft_bytes(configured=profile.soft_bytes, effective_hard=hard),
    )


# ------------------------------------------------------------------------ guard checks
#
# Each guard is a pure function returning the refusal it found, or None. Returning the
# `BudgetStop` instead of raising it lets the watchdog collect several at once and report
# every reason, while the ledger simply raises the first.


def memory_stop(profile: ResourceProfile, snapshot: ResourceSnapshot) -> BudgetStop | None:
    caps = effective_caps(profile, snapshot)
    if snapshot.process_rss_bytes is None:
        # The guard cannot watch what it cannot see. Carrying on would leave both the hard
        # cap and the soft warning permanently unreachable while every snapshot still
        # looked like a successful measurement.
        return BudgetStop(
            "RESOURCE_FAILURE",
            "this run's own memory use could not be measured "
            f"({snapshot.process_measurement_method}); refusing rather than treating an "
            "unknown measurement as zero",
        )
    if not caps.allows_start:
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"measured memory cap is 0 bytes: {snapshot.total_bytes} total, "
            f"{snapshot.available_bytes} available, {snapshot.process_rss_bytes} already "
            f"resident here, {profile.reserve_bytes} reserved for the OS",
        )
    if snapshot.process_rss_bytes >= caps.hard_bytes:
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"process tree holds {snapshot.process_rss_bytes} bytes, at or above the "
            f"measured hard cap of {caps.hard_bytes}",
        )
    if snapshot.pressure.status == "critical":
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"OS memory pressure is critical (raw {snapshot.pressure.raw!r} via "
            f"{snapshot.pressure.method})",
        )
    return None


def swap_growth_stop(
    profile: ResourceProfile, *, baseline_swap_bytes: int, snapshot: ResourceSnapshot
) -> BudgetStop | None:
    growth = snapshot.swap_used_bytes - baseline_swap_bytes
    if growth > profile.max_swap_growth_bytes:
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"swap grew by {growth} bytes since the session started, beyond the "
            f"{profile.max_swap_growth_bytes} this profile allows",
        )
    return None


def disk_stop(
    profile: ResourceProfile,
    *,
    session_output_bytes: int,
    predicted_bytes: int,
    free_bytes: int,
) -> BudgetStop | None:
    """Free space first, then the session's own budget.

    Both leave `DISK_FAILURE_RESERVE_BYTES` untouched, so that whatever happens next can
    still be written down.
    """
    needed = predicted_bytes + DISK_FAILURE_RESERVE_BYTES
    if free_bytes < needed:
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"free disk {free_bytes} bytes is below the {predicted_bytes} predicted for "
            f"this output plus {DISK_FAILURE_RESERVE_BYTES} reserved for the failure and "
            "ledger records",
        )
    if session_output_bytes + needed > profile.disk_budget_bytes:
        return BudgetStop(
            "INCOMPLETE_BUDGET",
            f"session disk budget {profile.disk_budget_bytes} bytes cannot hold the "
            f"{session_output_bytes} already written plus {predicted_bytes} predicted and "
            f"{DISK_FAILURE_RESERVE_BYTES} reserved for the records",
        )
    return None


def disk_headroom_stop(
    profile: ResourceProfile, *, session_output_bytes: int, free_bytes: int
) -> BudgetStop | None:
    """What is left on disk while a job is *already running*.

    `disk_stop` asks whether a job that has not started can fit. This asks the narrower
    question that only matters once one is running: is there still room to write down what
    happened? `DISK_FAILURE_RESERVE_BYTES` is exactly that room, and it is the reason the
    reserve exists at all — a job that eats into it leaves the failure record it is about
    to need unwritable, and the run ends in silence instead of a recorded stop. SPEC 18.4
    forbids exactly that: a resource failure nobody wrote down cannot be told apart from a
    physical result.

    The two refusals differ in their remedy, and `ResourceWatchdog` acts on that
    difference. A disk with no room left is not helped by finishing the chunk in hand,
    because finishing it writes more; a session that has merely committed its whole disk
    allowance is out of budget rather than out of room, and may still land what it holds.
    """
    if free_bytes < DISK_FAILURE_RESERVE_BYTES:
        return BudgetStop(
            "RESOURCE_FAILURE",
            f"free disk {free_bytes} bytes has fallen below the "
            f"{DISK_FAILURE_RESERVE_BYTES} kept free so that the failure record and the "
            "ledger can still be written",
        )
    if session_output_bytes + DISK_FAILURE_RESERVE_BYTES > profile.disk_budget_bytes:
        return BudgetStop(
            "INCOMPLETE_BUDGET",
            f"session has committed {session_output_bytes} bytes of its "
            f"{profile.disk_budget_bytes} byte disk budget, leaving less than the "
            f"{DISK_FAILURE_RESERVE_BYTES} reserved for the records",
        )
    return None


def wall_stop(
    profile: ResourceProfile, *, elapsed_s: float, estimated_s: float
) -> BudgetStop | None:
    if estimated_s > profile.job_timeout_s:
        return BudgetStop(
            "INCOMPLETE_BUDGET",
            f"estimated {estimated_s}s exceeds the job timeout of {profile.job_timeout_s}s",
        )
    if elapsed_s + estimated_s > profile.wall_budget_s:
        return BudgetStop(
            "INCOMPLETE_BUDGET",
            f"session wall budget {profile.wall_budget_s}s cannot hold {estimated_s}s more "
            f"after {elapsed_s}s elapsed",
        )
    return None


# --------------------------------------------------------------------------- watchdog


class WatchdogDecision(StrictModel):
    """What the poller concluded. A value, not an action: the caller does the acting."""

    action: WatchdogAction
    accept_new_jobs: bool
    #: On a soft breach: release finished allocations and caches before growing further.
    release_completed: bool
    #: The status to record if this decision stops work. None while nothing is wrong.
    status: ForwardStatus | None
    caps: ResourceCaps
    reasons: tuple[str, ...]
    #: What this decision could NOT see — an unavailable OS pressure signal, say. Reported
    #: separately from `reasons` so it is never mistaken for a cause of a stop.
    limitations: tuple[str, ...]

    @property
    def terminate_process_group(self) -> bool:
        return self.action == "terminate"


class ResourceWatchdog:
    """Polls snapshots and escalates: warn, then drain, then terminate.

    The escalation needs memory of what it already saw, which is why this is a small
    object rather than a function. It holds no process handle and performs no kill: the
    worker task owns the Julia process group and acts on `terminate_process_group`.

    It does not own the poll loop either. The caller samples at `profile.poll_interval_s`
    and hands each snapshot to `observe`, which is what keeps every escalation reachable
    in a test by describing four instants instead of waiting through them.
    """

    def __init__(self, profile: ResourceProfile, *, baseline: ResourceSnapshot) -> None:
        self.profile = profile
        self.baseline_swap_bytes = baseline.swap_used_bytes
        self.started_monotonic_s = baseline.monotonic_s
        # RSS at the moment the current breach began; None while no breach is open.
        self._breach_rss_bytes: int | None = None
        self._terminal: WatchdogDecision | None = None

    def _decide(
        self,
        action: WatchdogAction,
        *,
        caps: ResourceCaps,
        limitations: tuple[str, ...],
        status: ForwardStatus | None = None,
        reasons: tuple[str, ...] = (),
    ) -> WatchdogDecision:
        return WatchdogDecision(
            action=action,
            accept_new_jobs=action in ("continue", "warn"),
            release_completed=action == "warn",
            status=status,
            caps=caps,
            reasons=reasons,
            limitations=limitations,
        )

    def observe(
        self,
        snapshot: ResourceSnapshot,
        *,
        job_started_monotonic_s: float | None = None,
        session_output_bytes: int = 0,
    ) -> WatchdogDecision:
        if self._terminal is not None:
            # A termination does not un-happen because the next sample looks calm.
            return self._terminal
        caps = effective_caps(self.profile, snapshot)
        limitations: tuple[str, ...] = ()
        if snapshot.pressure.status == "unknown":
            limitations = (
                f"OS memory pressure unavailable ({snapshot.pressure.method}); the "
                "available-memory, cap and swap guards still apply, the kernel's own "
                "pressure signal does not",
            )

        if job_started_monotonic_s is not None:
            job_elapsed_s = snapshot.monotonic_s - job_started_monotonic_s
            if job_elapsed_s > self.profile.job_timeout_s:
                self._terminal = self._decide(
                    "terminate",
                    caps=caps,
                    limitations=limitations,
                    status="TIMEOUT",
                    reasons=(
                        f"job has run {job_elapsed_s}s, past its {self.profile.job_timeout_s}s "
                        "timeout",
                    ),
                )
                return self._terminal

        # Free space is checked with the job timeout rather than with the breaches below,
        # because it shares their remedy and not the breaches' one: there is no draining
        # out of a disk that has already taken the room the failure record needs.
        disk = disk_headroom_stop(
            self.profile,
            session_output_bytes=session_output_bytes,
            free_bytes=snapshot.disk_free_bytes,
        )
        if disk is not None and disk.status == "RESOURCE_FAILURE":
            self._terminal = self._decide(
                "terminate",
                caps=caps,
                limitations=limitations,
                status=disk.status,
                reasons=(disk.reason,),
            )
            return self._terminal

        breaches = [
            stop
            for stop in (
                memory_stop(self.profile, snapshot),
                swap_growth_stop(
                    self.profile,
                    baseline_swap_bytes=self.baseline_swap_bytes,
                    snapshot=snapshot,
                ),
            )
            if stop is not None
        ]
        # None when the tree could not be read; every comparison below therefore checks.
        # Growth cannot be established on an unreadable tree, so such a breach stays a
        # drain and it is the job timeout that eventually ends it.
        rss = snapshot.process_rss_bytes
        if breaches:
            reasons = tuple(stop.reason for stop in breaches)
            if (
                self._breach_rss_bytes is not None
                and rss is not None
                and rss > self._breach_rss_bytes
            ):
                # Asked to finish the chunk, and grew anyway: stop asking.
                self._terminal = self._decide(
                    "terminate",
                    caps=caps,
                    limitations=limitations,
                    status="RESOURCE_FAILURE",
                    reasons=(
                        *reasons,
                        f"the process tree kept growing ({self._breach_rss_bytes} -> "
                        f"{rss} bytes) after being asked to stop",
                    ),
                )
                return self._terminal
            if self._breach_rss_bytes is None:
                self._breach_rss_bytes = rss
            return self._decide(
                "drain",
                caps=caps,
                limitations=limitations,
                status="RESOURCE_FAILURE",
                reasons=reasons,
            )
        self._breach_rss_bytes = None

        if disk is not None:
            # Out of disk allowance, not out of room: land the chunk in hand, start
            # nothing new. The same remedy as running out of session wall time below.
            return self._decide(
                "drain",
                caps=caps,
                limitations=limitations,
                status=disk.status,
                reasons=(disk.reason,),
            )

        elapsed_s = snapshot.monotonic_s - self.started_monotonic_s
        if elapsed_s > self.profile.wall_budget_s:
            # Out of time is not out of memory: finish the chunk in hand, start nothing new.
            return self._decide(
                "drain",
                caps=caps,
                limitations=limitations,
                status="INCOMPLETE_BUDGET",
                reasons=(
                    f"session has run {elapsed_s}s, past its {self.profile.wall_budget_s}s "
                    "wall budget",
                ),
            )

        if rss is not None and rss >= caps.soft_bytes:
            return self._decide(
                "warn",
                caps=caps,
                limitations=limitations,
                reasons=(
                    f"process tree holds {rss} bytes, at or above the {caps.soft_bytes} "
                    "soft threshold",
                ),
            )
        return self._decide("continue", caps=caps, limitations=limitations)


# ----------------------------------------------------------------------------- ledger


class BudgetTotals(StrictModel):
    """Summed cost of a set of attempts.

    Not a second `CostRecord`: that record describes exactly one attempt and caps its
    `retry_count` at the retry policy, which no sum can satisfy. This is the aggregate a
    session reports and a resume carries forward.
    """

    wall_s: float = Field(default=0.0, ge=0.0)
    cpu_s: float = Field(default=0.0, ge=0.0)
    output_bytes: int = Field(default=0, ge=0)
    forwards: int = Field(default=0, ge=0)

    def plus(self, other: BudgetTotals) -> BudgetTotals:
        return BudgetTotals(
            wall_s=self.wall_s + other.wall_s,
            cpu_s=self.cpu_s + other.cpu_s,
            output_bytes=self.output_bytes + other.output_bytes,
            forwards=self.forwards + other.forwards,
        )


class LedgerEntry(StrictModel):
    """One attempt. `job_id` is unique per attempt; `model_hash` identifies the physics."""

    job_id: str = Field(min_length=1)
    model_hash: Sha256
    case_sha256: Sha256
    attempt: int = Field(ge=1, le=MAX_ATTEMPTS)
    state: EntryState
    reserved_s: float = Field(ge=0.0)
    reserved_bytes: int = Field(ge=0)
    status: ForwardStatus | None = None
    cost: CostRecord | None = None
    checkpoint: RestartRef | None = None


class SessionLedger(StrictModel):
    """The persisted account of one session, and of every session it resumed."""

    schema_version: Literal["ledger-1"] = LEDGER_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    profile: ResourceProfile
    parent_session_id: str | None = None
    parent_ledger_sha256: Sha256 | None = None
    #: Every attempt of every earlier session in this chain, carried so that the total
    #: cost is not reset by a resume and a completed job can still be recognised.
    inherited_entries: tuple[LedgerEntry, ...] = ()
    entries: tuple[LedgerEntry, ...] = ()
    stop_reason: str | None = None


def _totals(entries: tuple[LedgerEntry, ...]) -> BudgetTotals:
    """Cost of the attempts that actually ran. Every attempt counts as a forward."""
    costs = [entry.cost for entry in entries if entry.cost is not None]
    return BudgetTotals(
        wall_s=sum(cost.wall_s for cost in costs),
        cpu_s=sum(cost.cpu_s for cost in costs),
        output_bytes=sum(cost.output_bytes for cost in costs),
        forwards=len(entries),
    )


class BudgetLedger:
    """The session's account, written atomically on every change.

    The probe is injected: it is the single source of both the machine's state and the
    clock, so a budget boundary can be described in a test rather than waited for.
    """

    def __init__(
        self,
        record: SessionLedger,
        *,
        path: Path,
        probe: Callable[[], ResourceSnapshot],
        baseline: ResourceSnapshot,
    ) -> None:
        self._record = record
        self._path = path
        self._probe = probe
        self._started_monotonic_s = baseline.monotonic_s
        self._baseline_swap_bytes = baseline.swap_used_bytes

    # ---------------------------------------------------------------- construction

    @classmethod
    def start(
        cls,
        *,
        profile: ResourceProfile,
        path: Path,
        session_id: str,
        probe: Callable[[], ResourceSnapshot],
        now: datetime | None = None,
        parent: SessionLedger | None = None,
        parent_ledger_sha256: str | None = None,
    ) -> BudgetLedger:
        baseline = probe()
        inherited = (*parent.inherited_entries, *parent.entries) if parent is not None else ()
        record = SessionLedger(
            session_id=session_id,
            started_at=(now or datetime.now(UTC)).isoformat(),
            profile=profile,
            parent_session_id=parent.session_id if parent is not None else None,
            parent_ledger_sha256=parent_ledger_sha256,
            inherited_entries=inherited,
        )
        ledger = cls(record, path=path, probe=probe, baseline=baseline)
        ledger._write()
        return ledger

    @classmethod
    def resume(
        cls,
        *,
        parent_path: Path,
        path: Path,
        session_id: str,
        probe: Callable[[], ResourceSnapshot],
        profile: ResourceProfile | None = None,
        now: datetime | None = None,
    ) -> BudgetLedger:
        """Open a new session continuing `parent_path`.

        The new session gets its own wall budget — a resume is a new sitting, not a
        continuation of the old clock — and records the parent's session id and the
        SHA-256 of its exact bytes, so the chain is traceable. It inherits the parent's
        entries, which is what keeps the cumulative cost and the per-model attempt count
        from being reset by resuming.
        """
        parent = load_ledger(parent_path)
        return cls.start(
            profile=profile if profile is not None else parent.profile,
            path=path,
            session_id=session_id,
            probe=probe,
            now=now,
            parent=parent,
            parent_ledger_sha256=sha256_file(parent_path),
        )

    # ------------------------------------------------------------------- accessors

    @property
    def record(self) -> SessionLedger:
        return self._record

    @property
    def path(self) -> Path:
        return self._path

    @property
    def attempted_job_ids(self) -> tuple[str, ...]:
        return tuple(entry.job_id for entry in self._record.entries)

    @property
    def completed_job_ids(self) -> tuple[str, ...]:
        return tuple(e.job_id for e in self._record.entries if e.state == "COMPLETE")

    @property
    def pending_job_ids(self) -> tuple[str, ...]:
        return tuple(e.job_id for e in self._record.entries if e.state == "PENDING")

    def session_totals(self) -> BudgetTotals:
        return _totals(self._record.entries)

    def cumulative_totals(self) -> BudgetTotals:
        """Everything this chain of sessions has spent. A resume does not reset it."""
        return _totals(self._record.inherited_entries).plus(self.session_totals())

    def _all_entries(self) -> tuple[LedgerEntry, ...]:
        return (*self._record.inherited_entries, *self._record.entries)

    def committed_output_bytes(self) -> int:
        """What this session has written plus what it has already promised to write.

        A reservation that has not resolved yet is still owed disk: counting only finished
        output would let two outstanding jobs each pass a check that together they fail,
        and would silently forget the space owed by a job whose process died.
        """
        return self.session_totals().output_bytes + sum(
            entry.reserved_bytes for entry in self._record.entries if entry.state == "PENDING"
        )

    def is_already_complete(self, *, model_hash: str, case_sha256: str) -> bool:
        """Whether this exact physical model has a verified completed attempt on record.

        Keyed on `model_hash` because `job_id` is unique per attempt (plan 3.1) and so
        never repeats across a resume. The recorded input hash must match the case being
        planned: a regenerated case is different work, however familiar its model hash
        looks. An attempt that is pending or failed is never a skip.
        """
        return any(
            entry.state == "COMPLETE"
            and entry.model_hash == model_hash
            and entry.case_sha256 == case_sha256
            for entry in self._all_entries()
        )

    # ---------------------------------------------------------------------- writes

    def _write(self) -> None:
        write_json_atomic(self._path, self._record.model_dump(mode="json"))

    def _replace(self, **fields: object) -> None:
        self._record = SessionLedger.model_validate({**self._record.model_dump(), **fields})
        self._write()

    def reserve(
        self,
        job_id: str,
        estimated_s: float,
        estimated_bytes: int,
        *,
        case_sha256: str,
        model_hash: str,
        attempt: int = 1,
    ) -> None:
        """Check every budget, then book the attempt. Raises `BudgetStop` if refused.

        The attempt is recorded before the job runs, and it stays recorded whatever
        happens to the job: a crash leaves a PENDING entry, which is exactly what tells a
        resumed session that this work is unfinished rather than untried.
        """
        if self._record.stop_reason is not None:
            raise BudgetStop("INCOMPLETE_BUDGET", f"session stopped: {self._record.stop_reason}")
        if any(entry.job_id == job_id for entry in self._all_entries()):
            raise BudgetStop(
                "PROTOCOL_FAILURE",
                f"job_id {job_id!r} is already on the ledger; a job id is unique per attempt",
            )
        prior = sum(1 for entry in self._all_entries() if entry.model_hash == model_hash)
        if prior >= MAX_ATTEMPTS:
            raise BudgetStop(
                "PROTOCOL_FAILURE",
                f"model {model_hash} already has {prior} attempts on record; SPEC 3.3 "
                f"allows at most {MAX_ATTEMPTS}",
            )
        if len(self._record.entries) >= self._record.profile.max_new_forward:
            raise BudgetStop(
                "INCOMPLETE_BUDGET",
                f"session has already spent its whole budget of "
                f"{self._record.profile.max_new_forward} new forwards",
            )
        snapshot = self._probe()
        profile = self._record.profile
        # Resources first, then the schedule: "there is nowhere to run" is a more
        # fundamental refusal than "there is not enough time left to run".
        checks = (
            memory_stop(profile, snapshot),
            swap_growth_stop(
                profile, baseline_swap_bytes=self._baseline_swap_bytes, snapshot=snapshot
            ),
            disk_stop(
                profile,
                session_output_bytes=self.committed_output_bytes(),
                predicted_bytes=estimated_bytes,
                free_bytes=snapshot.disk_free_bytes,
            ),
            wall_stop(
                profile,
                elapsed_s=snapshot.monotonic_s - self._started_monotonic_s,
                estimated_s=estimated_s,
            ),
        )
        for stop in checks:
            if stop is not None:
                raise stop
        entry = LedgerEntry(
            job_id=job_id,
            model_hash=model_hash,
            case_sha256=case_sha256,
            attempt=attempt,
            state="PENDING",
            reserved_s=estimated_s,
            reserved_bytes=estimated_bytes,
        )
        self._replace(entries=(*self._record.entries, entry))

    def complete(self, job_id: str, result: ForwardResult) -> None:
        """Resolve a reservation with its measured cost. Failure resolves it too.

        A failed attempt becomes FAILED rather than disappearing: it was paid for, it
        counts against the forward budget, and SPEC 18.4 needs the reason on record
        instead of an unexplained absence.

        The result must be the one this reservation is waiting for, in all three of its
        identities. Two attempts on one physical model share a case hash, so checking only
        that would let one attempt's result be filed against the other's entry: the wrong
        cost and checkpoint on one row, and a second row left PENDING forever, holding its
        reserved disk against the session for the rest of the run.
        """
        index = next(
            (
                i
                for i, entry in enumerate(self._record.entries)
                if entry.job_id == job_id and entry.state == "PENDING"
            ),
            None,
        )
        if index is None:
            raise BudgetStop(
                "PROTOCOL_FAILURE",
                f"no pending reservation for job_id {job_id!r}; a job must be reserved "
                "before it is completed",
            )
        entry = self._record.entries[index]
        for label, reserved, reported in (
            ("job_id", entry.job_id, result.job_id),
            ("model_hash", entry.model_hash, result.model_hash),
            ("case_sha256", entry.case_sha256, result.case_sha256),
        ):
            if reserved != reported:
                raise BudgetStop(
                    "PROTOCOL_FAILURE",
                    f"job {job_id!r} was reserved with {label} {reserved} but the result "
                    f"reports {reported}",
                )
        resolved = LedgerEntry.model_validate(
            {
                **entry.model_dump(),
                "state": "COMPLETE" if result.status == "COMPLETE" else "FAILED",
                "status": result.status,
                "cost": result.cost.model_dump(),
                "checkpoint": result.restart.model_dump() if result.restart else None,
            }
        )
        entries = list(self._record.entries)
        entries[index] = resolved
        self._replace(entries=tuple(entries))

    def stop(self, reason: str) -> None:
        """Close the session. Completed and pending work both stay on the record."""
        if self._record.stop_reason is None:
            self._replace(stop_reason=reason)


def load_ledger(path: Path) -> SessionLedger:
    """Read a ledger, refusing anything that is not one rather than starting empty."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read parent ledger {path}: {exc}") from exc
    try:
        return SessionLedger.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"parent ledger {path} is not a valid {LEDGER_SCHEMA_VERSION} record:\n{exc}"
        ) from exc
