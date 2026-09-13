"""One long-lived Julia process, and the isolation of the jobs it runs.

This module is the transport. It owns a process and a line protocol, and nothing else:
the budget policy lives in `simulator.budget` and is only *acted on* here, the physics
lives behind `julia/worker/main.jl`, and the records travel as `JobDescriptor` in and
`ForwardResult` out. Paying Julia's cold start once and then feeding it immutable job
descriptors is the whole point — a process per forward would spend more time compiling
than simulating.

Four rules shape the implementation, and each of them is a way a transport like this
usually goes wrong:

* **No unbounded wait.** Every wait has a deadline on a monotonic clock and is taken in
  slices no longer than `ResourceProfile.poll_interval_s`, so a stop is noticed within a
  poll rather than at the end of a blocking read. The cold start is bounded by
  `ResourceProfile.startup_timeout_s`; a running job is bounded by the watchdog, which
  reads the monotonic clock carried in each `ResourceSnapshot`. A system clock that jumps
  therefore cannot extend either budget.
* **No unbounded read.** A reader thread drains stdout continuously into a bounded queue,
  so the child can never block writing to a full pipe while this side is busy, and the
  control loop never calls `readline()` itself. Each line is read with a character limit:
  a worker that emits no newline at all is a protocol violation, not a reason to grow a
  buffer until the machine gives up. Native Julia logging never reaches this pipe — it is
  redirected into per-job stdout/stderr files, and the process-level stderr goes to a log
  file rather than to a second pipe nobody is draining.
* **No orphaned process.** The child is started with `start_new_session=True`, which makes
  it the leader of its own process group, so a termination reaches every descendant it
  spawned rather than only the one pid this side knows about. Shutdown is an escalation:
  a `shutdown` request, then SIGTERM to the group, then SIGKILL, each with a five second
  grace, and a final sweep of the group before the leader is reaped — the leader is still
  a zombie at that moment, so its pid, and therefore the group id, cannot have been reused
  by anyone else. Exceptions and Ctrl-C reach that path through `finally`.
* **No unexplained result.** A job that produced no reply is classified from evidence.
  A guard breach or a timeout observed by the watchdog is recorded as `RESOURCE_FAILURE`
  or `TIMEOUT`; a process that merely died, however violently, is `PROTOCOL_FAILURE` with
  its exit code or signal. A force-kill on its own is not a diagnosis of memory exhaustion
  (SPEC 18.4: a resource failure must not become a physical zero likelihood by default).

Job identity is checked on both sides. Python stamps what only it knows — the environment
lock hash and the solver-config digest the descriptor declares — into
`ForwardResult.solver_metadata`, and `main.jl` re-hashes the bytes of the case, of every
array the case references and of the solver configuration before it runs anything at all.
Until the JutulDarcy adapter lands, `run` can only answer `INVALID_INPUT: adapter
unavailable`, and this side refuses a `COMPLETE` reply outright: a transport must not be
able to manufacture a successful physics result without a solver.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from types import TracebackType
from typing import IO, Any, Literal, get_args

from pydantic import Field

from so_recon.config.resources import ResourceProfile, require_resource_profile
from so_recon.config.schema import StrictModel
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.simulator.budget import BudgetLedger, ResourceWatchdog, WatchdogDecision
from so_recon.simulator.contracts import (
    CostRecord,
    ForwardResult,
    ForwardStatus,
    JobDescriptor,
)

log = logging.getLogger(__name__)

#: Bumped whenever the line protocol changes shape. Both sides state it; a mismatch is a
#: refusal to start rather than a run that half understands its own worker.
PROTOCOL_VERSION = "worker-1"

#: The statuses a reply may carry. This IS the contract's set (plan 3.3) rather than a
#: copy of it, so a status no `ForwardResult` could hold cannot cross the transport.
WORKER_STATUSES: frozenset[str] = frozenset(get_args(ForwardStatus))

#: The reader refuses a line longer than this without a newline. A megabyte is far more
#: than any legitimate frame and far less than enough to exhaust a machine.
MAX_PROTOCOL_LINE_CHARS = 1 << 20

#: How many frames are retained while the control loop is busy. On overflow the oldest is
#: dropped and counted: the reader must never stop draining the pipe to apply back
#: pressure, because the thing it would be applying back pressure to is the process it is
#: also responsible for terminating.
MAX_PENDING_FRAMES = 256

#: Escalation grace periods. `shutdown` first, then the signals.
SHUTDOWN_GRACE_S = 5.0
TERMINATE_GRACE_S = 5.0

#: The control-plane request an operator makes to stop after the current unit of work.
#: It is a file in the session directory and it changes no job input; Task 8 reads it at
#: native chunk boundaries. `PersistentJuliaWorker.stop_requested` merely recognises it.
STOP_FILE_NAME = "stop-after-chunk"

#: A job id becomes a file name (the descriptor, the per-job logs), so it is restricted to
#: characters that cannot escape a directory or surprise a shell.
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkerProtocolError(RuntimeError):
    """The worker said something the protocol does not allow, or said nothing at all."""


class WorkerStartupError(WorkerProtocolError):
    """The worker never reached READY within its startup budget."""


# --------------------------------------------------------------------- reply kernels


def validate_reply(reply: dict[str, object], job_id: str) -> None:
    """Refuse a reply that is not this job's, or that carries a status no record holds.

    A stale reply is the dangerous case and the reason this is checked at all: a worker
    that answered the previous job late would otherwise have its answer filed against the
    current one, and both the cost and the physics would be attributed to the wrong case.
    """
    if reply.get("job_id") != job_id:
        raise ValueError(
            f"job_id mismatch: the reply carries {reply.get('job_id')!r} but this job is "
            f"{job_id!r}; a reply is only ever accepted for the job it names"
        )
    if reply.get("status") not in WORKER_STATUSES:
        raise ValueError(
            f"unknown worker status {reply.get('status')!r}; the contract allows "
            f"{sorted(WORKER_STATUSES)}"
        )


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return "unknown signal"


def classify_termination(
    *, returncode: int | None, decision: WatchdogDecision | None
) -> tuple[ForwardStatus, str]:
    """Say why a job produced no reply, using only evidence that was actually collected.

    A guard breach or an expired job timeout is evidence: the watchdog looked at a measured
    snapshot and said so, and the decision it returned carries both the status and the
    reasons. Anything else is a protocol failure that names the exit code or the signal.

    The distinction is deliberate and normative. `exit -9` on its own says the kernel or
    someone else killed the process; it does not say the process ran out of memory, and
    recording `RESOURCE_FAILURE` from a signal number alone would invent an OOM diagnosis
    that no measurement supports (SPEC 18.4).
    """
    if decision is not None and decision.status is not None:
        detail = "; ".join(decision.reasons) or f"watchdog action {decision.action}"
        if decision.limitations:
            detail = f"{detail} (not observed: {'; '.join(decision.limitations)})"
        return decision.status, detail
    if returncode is None:
        return (
            "PROTOCOL_FAILURE",
            "the worker process is still running but produced no reply for this job",
        )
    if returncode < 0:
        number = -returncode
        return (
            "PROTOCOL_FAILURE",
            f"the worker process was killed by signal {number} ({_signal_name(number)}) "
            "with no guard or OS evidence of a resource breach; a force-kill on its own "
            "does not establish that the process ran out of memory",
        )
    return (
        "PROTOCOL_FAILURE",
        f"the worker process exited with code {returncode} before replying to this job",
    )


# ------------------------------------------------------------------ control plane file


def request_stop_after_chunk(session_dir: Path, *, reason: str | None = None) -> Path:
    """Ask the session to stop after the work in hand. Touches no job input.

    Written atomically, because a half-written request is a request nobody can read. It
    lives beside the session's own bookkeeping and never inside a case, a descriptor or a
    result directory: an immutable input stays immutable, whatever an operator asks for.
    """
    path = session_dir / STOP_FILE_NAME
    write_json_atomic(path, {"request": STOP_FILE_NAME, "reason": reason})
    return path


def stop_after_chunk_requested(session_dir: Path) -> bool:
    return (session_dir / STOP_FILE_NAME).is_file()


# -------------------------------------------------------------------- protocol reader


@dataclass(frozen=True)
class ProtocolViolation:
    """A line that could not be a frame. Carried to the control loop as a value."""

    reason: str


Frame = dict[str, Any] | ProtocolViolation


class _ProtocolReader:
    """Drains the worker's stdout in a thread, so the pipe is never left to fill up."""

    def __init__(self, stream: IO[str], *, max_line_chars: int, max_pending: int) -> None:
        self._stream = stream
        self._max_line_chars = max_line_chars
        self._frames: queue.Queue[Frame] = queue.Queue(maxsize=max_pending)
        self.finished = threading.Event()
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="julia-protocol-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get(self, timeout_s: float) -> Frame | None:
        try:
            return self._frames.get(timeout=max(timeout_s, 0.0))
        except queue.Empty:
            return None

    def join(self, timeout_s: float) -> None:
        self._thread.join(timeout_s)

    def _offer(self, frame: Frame) -> None:
        """Never blocks. A full queue loses its oldest frame, and says so."""
        while True:
            try:
                self._frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:  # another consumer emptied it between the two calls
                    continue
                self.dropped += 1
                log.warning(
                    "worker protocol queue is full; dropped the oldest of %d frames. "
                    "stdout carries protocol frames only, so this means the worker is "
                    "writing frames nobody asked for",
                    self.dropped,
                )

    def _discard_to_newline(self) -> None:
        while True:
            chunk = self._stream.readline(self._max_line_chars + 1)
            if not chunk or chunk.endswith("\n"):
                return

    def _run(self) -> None:
        try:
            while True:
                # Bounded: `readline(size)` stops at the limit, so an endless line is
                # refused instead of being accumulated until the machine gives up.
                line = self._stream.readline(self._max_line_chars + 1)
                if not line:
                    return
                if len(line) > self._max_line_chars and not line.endswith("\n"):
                    self._offer(
                        ProtocolViolation(
                            f"the worker emitted more than {self._max_line_chars} "
                            "characters with no newline; the protocol reader is bounded "
                            "and refuses to keep reading one frame"
                        )
                    )
                    self._discard_to_newline()
                    continue
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except ValueError as exc:
                    self._offer(
                        ProtocolViolation(
                            f"the worker wrote a line to stdout that is not JSON ({exc}); "
                            f"stdout carries protocol frames only, got {text[:200]!r}"
                        )
                    )
                    continue
                if not isinstance(payload, dict):
                    self._offer(
                        ProtocolViolation(
                            f"a protocol frame must be a JSON object, got {type(payload).__name__}"
                        )
                    )
                    continue
                self._offer(payload)
        except (OSError, ValueError):  # the stream was closed under us during shutdown
            return
        finally:
            self.finished.set()


# --------------------------------------------------------------------- typed replies


class WorkerHandshake(StrictModel):
    """The READY frame: who the worker is and what it was started with."""

    protocol: str = Field(min_length=1)
    pid: int = Field(gt=0)
    julia_threads: int
    blas_threads: int
    versions: dict[str, str]


class PingReply(StrictModel):
    """A diagnostic echo. Never a physics result, and never becomes a `ForwardResult`."""

    job_id: str = Field(min_length=1)
    status: ForwardStatus
    protocol: str = Field(min_length=1)
    pid: int = Field(gt=0)
    #: name -> SHA-256 of the bytes the worker read at the path under that name.
    inputs: dict[str, str]


# ------------------------------------------------------------------------- the worker


class PersistentJuliaWorker:
    """One Julia process, kept alive across jobs, each job isolated by its descriptor.

    Construction launches the process and waits for READY inside
    `profile.startup_timeout_s`. It is a context manager because the only safe way to own
    a process group is to have a `finally` that ends it.

    `probe` and `clock` are injected. They are the two measurement boundaries of the
    class: the probe is where every resource number and the watchdog's own notion of time
    come from, and the clock is what the startup and shutdown deadlines are measured on.
    Injecting them is what lets a hung start, an exhausted job timeout and a memory breach
    be described in a test rather than waited for, while the policy being exercised stays
    the real `budget.ResourceWatchdog`.
    """

    def __init__(
        self,
        executable: Path,
        project: Path,
        session_dir: Path,
        profile: ResourceProfile,
        *,
        paths: ProjectPaths,
        probe: Callable[[], ResourceSnapshot] | None = None,
        clock: Callable[[], float] = time.monotonic,
        command: str = "forward",
    ) -> None:
        # This process exists to run physics, so it needs an approved, measured budget
        # before it spends anything at all (COMPUTE §§2, 5, 7, 10; SPEC 18.4).
        self._profile = require_resource_profile(profile, command=command)
        self._paths = paths
        self._session_dir = session_dir
        self._clock = clock
        self._probe: Callable[[], ResourceSnapshot] = (
            probe if probe is not None else (lambda: probe_resources(None, session_dir))
        )
        self._environment_lock_hash = environment_lock_hash(paths)
        self._closed = False
        self._broken = False

        script = project / "worker" / "main.jl"
        if not script.is_file():
            raise FileNotFoundError(
                f"the persistent worker entry point {script} does not exist; "
                "julia/worker/main.jl is what a session runs"
            )
        session_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_log_path = session_dir / "worker.stderr.log"
        self._stderr_log = self._stderr_log_path.open("ab")

        env = os.environ.copy()
        # COMPUTE §5: 4 Julia threads, 1 BLAS thread, 1 worker. They come from the
        # profile, never from whatever the ambient shell happens to export.
        env["JULIA_NUM_THREADS"] = str(self._profile.julia_threads)
        env["OPENBLAS_NUM_THREADS"] = str(self._profile.blas_threads)
        env["OMP_NUM_THREADS"] = str(self._profile.blas_threads)
        try:
            self._proc: Popen[str] = Popen(
                [
                    str(executable),
                    f"--project={project}",
                    "--startup-file=no",
                    "--color=no",
                    str(script),
                ],
                stdin=PIPE,
                stdout=PIPE,
                stderr=self._stderr_log,
                text=True,
                env=env,
                cwd=paths.root,
                # Its own session, so it leads its own process group and a termination
                # reaches every descendant rather than only this pid.
                start_new_session=True,
            )
        except BaseException:
            self._stderr_log.close()
            raise
        if self._proc.stdout is None:  # pragma: no cover - Popen was given a PIPE
            self._proc.kill()
            self._proc.wait()
            self._stderr_log.close()
            raise WorkerProtocolError("the worker was started without a stdout pipe")
        self._reader = _ProtocolReader(
            self._proc.stdout,
            max_line_chars=MAX_PROTOCOL_LINE_CHARS,
            max_pending=MAX_PENDING_FRAMES,
        )
        try:
            self._reader.start()
            self._handshake = self._await_ready()
        except BaseException:
            # A worker that never became usable still owns a process group.
            self._broken = True
            self.close()
            raise

    # ------------------------------------------------------------------ accessors

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    @property
    def handshake(self) -> WorkerHandshake:
        return self._handshake

    @property
    def stop_requested(self) -> bool:
        """Whether an operator has asked this session to stop after the work in hand."""
        return stop_after_chunk_requested(self._session_dir)

    def __enter__(self) -> PersistentJuliaWorker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    # ------------------------------------------------------------------- protocol

    def _await_ready(self) -> WorkerHandshake:
        """Wait for READY, bounded by the profile's startup budget on a monotonic clock."""
        deadline = self._clock() + self._profile.startup_timeout_s
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                raise WorkerStartupError(
                    f"the worker did not reach READY within "
                    f"{self._profile.startup_timeout_s}s; see {self._stderr_log_path}"
                )
            frame = self._reader.get(min(self._profile.poll_interval_s, remaining))
            if frame is None:
                if self._exhausted():
                    raise WorkerStartupError(
                        f"the worker exited with {self._proc.returncode} before reaching "
                        f"READY; see {self._stderr_log_path}"
                    )
                continue
            if isinstance(frame, ProtocolViolation):
                raise WorkerStartupError(f"the worker's first frame was unusable: {frame.reason}")
            if frame.get("event") != "ready":
                raise WorkerStartupError(f"expected a READY frame from the worker, got {frame!r}")
            handshake = WorkerHandshake.model_validate(
                {k: v for k, v in frame.items() if k != "event"}
            )
            if handshake.protocol != PROTOCOL_VERSION:
                raise WorkerStartupError(
                    f"the worker speaks protocol {handshake.protocol!r}, this session "
                    f"speaks {PROTOCOL_VERSION!r}"
                )
            return handshake

    def _exhausted(self) -> bool:
        """True when the process has exited and every frame it wrote has been consumed."""
        return self._proc.poll() is not None and self._reader.finished.is_set()

    def _send(self, request: Mapping[str, object]) -> None:
        if self._proc.stdin is None:  # pragma: no cover - Popen was given a PIPE
            raise WorkerProtocolError("the worker has no stdin to write to")
        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise WorkerProtocolError(
                f"could not send {request.get('op')!r} to the worker: {exc}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("this worker is closed; start a new one to run more work")
        if self._broken:
            raise RuntimeError(
                "this worker was terminated mid-job and cannot be resynchronised; start a new one"
            )

    # ----------------------------------------------------------------------- ping

    def ping(self, job_id: str, inputs: Mapping[str, Path]) -> PingReply:
        """Diagnostic round trip: the worker re-hashes the named files and echoes them.

        Not a job. It runs no physics, reserves nothing and never produces a
        `ForwardResult`; it exists so that the protocol itself can be exercised against a
        real Julia process without a solver.
        """
        self._require_open()
        request = {
            "op": "ping",
            "job_id": job_id,
            "protocol": PROTOCOL_VERSION,
            "root": str(self._paths.root),
            "inputs": {name: self._paths.relative(path) for name, path in inputs.items()},
        }
        self._send(request)
        deadline = self._clock() + self._profile.job_timeout_s
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                raise WorkerProtocolError(
                    f"the worker did not answer ping {job_id!r} within "
                    f"{self._profile.job_timeout_s}s"
                )
            frame = self._reader.get(min(self._profile.poll_interval_s, remaining))
            if frame is None:
                if self._exhausted():
                    raise WorkerProtocolError(
                        f"the worker exited with {self._proc.returncode} without answering "
                        f"ping {job_id!r}"
                    )
                continue
            if isinstance(frame, ProtocolViolation):
                raise WorkerProtocolError(frame.reason)
            validate_reply(frame, job_id)
            return PingReply.model_validate(frame)

    # --------------------------------------------------------------------- submit

    def submit(
        self,
        job: JobDescriptor,
        ledger: BudgetLedger,
        *,
        estimated_s: float | None = None,
        estimated_bytes: int | None = None,
    ) -> ForwardResult:
        """Run one job to a classified result, and account for it either way.

        The reservation is taken before the job starts and resolved after it ends,
        whatever it ended as: SPEC 3.3 counts every attempt against the forward budget,
        and a failed attempt that vanished from the ledger would buy free work.

        The default estimates are policy, not physics. A job may take at most
        `job_timeout_s`, and may fairly claim its share of the session's disk budget —
        `disk_budget_bytes / max_new_forward`. Task 8 knows the case and passes a measured
        prediction instead.
        """
        self._require_open()
        if not _SAFE_JOB_ID.match(job.job_id):
            raise ValueError(
                f"job_id {job.job_id!r} is not usable as a file name; a job id may contain "
                "only letters, digits, '.', '_' and '-' because it names the descriptor "
                "and the per-job log files"
            )
        ledger.reserve(
            job.job_id,
            float(self._profile.job_timeout_s) if estimated_s is None else estimated_s,
            self._fair_share_bytes() if estimated_bytes is None else estimated_bytes,
            case_sha256=job.case_sha256,
            model_hash=job.model_hash,
            attempt=job.attempt,
        )
        # What the session owes the disk, read once now: the watchdog needs it to tell a
        # disk that is out of allowance from one that is out of room, and with a single
        # worker nothing else can change it while this job runs.
        committed_bytes = ledger.committed_output_bytes()
        try:
            result = self._run_job(job, committed_bytes)
        except BaseException:
            # An exception or a Ctrl-C leaves the protocol out of step, and the job's
            # process group still running. End the group; the reservation stays PENDING,
            # which is exactly what tells a resumed session this work is unfinished.
            self._terminate_now()
            raise
        ledger.complete(job.job_id, result)
        return result

    def _fair_share_bytes(self) -> int:
        return max(1, self._profile.disk_budget_bytes // self._profile.max_new_forward)

    def _write_descriptor(self, job: JobDescriptor) -> Path:
        path = self._session_dir / "jobs" / f"{job.job_id}.json"
        write_json_atomic(path, job.model_dump(mode="json"))
        return path

    def _run_job(self, job: JobDescriptor, session_output_bytes: int) -> ForwardResult:
        job_path = self._write_descriptor(job)
        log_dir = self._session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{job.job_id}.stdout"
        stderr_path = log_dir / f"{job.job_id}.stderr"

        baseline = self._probe()
        watchdog = ResourceWatchdog(self._profile, baseline=baseline)
        meter = _JobMeter(baseline)
        try:
            self._send(
                {
                    "op": "run",
                    "job_id": job.job_id,
                    "job_path": str(job_path),
                    "root": str(self._paths.root),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "protocol": PROTOCOL_VERSION,
                }
            )
        except WorkerProtocolError as exc:
            return self._classified(job, "PROTOCOL_FAILURE", str(exc), meter.finish(self._probe()))

        frame, decision = self._await_job_frame(watchdog, meter, session_output_bytes)
        cost_inputs = meter.finish(self._probe())
        if frame is None:
            status, reason = classify_termination(returncode=self._proc.poll(), decision=decision)
            return self._classified(job, status, reason, cost_inputs)
        if isinstance(frame, ProtocolViolation):
            return self._classified(job, "PROTOCOL_FAILURE", frame.reason, cost_inputs)
        try:
            validate_reply(frame, job.job_id)
        except ValueError as exc:
            return self._classified(job, "PROTOCOL_FAILURE", str(exc), cost_inputs)
        return self._result_from_reply(job, frame, cost_inputs)

    def _await_job_frame(
        self, watchdog: ResourceWatchdog, meter: _JobMeter, session_output_bytes: int
    ) -> tuple[Frame | None, WatchdogDecision | None]:
        """Wait in poll-sized slices, watching the machine between them.

        No `readline()` happens here: the reader thread already holds the pipe, and this
        loop only takes what it produced. The exit conditions are a frame, a dead process
        with nothing left queued, or a watchdog decision to terminate.
        """
        last: WatchdogDecision | None = None
        while True:
            frame = self._reader.get(self._profile.poll_interval_s)
            if frame is not None:
                return frame, None
            if self._exhausted():
                # The MOST RECENT observation is the evidence, not an older breach the
                # tree has since recovered from: a process that died while measurably
                # healthy did not die of a resource exhaustion anyone observed.
                return None, last
            snapshot = self._probe()
            meter.observe(snapshot)
            last = watchdog.observe(
                snapshot,
                job_started_monotonic_s=meter.started_monotonic_s,
                session_output_bytes=session_output_bytes,
            )
            if last.terminate_process_group:
                self._terminate_now()
                return None, last

    # ----------------------------------------------------------------- result building

    def _classified(
        self, job: JobDescriptor, status: ForwardStatus, reason: str, cost_inputs: _CostInputs
    ) -> ForwardResult:
        return self._build_result(
            job,
            status=status,
            reason=reason,
            physics_class="unknown",
            cost_inputs=cost_inputs,
            extra_metadata={},
        )

    def _result_from_reply(
        self, job: JobDescriptor, reply: dict[str, Any], cost_inputs: _CostInputs
    ) -> ForwardResult:
        status: ForwardStatus = reply["status"]
        if status == "COMPLETE":
            # Defence in depth for the sequencing of this stage: no solver adapter is
            # wired yet, so nothing downstream may treat a reply as a finished physics
            # result. Task 5 replaces this branch with the real COMPLETE decoding.
            return self._classified(
                job,
                "PROTOCOL_FAILURE",
                "the worker reported COMPLETE, but no JutulDarcy adapter is wired into "
                "this build; a transport cannot produce a physics result without a solver",
                cost_inputs,
            )
        reason = reply.get("reason")
        if not isinstance(reason, str) or not reason:
            return self._classified(
                job,
                "PROTOCOL_FAILURE",
                f"the worker reported {status} without a reason; an unsuccessful result "
                "must say what is missing (plan 3.2)",
                cost_inputs,
            )
        metadata: dict[str, str] = {}
        result_path = reply.get("result_path")
        if isinstance(result_path, str) and result_path:
            try:
                resolved = self._paths.resolve(result_path)
                digest = sha256_file(resolved)
            except (OSError, ValueError) as exc:
                return self._classified(
                    job,
                    "PROTOCOL_FAILURE",
                    f"the worker reported a result at {result_path!r} that cannot be read "
                    f"back: {exc}",
                    cost_inputs,
                )
            result_dir = self._paths.resolve(job.result_dir)
            if not resolved.is_relative_to(result_dir):
                # A job writes into the directory its own descriptor named, and nowhere
                # else: a result filed outside it would attribute this job's output to
                # some other job, or overwrite an artifact that is not a result at all.
                return self._classified(
                    job,
                    "PROTOCOL_FAILURE",
                    f"the worker reported a result at {result_path!r}, outside this job's "
                    f"own result directory {job.result_dir!r}",
                    cost_inputs,
                )
            if digest != reply.get("result_sha256"):
                return self._classified(
                    job,
                    "PROTOCOL_FAILURE",
                    f"the worker reported result_sha256 {reply.get('result_sha256')!r} for "
                    f"{result_path!r}, whose bytes hash to {digest}",
                    cost_inputs,
                )
            metadata["result_path"] = result_path
            metadata["result_sha256"] = digest
            cost_inputs = cost_inputs.with_output_bytes(_directory_bytes(result_dir))
        declared = reply.get("physics_class")
        return self._build_result(
            job,
            status=status,
            reason=reason,
            physics_class=declared if isinstance(declared, str) and declared else "unknown",
            cost_inputs=cost_inputs,
            extra_metadata=metadata,
        )

    def _build_result(
        self,
        job: JobDescriptor,
        *,
        status: ForwardStatus,
        reason: str,
        physics_class: str,
        cost_inputs: _CostInputs,
        extra_metadata: Mapping[str, str],
    ) -> ForwardResult:
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            # What the result was produced under, beyond the case itself: the locked
            # environment and the solver configuration the descriptor pinned (plan 3.1).
            "environment_lock_hash": self._environment_lock_hash,
            "solver_config_sha256": job.solver_config_sha256,
            "solver_config_path": job.solver_config_path,
            "resource_profile": self._profile.profile,
            "julia_threads": str(self._profile.julia_threads),
            "blas_threads": str(self._profile.blas_threads),
            "worker_pid": str(self._proc.pid),
            **{f"version.{name}": value for name, value in self._handshake.versions.items()},
            **extra_metadata,
        }
        return ForwardResult(
            job_id=job.job_id,
            case_sha256=job.case_sha256,
            model_hash=job.model_hash,
            physics_class=physics_class,
            status=status,
            reason=reason,
            completed_time_s=0.0,
            times_s=(),
            states={},
            solver_metadata=metadata,
            cost=cost_inputs.to_record(attempt=job.attempt),
            parent_attempt_ids=(),
        )

    # ---------------------------------------------------------------- process control

    def _signal_group(self, sig: signal.Signals) -> None:
        """Signal the whole group. The child leads it, so its pid is the group id.

        A group that no longer exists is not an error here: every caller is trying to make
        sure nothing survives, and "nothing did" is the outcome they wanted.
        """
        with contextlib.suppress(OSError):
            os.killpg(self._proc.pid, sig)

    def _wait_for_exit(self, budget_s: float) -> bool:
        """Wait in poll-sized slices until the process exits or the budget runs out."""
        deadline = self._clock() + budget_s
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                return self._proc.poll() is not None
            try:
                self._proc.wait(timeout=max(0.0, min(self._profile.poll_interval_s, remaining)))
                return True
            except TimeoutExpired:
                continue

    def _terminate_now(self) -> None:
        """End the process group without asking. Used when the protocol cannot continue."""
        self._broken = True
        self._signal_group(signal.SIGTERM)
        if not self._wait_for_exit(TERMINATE_GRACE_S):
            self._signal_group(signal.SIGKILL)
            self._wait_for_exit(TERMINATE_GRACE_S)

    def close(self) -> None:
        """Shutdown, escalate, sweep the group, reap. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._broken and self._proc.poll() is None:
                with contextlib.suppress(WorkerProtocolError):
                    self._send({"op": "shutdown", "protocol": PROTOCOL_VERSION})
            self._close_stdin()
            if not self._wait_for_exit(SHUTDOWN_GRACE_S):
                self._signal_group(signal.SIGTERM)
                if not self._wait_for_exit(TERMINATE_GRACE_S):
                    self._signal_group(signal.SIGKILL)
                    self._wait_for_exit(TERMINATE_GRACE_S)
        finally:
            # Sweep before reaping: the leader is still a zombie here, so its pid — and
            # therefore the group id — cannot yet have been handed to anyone else. Any
            # descendant it leaked is still in that group, and this is what ends it.
            self._signal_group(signal.SIGKILL)
            # SIGKILL has already been sent, so this wait is a reap, not a hope.
            with contextlib.suppress(TimeoutExpired):
                self._proc.wait(timeout=TERMINATE_GRACE_S)
            self._reader.join(TERMINATE_GRACE_S)
            self._close_stdin()
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._stderr_log.close()

    def _close_stdin(self) -> None:
        if self._proc.stdin is None or self._proc.stdin.closed:
            return
        with contextlib.suppress(BrokenPipeError, OSError):
            self._proc.stdin.close()


# ------------------------------------------------------------------------ cost meter


def _directory_bytes(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


@dataclass(frozen=True)
class _CostInputs:
    """What the snapshots around a job said, before it becomes a `CostRecord`."""

    wall_s: float
    cpu_s: float
    peak_rss_bytes: int
    output_bytes: int
    notes: tuple[str, ...]

    def with_output_bytes(self, value: int) -> _CostInputs:
        return _CostInputs(
            wall_s=self.wall_s,
            cpu_s=self.cpu_s,
            peak_rss_bytes=self.peak_rss_bytes,
            output_bytes=value,
            notes=self.notes,
        )

    def to_record(self, *, attempt: int) -> CostRecord:
        return CostRecord(
            wall_s=self.wall_s,
            cpu_s=self.cpu_s,
            peak_rss_bytes=self.peak_rss_bytes,
            output_bytes=self.output_bytes,
            # No physics ran in this build, so there are no solver counters to report.
            # Task 5 fills these from the adapter's own statistics.
            accepted_steps=0,
            cut_steps=0,
            nonlinear_iterations=0,
            retry_count=attempt - 1,
            measurement_method="; ".join(self.notes),
        )


class _JobMeter:
    """Accumulates what the probe saw while one job ran.

    `CostRecord` has no nullable field, so an unreadable process tree cannot be recorded
    as `None` here. It is recorded as 0 *and named in `measurement_method`*, which is the
    difference between a measurement and an assumption: nothing downstream can mistake the
    zero for an observation that the job used no memory.
    """

    def __init__(self, baseline: ResourceSnapshot) -> None:
        self._baseline = baseline
        self.started_monotonic_s = baseline.monotonic_s
        self._peak_rss: int | None = baseline.process_rss_bytes
        self._unreadable: str | None = (
            None if baseline.process_rss_bytes is not None else baseline.process_measurement_method
        )

    def observe(self, snapshot: ResourceSnapshot) -> None:
        if snapshot.process_rss_bytes is None:
            self._unreadable = snapshot.process_measurement_method
            return
        self._peak_rss = (
            snapshot.process_rss_bytes
            if self._peak_rss is None
            else max(self._peak_rss, snapshot.process_rss_bytes)
        )

    def finish(self, final: ResourceSnapshot) -> _CostInputs:
        self.observe(final)
        notes = ["wall_s from the monotonic clock carried in the resource snapshots"]
        if self._baseline.process_cpu_s is None or final.process_cpu_s is None:
            cpu_s = 0.0
            notes.append(
                f"cpu_s could not be measured ({final.process_measurement_method}), recorded as 0"
            )
        else:
            cpu_s = max(0.0, final.process_cpu_s - self._baseline.process_cpu_s)
            notes.append("cpu_s from the process-tree CPU time difference")
        if self._peak_rss is None:
            peak = 0
            notes.append(
                f"peak_rss_bytes could not be measured ({self._unreadable}), recorded as 0"
            )
        else:
            peak = self._peak_rss
            if self._unreadable is not None:
                notes.append(
                    f"peak_rss_bytes is a partial maximum: at least one sample was "
                    f"unreadable ({self._unreadable})"
                )
            else:
                notes.append("peak_rss_bytes is the maximum process-tree RSS sampled")
        return _CostInputs(
            wall_s=max(0.0, final.monotonic_s - self._baseline.monotonic_s),
            cpu_s=cpu_s,
            peak_rss_bytes=peak,
            output_bytes=0,
            notes=tuple(notes),
        )
