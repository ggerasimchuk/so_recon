"""Transport tests for the persistent Julia worker.

Everything here is about the *transport*: framing, deadlines, process lifetime and the
classification of a job that produced no reply. None of it is about physics. The fake
executable below exists only so that a hung process, a malformed frame, a stale reply, a
crash, a flooded stderr and an over-long line can be described exactly instead of being
waited for. It is never a stand-in for a solver, and no test here lets it report a
COMPLETE physics result — the worker refuses one outright while no adapter is wired.

Two injections keep every wait bounded and deterministic:

* `clock` is the monotonic clock the startup and shutdown deadlines are measured on. A
  test hands the worker a clock that advances in large steps, so a budget of 300 s is
  exhausted after one poll interval of real time. The test that asserts the constructor
  does not wait forever therefore also asserts, in real seconds, that it did not.
* `probe` is the resource-snapshot source. `budget.ResourceWatchdog` owns the job timeout
  and every memory rule and reads nothing but snapshots, so "this job has run past its
  timeout" is a snapshot whose `monotonic_s` says so — no sleeping, and the policy under
  test is the real one rather than a copy of it.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

import psutil
import pytest

from so_recon.config.resources import (
    P0_VERIFY_PROFILE,
    ResourceProfile,
    UnapprovedResourceProfileError,
)
from so_recon.environment.resources import MemoryPressure, ResourceSnapshot
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.budget import BudgetLedger, BudgetStop, ResourceWatchdog
from so_recon.simulator.contracts import ForwardStatus, JobDescriptor, OutputRequest
from so_recon.simulator.worker import (
    MAX_PROTOCOL_LINE_CHARS,
    PROTOCOL_VERSION,
    STOP_FILE_NAME,
    WORKER_STATUSES,
    PersistentJuliaWorker,
    WorkerStartupError,
    classify_termination,
    request_stop_after_chunk,
    stop_after_chunk_requested,
    validate_reply,
)

GIB = 1024**3

# A fake worker, driven entirely by the scenario JSON named in $FAKE_WORKER_SCENARIO.
# It ignores argv, because the worker launches it exactly the way it launches julia:
# `<exe> --project=... --startup-file=no --color=no <script>`.
FAKE_WORKER = r"""
import json
import os
import subprocess
import sys
import time

scenario = json.loads(os.environ["FAKE_WORKER_SCENARIO"])


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


spawn = scenario.get("spawn_grandchild")
if spawn:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(spawn, "w") as fh:
        fh.write(str(child.pid))
        fh.flush()
        os.fsync(fh.fileno())

if scenario.get("ready", True):
    emit({
        "event": "ready",
        "protocol": scenario.get("protocol", "worker-1"),
        "pid": os.getpid(),
        "julia_threads": int(os.environ.get("JULIA_NUM_THREADS", "-1")),
        "blas_threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "-1")),
        "versions": {"julia": "fake-0.0"},
    })

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        request = json.loads(line)
    except ValueError:
        continue
    op = request.get("op")
    if op == "shutdown":
        if scenario.get("exit_on_shutdown", True):
            sys.exit(int(scenario.get("shutdown_exit_code", 0)))
        while True:
            time.sleep(3600)
    queued = scenario.get("behaviours")
    if queued:
        behaviour = queued.pop(0) if len(queued) > 1 else queued[0]
    else:
        behaviour = scenario.get("behaviour", "reply")
    if behaviour == "silent":
        continue
    if behaviour == "crash":
        os._exit(int(scenario.get("crash_exit_code", 3)))
    if behaviour == "kill9":
        os.kill(os.getpid(), 9)
    if behaviour == "malformed":
        sys.stdout.write("this line is not json at all\n")
        sys.stdout.flush()
        continue
    if behaviour == "oversized":
        sys.stdout.write("{" + "p" * int(scenario["oversize_chars"]))
        sys.stdout.flush()
        continue
    if behaviour == "flood_stderr":
        block = "julia native log line, not protocol\n" * 1024
        for _ in range(int(scenario.get("flood_blocks", 64))):
            sys.stderr.write(block)
        sys.stderr.flush()
    reply = dict(scenario.get("reply", {}))
    if scenario.get("echo_job_id", True):
        reply["job_id"] = request.get("job_id")
    if behaviour == "big_reply":
        reply["padding"] = "x" * int(scenario["padding_chars"])
    if op == "ping":
        reply.setdefault("status", "COMPLETE")
        reply.setdefault("protocol", "worker-1")
        reply.setdefault("pid", os.getpid())
        reply.setdefault("inputs", {n: "f" * 64 for n in request.get("inputs", {})})
    emit(reply)
"""


def snapshot(
    *,
    monotonic_s: float = 100.0,
    rss: int | None = 2 * GIB,
    cpu_s: float | None = 1.0,
    available: int = 20 * GIB,
    total: int = 32 * GIB,
    swap: int = 0,
    disk_free: int = 100 * GIB,
) -> ResourceSnapshot:
    """A healthy machine, unless a test says otherwise."""
    return ResourceSnapshot(
        total_bytes=total,
        available_bytes=available,
        process_rss_bytes=rss,
        process_cpu_s=cpu_s,
        process_measurement_method="test fixture",
        swap_used_bytes=swap,
        pressure=MemoryPressure(status="normal", raw="1", method="test fixture"),
        disk_free_bytes=disk_free,
        monotonic_s=monotonic_s,
    )


def stepping_clock(step_s: float) -> Callable[[], float]:
    """A monotonic clock that advances by `step_s` every time it is read."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += step_s
        return state["t"]

    return clock


class ScriptedProbe:
    """Hands out a fixed series of snapshots, repeating the last one forever."""

    def __init__(self, *snapshots: ResourceSnapshot) -> None:
        self._snapshots = list(snapshots)
        self.calls = 0

    def __call__(self) -> ResourceSnapshot:
        self.calls += 1
        return self._snapshots[min(self.calls - 1, len(self._snapshots) - 1)]


# ------------------------------------------------------------------ shared fixtures


@pytest.fixture
def project(tmp_project: Path) -> ProjectPaths:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    (tmp_project / "julia" / "worker").mkdir(parents=True, exist_ok=True)
    (tmp_project / "julia" / "worker" / "main.jl").write_text("# stub\n", encoding="utf-8")
    return paths


@pytest.fixture
def fake_executable(tmp_path: Path) -> Path:
    """A `sh` wrapper that drops every argument and runs the fake worker."""
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    exe = tmp_path / "fake-julia"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n', encoding="utf-8")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def scenario(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def configure(**fields: Any) -> None:
        monkeypatch.setenv("FAKE_WORKER_SCENARIO", json.dumps(fields))

    configure()
    return configure


def make_ledger(paths: ProjectPaths) -> BudgetLedger:
    return BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=paths.artifacts / "ledger.json",
        session_id="session-transport",
        probe=ScriptedProbe(snapshot()),
    )


def write_job_inputs(paths: ProjectPaths, job_id: str = "job-0001") -> JobDescriptor:
    """A descriptor over real files, so every digest in it is the digest of real bytes."""
    case_path = paths.artifacts / "cases" / f"{job_id}-case.json"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps({"schema_version": "case-1"}), encoding="utf-8")
    solver_path = paths.artifacts / "cases" / f"{job_id}-solver.json"
    solver_path.write_text(json.dumps({"tolerance": 1e-6}), encoding="utf-8")
    return JobDescriptor(
        job_id=job_id,
        case_path=paths.relative(case_path),
        case_sha256=sha256_file(case_path),
        model_hash="a" * 64,
        solver_config_path=paths.relative(solver_path),
        solver_config_sha256=sha256_file(solver_path),
        output_request=OutputRequest(state_times_s=(86400.0,), keep_native_restart=False),
        seed=7,
        result_dir=f"artifacts/results/{job_id}",
        attempt=1,
    )


def start_worker(
    paths: ProjectPaths,
    executable: Path,
    *,
    probe: Callable[[], ResourceSnapshot] | None = None,
    clock: Callable[[], float] | None = None,
    profile: ResourceProfile = P0_VERIFY_PROFILE,
) -> PersistentJuliaWorker:
    return PersistentJuliaWorker(
        executable,
        paths.julia,
        paths.artifacts / "session",
        profile,
        paths=paths,
        probe=probe if probe is not None else ScriptedProbe(snapshot()),
        clock=clock if clock is not None else time.monotonic,
    )


@pytest.fixture
def worker_factory(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> Iterator[Callable[..., PersistentJuliaWorker]]:
    started: list[PersistentJuliaWorker] = []

    def factory(**kwargs: Any) -> PersistentJuliaWorker:
        worker = start_worker(project, fake_executable, **kwargs)
        started.append(worker)
        return worker

    yield factory
    for worker in started:
        worker.close()


# ------------------------------------------------------------------- the reply kernel


def test_stale_reply_is_not_accepted() -> None:
    with pytest.raises(ValueError, match="job_id"):
        validate_reply({"job_id": "previous", "status": "COMPLETE"}, "current")


def test_every_contract_status_is_accepted_and_nothing_else() -> None:
    # The worker's accepted set IS the contract's set: a status no record can hold must
    # not survive the transport either.
    assert frozenset(get_args(ForwardStatus)) == WORKER_STATUSES
    for status in WORKER_STATUSES:
        validate_reply({"job_id": "j", "status": status}, "j")


@pytest.mark.parametrize("status", ["ok", "complete", "", None, "DONE"])
def test_an_unknown_status_is_refused(status: object) -> None:
    with pytest.raises(ValueError, match="status"):
        validate_reply({"job_id": "j", "status": status}, "j")


def test_a_reply_with_no_job_id_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="job_id"):
        validate_reply({"status": "COMPLETE"}, "j")


# ---------------------------------------------------------- termination classification


def test_a_force_kill_alone_is_not_an_out_of_memory_diagnosis() -> None:
    status, reason = classify_termination(returncode=-9, decision=None)
    assert status == "PROTOCOL_FAILURE"
    assert "signal 9" in reason and "SIGKILL" in reason
    assert "memory" not in reason.lower() or "does not establish" in reason


def test_a_guard_breach_is_what_makes_a_death_a_resource_failure() -> None:
    watchdog = ResourceWatchdog(P0_VERIFY_PROFILE, baseline=snapshot())
    # A process tree at the hard cap: a real breach, found by the real guard.
    decision = watchdog.observe(snapshot(rss=31 * GIB, available=0))
    assert decision.status == "RESOURCE_FAILURE"
    status, reason = classify_termination(returncode=-9, decision=decision)
    assert status == "RESOURCE_FAILURE"
    assert "hard cap" in reason


def test_a_job_past_its_timeout_is_classified_as_a_timeout() -> None:
    watchdog = ResourceWatchdog(P0_VERIFY_PROFILE, baseline=snapshot(monotonic_s=100.0))
    decision = watchdog.observe(
        snapshot(monotonic_s=100.0 + P0_VERIFY_PROFILE.job_timeout_s + 1.0),
        job_started_monotonic_s=100.0,
    )
    assert decision.terminate_process_group
    status, reason = classify_termination(returncode=-15, decision=decision)
    assert status == "TIMEOUT"
    assert "timeout" in reason


def test_a_plain_nonzero_exit_names_its_code() -> None:
    status, reason = classify_termination(returncode=3, decision=None)
    assert status == "PROTOCOL_FAILURE"
    assert "3" in reason


def test_a_signal_is_named_by_the_operating_system_not_by_a_hardcoded_number() -> None:
    status, reason = classify_termination(returncode=-int(signal.SIGTERM), decision=None)
    assert status == "PROTOCOL_FAILURE"
    assert signal.Signals(signal.SIGTERM).name in reason


# ------------------------------------------------------------------ stop control plane


def test_the_stop_file_is_a_control_plane_request_that_touches_no_job_input(
    project: ProjectPaths,
) -> None:
    job = write_job_inputs(project)
    names = (job.case_path, job.solver_config_path)
    before = {name: sha256_file(project.resolve(name)) for name in names}
    session_dir = project.artifacts / "session"
    assert stop_after_chunk_requested(session_dir) is False

    stop_file = request_stop_after_chunk(session_dir)

    assert stop_file.name == STOP_FILE_NAME
    assert stop_after_chunk_requested(session_dir) is True
    assert {name: sha256_file(project.resolve(name)) for name in names} == before
    # It is a request, not a mutation of anything the job runs on.
    assert json.loads(stop_file.read_text(encoding="utf-8"))["request"] == STOP_FILE_NAME


# ------------------------------------------------------------------ process lifecycle


def test_the_worker_refuses_a_relabelled_resource_profile(
    project: ProjectPaths, fake_executable: Path
) -> None:
    relabelled = P0_VERIFY_PROFILE.model_copy(update={"max_new_forward": 100000})
    with pytest.raises(UnapprovedResourceProfileError, match="P0_VERIFY"):
        start_worker(project, fake_executable, profile=relabelled)


def test_a_missing_worker_script_is_refused_before_anything_is_launched(
    project: ProjectPaths, fake_executable: Path
) -> None:
    (project.julia / "worker" / "main.jl").unlink()
    with pytest.raises(FileNotFoundError, match="main.jl"):
        start_worker(project, fake_executable)


def test_the_constructor_does_not_wait_forever_for_ready(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> None:
    scenario(ready=False)
    assert P0_VERIFY_PROFILE.startup_timeout_s == 300  # what it would otherwise wait
    started = time.monotonic()
    with pytest.raises(WorkerStartupError, match="READY"):
        start_worker(project, fake_executable, clock=stepping_clock(120.0))
    elapsed = time.monotonic() - started
    # The deadline is on the injected monotonic clock, so 300 s of budget cost one poll.
    assert elapsed < 30.0, elapsed


def test_the_handshake_reports_the_threads_the_profile_asked_for(
    worker_factory: Callable[..., PersistentJuliaWorker],
) -> None:
    worker = worker_factory()
    assert worker.handshake.protocol == PROTOCOL_VERSION
    assert worker.handshake.pid == worker.pid
    # Global constraint: 4 Julia threads, 1 BLAS thread, both taken from the profile.
    assert worker.handshake.julia_threads == P0_VERIFY_PROFILE.julia_threads == 4
    assert worker.handshake.blas_threads == P0_VERIFY_PROFILE.blas_threads == 1


def test_close_kills_descendants_the_worker_leaked(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> None:
    pid_file = project.artifacts / "grandchild.pid"
    scenario(spawn_grandchild=str(pid_file))
    worker = start_worker(project, fake_executable)
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    assert psutil.pid_exists(grandchild)

    worker.close()

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and psutil.pid_exists(grandchild):
        time.sleep(0.05)
    assert not psutil.pid_exists(grandchild), "a descendant outlived the worker's process group"


def test_close_is_idempotent_and_reaps_the_process(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> None:
    scenario()
    worker = start_worker(project, fake_executable)
    worker.close()
    worker.close()
    assert worker.returncode == 0


def test_the_context_manager_closes_even_when_the_body_raises(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> None:
    scenario()
    captured: list[int] = []
    with pytest.raises(RuntimeError, match="boom"), start_worker(project, fake_executable) as w:
        captured.append(w.pid)
        raise RuntimeError("boom")
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and psutil.pid_exists(captured[0]):
        time.sleep(0.05)
    assert not psutil.pid_exists(captured[0])


# --------------------------------------------------------------------- job transport


def test_a_stale_reply_for_another_job_is_a_protocol_failure(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(echo_job_id=False, reply={"job_id": "previous", "status": "INVALID_INPUT"})
    worker = worker_factory()
    ledger = make_ledger(project)

    result = worker.submit(write_job_inputs(project), ledger)

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "job_id" in result.reason
    assert ledger.record.entries[0].state == "FAILED"
    assert ledger.record.entries[0].status == "PROTOCOL_FAILURE"


def test_a_malformed_frame_is_a_protocol_failure(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="malformed")
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "not JSON" in result.reason


def test_a_worker_that_crashes_before_replying_reports_its_exit_code(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="crash", crash_exit_code=7)
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "7" in result.reason


def test_a_force_killed_worker_is_not_reported_as_out_of_memory(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="kill9")
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    # No guard breach and no OS evidence: the signal alone proves nothing about memory.
    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "signal 9" in result.reason


def test_a_flooded_stderr_never_blocks_the_reply(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(
        behaviour="flood_stderr",
        flood_blocks=64,
        reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"},
    )
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "INVALID_INPUT"
    log = project.artifacts / "session" / "worker.stderr.log"
    # Two megabytes of native logging landed in a file, not in a pipe nobody drained.
    assert log.stat().st_size > 2 * 1024 * 1024


def test_a_reply_larger_than_the_pipe_buffer_is_read_without_deadlock(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(
        behaviour="big_reply",
        padding_chars=512 * 1024,  # far beyond any platform pipe buffer
        reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"},
    )
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "INVALID_INPUT"


def test_an_unterminated_line_is_refused_rather_than_read_without_bound(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="oversized", oversize_chars=MAX_PROTOCOL_LINE_CHARS + 4096)
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and str(MAX_PROTOCOL_LINE_CHARS) in result.reason


def test_a_result_whose_bytes_do_not_match_the_reported_digest_is_refused(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    job = write_job_inputs(project)
    result_file = project.root / job.result_dir / "result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text('{"status": "INVALID_INPUT"}', encoding="utf-8")
    scenario(
        reply={
            "status": "INVALID_INPUT",
            "reason": "adapter unavailable",
            "result_path": f"{job.result_dir}/result.json",
            "result_sha256": "b" * 64,
        }
    )
    worker = worker_factory()

    result = worker.submit(job, make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "sha256" in result.reason


def test_a_result_filed_outside_the_job_s_own_directory_is_refused(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    job = write_job_inputs(project)
    elsewhere = project.artifacts / "results" / "job-somebody-else" / "result.json"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text('{"status": "INVALID_INPUT"}', encoding="utf-8")
    scenario(
        reply={
            "status": "INVALID_INPUT",
            "reason": "adapter unavailable",
            "result_path": project.relative(elsewhere),
            "result_sha256": sha256_file(elsewhere),
        }
    )
    worker = worker_factory()

    result = worker.submit(job, make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and job.result_dir in result.reason


def test_a_verified_failure_record_is_carried_into_the_result_and_the_ledger(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    job = write_job_inputs(project)
    result_file = project.root / job.result_dir / "result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text('{"status": "INVALID_INPUT"}', encoding="utf-8")
    scenario(
        reply={
            "status": "INVALID_INPUT",
            "reason": "adapter unavailable",
            "physics_class": "OW",
            "result_path": f"{job.result_dir}/result.json",
            "result_sha256": sha256_file(result_file),
        }
    )
    worker = worker_factory(
        probe=ScriptedProbe(
            snapshot(monotonic_s=100.0, cpu_s=1.0, rss=2 * GIB),
            snapshot(monotonic_s=103.0, cpu_s=2.5, rss=3 * GIB),
        )
    )
    ledger = make_ledger(project)

    result = worker.submit(job, ledger)

    assert (result.status, result.physics_class) == ("INVALID_INPUT", "OW")
    assert result.reason == "adapter unavailable"
    assert result.job_id == job.job_id
    assert result.case_sha256 == job.case_sha256 and result.model_hash == job.model_hash
    # Job identity binds the solver lock and the environment lock (plan 3.1, at job level).
    assert result.solver_metadata["solver_config_sha256"] == job.solver_config_sha256
    assert len(result.solver_metadata["environment_lock_hash"]) == 64
    assert result.solver_metadata["result_sha256"] == sha256_file(result_file)
    assert result.cost.wall_s == pytest.approx(3.0)
    assert result.cost.cpu_s == pytest.approx(1.5)
    assert result.cost.peak_rss_bytes == 3 * GIB
    assert result.cost.output_bytes == result_file.stat().st_size
    assert ledger.record.entries[0].state == "FAILED"


def test_an_unmeasurable_process_tree_is_named_rather_than_recorded_as_zero(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"})
    unreadable = ResourceSnapshot(
        total_bytes=32 * GIB,
        available_bytes=20 * GIB,
        process_rss_bytes=None,
        process_cpu_s=None,
        process_measurement_method="process tree unreadable in this test",
        swap_used_bytes=0,
        pressure=MemoryPressure(status="normal", raw="1", method="test fixture"),
        disk_free_bytes=100 * GIB,
        monotonic_s=5.0,
    )
    worker = worker_factory(probe=ScriptedProbe(unreadable))

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.cost.peak_rss_bytes == 0
    assert result.cost.cpu_s == 0.0
    assert "unreadable in this test" in result.cost.measurement_method
    assert "recorded as 0" in result.cost.measurement_method


def test_a_complete_reply_is_refused_while_no_solver_adapter_is_wired(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(reply={"status": "COMPLETE"})
    worker = worker_factory()

    result = worker.submit(write_job_inputs(project), make_ledger(project))

    assert result.status == "PROTOCOL_FAILURE"
    assert result.reason is not None and "COMPLETE" in result.reason


def test_a_job_past_its_timeout_is_terminated_and_recorded(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="silent", exit_on_shutdown=False)
    over = float(P0_VERIFY_PROFILE.job_timeout_s) + 1.0
    worker = worker_factory(
        probe=ScriptedProbe(snapshot(monotonic_s=10.0), snapshot(monotonic_s=10.0 + over))
    )
    pid = worker.pid
    ledger = make_ledger(project)

    result = worker.submit(write_job_inputs(project), ledger)

    assert result.status == "TIMEOUT"
    assert result.reason is not None and "timeout" in result.reason
    assert ledger.record.entries[0].state == "FAILED"
    assert ledger.record.entries[0].status == "TIMEOUT"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and psutil.pid_exists(pid):
        time.sleep(0.05)
    assert not psutil.pid_exists(pid), "the hung job's process group survived the watchdog"


def test_a_resource_breach_while_a_job_runs_is_recorded_as_a_resource_failure(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(behaviour="silent", exit_on_shutdown=False)
    # A breach, then growth despite the drain request: budget.py escalates to terminate.
    worker = worker_factory(
        probe=ScriptedProbe(
            snapshot(monotonic_s=10.0),
            snapshot(monotonic_s=11.0, rss=31 * GIB, available=0),
            snapshot(monotonic_s=12.0, rss=31 * GIB + 1, available=0),
        )
    )
    ledger = make_ledger(project)

    result = worker.submit(write_job_inputs(project), ledger)

    assert result.status == "RESOURCE_FAILURE"
    assert result.reason is not None and "hard cap" in result.reason
    assert ledger.record.entries[0].state == "FAILED"
    assert ledger.record.entries[0].status == "RESOURCE_FAILURE"


def test_a_disk_that_fills_while_a_job_runs_is_terminated_and_recorded(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    """Step 4.6's disk case, on the axis that only exists once a job is running.

    The pre-flight check (`disk_stop` in `BudgetLedger.reserve`) asks whether a job can
    fit. This is the other half: the disk fell into the room reserved for the failure
    record *while* the job was running, and it is that record which must still get
    written. So the job is terminated, classified and recorded, and the earlier attempt
    on the same ledger survives untouched.
    """
    scenario(
        behaviours=["reply", "silent"],
        exit_on_shutdown=False,
        reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"},
    )
    worker = worker_factory(
        probe=ScriptedProbe(
            snapshot(monotonic_s=10.0),  # job A baseline
            snapshot(monotonic_s=11.0),  # job A final
            snapshot(monotonic_s=12.0),  # job B baseline
            snapshot(monotonic_s=13.0, disk_free=1024),  # job B: the reserve is gone
        )
    )
    ledger = make_ledger(project)
    finished = worker.submit(write_job_inputs(project, "job-a"), ledger)
    assert finished.status == "INVALID_INPUT"
    pid = worker.pid

    result = worker.submit(write_job_inputs(project, "job-b"), ledger)

    # 1. A classified result, not an unexplained empty one (SPEC 18.4).
    assert result.status == "RESOURCE_FAILURE"
    assert result.reason is not None and "free disk" in result.reason
    assert "failure record" in result.reason
    # 2. A FAILED run on the ledger.
    assert ledger.record.entries[1].job_id == "job-b"
    assert ledger.record.entries[1].state == "FAILED"
    assert ledger.record.entries[1].status == "RESOURCE_FAILURE"
    # 3. The earlier attempt's record is preserved, cost and all. Nothing can be COMPLETE
    #    while no adapter is wired, so what a stop must not lose here is the finished
    #    attempt itself; its checkpoint ref becomes possible in Task 8.
    assert ledger.record.entries[0].job_id == "job-a"
    assert ledger.record.entries[0].status == "INVALID_INPUT"
    assert ledger.record.entries[0].cost is not None
    assert ledger.cumulative_totals().forwards == 2
    # The process group really was ended, not merely decided about.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and psutil.pid_exists(pid):
        time.sleep(0.05)
    assert not psutil.pid_exists(pid), "the job's process group survived the disk guard"


def test_a_job_that_cannot_fit_on_disk_is_refused_before_it_starts(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"})
    worker = worker_factory()
    # A disk with less free space than this job's predicted output plus the reserve the
    # failure record and the ledger itself need.
    ledger = BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=project.artifacts / "ledger.json",
        session_id="session-full-disk",
        probe=ScriptedProbe(snapshot(disk_free=1024)),
    )

    with pytest.raises(BudgetStop) as raised:
        worker.submit(write_job_inputs(project), ledger)

    assert raised.value.status == "RESOURCE_FAILURE"
    assert "free disk" in raised.value.reason
    # Nothing started, so nothing is owed: no reservation was left behind.
    assert ledger.record.entries == ()


def test_a_finished_job_stays_on_the_ledger_after_a_stop_is_requested(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"})
    worker = worker_factory()
    ledger = make_ledger(project)
    assert worker.submit(write_job_inputs(project, "job-a"), ledger).status == "INVALID_INPUT"

    request_stop_after_chunk(project.artifacts / "session")
    assert worker.stop_requested is True
    ledger.stop("operator requested stop-after-chunk")

    assert ledger.record.entries[0].job_id == "job-a"
    assert ledger.record.entries[0].status == "INVALID_INPUT"
    assert ledger.cumulative_totals().forwards == 1


def test_the_worker_writes_the_descriptor_it_hands_to_julia(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario(reply={"status": "INVALID_INPUT", "reason": "adapter unavailable"})
    worker = worker_factory()
    job = write_job_inputs(project)

    worker.submit(job, make_ledger(project))

    written = project.artifacts / "session" / "jobs" / f"{job.job_id}.json"
    assert JobDescriptor.model_validate_json(written.read_text(encoding="utf-8")) == job


def test_a_ping_echoes_the_inputs_it_was_given(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario()
    worker = worker_factory()
    job = write_job_inputs(project)

    reply = worker.ping("ping-1", {"case": project.resolve(job.case_path)})

    assert reply.job_id == "ping-1"
    assert reply.pid == worker.pid
    assert set(reply.inputs) == {"case"}


def test_submitting_after_close_is_refused(
    project: ProjectPaths, fake_executable: Path, scenario: Callable[..., None]
) -> None:
    scenario()
    worker = start_worker(project, fake_executable)
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(write_job_inputs(project), make_ledger(project))


def test_a_job_id_that_is_not_a_safe_file_name_is_refused_before_it_is_reserved(
    project: ProjectPaths,
    worker_factory: Callable[..., PersistentJuliaWorker],
    scenario: Callable[..., None],
) -> None:
    scenario()
    worker = worker_factory()
    ledger = make_ledger(project)
    job = write_job_inputs(project).model_copy(update={"job_id": "../escape"})

    with pytest.raises(ValueError, match="job_id"):
        worker.submit(job, ledger)

    assert ledger.record.entries == ()
