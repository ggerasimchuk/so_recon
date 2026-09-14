"""E01.8 — the policy half of the forward driver, with no Julia anywhere near it.

Everything here is a decision rather than a computation: whether a failed attempt may be
tried again, what the one permitted retry is allowed to change, what a session predicts a
job will cost before it books it, what a `COMPLETE` result has to carry to be the result the
caller ASKED for, and what a worker whose memory kept growing is answered with.

They are tested apart from the solver on purpose. A retry policy that only ever runs behind
a real simulation is a policy nobody can show the boundary of: the interesting cases — the
second retry, the status that is never retried, the drift that repeats after a recycle — are
exactly the ones a healthy run never reaches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.budget import BudgetLedger, BudgetStop
from so_recon.simulator.contracts import (
    MAX_ATTEMPTS,
    SECONDS_PER_DAY,
    ArrayRef,
    CostRecord,
    ForwardResult,
    ForwardStatus,
    JobDescriptor,
    OutputRequest,
    RestartRef,
)
from so_recon.simulator.forward import (
    BASE_MAX_NONLINEAR_ITERATIONS,
    MEMORY_DRIFT_FLOOR_BYTES,
    MEMORY_DRIFT_FRACTION,
    RETRY_MAX_NONLINEAR_ITERATIONS,
    ForwardRequestError,
    MemoryDriftMonitor,
    SolverConfig,
    check_request_is_deliverable,
    check_requested_outputs,
    memory_drift_limit_bytes,
    predicted_output_bytes,
    predicted_wall_s,
    read_solver_config,
    retry_descriptor,
    should_retry,
    write_solver_config,
)

DAY = SECONDS_PER_DAY
GIB = 1024**3


# --------------------------------------------------------------------------- 8.1 kernel


@pytest.mark.parametrize(
    "status,attempt,expected",
    [
        ("NUMERICAL_FAILURE", 0, True),
        ("NUMERICAL_FAILURE", 1, False),
        ("RESOURCE_FAILURE", 0, False),
        ("CONTROL_INFEASIBLE", 0, False),
        ("TIMEOUT", 0, False),
        ("INVALID_INPUT", 0, False),
    ],
)
def test_only_one_numerical_retry(status: str, attempt: int, expected: bool) -> None:
    assert should_retry(status, attempt) is expected


def test_no_other_status_is_ever_retried() -> None:
    """Every status the contract allows, checked — not only the six the plan spells out."""
    for status in (
        "COMPLETE",
        "INVALID_INPUT",
        "PHYSICALLY_INVALID",
        "CONTROL_INFEASIBLE",
        "RESOURCE_FAILURE",
        "TIMEOUT",
        "PROTOCOL_FAILURE",
        "INCOMPLETE_BUDGET",
    ):
        assert should_retry(status, 0) is False, status
    assert should_retry("NUMERICAL_FAILURE", 0) is True


def test_a_negative_attempt_is_refused_rather_than_treated_as_the_first() -> None:
    with pytest.raises(ValueError, match="attempt"):
        should_retry("NUMERICAL_FAILURE", -1)


# ------------------------------------------------------------------ the retry descriptor


def _paths(tmp_project: Path) -> ProjectPaths:
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    return paths


def _solver(paths: ProjectPaths, **overrides: object) -> tuple[str, str]:
    payload = {"max_timestep_days": 10.0, "max_nonlinear_iterations": BASE_MAX_NONLINEAR_ITERATIONS}
    payload.update(overrides)
    path = paths.artifacts / "solver" / "e01_solver.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return paths.relative(path), sha256_file(path)


def _descriptor(paths: ProjectPaths, **overrides: object) -> JobDescriptor:
    case_path = paths.artifacts / "case.json"
    case_path.write_text("{}", encoding="utf-8")
    solver_path, solver_sha = _solver(paths)
    fields: dict[str, object] = {
        "job_id": "job-e01-0001",
        "case_path": paths.relative(case_path),
        "case_sha256": "a" * 64,
        "model_hash": "b" * 64,
        "solver_config_path": solver_path,
        "solver_config_sha256": solver_sha,
        "output_request": OutputRequest(state_times_s=(0.0, 30.0 * DAY), keep_native_restart=True),
        "seed": 20260913,
        "result_dir": "artifacts/results/job-e01-0001",
        "attempt": 1,
    }
    fields.update(overrides)
    return JobDescriptor(**fields)  # type: ignore[arg-type]


def test_the_retry_keeps_the_physics_and_changes_only_the_solver(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    parent = _descriptor(paths)
    retry = retry_descriptor(parent, paths=paths)

    # The physical model is the SAME model: the original hash, the original case bytes and
    # the original requested outputs. A retry that re-rendered the case would be a different
    # forward wearing the first one's name.
    assert retry.model_hash == parent.model_hash
    assert retry.case_sha256 == parent.case_sha256
    assert retry.case_path == parent.case_path
    assert retry.output_request == parent.output_request
    assert retry.seed == parent.seed
    # A registered second attempt, with the attempt it descends from named on the record.
    assert retry.attempt == 2 == MAX_ATTEMPTS
    assert retry.parent_job_id == parent.job_id
    assert retry.job_id != parent.job_id
    assert retry.result_dir != parent.result_dir
    # Its own solver configuration, with its own digest.
    assert retry.solver_config_path != parent.solver_config_path
    assert retry.solver_config_sha256 != parent.solver_config_sha256
    assert retry.solver_config_sha256 == sha256_file(paths.resolve(retry.solver_config_path))

    before = read_solver_config(paths.resolve(parent.solver_config_path))
    after = read_solver_config(paths.resolve(retry.solver_config_path))
    assert after.max_timestep_days == before.max_timestep_days / 2
    assert before.max_nonlinear_iterations == BASE_MAX_NONLINEAR_ITERATIONS
    assert after.max_nonlinear_iterations == RETRY_MAX_NONLINEAR_ITERATIONS == 25


def test_the_retry_configuration_carries_nothing_but_the_two_numbers_it_moves(
    tmp_project: Path,
) -> None:
    """A tolerance cannot be relaxed by a retry, because the record has nowhere to put one."""
    paths = _paths(tmp_project)
    retry = retry_descriptor(_descriptor(paths), paths=paths)
    written = json.loads(paths.resolve(retry.solver_config_path).read_text(encoding="utf-8"))
    assert sorted(written) == ["max_nonlinear_iterations", "max_timestep_days"]
    with pytest.raises(ValueError):
        SolverConfig.model_validate({**written, "tol_cnv": 1e-1})


def test_a_solver_configuration_that_moved_under_the_descriptor_is_refused(
    tmp_project: Path,
) -> None:
    paths = _paths(tmp_project)
    parent = _descriptor(paths)
    paths.resolve(parent.solver_config_path).write_text(
        json.dumps({"max_timestep_days": 1.0, "max_nonlinear_iterations": 15}), encoding="utf-8"
    )
    with pytest.raises(ForwardRequestError, match="sha256"):
        retry_descriptor(parent, paths=paths)


def test_a_retry_of_a_retry_is_refused(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    retry = retry_descriptor(_descriptor(paths), paths=paths)
    with pytest.raises(ForwardRequestError, match="attempt"):
        retry_descriptor(retry, paths=paths)


def test_a_solver_configuration_round_trips_through_its_own_writer(tmp_project: Path) -> None:
    paths = _paths(tmp_project)
    config = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)
    relative, digest = write_solver_config(config, paths.artifacts / "solver" / "x.json", paths)
    assert digest == sha256_file(paths.resolve(relative))
    assert read_solver_config(paths.resolve(relative)) == config


# ------------------------------------------------------------------- budget accounting


def _snapshot(tmp_path: Path) -> ResourceSnapshot:
    """A real measurement with the memory fields pinned, as the integration tests do."""
    return probe_resources(None, tmp_path).model_copy(
        update={
            "total_bytes": 64 * GIB,
            "available_bytes": 32 * GIB,
            "process_rss_bytes": GIB,
            "swap_used_bytes": 0,
        }
    )


def _ledger(tmp_project: Path, probe: object) -> BudgetLedger:
    return BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=tmp_project / "artifacts" / "ledger.json",
        session_id="session-e01-policy",
        probe=probe,  # type: ignore[arg-type]
    )


def _state_ref(result_dir: str, *, n_times: int) -> ArrayRef:
    return ArrayRef(
        path=f"{result_dir}/states.h5",
        dataset="pressure_pa",
        sha256="c" * 64,
        shape=(n_times, 8),
        dtype="float64",
        unit="Pa",
        axis_order=("time", "cell"),
    )


def _result(job: JobDescriptor, status: ForwardStatus, *, wall_s: float) -> ForwardResult:
    complete = status == "COMPLETE"
    published: dict[str, Any] = {
        "completed_time_s": 0.0,
        "times_s": (),
        "states": {},
    }
    if complete:
        published = {
            "completed_time_s": 30.0 * DAY,
            "times_s": (30.0 * DAY,),
            "states": {"pressure_pa": _state_ref(job.result_dir, n_times=1)},
            "monthly_path": f"{job.result_dir}/monthly.parquet",
            "connections_path": f"{job.result_dir}/connections.parquet",
            "balances_path": f"{job.result_dir}/balances.parquet",
        }
    return ForwardResult(
        job_id=job.job_id,
        case_sha256=job.case_sha256,
        model_hash=job.model_hash,
        physics_class="OW",
        status=status,
        reason=None if complete else f"the solver reported {status}",
        solver_metadata={},
        **published,
        cost=CostRecord(
            wall_s=wall_s,
            cpu_s=wall_s,
            peak_rss_bytes=GIB,
            output_bytes=1024,
            accepted_steps=1,
            cut_steps=0,
            nonlinear_iterations=1,
            retry_count=job.attempt - 1,
            measurement_method="test",
        ),
        parent_attempt_ids=() if job.parent_job_id is None else (job.parent_job_id,),
    )


def test_both_attempts_are_paid_for_and_the_first_result_is_not_overwritten(
    tmp_project: Path,
) -> None:
    paths = _paths(tmp_project)
    ledger = _ledger(tmp_project, lambda: _snapshot(tmp_project))
    parent = _descriptor(paths)
    retry = retry_descriptor(parent, paths=paths)

    ledger.reserve(
        parent.job_id, 5.0, 1024, case_sha256=parent.case_sha256, model_hash=parent.model_hash
    )
    ledger.complete(parent.job_id, _result(parent, "NUMERICAL_FAILURE", wall_s=4.0))
    ledger.reserve(
        retry.job_id,
        5.0,
        1024,
        case_sha256=retry.case_sha256,
        model_hash=retry.model_hash,
        attempt=retry.attempt,
    )
    ledger.complete(retry.job_id, _result(retry, "COMPLETE", wall_s=6.0))

    # SPEC 3.3: every attempt counts against the forward budget, including the failed one.
    assert ledger.cumulative_totals().forwards == 2
    assert ledger.cumulative_totals().wall_s == pytest.approx(10.0)
    first, second = ledger.record.entries
    assert (first.job_id, first.status, first.attempt) == (parent.job_id, "NUMERICAL_FAILURE", 1)
    assert (second.job_id, second.status, second.attempt) == (retry.job_id, "COMPLETE", 2)
    # The first attempt's own measured cost is still the first attempt's.
    assert first.cost is not None and first.cost.wall_s == 4.0

    # And a third attempt on the same physical model is refused by the ledger itself.
    with pytest.raises(BudgetStop, match="at most 2"):
        ledger.reserve(
            "job-e01-0003",
            5.0,
            1024,
            case_sha256=parent.case_sha256,
            model_hash=parent.model_hash,
            attempt=2,
        )


def test_the_guards_are_rechecked_before_the_retry_is_booked(tmp_project: Path) -> None:
    """A retry is new work: it passes the same resource guards the first attempt did."""
    paths = _paths(tmp_project)
    exhausted = False

    def probe() -> ResourceSnapshot:
        snapshot = _snapshot(tmp_project)
        if exhausted:
            return snapshot.model_copy(update={"available_bytes": 1024})
        return snapshot

    ledger = _ledger(tmp_project, probe)
    parent = _descriptor(paths)
    retry = retry_descriptor(parent, paths=paths)
    ledger.reserve(
        parent.job_id, 5.0, 1024, case_sha256=parent.case_sha256, model_hash=parent.model_hash
    )
    ledger.complete(parent.job_id, _result(parent, "NUMERICAL_FAILURE", wall_s=4.0))

    exhausted = True
    with pytest.raises(BudgetStop) as raised:
        ledger.reserve(
            retry.job_id,
            5.0,
            1024,
            case_sha256=retry.case_sha256,
            model_hash=retry.model_hash,
            attempt=retry.attempt,
        )
    assert raised.value.status == "RESOURCE_FAILURE"


def test_the_predicted_wall_time_is_the_session_share_until_something_is_measured(
    tmp_project: Path,
) -> None:
    """The default reservation is the whole job timeout, which no P0 session can afford."""
    ledger = _ledger(tmp_project, lambda: _snapshot(tmp_project))
    profile = P0_VERIFY_PROFILE
    first = predicted_wall_s(ledger, profile)
    assert first == pytest.approx(profile.wall_budget_s / profile.max_new_forward)
    assert first < profile.job_timeout_s

    # With a measurement on the record, the prediction follows the measurement.
    paths = _paths(tmp_project)
    job = _descriptor(paths)
    ledger.reserve(job.job_id, first, 1024, case_sha256=job.case_sha256, model_hash=job.model_hash)
    ledger.complete(job.job_id, _result(job, "COMPLETE", wall_s=40.0))
    assert predicted_wall_s(ledger, profile) > 40.0
    assert predicted_wall_s(ledger, profile) <= profile.job_timeout_s


def test_a_p0_session_can_book_more_than_two_jobs_against_its_wall_budget(
    tmp_project: Path,
) -> None:
    """The deferral this closes: `job_timeout_s` as an estimate refuses the third job."""
    paths = _paths(tmp_project)
    ledger = _ledger(tmp_project, lambda: _snapshot(tmp_project))
    profile = P0_VERIFY_PROFILE
    for index in range(8):
        job = _descriptor(
            paths,
            job_id=f"job-e01-{index:04d}",
            model_hash=f"{index + 1:064x}",
            result_dir=f"artifacts/results/job-e01-{index:04d}",
            attempt=1,
        )
        ledger.reserve(
            job.job_id,
            predicted_wall_s(ledger, profile),
            predicted_output_bytes(ledger, profile, n_cells=128, n_times=13),
            case_sha256=job.case_sha256,
            model_hash=job.model_hash,
        )
        ledger.complete(job.job_id, _result(job, "COMPLETE", wall_s=1.0))
    assert ledger.session_totals().forwards == 8


def test_the_predicted_output_size_is_derived_from_the_case_it_is_for(
    tmp_project: Path,
) -> None:
    ledger = _ledger(tmp_project, lambda: _snapshot(tmp_project))
    small = predicted_output_bytes(ledger, P0_VERIFY_PROFILE, n_cells=8, n_times=2)
    large = predicted_output_bytes(ledger, P0_VERIFY_PROFILE, n_cells=1024, n_times=37)
    assert 0 < small < large
    assert large < P0_VERIFY_PROFILE.disk_budget_bytes


# ----------------------------------------------------- the cross-record COMPLETE checks


def _complete(**overrides: object) -> ForwardResult:
    times = tuple(overrides.pop("times_s", (0.0, 30.0 * DAY)))  # type: ignore[arg-type]
    ref = ArrayRef(
        path="artifacts/results/job/states.h5",
        dataset="pressure_pa",
        sha256="c" * 64,
        shape=(len(times), 8),
        dtype="float64",
        unit="Pa",
        axis_order=("time", "cell"),
    )
    fields: dict[str, object] = {
        "job_id": "job-e01-0001",
        "case_sha256": "a" * 64,
        "model_hash": "b" * 64,
        "physics_class": "OW",
        "status": "COMPLETE",
        "completed_time_s": times[-1],
        "times_s": times,
        "states": {"pressure_pa": ref},
        "monthly_path": "artifacts/results/job/monthly.parquet",
        "connections_path": "artifacts/results/job/connections.parquet",
        "balances_path": "artifacts/results/job/balances.parquet",
        "solver_metadata": {},
        "cost": CostRecord(
            wall_s=1.0,
            cpu_s=1.0,
            peak_rss_bytes=1,
            output_bytes=1,
            accepted_steps=1,
            cut_steps=0,
            nonlinear_iterations=1,
            retry_count=0,
            measurement_method="test",
        ),
        "parent_attempt_ids": (),
    }
    fields.update(overrides)
    return ForwardResult(**fields)  # type: ignore[arg-type]


def _restart_ref() -> RestartRef:
    return RestartRef(
        manifest_path="artifacts/results/job/checkpoint/restart_manifest.json",
        sha256="d" * 64,
        completed_report_step=2,
        completed_time_s=30.0 * DAY,
        model_hash="b" * 64,
        schedule_prefix_hash="e" * 64,
        environment_lock_hash="f" * 64,
    )


def test_a_complete_result_must_carry_the_whole_requested_axis() -> None:
    request = OutputRequest(state_times_s=(0.0, 15.0 * DAY, 30.0 * DAY), keep_native_restart=False)
    with pytest.raises(ForwardRequestError, match="1296000"):
        check_requested_outputs(_complete(), request)
    covered = _complete(times_s=(0.0, 15.0 * DAY, 30.0 * DAY))
    assert check_requested_outputs(covered, request) is None


def test_a_complete_result_must_carry_the_restart_that_was_asked_for() -> None:
    request = OutputRequest(state_times_s=(0.0, 30.0 * DAY), keep_native_restart=True)
    with pytest.raises(ForwardRequestError, match="keep_native_restart"):
        check_requested_outputs(_complete(), request)
    assert check_requested_outputs(_complete(restart=_restart_ref()), request) is None


def test_a_restart_from_a_different_model_is_refused() -> None:
    request = OutputRequest(state_times_s=(0.0, 30.0 * DAY), keep_native_restart=True)
    foreign = _restart_ref().model_copy(update={"model_hash": "9" * 64})
    with pytest.raises(ForwardRequestError, match="model_hash"):
        check_requested_outputs(_complete(restart=foreign), request)


def test_an_unsuccessful_result_is_not_measured_against_the_request() -> None:
    """Only a COMPLETE claims to have delivered; a refusal is judged by its own record."""
    request = OutputRequest(state_times_s=(0.0, 30.0 * DAY), keep_native_restart=True)
    refused = ForwardResult(
        job_id="job-e01-0001",
        case_sha256="a" * 64,
        model_hash="b" * 64,
        physics_class="OW",
        status="NUMERICAL_FAILURE",
        reason="the solver did not reach the end of its schedule",
        completed_time_s=0.0,
        times_s=(),
        states={},
        solver_metadata={},
        cost=CostRecord(
            wall_s=1.0,
            cpu_s=1.0,
            peak_rss_bytes=1,
            output_bytes=1,
            accepted_steps=1,
            cut_steps=0,
            nonlinear_iterations=1,
            retry_count=0,
            measurement_method="test",
        ),
        parent_attempt_ids=(),
    )
    assert check_requested_outputs(refused, request) is None


# ------------------------------------------------------------------ the memory drift kernel


def test_the_drift_threshold_is_the_larger_of_the_floor_and_the_fraction() -> None:
    assert memory_drift_limit_bytes(100 * 1024**2) == MEMORY_DRIFT_FLOOR_BYTES
    big = 4 * GIB
    assert memory_drift_limit_bytes(big) == int(MEMORY_DRIFT_FRACTION * big)
    assert MEMORY_DRIFT_FLOOR_BYTES == 256 * 1024**2
    assert MEMORY_DRIFT_FRACTION == 0.20


def test_a_steady_worker_is_left_alone() -> None:
    monitor = MemoryDriftMonitor(baseline_rss_bytes=2 * GIB)
    for _ in range(5):
        decision = monitor.observe(2 * GIB + 10 * 1024**2)
        assert decision.action == "continue"
        assert decision.status is None


def test_a_drifting_worker_is_recycled_once_and_then_refused() -> None:
    monitor = MemoryDriftMonitor(baseline_rss_bytes=2 * GIB)
    drifting = 2 * GIB + 700 * 1024**2

    first = monitor.observe(drifting)
    assert first.action == "recycle"
    assert first.status is None
    assert "recycle" in (first.reason or "")

    # A repeatable rise is a resource failure, not an endless recycle loop.
    second = monitor.observe(drifting)
    assert second.action == "stop"
    assert second.status == "RESOURCE_FAILURE"
    assert monitor.recycles == 1

    # And the verdict does not un-happen because the next sample looks calm.
    assert monitor.observe(2 * GIB).action == "stop"


def test_a_recycled_worker_rebaselines_when_it_comes_back_healthy() -> None:
    monitor = MemoryDriftMonitor(baseline_rss_bytes=2 * GIB)
    assert monitor.observe(2 * GIB + 700 * 1024**2).action == "recycle"
    monitor.rebaseline(1 * GIB)
    assert monitor.observe(1 * GIB + 1024**2).action == "continue"
    assert monitor.recycles == 1


# ------------------------------------------------- what this build can be asked to deliver


def test_a_request_for_per_substep_diagnostics_is_refused_rather_than_ignored() -> None:
    """`diagnostic_substeps` gates nothing here, so asking for it is answered, not dropped.

    Plan 7.7 makes the accepted-substep diagnostics table a minimal output written for every
    job, and E01 publishes no full per-substep states for the flag to gate instead. A caller
    that set it and silently got neither would be reading a result it did not ask for.
    """
    request = OutputRequest(
        state_times_s=(0.0, 30.0 * DAY), keep_native_restart=False, diagnostic_substeps=True
    )
    with pytest.raises(ForwardRequestError, match="diagnostic_substeps"):
        check_request_is_deliverable(request)
    # And the default request — the one every caller in this epic makes — is deliverable.
    assert (
        check_request_is_deliverable(
            OutputRequest(state_times_s=(0.0, 30.0 * DAY), keep_native_restart=False)
        )
        is None
    )
