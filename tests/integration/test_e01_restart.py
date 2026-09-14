"""E01.8: a native restart that is the same run, and a worker that keeps nothing.

Two claims, each worth a pair of real Julia processes and nothing more.

**A continuation is the same run.** `test_a_native_restart_reproduces_the_continuous_run`
runs six months three ways in one worker — continuous (one chunk), chunked (six), and
stopped after month three — then CLOSES that worker, opens a new one, and continues the
stopped run from its published checkpoint. The three results are compared on every shared
date: pressure, both saturations, the monthly volumes of both wells, the component
inventory and the connection summaries. The fixture is built so that this can fail: a
completion event lands strictly inside the part the checkpoint covers, and the two wells
SWAP ROLES exactly at the checkpoint, so a continuation that reused the prefix's controls,
double-counted the prefix or resumed from a re-initialised state gives different numbers.

**A worker keeps nothing between jobs.** `test_a_b_a_leaves_nothing_of_b_in_the_second_a`
runs A, B and A again in one process, where A and B differ in permeability, fluids, initial
pressure, initial saturation and completion masks. Both A results must be the first A, and
the same A run in a process that never saw B must be that too. Five warm repeats of A in
the same process are measured for retained memory, because a leak is the way "keeps
nothing" fails quietly.

Neither test depends on this host's free RAM: the resource guards are driven by a
`model_copy` of a real snapshot with the memory fields pinned, exactly as the earlier
integration tests do. The ONE place a real measurement is the point is the drift check,
which reads the worker process tree's own RSS.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import read_array
from so_recon.simulator.contracts import CaseBundle, ForwardResult, OutputRequest, RestartRef
from so_recon.simulator.forward import (
    ForwardRequestError,
    MemoryDriftMonitor,
    SolverConfig,
    check_requested_outputs,
    resume,
    simulate,
)
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.results import (
    RESULT_FILENAME,
    load_forward_result,
    write_forward_result,
)
from so_recon.simulator.worker import PersistentJuliaWorker, request_stop_after_chunk
from tests.restart_case import (
    RESTART_MONTHS,
    ROLE_SWITCH_MONTH,
    future_policy,
    isolation_cases,
    six_month_case,
    six_month_controls,
)

ROOT = Path(__file__).resolve().parents[2]

#: The tolerance the continuous and the restarted run are compared under.
#:
#: It is deliberately an explicit constant rather than a bare `==`: Task 9 owns the
#: verification tolerances of E01 and will replace THIS name with the one it defines. What is
#: measured here on Jutul 0.4.31 / JutulDarcy 0.3.11 is exact agreement — every published
#: state, volume and inventory of the restarted run equals the continuous run's to the last
#: bit — so the number below is headroom for a future solver, not a fudge for this one. A
#: restart that only agreed to, say, 1e-6 would be a different claim and would need a
#: physical argument for the difference; the assertions are written tight enough to notice.
RESTART_AGREEMENT_ATOL = 0.0
RESTART_AGREEMENT_RTOL = 0.0

#: How far a warm worker's retained RSS may sit above the post-warm baseline before this
#: test calls it drift. It is `forward.memory_drift_limit_bytes`' own threshold — the test
#: asserts on the DECISION the policy makes, not on a byte count of its own.
WARM_REPEATS = 5

SOLVER = SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15)

#: A job that cannot possibly finish inside a poll: two-hour maximum steps over half a year.
#: The watchdog kills it mid-chunk, which is what the force-kill claim needs to be about a
#: real process and not about a fixture that happened to be slow.
UNENDING_SOLVER = SolverConfig(max_timestep_days=0.05, max_nonlinear_iterations=15)


def _skip_unless_julia_is_installed() -> Path:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        return find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")


def _pinned_probe(session_dir: Path) -> Callable[[], ResourceSnapshot]:
    """A real measurement with only the memory fields pinned (see the module docstring)."""

    def probe() -> ResourceSnapshot:
        return probe_resources(None, session_dir).model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "process_rss_bytes": 1024**3,
                "swap_used_bytes": 0,
            }
        )

    return probe


def _ledger(paths: ProjectPaths, name: str, probe: Callable[[], ResourceSnapshot]) -> BudgetLedger:
    """One session per run of a given model.

    `BudgetLedger.reserve` allows SPEC 3.3's two attempts per physical model and no more,
    which is the whole point of it; running the same case a third time is therefore a third
    SESSION rather than a third attempt, and that is what these are.
    """
    return BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=paths.artifacts / f"ledger-{name}.json",
        session_id=f"session-{name}",
        probe=probe,
    )


def _states(result: ForwardResult, paths: ProjectPaths) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(read_array(ref, paths), dtype=np.float64)
        for name, ref in result.states.items()
    }


def _table(path: str | None, paths: ProjectPaths) -> list[dict[str, Any]]:
    assert path is not None
    rows: list[dict[str, Any]] = pq.read_table(paths.resolve(path)).to_pylist()
    return rows


def _digests(directory: Path) -> dict[str, str]:
    """Every file under a directory and its digest, so "unchanged" can be asserted."""
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _agree(left: NDArray[np.float64], right: NDArray[np.float64], what: str) -> None:
    assert left.shape == right.shape, what
    assert np.allclose(left, right, atol=RESTART_AGREEMENT_ATOL, rtol=RESTART_AGREEMENT_RTOL), (
        f"{what}: largest difference {np.abs(left - right).max()}"
    )


def _reload(result: ForwardResult, paths: ProjectPaths) -> ForwardResult:
    """Write the record out and read it back through every check it has to pass."""
    record_path = paths.resolve(result.solver_metadata["result_path"]).parent / RESULT_FILENAME
    write_forward_result(result, record_path)
    return load_forward_result(record_path, paths)


@pytest.mark.julia
def test_a_native_restart_reproduces_the_continuous_run(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    case = six_month_case(paths)
    edges = case.report_edges_s
    assert len(edges) == RESTART_MONTHS + 1

    session = paths.artifacts / "session-a"
    probe = _pinned_probe(session)
    continuous_request = OutputRequest(
        state_times_s=edges, keep_native_restart=False, chunk_months=RESTART_MONTHS
    )
    chunked_request = OutputRequest(state_times_s=edges, keep_native_restart=False, chunk_months=1)
    restartable = OutputRequest(state_times_s=edges, keep_native_restart=True, chunk_months=1)

    def run_context() -> RunContext:
        return RunContext.start(command="forward", argv=[], cfg=None, paths=paths)

    prefix_ledger = _ledger(paths, "prefix", probe)
    with PersistentJuliaWorker(
        julia_exe, ROOT / "julia", session, P0_VERIFY_PROFILE, paths=paths, probe=probe
    ) as worker:
        continuous = simulate(
            case,
            continuous_request,
            worker=worker,
            ctx=run_context(),
            ledger=_ledger(paths, "continuous", probe),
            solver_config=SOLVER,
        )
        chunked = simulate(
            case,
            chunked_request,
            worker=worker,
            ctx=run_context(),
            ledger=_ledger(paths, "chunked", probe),
            solver_config=SOLVER,
        )
        # The operator asks for the run to stop once March is finished. The request is a
        # file in the session directory and touches no job input.
        stop_path = request_stop_after_chunk(
            session,
            reason="the restart test stops after the third month",
            after_completed_time_s=edges[ROLE_SWITCH_MONTH],
        )
        prefix = simulate(
            case,
            restartable,
            worker=worker,
            ctx=run_context(),
            ledger=prefix_ledger,
            solver_config=SOLVER,
        )
        stop_path.unlink()
        prefix_pid = worker.pid

    # --- the chunked path against the continuous one, first ---------------------------
    assert continuous.status == "COMPLETE", continuous.reason
    assert chunked.status == "COMPLETE", chunked.reason
    assert continuous.times_s == chunked.times_s == tuple(edges)
    check_requested_outputs(continuous, continuous_request)
    check_requested_outputs(chunked, chunked_request)
    # One `simulate!` call against six, each continuing the last natively.
    assert len(json.loads(continuous.solver_metadata["chunk_diagnostics"])) == 1
    assert len(json.loads(chunked.solver_metadata["chunk_diagnostics"])) == RESTART_MONTHS
    # The same accepted substeps, not merely the same answer at the month ends.
    assert continuous.cost.accepted_steps == chunked.cost.accepted_steps
    assert continuous.cost.cut_steps == chunked.cost.cut_steps == 0
    assert continuous.cost.nonlinear_iterations == chunked.cost.nonlinear_iterations > 0

    reference = _states(continuous, paths)
    for name, values in _states(chunked, paths).items():
        _agree(reference[name], values, f"chunked states[{name!r}]")
    assert _table(chunked.monthly_path, paths) == _table(continuous.monthly_path, paths)
    assert _table(chunked.connections_path, paths) == _table(continuous.connections_path, paths)

    # --- and the stop: whole months kept, the rest not invented ------------------------
    assert prefix.status == "INCOMPLETE_BUDGET"
    assert prefix.reason is not None and "stop was requested" in prefix.reason
    # SPEC 18.4: the months nobody simulated are not published as months in which nothing
    # flowed. No outputs at all rather than three empty ones.
    assert prefix.monthly_path is None
    assert prefix.balances_path is None
    # The completed work is on the ledger, and it is on it as work that was paid for.
    assert prefix_ledger.completed_job_ids == ()
    assert [entry.status for entry in prefix_ledger.record.entries] == ["INCOMPLETE_BUDGET"]
    assert prefix_ledger.session_totals().forwards == 1

    checkpoint = prefix.restart
    assert checkpoint is not None
    assert checkpoint.completed_report_step == ROLE_SWITCH_MONTH
    assert checkpoint.completed_time_s == edges[ROLE_SWITCH_MONTH]
    assert checkpoint.model_hash == case.model_hash
    assert checkpoint.native_format == "Jutul-native"
    manifest_path = paths.resolve(checkpoint.manifest_path)
    assert sha256_file(manifest_path) == checkpoint.sha256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "restart-manifest-1"
    assert (manifest["Jutul"], manifest["JutulDarcy"]) == ("0.4.31", "0.3.11")
    assert manifest["environment_lock_hash"] == checkpoint.environment_lock_hash
    assert manifest["schedule_prefix_hash"] == checkpoint.schedule_prefix_hash
    assert manifest["case_sha256"] == prefix.case_sha256
    # Every native file, its step, its own digest — and the digests are of the bytes there.
    assert [entry["step"] for entry in manifest["files"]] == [1, 2, 3]
    for entry in manifest["files"]:
        native = manifest_path.parent / entry["path"]
        assert entry["path"] == f"jutul_{entry['step']}.jld2"
        assert sha256_file(native) == entry["sha256"]
        assert native.stat().st_size == entry["bytes"] > 0
    before_resume = _digests(manifest_path.parent)

    # --- a NEW process continues it ----------------------------------------------------
    resumed_session = paths.artifacts / "session-b"
    resume_probe = _pinned_probe(resumed_session)
    resumed_ledger = BudgetLedger.resume(
        parent_path=prefix_ledger.path,
        path=paths.artifacts / "ledger-resumed.json",
        session_id="session-resumed",
        probe=resume_probe,
    )
    with PersistentJuliaWorker(
        julia_exe,
        ROOT / "julia",
        resumed_session,
        P0_VERIFY_PROFILE,
        paths=paths,
        probe=resume_probe,
    ) as second:
        assert second.pid != prefix_pid
        continued = resume(
            case,
            checkpoint,
            future_policy(edges),
            worker=second,
            ctx=run_context(),
            ledger=resumed_ledger,
            solver_config=SOLVER,
            output_request=restartable,
        )
        # A checkpoint whose native bytes moved is INVALID_INPUT, and it is refused before a
        # single one of them is read. The manifest itself is untouched, so this is the file
        # check rather than the metadata check.
        corrupt_root = paths.artifacts / "corrupted"
        shutil.copytree(manifest_path.parent, corrupt_root)
        victim = corrupt_root / "jutul_2.jld2"
        victim.write_bytes(victim.read_bytes()[:-8] + b"\x00" * 8)
        corrupted = resume(
            case,
            checkpoint.model_copy(
                update={"manifest_path": paths.relative(corrupt_root / manifest_path.name)}
            ),
            future_policy(edges),
            worker=second,
            ctx=run_context(),
            ledger=_ledger(paths, "corrupt", resume_probe),
            solver_config=SOLVER,
            output_request=restartable,
        )
        assert corrupted.status == "INVALID_INPUT"
        assert corrupted.reason is not None
        assert "jutul_2.jld2" in corrupted.reason and "INVALID_INPUT" in corrupted.reason

        # A job the watchdog kills mid-chunk publishes nothing and damages nothing. The
        # process group really is signalled: this worker is unusable afterwards, which is
        # why it is the last thing it is asked to do.
        killed_ctx = run_context()
        killed = simulate(
            case,
            restartable,
            worker=_timeout_after_one_poll(second, resume_probe),
            ctx=killed_ctx,
            ledger=_ledger(paths, "killed", resume_probe),
            solver_config=UNENDING_SOLVER,
        )
        assert killed.status == "TIMEOUT", killed.reason
        assert killed.reason is not None and "timeout" in killed.reason
        killed_dir = killed_ctx.run_dir / killed.job_id

    # Nothing of the killed job was accepted: its result directory was never created, and
    # any staging the dead process left behind is staging — it holds no worker record, so
    # nothing can be read out of it as a result or resumed out of it as a checkpoint.
    assert not killed_dir.exists()
    for leftover in (paths.artifacts / "runs").rglob("*.staging-*"):
        assert not (leftover / "result.json").exists(), leftover
    # And the last completed checkpoint is exactly as it was, byte for byte.
    assert _digests(manifest_path.parent) == before_resume

    # --- the continuation IS the continuous run ----------------------------------------
    assert continued.status == "COMPLETE", continued.reason
    check_requested_outputs(continued, restartable)
    assert continued.times_s == tuple(edges)
    assert continued.completed_time_s == edges[-1]
    # An unchanged future policy is the same model, so the continuation is a second attempt
    # at the model the stop left unfinished rather than a new one.
    assert continued.model_hash == case.model_hash
    assert resumed_ledger.completed_job_ids == (continued.job_id,)
    assert resumed_ledger.cumulative_totals().forwards == 2

    # The published record survives every check it has to pass on the way back in.
    assert _reload(continued, paths) == continued

    for name, values in _states(continued, paths).items():
        _agree(reference[name], values, f"restarted states[{name!r}]")
    assert _table(continued.monthly_path, paths) == _table(continuous.monthly_path, paths)
    assert _table(continued.connections_path, paths) == _table(continuous.connections_path, paths)
    assert _table(continued.balances_path, paths) == _table(continuous.balances_path, paths)
    for key, value in continuous.solver_metadata.items():
        if key.startswith("balance") and key.endswith(("_within_spec_tolerance", "_relative")):
            assert continued.solver_metadata[key] == value, key

    # The prefix was not integrated twice and the initial snapshot was not duplicated: the
    # substep axis of the continuation is the continuous run's, one entry per substep.
    steps = _table(continued.solver_metadata["accepted_steps.parquet.path"], paths)
    assert steps == _table(continuous.solver_metadata["accepted_steps.parquet.path"], paths)
    assert len(steps) == continuous.cost.accepted_steps
    assert len({row["start_s"] for row in steps}) == len(steps)
    assert continued.times_s == continuous.times_s

    # And the LEDGER is charged for work, not for coverage. The continuation's published
    # integrals cover all six months, because it re-extracted the parent's three from the
    # parent's own `.jld2` files — but the parent's entry already holds those, so charging
    # them again would make a session's totals larger than the session.
    assert prefix.cost.accepted_steps > 0
    assert continued.cost.accepted_steps == len(steps) - prefix.cost.accepted_steps
    assert continued.cost.nonlinear_iterations == (
        continuous.cost.nonlinear_iterations - prefix.cost.nonlinear_iterations
    )
    assert "re-extracted from the checkpoint it resumed" in continued.cost.measurement_method
    chain = resumed_ledger.record
    charged = sum(
        entry.cost.accepted_steps
        for entry in (*chain.inherited_entries, *chain.entries)
        if entry.cost is not None
    )
    assert charged == continuous.cost.accepted_steps

    # And the disk it is charged for is the disk it published, not only the worker record
    # that preceded them: `publish_forward_result` writes `states.h5` and the four tables
    # into the job's directory, and `BudgetLedger.committed_output_bytes` sums this field.
    # (The `forward_result.json` that `_reload` wrote above is deliberately not in it: the
    # driver decides where that record goes, and it goes in after the job has been costed.)
    worker_record = paths.resolve(continued.solver_metadata["result_path"])
    published = [
        paths.resolve(continued.solver_metadata["states_path"]),
        paths.resolve(str(continued.monthly_path)),
        paths.resolve(str(continued.connections_path)),
        paths.resolve(str(continued.balances_path)),
        paths.resolve(continued.solver_metadata["accepted_steps.parquet.path"]),
    ]
    assert continued.cost.output_bytes >= worker_record.stat().st_size + sum(
        path.stat().st_size for path in published
    )
    assert continued.cost.output_bytes > worker_record.stat().st_size

    # A continuation whose PAST was rewritten is refused, and refused here rather than by
    # the solver: the digest is of the schedule prefix the checkpoint stopped after.
    rewritten = case.model_copy(
        update={
            "controls": tuple(
                segment.model_copy(update={"value": segment.value + 1.0})
                if segment.start_s == 0.0 and segment.well_id == "PRO1"
                else segment
                for segment in six_month_controls(edges)
            )
        }
    )
    with pytest.raises(ForwardRequestError, match="rewritten|prefix"):
        resume(
            rewritten,
            checkpoint,
            future_policy(edges),
            worker=second,
            ctx=run_context(),
            ledger=_ledger(paths, "rewritten", resume_probe),
            solver_config=SOLVER,
            output_request=restartable,
        )


def _timeout_after_one_poll(
    worker: PersistentJuliaWorker, probe: Callable[[], ResourceSnapshot]
) -> PersistentJuliaWorker:
    """Make the watchdog see this worker's next job as having run past its timeout.

    Only the monotonic clock the snapshots carry is moved, and only after the baseline has
    been taken — which is the clock the watchdog measures a job against. The kill that
    follows is the real one: SIGTERM to the process group, then SIGKILL.
    """
    calls = {"n": 0}
    jump = float(P0_VERIFY_PROFILE.job_timeout_s + 10)

    def jumping() -> ResourceSnapshot:
        calls["n"] += 1
        snapshot = probe()
        if calls["n"] <= 1:
            return snapshot
        return snapshot.model_copy(update={"monotonic_s": snapshot.monotonic_s + jump})

    worker._probe = jumping  # noqa: SLF001 - the probe is the class's injected seam
    return worker


@pytest.mark.julia
def test_a_b_a_leaves_nothing_of_b_in_the_second_a(tmp_project: Path) -> None:
    julia_exe = _skip_unless_julia_is_installed()
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    case_a, case_b = isolation_cases(paths)
    # A and B really are different physics, not the same case renamed.
    assert case_a.model_hash != case_b.model_hash
    assert case_a.rock.permeability_m2.sha256 != case_b.rock.permeability_m2.sha256
    assert case_a.fluids.viscosity_pa_s != case_b.fluids.viscosity_pa_s
    assert case_a.initial.pressure_pa != case_b.initial.pressure_pa  # type: ignore[union-attr]
    assert case_a.initial.sw != case_b.initial.sw  # type: ignore[union-attr]
    assert {c.connection_open for c in case_a.controls} != {
        c.connection_open for c in case_b.controls
    }

    session = paths.artifacts / "session-aba"
    probe = _pinned_probe(session)
    request = OutputRequest(
        state_times_s=case_a.report_edges_s, keep_native_restart=False, chunk_months=1
    )

    def run(worker: PersistentJuliaWorker, case: CaseBundle, name: str) -> ForwardResult:
        return simulate(
            case,
            request,
            worker=worker,
            ctx=RunContext.start(command="forward", argv=[], cfg=None, paths=paths),
            ledger=_ledger(paths, name, probe),
            solver_config=SOLVER,
        )

    warm: list[ForwardResult] = []
    drift_samples: list[int] = []
    with PersistentJuliaWorker(
        julia_exe, ROOT / "julia", session, P0_VERIFY_PROFILE, paths=paths, probe=probe
    ) as worker:
        pid = worker.pid
        first_a = run(worker, case_a, "aba-a1")
        # The baseline is taken AFTER the first job, not at start-up: Julia specialises the
        # whole solver on the first solve, and a baseline from before that would call the
        # specialisation a leak. It is the WORKER's own process tree, measured, and the one
        # place in this file where a real memory number is the point.
        monitor = MemoryDriftMonitor(
            baseline_rss_bytes=_worker_rss(worker, session),
        )
        b = run(worker, case_b, "aba-b")
        warm = [first_a]
        for index in range(WARM_REPEATS - 1):
            warm.append(run(worker, case_a, f"aba-warm{index}"))
            drift_samples.append(_worker_rss(worker, session))
        assert worker.pid == pid, "the same process answered every job"
        decisions = [monitor.observe(sample) for sample in drift_samples]
    assert len(warm) == WARM_REPEATS

    # --- the same process ran three different models and kept none of them --------------
    assert b.status == "COMPLETE", b.reason
    assert b.model_hash == case_b.model_hash
    assert all(result.status == "COMPLETE" for result in warm)
    reference = _states(first_a, paths)
    reference_monthly = _table(first_a.monthly_path, paths)
    for index, result in enumerate(warm[1:], start=1):
        for name, values in _states(result, paths).items():
            _agree(reference[name], values, f"warm run {index} states[{name!r}]")
        assert _table(result.monthly_path, paths) == reference_monthly, index
    # B's answer is a different answer, so agreement above is not agreement with anything.
    b_states = _states(b, paths)
    assert not np.allclose(b_states["pressure_pa"], reference["pressure_pa"])
    assert not np.allclose(b_states["sw"], reference["sw"])

    # --- and a process that never saw B agrees with both of them ------------------------
    fresh_session = paths.artifacts / "session-fresh"
    fresh_probe = _pinned_probe(fresh_session)
    with PersistentJuliaWorker(
        julia_exe,
        ROOT / "julia",
        fresh_session,
        P0_VERIFY_PROFILE,
        paths=paths,
        probe=fresh_probe,
    ) as fresh_worker:
        assert fresh_worker.pid != pid
        fresh = simulate(
            case_a,
            request,
            worker=fresh_worker,
            ctx=RunContext.start(command="forward", argv=[], cfg=None, paths=paths),
            ledger=_ledger(paths, "aba-fresh", fresh_probe),
            solver_config=SOLVER,
        )
    assert fresh.status == "COMPLETE", fresh.reason
    for name, values in _states(fresh, paths).items():
        _agree(reference[name], values, f"fresh worker states[{name!r}]")
    assert _table(fresh.monthly_path, paths) == reference_monthly
    # Identical physics, identical inputs, identical model hash — and a different job id, so
    # the agreement is between two runs rather than between a run and itself.
    assert fresh.model_hash == first_a.model_hash
    assert fresh.job_id != first_a.job_id

    # --- retained memory across the warm repeats ----------------------------------------
    # The assertion is on the POLICY's verdict rather than on a byte count of this test's
    # own, and over every sample rather than one: a single noisy measurement cannot fail it,
    # and a real leak crosses `max(256 MiB, 20% of the post-warm baseline)` and is caught.
    assert drift_samples and len(decisions) == len(drift_samples)
    assert [d.action for d in decisions] == ["continue"] * len(decisions), [
        (d.drift_bytes, d.limit_bytes) for d in decisions
    ]
    assert all(d.status is None for d in decisions)
    assert monitor.recycles == 0


def _worker_rss(worker: PersistentJuliaWorker, session_dir: Path) -> int:
    """The worker process tree's own RSS, after the worker collected its last job.

    `julia/worker/main.jl` runs `GC.gc()` in the `finally` of every request, so by the time
    a reply has been read the job's own allocations are collectable and collected. Measuring
    the WORKER's pid rather than this process's keeps pytest's own memory out of the number.
    """
    snapshot = probe_resources(worker.pid, session_dir)
    assert snapshot.process_rss_bytes is not None, snapshot.process_measurement_method
    return snapshot.process_rss_bytes


def test_a_restart_reference_of_a_different_model_is_refused(tmp_project: Path) -> None:
    """The cross-record check `load_forward_result` cannot make, made where both records are.

    No Julia: this is a statement about two records, and the point of keeping it here is
    that the same function the real restart test calls is the one being exercised.
    """
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    case = six_month_case(paths)
    request = OutputRequest(
        state_times_s=case.report_edges_s, keep_native_restart=True, chunk_months=1
    )
    foreign = RestartRef(
        manifest_path="artifacts/x/restart_manifest.json",
        sha256="a" * 64,
        completed_report_step=3,
        completed_time_s=case.report_edges_s[3],
        model_hash="b" * 64,
        schedule_prefix_hash="c" * 64,
        environment_lock_hash="d" * 64,
    )
    assert foreign.model_hash != case.model_hash
    with pytest.raises(ForwardRequestError, match="model_hash"):
        check_requested_outputs(_stub_complete(case, foreign), request)


def _stub_complete(case: CaseBundle, restart: RestartRef) -> ForwardResult:
    from so_recon.simulator.contracts import ArrayRef, CostRecord

    times = tuple(case.report_edges_s)
    return ForwardResult(
        job_id="job-e01-stub",
        case_sha256="e" * 64,
        model_hash=case.model_hash,
        physics_class="OW",
        status="COMPLETE",
        completed_time_s=times[-1],
        times_s=times,
        states={
            "pressure_pa": ArrayRef(
                path="artifacts/x/states.h5",
                dataset="pressure_pa",
                sha256="f" * 64,
                shape=(len(times), 8),
                dtype="float64",
                unit="Pa",
                axis_order=("time", "cell"),
            )
        },
        monthly_path="artifacts/x/monthly.parquet",
        connections_path="artifacts/x/connections.parquet",
        balances_path="artifacts/x/balances.parquet",
        restart=restart,
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
