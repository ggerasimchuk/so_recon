"""E01.0: the real persistent worker, exercised over the real protocol.

One Julia process handles everything here — two diagnostic pings, two jobs and a shutdown
— because that is the point of a persistent worker: the cold start is paid once and the
process is then reused without leaking state between jobs. A test that launched Julia per
assertion would both be slow and prove the opposite of what is being claimed.

What this asserts that the transport tests cannot:

* The handshake carries the *pinned* environment. Jutul 0.4.31, JutulDarcy 0.3.11,
  JSON 1.8.0 and HDF5 0.17.3 are read out of the running process, not out of a lock file.
* The process is stable across jobs: both pings and both jobs are answered by one pid.
* Julia re-hashes the bytes it was told to run on. A descriptor whose declared digests are
  wrong is refused with a digest mismatch, by the same code path that accepts a good one.
* `run` cannot succeed. The adapter builds the model, but this build integrates no time
  axis, so the answer is an explicit `outputs unavailable` refusal — and it still publishes
  a record into the job's own directory, whose bytes the Python side verifies against the
  digest it was sent. What the constructor actually built is checked in
  `test_e01_physics.py`; this file is about the transport and the identity check.
* stdout carried protocol frames only: every frame in this session parsed, and the native
  per-job log files were created beside them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import CASE_MANIFEST_FILENAME, write_case
from so_recon.simulator.contracts import SECONDS_PER_DAY, JobDescriptor, OutputRequest
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.worker import PROTOCOL_VERSION, PersistentJuliaWorker
from tests.forward_case import build_case, write_case_arrays

ROOT = Path(__file__).resolve().parents[2]

PINNED_JULIA_PACKAGES = {
    "Jutul": "0.4.31",
    "JutulDarcy": "0.3.11",
    "JSON": "1.8.0",
    "HDF5": "0.17.3",
}


@pytest.mark.julia
def test_one_julia_process_serves_pings_and_isolated_jobs(tmp_project: Path) -> None:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        julia_exe = find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")

    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    case = build_case(write_case_arrays(paths))
    ctx = RunContext.start(command="worker", argv=[], cfg=None, paths=paths)
    write_case(case, paths, ctx)
    case_path = ctx.run_dir / CASE_MANIFEST_FILENAME
    solver_path = paths.artifacts / "solver" / "e01_solver.json"
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    solver_path.write_text(
        json.dumps({"max_timestep_days": 30, "max_nonlinear_iterations": 15}), encoding="utf-8"
    )

    session_dir = paths.artifacts / "session"

    def probe() -> ResourceSnapshot:
        return probe_resources(None, session_dir)

    # Both the ledger and the worker's watchdog are given a machine with room,
    # deliberately. `memory_stop`'s cap is `min(configured, total - reserve,
    # rss + available - reserve)`, so it collapses whenever this laptop's free memory
    # approaches the 6 GiB OS reserve — it already refused this test once on a busy
    # machine. The watchdog reads the same guard from inside the job loop while Julia is
    # JIT-compiling (which IS growing RSS), so leaving the real probe on the worker would
    # make a stage-gate test fail because something else was running. Whether this host
    # has memory to spare is Task 3's subject, measured by Task 3's own tests; what this
    # test is for is the protocol.
    #
    # Only the memory fields are pinned: it is a `model_copy` of a real measurement, so
    # the monotonic clock, the CPU time, the free disk and the measurement method the
    # cost record reports all still come from `probe_resources`.
    def unloaded_machine() -> ResourceSnapshot:
        return probe().model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "process_rss_bytes": 1024**3,
                "swap_used_bytes": 0,
            }
        )

    ledger = BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=paths.artifacts / "ledger.json",
        session_id="session-e01-worker",
        probe=unloaded_machine,
    )

    def descriptor(job_id: str, *, case_sha256: str, model_hash: str) -> JobDescriptor:
        return JobDescriptor(
            job_id=job_id,
            case_path=paths.relative(case_path),
            case_sha256=case_sha256,
            model_hash=model_hash,
            solver_config_path=paths.relative(solver_path),
            solver_config_sha256=sha256_file(solver_path),
            output_request=OutputRequest(
                state_times_s=(15.0 * SECONDS_PER_DAY, 30.0 * SECONDS_PER_DAY),
                keep_native_restart=False,
            ),
            seed=20260913,
            result_dir=f"artifacts/results/{job_id}",
            attempt=1,
        )

    with PersistentJuliaWorker(
        julia_exe,
        ROOT / "julia",
        session_dir,
        P0_VERIFY_PROFILE,
        paths=paths,
        probe=unloaded_machine,
    ) as worker:
        handshake = worker.handshake
        assert handshake.protocol == PROTOCOL_VERSION
        assert handshake.pid == worker.pid
        # COMPUTE §5, taken from the profile and read back out of the running process.
        assert handshake.julia_threads == 4
        assert handshake.blas_threads == 1
        for name, version in PINNED_JULIA_PACKAGES.items():
            assert handshake.versions[name] == version, (name, handshake.versions)

        first = worker.ping("ping-0001", {"case": case_path, "solver": solver_path})
        second = worker.ping("ping-0002", {"case": case_path})

        # Exact job ids, exact digests, and one process behind both answers.
        assert (first.job_id, second.job_id) == ("ping-0001", "ping-0002")
        assert first.pid == second.pid == worker.pid
        assert first.inputs == {
            "case": sha256_file(case_path),
            "solver": sha256_file(solver_path),
        }
        assert second.inputs == {"case": sha256_file(case_path)}

        good = descriptor(
            "job-e01-0001", case_sha256=sha256_file(case_path), model_hash=case.model_hash
        )
        result = worker.submit(good, ledger)

        # A model was built, but a forward result needs states this build cannot produce,
        # so the answer names what is missing. The physics class was read out of the
        # verified case.
        assert result.status == "INVALID_INPUT"
        assert result.reason is not None and result.reason.startswith("outputs unavailable")
        assert result.physics_class == "OW"
        assert result.job_id == good.job_id
        assert result.case_sha256 == good.case_sha256
        assert result.model_hash == good.model_hash
        assert result.solver_metadata["solver_config_sha256"] == good.solver_config_sha256
        assert len(result.solver_metadata["environment_lock_hash"]) == 64
        assert result.solver_metadata["version.JutulDarcy"] == "0.3.11"

        # The failure record was published into the job's own directory, and the digest
        # the worker reported is the digest of the bytes that are actually there.
        record_path = paths.resolve(result.solver_metadata["result_path"])
        assert record_path.parent == paths.root / good.result_dir
        assert sha256_file(record_path) == result.solver_metadata["result_sha256"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["status"] == "INVALID_INPUT"
        assert record["checked_inputs"]["case"] == good.case_sha256
        assert record["checked_inputs"]["solver_config"] == good.solver_config_sha256
        # Every array the case references was re-hashed too, not only the manifest.
        array_checks = {k: v for k, v in record["checked_inputs"].items() if k.startswith("case.")}
        assert len(array_checks) == 7, sorted(array_checks)
        assert all(isinstance(v, str) and len(v) == 64 for v in array_checks.values())

        # Native logging went to the job's own files, not to the protocol stream. The
        # line below was printed by Julia to stdout while the job ran; every frame this
        # session read still parsed, so the redirect did not leak into the protocol and
        # the protocol did not leak into the log.
        job_stdout = session_dir / "logs" / f"{good.job_id}.stdout"
        assert (session_dir / "logs" / f"{good.job_id}.stderr").is_file()
        assert f"so-recon worker: job {good.job_id}" in job_stdout.read_text(encoding="utf-8")

        # A descriptor that lies about the bytes is refused by the same path.
        tampered = descriptor("job-e01-0002", case_sha256="c" * 64, model_hash="d" * 64)
        refused = worker.submit(tampered, ledger)

        assert refused.status == "INVALID_INPUT"
        assert refused.reason is not None
        assert "sha256" in refused.reason and sha256_file(case_path) in refused.reason
        assert worker.pid == handshake.pid  # still the same process after four requests

    assert worker.returncode == 0, "the worker did not exit cleanly on shutdown"
    assert [entry.status for entry in ledger.record.entries] == ["INVALID_INPUT"] * 2
    assert all(entry.state == "FAILED" for entry in ledger.record.entries)
