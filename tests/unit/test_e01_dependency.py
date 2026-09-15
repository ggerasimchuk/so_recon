"""E02.1: the executable border with E01.

`require_e01_ow` is the only door between E02 and the accepted oil-water forward scope. It
reads the machine report `so-recon e01-report` published — the same `E01.json` the stage
command writes beside its Markdown page — and maps it into the `e01-ow-dependency-1`
evidence protocol. It refuses everything it cannot prove: a status that outruns its own
matrix, a restart row that is not there, a recorded SHA that no longer matches the artifact
on disk, a working tree that was dirty when the physics ran, an evidence commit that is not
an ancestor of the one E02 is being built on, and any change to the physics, configs or
locks since — because a changed component needs its native gate run again, not a gate that
looks the other way.

Every fixture here is a real repository: real files, real SHA-256 digests taken from those
files, real git commits. Nothing in this module asserts against the literal string "PASS"
that some payload happens to carry; the string is exactly what must not be believed on its
own.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from so_recon.inference.contracts import E01DependencyError
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.simulator.commands import MANDATORY_CHECKS
from so_recon.validation.e01_dependency import (
    E01_OW_EVIDENCE_SCHEMA_VERSION,
    PHYSICS_LINEAGE_PATHS,
    require_e01_ow,
)

P0_RUN = "artifacts/runs/20260914T203658Z-verify-physics-072fe0e7"
BENCH_RUN = "artifacts/runs/20260914T203415Z-benchmark-forward-c28739d6"
REPORT_RUN = "artifacts/runs/20260914T210000Z-e01-report-0a0a0a0a"
RESTART_EVIDENCE = f"{P0_RUN}/verification/restart.json"
BENCHMARK_EVIDENCE = f"{BENCH_RUN}/e01_benchmark.json"


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=E02 test",
            "-c",
            "user.email=e02@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.skip(f"git {args[0]} unavailable in this environment: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _check(name: str, status: str = "PASS", evidence: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "metrics": {} if status == "NOT_RUN" else {f"{name}_error": 0.25},
        "thresholds": {f"{name}_error_max": 1.0},
        "input_hashes": {},
        "evidence_paths": list(evidence),
        "reason": None if status == "PASS" else f"{name}: injected by the fixture",
    }


def _job(job_id: str, status: str = "COMPLETE") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "group": "p0",
        "kind": "forward",
        "profile": "P0_VERIFY",
        "accounting": "ledger",
        "expected_outcome": "COMPLETE",
        "status": status,
        "wall_s": 1.5,
        "cpu_s": 1.4,
        "peak_rss_bytes": 1024,
        "output_bytes": 2048,
        "native_chunk_calls": 3,
        "accepted_steps": 12,
        "cut_steps": 0,
        "nonlinear_iterations": 30,
        "retry_count": 0,
        "reason": None if status == "COMPLETE" else f"{job_id}: injected by the fixture",
    }


def _command(run_dir: str, command: str, commit: str) -> dict[str, Any]:
    return {
        "run_id": Path(run_dir).name,
        "run_dir": run_dir,
        "command": command,
        "argv": ["so-recon", command],
        "status": "PASS",
        "exit_code": 0,
        "created_at": "2026-09-14T20:36:58+00:00",
        "finished_at": "2026-09-14T20:40:00+00:00",
        "git_commit": commit,
        "git_dirty": False,
        "spec_version": "4.0",
        "config_version": "E01.1",
        "resolved_config_hash": "9" * 64,
        "environment_lock_hash": "8" * 64,
        "notes": [],
    }


@dataclass
class World:
    """A repository that really holds what its E01 report claims it holds."""

    root: Path
    paths: ProjectPaths
    payload: dict[str, Any]
    accepted_commit: str

    def publish(self) -> Path:
        path = self.root / REPORT_RUN / "E01.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def rehash(self) -> None:
        """Re-record the hashes the report states, after the fixture changed a file."""
        self.payload["tolerances_sha256"] = sha256_file(self.root / "configs/e01_tolerances.yml")
        self.payload["job_plan_sha256"] = sha256_file(self.root / "configs/e01_jobs.json")
        self.payload["lock_hashes"] = {
            name: sha256_file(self.root / name)
            for name in ("uv.lock", "julia/Manifest.toml", "julia/.julia-version")
        }
        self.payload["source_manifest"] = {
            "source_manifest.json": sha256_file(
                self.root / "reports/manifests/source_manifest.json"
            )
        }


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def world(tmp_path: Path) -> World:
    root = tmp_path / "repo"
    _write(root, "pyproject.toml", "[project]\nname='x'\n")
    _write(root, "README.md", "# fixture\n")
    _write(root, "src/so_recon/__init__.py", "")
    _write(root, "src/so_recon/simulator/forward.py", "# forward\n")
    _write(root, "src/so_recon/synthetic/p1.py", "# p1\n")
    _write(root, "configs/e01.yml", 'spec_version: "4.0"\n')
    _write(root, "configs/e01_tolerances.yml", "schema_version: e01-tolerances-2\n")
    _write(root, "configs/e01_jobs.json", '{"jobs": []}\n')
    _write(root, "uv.lock", "version = 1\n")
    _write(root, "julia/Manifest.toml", 'julia_version = "1.12.7"\n')
    _write(root, "julia/.julia-version", "1.12.7\n")
    _write(root, "reports/manifests/source_manifest.json", '{"manifest_version": "2"}\n')
    _write(root, RESTART_EVIDENCE, '{"restart_pressure_relative": 0.0}\n')
    _write(root, BENCHMARK_EVIDENCE, '{"warm": {"n": 5}}\n')
    _write(root, f"{P0_RUN}/run.json", '{"run_id": "p0"}\n')

    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "e01 accepted")
    accepted = _git(root, "rev-parse", "HEAD")
    _write(root, "README.md", "# fixture\n\nE02 starts here.\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "start e02")

    checks = [_check(name) for name in sorted(MANDATORY_CHECKS) if name != "restart_round_trip"]
    checks.append(_check("restart_round_trip", evidence=(RESTART_EVIDENCE,)))
    payload: dict[str, Any] = {
        "schema_version": "e01-stage-1",
        "stage": "E01",
        "status": "PASS_WITH_LIMITATIONS",
        "status_reason": "every mandatory oil-water check passed; black oil is NOT_RUN",
        "generated_at": "2026-09-14T20:58:09+00:00",
        "spec_version": "4.0",
        "config_version": "E01.1",
        "resolved_config_hash": "9" * 64,
        "environment_lock_hash": "8" * 64,
        "lock_hashes": {},
        "tolerances_path": "configs/e01_tolerances.yml",
        "tolerances_sha256": "",
        "job_plan_path": "configs/e01_jobs.json",
        "job_plan_sha256": "",
        "source_manifest": {},
        "git_commit": accepted,
        "git_commits": [accepted],
        "git_dirty": False,
        "commands": [
            _command(P0_RUN, "verify-physics", accepted),
            _command(BENCH_RUN, "benchmark-forward", accepted),
        ],
        "suites": {"p0": "exit 0"},
        "checks": checks,
        "required_checks": sorted(MANDATORY_CHECKS),
        "complete_checks": sorted(MANDATORY_CHECKS),
        "failed_checks": [],
        "unrun_checks": [],
        "jobs": [_job("p0_bl64")],
        "remaining_job_ids": [],
        "black_oil_remaining_job_ids": [],
        "limitations": [],
        "ow_gate": "PASS",
        "bo_status": "NOT_RUN",
        "physical_class": "oil_water",
        "fluid_claim": "educational two-phase oil-water on synthetic truth",
        "restart_round_trip": _check("restart_round_trip", evidence=(RESTART_EVIDENCE,)),
        "benchmark": {"p0": {"warm": {"n": 5}}},
        "costs": {
            "ledger_forwards": 1,
            "ledger_wall_s": 1.5,
            "ledger_cpu_s": 1.4,
            "ledger_output_bytes": 2048,
            "launcher_forwards": 0,
            "launcher_wall_s": 0.0,
            "peak_rss_bytes": 1024,
            "restart_write_s": 0.2,
            "restart_read_s": 0.1,
            "note": "two accounting routes, kept apart",
        },
        "budget": {
            "measured_unit_cost_s": {"p0": 1.5},
            "jobs_this_matrix": {"p0": 1},
            "headroom": {"note": "E01's own measured unit costs"},
        },
        "artifacts": {"p0.verification.restart": RESTART_EVIDENCE},
        "plots": [],
        "acceptance_notes": ["synthetic truth only"],
        "next_stage": "E02 is permitted on an accepted oil-water forward scope only",
    }
    built = World(
        root=root, paths=ProjectPaths.default(root), payload=payload, accepted_commit=accepted
    )
    built.rehash()
    return built


# --------------------------------------------------------------------------------------
# what the gate accepts
# --------------------------------------------------------------------------------------


def test_accepted_e01_evidence_is_mapped_into_the_dependency_protocol(world: World) -> None:
    evidence = require_e01_ow(world.publish(), world.paths)
    assert evidence["schema_version"] == E01_OW_EVIDENCE_SCHEMA_VERSION
    assert evidence["accepted_commit"] == world.accepted_commit
    assert evidence["ow_status"] == "PASS"
    assert evidence["benchmark_ref"] == BENCHMARK_EVIDENCE
    assert evidence["restart_ref"] == RESTART_EVIDENCE
    assert evidence["report_ref"] == f"{REPORT_RUN}/E01.json"
    checks = evidence["checks"]
    assert isinstance(checks, tuple | list)
    assert {str(row["check_id"]) for row in checks} == set(MANDATORY_CHECKS)
    assert all(row["status"] == "PASS" for row in checks)
    # The code tree and the locks are named by content, not by the report's own adjectives.
    assert isinstance(evidence["code_tree_hash"], str)
    assert len(str(evidence["code_tree_hash"])) == 40
    assert len(str(evidence["lock_hash"])) == 64


def test_black_oil_not_run_does_not_block_the_oil_water_scope(world: World) -> None:
    assert world.payload["bo_status"] == "NOT_RUN"
    evidence = require_e01_ow(world.publish(), world.paths)
    assert evidence["ow_status"] == "PASS"


def test_benchmark_group_inside_verify_physics_is_the_current_measured_cost_ref(
    world: World,
) -> None:
    benchmark = {"warm41": {"n": 5, "mean_s": 4.2}}
    _write(
        world.root,
        f"{P0_RUN}/e01_suite.json",
        json.dumps({"suite": "p1", "benchmark": benchmark}),
    )
    world.payload["commands"] = [_command(P0_RUN, "verify-physics", world.accepted_commit)]
    world.payload["benchmark"] = {"p1": benchmark}
    evidence = require_e01_ow(world.publish(), world.paths)
    assert evidence["benchmark_ref"] == f"{P0_RUN}/e01_suite.json"


def test_current_restart_job_directory_resolves_to_its_checkpoint_manifest(world: World) -> None:
    job_dir = f"{P0_RUN}/job-a1"
    manifest = f"{job_dir}/checkpoint/restart_manifest.json"
    _write(world.root, manifest, '{"completed_report_step": 3}\n')
    world.payload["restart_round_trip"]["evidence_paths"] = [job_dir]
    world.payload["checks"] = [
        {**check, "evidence_paths": [job_dir]} if check["name"] == "restart_round_trip" else check
        for check in world.payload["checks"]
    ]
    evidence = require_e01_ow(world.publish(), world.paths)
    assert evidence["restart_ref"] == manifest


def test_a_file_e02_adds_of_its_own_is_not_a_change_to_e01_physics(world: World) -> None:
    """E02 writes its own configuration and its own modules; neither is E01's physics."""
    _write(world.root, "configs/e02.yml", 'spec_version: "4.0"\n')
    _write(world.root, "src/so_recon/inference/contracts.py", "# E02\n")
    assert require_e01_ow(world.publish(), world.paths)["ow_status"] == "PASS"


def test_a_full_pass_is_accepted_too(world: World) -> None:
    world.payload["status"] = "PASS"
    world.payload["bo_status"] = "PASS"
    assert require_e01_ow(world.publish(), world.paths)["ow_status"] == "PASS"


# --------------------------------------------------------------------------------------
# what the gate refuses
# --------------------------------------------------------------------------------------


def test_a_pass_without_a_restart_row_is_refused(world: World) -> None:
    """The claimed status is not the evidence; the restart row is."""
    world.payload["restart_round_trip"] = {}
    with pytest.raises(E01DependencyError, match="restart"):
        require_e01_ow(world.publish(), world.paths)


def test_a_restart_row_that_names_no_artifact_is_refused(world: World) -> None:
    world.payload["restart_round_trip"]["evidence_paths"] = []
    with pytest.raises(E01DependencyError, match="restart"):
        require_e01_ow(world.publish(), world.paths)


def test_evidence_whose_recorded_sha_no_longer_matches_the_artifact_is_refused(
    world: World,
) -> None:
    (world.root / "configs" / "e01_tolerances.yml").write_text(
        "schema_version: e01-tolerances-2\nbalance_cumulative_relative_max: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(E01DependencyError, match="configs/e01_tolerances.yml"):
        require_e01_ow(world.publish(), world.paths)


def test_a_changed_lock_is_refused(world: World) -> None:
    (world.root / "uv.lock").write_text("version = 1\n# scipy\n", encoding="utf-8")
    with pytest.raises(E01DependencyError, match="uv.lock"):
        require_e01_ow(world.publish(), world.paths)


def test_a_failed_mandatory_check_is_refused(world: World) -> None:
    world.payload["checks"] = [
        _check(row["name"], status="FAIL") if row["name"] == "five_spot_refinement" else row
        for row in world.payload["checks"]
    ]
    with pytest.raises(E01DependencyError, match="five_spot_refinement"):
        require_e01_ow(world.publish(), world.paths)


def test_a_missing_mandatory_check_is_refused(world: World) -> None:
    world.payload["checks"] = [
        row for row in world.payload["checks"] if row["name"] != "worker_isolation"
    ]
    with pytest.raises(E01DependencyError, match="worker_isolation"):
        require_e01_ow(world.publish(), world.paths)


def test_a_resource_failure_is_never_a_pass(world: World) -> None:
    """SPEC 18.4: a job the memory guard stopped did not produce a physical result."""
    world.payload["jobs"] = [_job("p0_bl64"), _job("p1_world41", status="RESOURCE_FAILURE")]
    with pytest.raises(E01DependencyError, match="RESOURCE_FAILURE"):
        require_e01_ow(world.publish(), world.paths)


def test_a_stage_that_never_ran_is_refused(world: World) -> None:
    world.payload["status"] = "NOT_RUN"
    world.payload["ow_gate"] = "NOT_RUN"
    with pytest.raises(E01DependencyError, match="NOT_RUN"):
        require_e01_ow(world.publish(), world.paths)


def test_a_failed_black_oil_capability_is_refused(world: World) -> None:
    world.payload["bo_status"] = "FAIL"
    with pytest.raises(E01DependencyError, match="black oil|bo_status"):
        require_e01_ow(world.publish(), world.paths)


def test_a_dirty_working_tree_publishes_no_identified_code_state(world: World) -> None:
    world.payload["git_dirty"] = True
    with pytest.raises(E01DependencyError, match="dirty"):
        require_e01_ow(world.publish(), world.paths)


def test_evidence_without_a_commit_is_refused(world: World) -> None:
    world.payload["git_commit"] = None
    with pytest.raises(E01DependencyError, match="commit"):
        require_e01_ow(world.publish(), world.paths)


def test_an_evidence_commit_outside_the_current_ancestry_is_refused(world: World) -> None:
    _git(world.root, "checkout", "-q", "-b", "side", world.accepted_commit)
    (world.root / "README.md").write_text("# a different history\n", encoding="utf-8")
    _git(world.root, "add", "-A")
    _git(world.root, "commit", "-q", "-m", "side history")
    side = _git(world.root, "rev-parse", "HEAD")
    _git(world.root, "checkout", "-q", "-")
    world.payload["git_commit"] = side
    for record in world.payload["commands"]:
        record["git_commit"] = side
    with pytest.raises(E01DependencyError, match="ancestor"):
        require_e01_ow(world.publish(), world.paths)


def test_a_change_in_the_physics_tree_since_the_evidence_is_refused(world: World) -> None:
    assert "src/so_recon/simulator" in PHYSICS_LINEAGE_PATHS
    (world.root / "src" / "so_recon" / "simulator" / "forward.py").write_text(
        "# forward, edited after the evidence was published\n", encoding="utf-8"
    )
    with pytest.raises(E01DependencyError, match="src/so_recon/simulator/forward.py"):
        require_e01_ow(world.publish(), world.paths)


def test_a_changed_e01_configuration_since_the_evidence_is_refused(world: World) -> None:
    (world.root / "configs" / "e01.yml").write_text(
        'spec_version: "4.0"\nconfig_version: "E01.2"\n', encoding="utf-8"
    )
    with pytest.raises(E01DependencyError, match="configs/e01.yml"):
        require_e01_ow(world.publish(), world.paths)


def test_a_new_untracked_physics_file_since_the_evidence_is_refused(world: World) -> None:
    _write(world.root, "julia/extra_physics.jl", "# added after the evidence\n")
    with pytest.raises(E01DependencyError, match="julia/extra_physics.jl"):
        require_e01_ow(world.publish(), world.paths)


def test_an_absolute_evidence_path_is_refused(world: World) -> None:
    """A published record states project-relative paths; an absolute one is a fingerprint."""
    world.payload["checks"] = [
        {**row, "evidence_paths": [f"/private{RESTART_EVIDENCE}"]}
        if row["name"] == "restart_round_trip"
        else row
        for row in world.payload["checks"]
    ]
    with pytest.raises(E01DependencyError, match="e01-ow-dependency-1"):
        require_e01_ow(world.publish(), world.paths)


def test_a_referenced_artifact_that_is_not_on_disk_is_refused(world: World) -> None:
    (world.root / BENCHMARK_EVIDENCE).unlink()
    with pytest.raises(E01DependencyError, match="e01_benchmark.json"):
        require_e01_ow(world.publish(), world.paths)


def test_a_report_without_a_benchmark_run_is_refused(world: World) -> None:
    world.payload["commands"] = [_command(P0_RUN, "verify-physics", world.accepted_commit)]
    with pytest.raises(E01DependencyError, match="benchmark"):
        require_e01_ow(world.publish(), world.paths)


def test_a_missing_evidence_file_is_refused(world: World) -> None:
    with pytest.raises(E01DependencyError, match="not found|no such"):
        require_e01_ow(world.root / REPORT_RUN / "absent.json", world.paths)


def test_a_malformed_evidence_file_is_refused(world: World) -> None:
    path = world.publish()
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(E01DependencyError, match="JSON|json"):
        require_e01_ow(path, world.paths)


def test_evidence_from_another_schema_is_refused(world: World) -> None:
    world.payload["schema_version"] = "e01-stage-2"
    with pytest.raises(E01DependencyError, match="e01-stage-1"):
        require_e01_ow(world.publish(), world.paths)


def test_an_incomplete_machine_report_is_refused(world: World) -> None:
    """A hand-written document that only claims to be the report is not the report."""
    del world.payload["costs"]
    with pytest.raises(E01DependencyError, match="machine report|costs"):
        require_e01_ow(world.publish(), world.paths)


def test_evidence_outside_the_repository_is_refused(world: World, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "E01.json"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text(json.dumps(world.payload), encoding="utf-8")
    with pytest.raises(E01DependencyError, match="outside"):
        require_e01_ow(outside, world.paths)
