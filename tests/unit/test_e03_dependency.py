"""Unit tests of the E03 dependency gate on explicitly synthetic fixtures.

No scientific evidence is read here: every fixture is manufactured under `tmp_path`
with real files and real sha256 digests, so the refusals these tests pin are the ones
the gate must produce when actual artifacts are missing, swapped or incomplete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import so_recon.validation.e03_dependency as gate
from so_recon.inference.contracts import E01DependencyError
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.validation.e02_report import REQUIRED_CHECKS

COMMIT = "1" * 40


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _ref_for(path: Path, paths: ProjectPaths, *, schema: str = "fixture-1") -> dict[str, Any]:
    return {
        "artifact_id": sha256_file(path),
        "path": paths.relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": "application/json",
        "schema_version": schema,
        "producer_run_id": "fixture-run",
        "parent_artifact_ids": [],
        "created_at": "2026-09-16T00:00:00+00:00",
    }


class E02Fixture:
    """A minimal but internally consistent E02 evidence tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = ProjectPaths.default(root)
        self.paths.ensure_dirs()
        self.lock_hash = environment_lock_hash(self.paths)
        self.run_dirs: list[Path] = []
        self.experiment_path: dict[str, Path] = {}

    def add_status_run(self) -> Path:
        run_dir = self.root / "artifacts" / "runs" / f"status-{len(self.run_dirs):02d}"
        _write_json(
            run_dir / "e02_status.json",
            {
                "schema_version": "e02-status-1",
                "suite": "math",
                "algorithm_status": "COMPLETE",
                "convergence_status": "PASS",
                "beta": 1.0,
                "checks": {name: True for name in REQUIRED_CHECKS},
                "diagnostics": {"unique_ancestors": 32},
            },
        )
        _write_json(run_dir / "run.json", {"git_commit": COMMIT, "git_dirty": False})
        self.run_dirs.append(run_dir)
        return run_dir

    def add_noise_recovery(self, *, replicates: int = 200, run_id: str = "noise") -> Path:
        run_dir = self.root / "artifacts" / "runs" / run_id
        bank = _write_json(run_dir / "prediction_bank.json", {"fixture": "bank"})
        _write_json(
            run_dir / "noise_recovery.json",
            {
                "schema_version": "e02-noise-recovery-1",
                "status": "PASS",
                "repetitions": replicates,
                "quadrature": {"new_physical_forwards": 0},
                "parents": [_ref_for(bank, self.paths)],
            },
        )
        _write_json(run_dir / "run.json", {"git_commit": COMMIT, "git_dirty": False})
        self.run_dirs.append(run_dir)
        return run_dir

    def add_experiment(self, experiment_id: str) -> Path:
        run_dir = self.root / "artifacts" / "runs" / f"exp-{experiment_id}"
        truth_dir = run_dir / "truth"
        _write_json(truth_dir / "generator.json", {"fixture": experiment_id})
        _write_json(truth_dir / "forward.json", {"fixture": experiment_id})
        _write_json(run_dir / "inference_input.json", {"fixture": "allowed"})
        experiment = _write_json(
            run_dir / "physical_experiment.json",
            {
                "schema_version": "e02-physical-experiment-1",
                "status": "PASS",
                "experiment_id": experiment_id,
                "truth": {
                    "generator": _ref_for(truth_dir / "generator.json", self.paths),
                    "forward": _ref_for(truth_dir / "forward.json", self.paths),
                },
                "inference_input": _ref_for(run_dir / "inference_input.json", self.paths),
            },
        )
        _write_json(run_dir / "run.json", {"git_commit": COMMIT, "git_dirty": False})
        self.experiment_path[experiment_id] = experiment
        self.run_dirs.append(run_dir)
        return experiment

    def add_smc_run(
        self,
        experiment_id: str,
        *,
        n_particles: int,
        seed: int,
        status: str = "COMPLETE",
        beta: float = 1.0,
    ) -> Path:
        run_dir = self.root / "artifacts" / "runs" / f"smc-{experiment_id}-n{n_particles}-s{seed}"
        checkpoint = _write_json(run_dir / "checkpoint" / "manifest.json", {"fixture": "state"})
        _write_json(
            run_dir / "physical_smc.json",
            {
                "schema_version": "e02-physical-smc-run-1",
                "experiment": _ref_for(self.experiment_path[experiment_id], self.paths),
                "experiment_id": experiment_id,
                "design_id": "e02-t1-v1",
                "algorithm_status": status,
                "beta": beta,
                "n_particles": n_particles,
                "seed": seed,
                "unique_ancestors": n_particles,
                "target_hashes": {"lock_hash": self.lock_hash},
                "checkpoint": _ref_for(checkpoint, self.paths),
            },
        )
        _write_json(
            run_dir / "posterior_bundle.json",
            {"schema_version": "e02-posterior-bundle-1", "algorithm_status": status, "beta": beta},
        )
        _write_json(run_dir / "run.json", {"git_commit": COMMIT, "git_dirty": False})
        self.run_dirs.append(run_dir)
        return run_dir

    def add_comparison(self, experiment_id: str, *, status: str = "PASS") -> Path:
        run_dir = self.root / "artifacts" / "runs" / f"cmp-{experiment_id}"
        _write_json(
            run_dir / "physical_smc_comparison.json",
            {
                "schema_version": "e02-physical-smc-comparison-1",
                "status": status,
                "experiment": _ref_for(self.experiment_path[experiment_id], self.paths),
            },
        )
        _write_json(run_dir / "run.json", {"git_commit": COMMIT, "git_dirty": False})
        self.run_dirs.append(run_dir)
        return run_dir

    def build_complete(self) -> E02Fixture:
        self.add_status_run()
        self.add_noise_recovery()
        for experiment_id in gate.NATIVE_MATRIX_EXPERIMENTS:
            self.add_experiment(experiment_id)
            for n in gate.NATIVE_PARTICLE_COUNTS:
                for seed in gate.NATIVE_INFERENCE_SEEDS:
                    self.add_smc_run(experiment_id, n_particles=n, seed=seed)
            self.add_comparison(experiment_id)
        return self


@pytest.fixture()
def stub_ow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "require_e01_ow",
        lambda path, paths: {"ow_status": "PASS", "fixture": True},
    )


@pytest.fixture()
def stub_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_require_clean_lineage", lambda run_dirs, paths: COMMIT)


def _require(fixture: E02Fixture, *, e01: Path | None = None) -> dict[str, Any]:
    return gate.require_e02_inverse(
        e01_evidence_path=e01 or (fixture.root / "E01.json"),
        run_dirs=fixture.run_dirs,
        paths=fixture.paths,
    )


def test_complete_evidence_passes_and_counts_every_cell(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    evidence = _require(fixture)
    assert evidence["stage_status"] == "PASS"
    assert len(evidence["native_cells"]) == 16
    assert set(evidence["comparisons"]) == set(gate.NATIVE_MATRIX_EXPERIMENTS)


def test_no_run_dirs_is_an_explainable_refusal(tmp_path: Path, stub_ow) -> None:
    fixture = E02Fixture(tmp_path)
    with pytest.raises(E01DependencyError, match="no E02 run directories"):
        _require(fixture)


def test_missing_native_artifacts_refuse_instead_of_passing(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path)
    fixture.add_status_run()
    fixture.add_noise_recovery()
    with pytest.raises(E01DependencyError, match="no physical_smc.json"):
        _require(fixture)


def test_incomplete_matrix_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    # drop one cell: remove the N64/seed12 run of the first experiment
    dropped = (
        fixture.root / "artifacts" / "runs" / f"smc-{gate.NATIVE_MATRIX_EXPERIMENTS[0]}-n64-s12"
    )
    (dropped / "physical_smc.json").unlink()
    with pytest.raises(E01DependencyError, match="incomplete"):
        _require(fixture)


def test_intermediate_beta_never_closes_the_gate(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    run_dir = (
        fixture.root / "artifacts" / "runs" / f"smc-{gate.NATIVE_MATRIX_EXPERIMENTS[0]}-n64-s12"
    )
    payload = json.loads((run_dir / "physical_smc.json").read_text(encoding="utf-8"))
    payload["beta"] = 0.5
    _write_json(run_dir / "physical_smc.json", payload)
    with pytest.raises(E01DependencyError, match="algorithm_status='COMPLETE' at beta=0.5"):
        _require(fixture)


def test_swapped_experiment_file_is_refused_by_hash(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    experiment = fixture.experiment_path[gate.NATIVE_MATRIX_EXPERIMENTS[0]]
    payload = json.loads(experiment.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    _write_json(experiment, payload)
    with pytest.raises(E01DependencyError, match="changed since it was published"):
        _require(fixture)


def test_noise_recovery_with_wrong_scope_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    (fixture.root / "artifacts" / "runs" / "noise" / "noise_recovery.json").unlink()
    fixture.add_noise_recovery(replicates=50)
    with pytest.raises(E01DependencyError, match="scope"):
        _require(fixture)


def test_substituted_noise_recovery_parent_is_refused(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    bank = fixture.root / "artifacts" / "runs" / "noise" / "prediction_bank.json"
    _write_json(bank, {"fixture": "swapped"})
    with pytest.raises(E01DependencyError, match="changed since it was published"):
        _require(fixture)


def test_second_noise_recovery_artifact_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    fixture.add_noise_recovery(run_id="noise-again")
    with pytest.raises(E01DependencyError, match="noise_recovery.json artifacts were supplied"):
        _require(fixture)


def test_noise_recovery_with_non_object_quadrature_refuses_not_attributeerror(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    report = fixture.root / "artifacts" / "runs" / "noise" / "noise_recovery.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["quadrature"] = None
    _write_json(report, payload)
    with pytest.raises(E01DependencyError, match="quadrature"):
        _require(fixture)


def test_failing_stage_check_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    status = fixture.root / "artifacts" / "runs" / "status-00" / "e02_status.json"
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["checks"]["reduced_reference"] = False
    _write_json(status, payload)
    with pytest.raises(E01DependencyError, match="stage report"):
        _require(fixture)


def test_missing_comparison_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    (
        fixture.root
        / "artifacts"
        / "runs"
        / f"cmp-{gate.NATIVE_MATRIX_EXPERIMENTS[1]}"
        / "physical_smc_comparison.json"
    ).unlink()
    with pytest.raises(E01DependencyError, match="comparison"):
        _require(fixture)


def test_changed_physics_lineage_is_refused(
    tmp_path: Path, stub_ow, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()

    class FakeProc:
        returncode = 0
        stdout = "src/so_recon/synthetic/inverse_designs.py"
        stderr = ""

    monkeypatch.setattr(gate, "_git", lambda paths, *args: FakeProc())
    with pytest.raises(E01DependencyError, match="changed since the evidence commit"):
        _require(fixture)


def test_dirty_run_record_is_refused(tmp_path: Path, stub_ow) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    smc_record = (
        fixture.root
        / "artifacts"
        / "runs"
        / f"smc-{gate.NATIVE_MATRIX_EXPERIMENTS[0]}-n32-s11"
        / "run.json"
    )
    _write_json(smc_record, {"git_commit": COMMIT, "git_dirty": True})
    with pytest.raises(E01DependencyError, match="dirty tree"):
        _require(fixture)


@pytest.mark.parametrize(
    "record",
    [
        "artifacts/runs/noise/run.json",
        "artifacts/runs/cmp-e02-t1-v1-s141/run.json",
        "artifacts/runs/exp-e02-t1-v1-s141/run.json",
    ],
)
def test_dirty_record_of_any_cited_run_dir_is_refused(tmp_path: Path, stub_ow, record: str) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    _write_json(fixture.root / record, {"git_commit": COMMIT, "git_dirty": True})
    with pytest.raises(E01DependencyError, match="dirty tree"):
        _require(fixture)


def test_changed_lock_is_refused(tmp_path: Path, stub_ow, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    smc = (
        fixture.root
        / "artifacts"
        / "runs"
        / f"smc-{gate.NATIVE_MATRIX_EXPERIMENTS[0]}-n32-s11"
        / "physical_smc.json"
    )
    payload = json.loads(smc.read_text(encoding="utf-8"))
    payload["target_hashes"]["lock_hash"] = "f" * 64
    _write_json(smc, payload)
    with pytest.raises(E01DependencyError, match="environment lock changed"):
        _require(fixture)


def _first_smc_payload(fixture: E02Fixture) -> tuple[Path, dict[str, Any]]:
    smc = (
        fixture.root
        / "artifacts"
        / "runs"
        / f"smc-{gate.NATIVE_MATRIX_EXPERIMENTS[0]}-n32-s11"
        / "physical_smc.json"
    )
    return smc, json.loads(smc.read_text(encoding="utf-8"))


def test_smc_run_with_malformed_cell_refuses_not_typeerror(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    smc, payload = _first_smc_payload(fixture)
    del payload["n_particles"]
    _write_json(smc, payload)
    with pytest.raises(E01DependencyError, match="malformed cell"):
        _require(fixture)


def test_smc_run_without_experiment_ref_refuses_not_keyerror(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    smc, payload = _first_smc_payload(fixture)
    del payload["experiment"]
    _write_json(smc, payload)
    with pytest.raises(E01DependencyError, match="carries no experiment ArtifactRef"):
        _require(fixture)


def test_smc_run_with_non_object_target_hashes_refuses(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    smc, payload = _first_smc_payload(fixture)
    payload["target_hashes"] = None
    _write_json(smc, payload)
    with pytest.raises(E01DependencyError, match="target_hashes is not an object"):
        _require(fixture)


def test_comparison_with_invalid_experiment_ref_refuses_not_validationerror(
    tmp_path: Path, stub_ow, stub_lineage
) -> None:
    fixture = E02Fixture(tmp_path).build_complete()
    comparison = (
        fixture.root / "artifacts" / "runs" / "cmp-e02-t1-v1-s141" / "physical_smc_comparison.json"
    )
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["experiment"] = {"path": "artifacts/runs/exp-e02-t1-v1-s141/physical_experiment.json"}
    _write_json(comparison, payload)
    with pytest.raises(E01DependencyError, match="does not carry a valid experiment"):
        _require(fixture)


def test_ow_refusal_propagates(tmp_path: Path, stub_lineage) -> None:
    fixture = E02Fixture(tmp_path).build_complete()

    def refuse(path: Path, paths: ProjectPaths) -> dict[str, Any]:
        raise E01DependencyError("E01 evidence not found at E01.json")

    import so_recon.validation.e03_dependency as dependency

    original = dependency.require_e01_ow
    dependency.require_e01_ow = refuse
    try:
        with pytest.raises(E01DependencyError, match="E01 evidence"):
            _require(fixture)
    finally:
        dependency.require_e01_ow = original
