"""The executable border with E02 for E03's scientific scope (plan E03 §2.2).

`require_e02_inverse` is the only place in E03 that may say «the E02 probabilistic
inverse is accepted». Like `require_e01_ow` before it, it takes nothing on a payload's
word: it re-derives the verdict from the machine artifacts the E02 commands published —
per-run `e02_status.json`, `noise_recovery.json`, `physical_experiment.json`,
`physical_smc.json`, `posterior_bundle.json` and `physical_smc_comparison.json` — and
verifies the sha256 of every file it cites.

What the gate must prove, in order (plan §2.2):

1. the accepted E01 oil-water evidence, through `require_e01_ow` itself;
2. the reduced reference and reduced-SMC diagnostics accepted by the E02 stage report
   machinery (`build_e02_report` over the supplied run dirs);
3. a noise-recovery report of the correct scope — 200 replicates, zero new forwards —
   whose published parent ArtifactRefs still re-hash;
4. the remaining native matrix: every registered (experiment, N, seed) cell COMPLETE at
   beta = 1 with a posterior bundle, plus a passing convergence comparison per
   experiment;
5. the recorded source/lock lineage: a clean commit recorded by the run record of every
   cited run directory, physics tree unchanged since the evidence was taken,
   environment lock still current.

Missing gitignored evidence is an explainable refusal, never a silent pass, and this
module never writes placeholder metrics. Unit tests drive it with synthetic fixtures
under `tmp_path`; scientific evidence is what the real runs publish.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from so_recon.config.schema import StrictModel
from so_recon.inference.contracts import E01DependencyError
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import environment_lock_hash
from so_recon.simulator.contracts import RelativePath, Sha256
from so_recon.validation.e01_dependency import require_e01_ow
from so_recon.validation.e02_report import build_e02_report
from so_recon.validation.noise_recovery import NOISE_RECOVERY_SCHEMA
from so_recon.validation.physical_smc import (
    PHYSICAL_EXPERIMENT_SCHEMA,
    PHYSICAL_SMC_RUN_SCHEMA,
)

E02_INVERSE_EVIDENCE_SCHEMA_VERSION: Literal["e02-inverse-dependency-1"] = (
    "e02-inverse-dependency-1"
)

#: The E02 native matrix this gate closes (plan §2.1): four registered setups ×
#: N32/N64 × inference seeds 11/12 = 16 posterior runs. The superseded T2-v1 is not a
#: cell and never re-enters as one.
NATIVE_MATRIX_EXPERIMENTS: tuple[str, ...] = (
    "e02-t1-v1-s141",
    "e02-t1-v1-s142",
    "e02-t2-v2-s143",
    "e02-t4-v1-s144",
)
NATIVE_PARTICLE_COUNTS: tuple[int, ...] = (32, 64)
NATIVE_INFERENCE_SEEDS: tuple[int, ...] = (11, 12)

#: The noise-recovery scope the E02 report committed to: 200 independent replicates on
#: one immutable physical bank, zero new forwards. A shorter rerun is a different scope.
NOISE_RECOVERY_REPLICATES = 200

#: The parts of the tree whose change invalidates a PHYSICAL inverse verdict: solver,
#: case construction, latent priors/renderers, likelihood stack and the E02 configs.
#: E03's own modules (ml/, validation/e03_*, configs/e03*) are deliberately absent: an
#: E03 addition is not a change to E02's physics, and re-running the matrix at the new
#: commit is what re-establishes the lineage, not widening the unchanged set.
E02_PHYSICS_LINEAGE_PATHS: tuple[str, ...] = (
    "julia",
    "src/so_recon/simulator",
    "src/so_recon/synthetic",
    "src/so_recon/geology",
    "src/so_recon/inference",
    "src/so_recon/observation",
    "src/so_recon/validation/physical_smc.py",
    "src/so_recon/validation/reduced_smc.py",
    "src/so_recon/validation/reference_inverse.py",
    "src/so_recon/validation/noise_recovery.py",
    "configs/e02.yml",
    "configs/e02_experiments.json",
)

GIT_TIMEOUT_S = 60


class NativeCell(StrictModel):
    """One cell of the native matrix, as the gate verified it."""

    experiment_id: str
    n_particles: int = Field(ge=2)
    seed: int = Field(ge=0)
    run_path: RelativePath
    algorithm_status: Literal["COMPLETE"]
    beta: float = Field(ge=1.0, le=1.0)


class E02InverseEvidence(StrictModel):
    """The `e02-inverse-dependency-1` document: what E03 may assume about E02."""

    schema_version: Literal["e02-inverse-dependency-1"] = E02_INVERSE_EVIDENCE_SCHEMA_VERSION
    ow: dict[str, Any]
    stage_status: Literal["PASS"]
    stage_checks: dict[str, bool]
    noise_recovery_path: RelativePath
    noise_recovery_replicates: int = Field(ge=1)
    experiments: dict[str, RelativePath]
    native_cells: tuple[NativeCell, ...]
    comparisons: dict[str, RelativePath]
    accepted_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_hash: Sha256
    artifacts: tuple[str, ...]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _refuse(reason: str) -> E01DependencyError:
    return E01DependencyError(reason)


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise _refuse(
            f"E02 {role} not found at {path}: gitignored run evidence is not "
            "present on this checkout. Re-run the E02 commands that publish it "
            "and point the gate at their run directories"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(f"{path}: {role} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _refuse(f"{path}: {role} must be a JSON object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, *, path: Path, role: str) -> None:
    declared = payload.get("schema_version")
    if declared != expected:
        raise _refuse(f"{path}: {role} declares schema_version {declared!r}, expected {expected!r}")


def _verify_ref(ref: Mapping[str, Any], paths: ProjectPaths, *, role: str) -> Path:
    """Re-hash a cited ArtifactRef: a swapped or truncated file is a refusal."""
    try:
        artifact = ArtifactRef.model_validate(ref)
    except ValidationError as exc:
        raise _refuse(f"{role} does not carry a valid ArtifactRef: {exc}") from exc
    try:
        path = paths.resolve(artifact.path)
    except ValueError as exc:
        raise _refuse(f"{role} points outside the repository at {artifact.path!r}") from exc
    if not path.is_file():
        raise _refuse(
            f"{role} cites {artifact.path}, which is absent: the parent evidence of the "
            "native matrix (experiments, truth forwards, checkpoints) travels with the "
            "gitignored run tree and is missing on this checkout"
        )
    digest = sha256_file(path)
    if digest != artifact.sha256:
        raise _refuse(
            f"{role}: {artifact.path} changed since it was published "
            f"(recorded {artifact.sha256[:12]}…, found {digest[:12]}…): republish the "
            "evidence at the current sources instead of mixing lineages"
        )
    return path


def _git(paths: ProjectPaths, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=paths.root,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )


def _require_clean_lineage(run_dirs: Sequence[Path], paths: ProjectPaths) -> str:
    """The physics tree today is the tree the physical runs were executed on."""
    commits: set[str] = set()
    for run_dir in run_dirs:
        record = _read_json(run_dir / "run.json", role="run record")
        if record.get("git_dirty") is not False:
            raise _refuse(
                f"{run_dir}/run.json: evidence recorded on a dirty tree (git_dirty="
                f"{record.get('git_dirty')!r}); a dirty run cannot identify its sources"
            )
        commit = record.get("git_commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise _refuse(f"{run_dir}/run.json: no identified 40-hex commit")
        commits.add(commit)
    if not commits:
        raise _refuse("no run records to derive the accepted commit from")
    if len(commits) != 1:
        raise _refuse(
            f"physical evidence spans commits {sorted(commits)}: the native matrix must "
            "be one registered session lineage, not a mix"
        )
    commit = next(iter(commits))
    changed = _git(paths, "diff", "--name-only", commit, "--", *E02_PHYSICS_LINEAGE_PATHS)
    if changed.returncode != 0:
        raise _refuse(
            f"the physics tree could not be compared with {commit}: {changed.stderr.strip()}"
        )
    added = _git(
        paths, "ls-files", "--others", "--exclude-standard", "--", *E02_PHYSICS_LINEAGE_PATHS
    )
    if added.returncode != 0:
        raise _refuse(f"untracked physics files could not be listed: {added.stderr.strip()}")
    names = sorted({*changed.stdout.split(), *added.stdout.split()})
    if names:
        raise _refuse(
            f"the E02 physics/config tree changed since the evidence commit {commit[:12]}: "
            f"{names}. Re-run the remaining native matrix at the current commit and "
            "republish; the lineage is not widened to cover E03 edits"
        )
    return commit


def _require_current_lock(target_hashes: Mapping[str, Any], paths: ProjectPaths) -> str:
    recorded = target_hashes.get("lock_hash")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise _refuse(
            "physical run target_hashes carry no lock_hash: the environment "
            "lineage of the native runs cannot be verified"
        )
    current = environment_lock_hash(paths)
    if recorded != current:
        raise _refuse(
            "the environment lock changed since the E02 runs: adding a dependency "
            "(uv.lock/julia Manifest) invalidates the recorded lineage. Re-run the "
            "native matrix under the current lock and republish"
        )
    return recorded


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------


def require_e02_inverse(
    *,
    e01_evidence_path: Path,
    run_dirs: Sequence[Path],
    paths: ProjectPaths,
) -> dict[str, object]:
    """Prove the accepted E02 probabilistic-inverse scope, or raise.

    `run_dirs` are E02 run directories — the `artifacts/runs/<run_id>` folders the E02
    commands wrote. The gate itself selects what each stage of the proof needs from
    them; a directory without a recognized artifact is simply not evidence of that
    kind, which keeps the caller honest about pointing at ALL of them.
    """
    ow = require_e01_ow(e01_evidence_path, paths)

    dirs = tuple(dict.fromkeys(Path(d) for d in run_dirs))
    if not dirs:
        raise _refuse(
            "no E02 run directories were supplied: the E03 scientific scope needs the "
            "E02 stage evidence, and absence of evidence is a refusal, not a zero"
        )

    # --- stage status: reduced reference, reduced SMC, toy checks, artifacts, ancestry
    status_dirs = tuple(d for d in dirs if (d / "e02_status.json").is_file())
    report = build_e02_report(status_dirs, paths)
    if report["status"] != "PASS":
        raise _refuse(
            "the E02 stage report over the supplied runs is "
            f"{report['status']}: missing={report['missing_checks']} "
            f"reasons={report['reasons']}"
        )
    stage_checks = {str(k): bool(v) for k, v in report["checks"].items()}

    # --- noise recovery with the committed scope
    noise_candidates: list[Path] = []
    for run_dir in dirs:
        candidate = run_dir / "noise_recovery.json"
        if not candidate.is_file():
            continue
        payload = _read_json(candidate, role="noise recovery")
        _require_schema(payload, NOISE_RECOVERY_SCHEMA, path=candidate, role="noise recovery")
        quadrature = payload.get("quadrature")
        if not isinstance(quadrature, Mapping):
            raise _refuse(
                f"{candidate}: noise recovery carries no quadrature object; the "
                "zero-new-forwards scope cannot be read from the artifact"
            )
        repetitions = payload.get("repetitions")
        new_forwards = quadrature.get("new_physical_forwards")
        if payload.get("status") != "PASS":
            raise _refuse(f"{candidate}: noise recovery status is {payload.get('status')!r}")
        if repetitions != NOISE_RECOVERY_REPLICATES or new_forwards != 0:
            raise _refuse(
                f"{candidate}: noise recovery scope is {repetitions} replicates with "
                f"{new_forwards} new forwards; the accepted evidence is 200 replicates "
                "on the immutable bank. A different scope is a new experiment, not a "
                "replacement dependency"
            )
        parents = payload.get("parents")
        if not isinstance(parents, list) or not parents:
            raise _refuse(
                f"{candidate}: noise recovery publishes no parents ArtifactRefs; the "
                "reduced evidence it was calibrated against cannot be re-verified"
            )
        for index, ref in enumerate(parents):
            if not isinstance(ref, Mapping):
                raise _refuse(
                    f"{candidate}: noise recovery parent #{index} is not a JSON object, "
                    "not a citable ArtifactRef"
                )
            _verify_ref(ref, paths, role=f"noise recovery parent #{index}")
        noise_candidates.append(candidate)
    if not noise_candidates:
        raise _refuse(
            "no noise_recovery.json in the supplied E02 run dirs: the calibration "
            "dependency of plan §2.2 is machine-checked, not assumed from Markdown"
        )
    if len(noise_candidates) > 1:
        raise _refuse(
            f"{len(noise_candidates)} valid noise_recovery.json artifacts were supplied "
            f"({sorted(str(p) for p in noise_candidates)}): the calibration dependency "
            "is one registered 200-replicate session, not a choice between candidates"
        )
    noise_path = noise_candidates[0]

    # --- the native matrix
    runs: list[dict[str, Any]] = []
    run_files: list[Path] = []
    for run_dir in dirs:
        candidate = run_dir / "physical_smc.json"
        if not candidate.is_file():
            continue
        payload = _read_json(candidate, role="physical SMC run")
        _require_schema(payload, PHYSICAL_SMC_RUN_SCHEMA, path=candidate, role="physical SMC run")
        runs.append(payload)
        run_files.append(candidate)
    if not runs:
        raise _refuse(
            "no physical_smc.json artifacts in the supplied run dirs: the native matrix "
            "of plan §2.1 (16 posterior runs) is the dependency E03 must not skip"
        )

    cells: dict[tuple[str, int, int], NativeCell] = {}
    verified_artifacts: set[str] = set()
    experiments: dict[str, Path] = {}
    lock_hashes: set[str] = set()
    for payload, path in zip(runs, run_files, strict=True):
        experiment_id = str(payload.get("experiment_id"))
        n_particles: Any = payload.get("n_particles")
        seed: Any = payload.get("seed")
        try:
            n_value = int(n_particles)
            seed_value = int(seed)
            key = (experiment_id, n_value, seed_value)
            beta = float(payload.get("beta", 0.0))
        except (TypeError, ValueError) as exc:
            raise _refuse(
                f"{path}: physical SMC run carries a malformed cell "
                f"(n_particles={n_particles!r}, seed={seed!r}, "
                f"beta={payload.get('beta')!r}): {exc}"
            ) from exc
        if key in cells:
            raise _refuse(
                f"the native matrix cell {key} appears twice (e.g. {path}): a cell is one "
                "registered run, not a best-of collection"
            )
        if payload.get("algorithm_status") != "COMPLETE" or beta != 1.0:
            raise _refuse(
                f"{path}: algorithm_status={payload.get('algorithm_status')!r} at beta="
                f"{payload.get('beta')!r}; an incomplete tempering is not a posterior and "
                "cannot close the dependency"
            )
        posterior = path.parent / "posterior_bundle.json"
        if not posterior.is_file():
            raise _refuse(
                f"{path}: no posterior_bundle.json beside the run: the published bundle, "
                "not the raw state, is the unit of E02 evidence"
            )
        for ref_role in ("experiment", "checkpoint"):
            if not isinstance(payload.get(ref_role), Mapping):
                raise _refuse(
                    f"{path}: physical SMC run carries no {ref_role} ArtifactRef; the "
                    "parent evidence of a native run cannot be re-verified"
                )
        experiment_path = _verify_ref(
            payload["experiment"], paths, role=f"{experiment_id} experiment"
        )
        _verify_ref(payload["checkpoint"], paths, role=f"{experiment_id} checkpoint")
        target_hashes = payload.get("target_hashes")
        if not isinstance(target_hashes, Mapping):
            raise _refuse(
                f"{path}: physical SMC run target_hashes is not an object; the "
                "environment lineage of the native runs cannot be verified"
            )
        experiments.setdefault(experiment_id, experiment_path)
        lock_hashes.add(_require_current_lock(target_hashes, paths))
        cells[key] = NativeCell(
            experiment_id=experiment_id,
            n_particles=n_value,
            seed=seed_value,
            run_path=paths.relative(path),
            algorithm_status="COMPLETE",
            beta=1.0,
        )
        verified_artifacts.add(paths.relative(path))

    expected = {
        (experiment, n, seed)
        for experiment in NATIVE_MATRIX_EXPERIMENTS
        for n in NATIVE_PARTICLE_COUNTS
        for seed in NATIVE_INFERENCE_SEEDS
    }
    missing = sorted(expected - cells.keys())
    if missing:
        raise _refuse(
            f"the registered native matrix is incomplete: missing cells "
            f"{[(e, n, s) for e, n, s in missing]}. Plan §2.1 fixes the 16-run matrix; a "
            "subset under the same name is not the dependency"
        )

    for experiment_id, experiment_path in experiments.items():
        payload = _read_json(experiment_path, role=f"{experiment_id} physical experiment")
        _require_schema(
            payload, PHYSICAL_EXPERIMENT_SCHEMA, path=experiment_path, role="physical experiment"
        )
        if payload.get("status") != "PASS":
            raise _refuse(
                f"{experiment_path}: physical experiment status is {payload.get('status')!r}"
            )
        truth = payload.get("truth", {})
        for role, ref in (("generator", truth.get("generator")), ("forward", truth.get("forward"))):
            if ref is None:
                raise _refuse(f"{experiment_path}: truth {role} reference is absent")
            _verify_ref(ref, paths, role=f"{experiment_id} truth {role}")
        inference_input = payload.get("inference_input")
        if inference_input is not None:
            _verify_ref(inference_input, paths, role=f"{experiment_id} inference input")
        verified_artifacts.add(paths.relative(experiment_path))

    # --- convergence comparisons per experiment
    comparisons: dict[str, Path] = {}
    for run_dir in dirs:
        candidate = run_dir / "physical_smc_comparison.json"
        if not candidate.is_file():
            continue
        payload = _read_json(candidate, role="physical comparison")
        experiment_ref = payload.get("experiment")
        if not isinstance(experiment_ref, Mapping):
            raise _refuse(
                f"{candidate}: physical comparison carries no experiment ArtifactRef "
                "to attribute its verdict to"
            )
        try:
            parent = ArtifactRef.model_validate(experiment_ref)
        except ValidationError as exc:
            raise _refuse(
                f"{candidate}: physical comparison does not carry a valid experiment "
                f"ArtifactRef: {exc}"
            ) from exc
        matched = next(
            (eid for eid, path in experiments.items() if sha256_file(path) == parent.sha256),
            None,
        )
        if matched is None:
            continue
        if payload.get("status") != "PASS":
            raise _refuse(
                f"{candidate}: convergence screen for {matched} is "
                f"{payload.get('status')!r}; POSTERIOR_NOT_CONVERGED closes no gate"
            )
        comparisons[matched] = candidate
        verified_artifacts.add(paths.relative(candidate))
    lacking = sorted(set(experiments) - comparisons.keys())
    if lacking:
        raise _refuse(
            f"no passing physical_smc_comparison.json for {lacking}: the convergence "
            "screen of the E02 matrix is part of the dependency"
        )

    # --- lineage: every cited run directory must record the same clean session
    lineage_dirs = {
        noise_path.parent,
        *(p.parent for p in run_files),
        *(p.parent for p in experiments.values()),
        *(p.parent for p in comparisons.values()),
    }
    commit = _require_clean_lineage(tuple(sorted(lineage_dirs)), paths)

    try:
        evidence = E02InverseEvidence(
            ow=ow,
            stage_status="PASS",
            stage_checks=stage_checks,
            noise_recovery_path=paths.relative(noise_path),
            noise_recovery_replicates=NOISE_RECOVERY_REPLICATES,
            experiments={eid: paths.relative(path) for eid, path in sorted(experiments.items())},
            native_cells=tuple(cells[key] for key in sorted(cells)),
            comparisons={eid: paths.relative(path) for eid, path in sorted(comparisons.items())},
            accepted_commit=commit,
            lock_hash=next(iter(lock_hashes)),
            artifacts=tuple(sorted(verified_artifacts)),
        )
    except ValidationError as exc:
        raise _refuse(
            f"the verified E02 evidence does not form a valid "
            f"{E02_INVERSE_EVIDENCE_SCHEMA_VERSION} document: {exc}"
        ) from exc
    return evidence.model_dump(mode="json")


__all__ = [
    "E02_INVERSE_EVIDENCE_SCHEMA_VERSION",
    "E02_PHYSICS_LINEAGE_PATHS",
    "E02InverseEvidence",
    "GIT_TIMEOUT_S",
    "NATIVE_INFERENCE_SEEDS",
    "NATIVE_MATRIX_EXPERIMENTS",
    "NATIVE_PARTICLE_COUNTS",
    "NOISE_RECOVERY_REPLICATES",
    "NativeCell",
    "require_e02_inverse",
]
