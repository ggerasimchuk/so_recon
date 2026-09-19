"""One explicitly selected action from the registered T1/T2/T4 native matrix."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from so_recon.config.inference import InferenceConfig
from so_recon.config.resources import P1_LOOP_PROFILE, ResourceProfile, resource_profile
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.validation.e01_dependency import require_e01_ow
from so_recon.validation.physical_smc import (
    PHYSICAL_EXPERIMENT_SCHEMA,
    PHYSICAL_SMC_RUN_SCHEMA,
    prepare_physical_experiment,
    run_physical_smc,
)

ROOT = Path(__file__).resolve().parents[2]


def _resource_probe(session: Path) -> Callable[[], ResourceSnapshot]:
    def probe() -> ResourceSnapshot:
        return probe_resources(None, session)

    return probe


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        pytest.fail(f"{name} is required for the selected physical E02 matrix action")
    return value


def _path(name: str, paths: ProjectPaths) -> Path:
    value = Path(_required(name))
    return value if value.is_absolute() else paths.root / value


def _profile() -> ResourceProfile:
    """The session's resource profile: P1_LOOP unless E02_PHYSICAL_PROFILE overrides it.

    The matrix's own P1_E02_MATRIX preset (six-hour wall budget, COMPUTE §§2, 10 fix the
    rest) is reachable only by naming it here explicitly; an absent variable keeps
    exactly today's P1_LOOP behaviour so nothing already recorded changes meaning. An
    unknown name fails loudly through `resource_profile` rather than falling back to a
    default — a silent fallback would run a six-hour campaign under a one-hour cap, the
    defect this override exists to remove.
    """
    name = os.environ.get("E02_PHYSICAL_PROFILE")
    if name is None:
        return P1_LOOP_PROFILE
    return resource_profile(name)


def _julia() -> Path:
    try:
        return find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"native physical E02 matrix requires Julia: {exc}")


def _experiment() -> dict[str, object]:
    experiment_id = _required("E02_PHYSICAL_EXPERIMENT")
    manifest = json.loads((ROOT / "configs/e02_experiments.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["experiments"] if row["experiment_id"] == experiment_id]
    if len(matches) != 1:
        pytest.fail(f"unknown or repeated registered physical experiment {experiment_id!r}")
    return cast(dict[str, object], matches[0])


@pytest.mark.julia
def test_registered_physical_matrix_action() -> None:
    paths = ProjectPaths.default(ROOT)
    paths.ensure_dirs()
    dependency = require_e01_ow(_path("E02_E01_REPORT", paths), paths)
    action = _required("E02_PHYSICAL_ACTION")
    experiment = _experiment()
    experiment_id = str(experiment["experiment_id"])
    if action not in {"setup", "smc", "resume"}:
        pytest.fail("E02_PHYSICAL_ACTION must be setup, smc or resume")
    if action == "setup":
        suffix = "setup"
        parent_ids: tuple[str, ...] = ()
        parent_ref = None
        resume_ref = None
        parent_ledger_path = None
    else:
        parent_path = _path("E02_PHYSICAL_PARENT", paths)
        parent_run = json.loads((parent_path.parent / "run.json").read_text(encoding="utf-8"))
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        if action == "smc":
            parent_ref = ArtifactRef.model_validate(parent_run["outputs"]["physical_experiment"])
            resume_ref = None
            parent_ledger_path = None
            if paths.resolve(parent_ref.path) != parent_path:
                pytest.fail("E02_PHYSICAL_PARENT does not match its producer run")
            if parent_payload.get("schema_version") != PHYSICAL_EXPERIMENT_SCHEMA:
                pytest.fail("E02_PHYSICAL_PARENT is not a physical experiment")
            particles = int(_required("E02_PHYSICAL_PARTICLES"))
            inference_seed = int(_required("E02_PHYSICAL_INFERENCE_SEED"))
        else:
            if parent_payload.get("schema_version") != PHYSICAL_SMC_RUN_SCHEMA:
                pytest.fail("resume parent is not a physical SMC run")
            parent_ref = ArtifactRef.model_validate(parent_payload["experiment"])
            resume_ref = ArtifactRef.model_validate(parent_payload["checkpoint"])
            previous_ledger = ArtifactRef.model_validate(parent_run["outputs"]["budget_ledger"])
            parent_ledger_path = paths.resolve(previous_ledger.path)
            config_payload = cast(dict[str, object], parent_payload["config"])
            particles = int(cast(int | str, config_payload["n_particles"]))
            inference_seed = int(cast(int | str, config_payload["seed"]))
        if parent_payload.get("experiment_id") != experiment_id:
            pytest.fail("E02_PHYSICAL_PARENT belongs to another experiment")
        if particles not in {32, 64} or inference_seed not in {11, 12}:
            pytest.fail("registered physical SMC matrix is N32/N64 with seeds 11/12")
        suffix = f"{'resume-' if action == 'resume' else ''}n{particles}-s{inference_seed}"
        parent_ids = (
            (resume_ref.producer_run_id, parent_ref.producer_run_id)
            if resume_ref is not None
            else (parent_ref.producer_run_id,)
        )

    session_id = os.environ.get("E02_PHYSICAL_SESSION_ID", f"{experiment_id}-{suffix}")
    session = paths.artifacts / session_id
    probe = _resource_probe(session)
    profile = _profile()
    ledger = (
        BudgetLedger.start(
            profile=profile,
            path=session / "ledger.json",
            session_id=session_id,
            probe=probe,
        )
        if parent_ledger_path is None
        else BudgetLedger.resume(
            parent_path=parent_ledger_path,
            profile=profile,
            path=session / "ledger.json",
            session_id=session_id,
            probe=probe,
        )
    )
    raw_inputs = {
        "e01_dependency": sha256_json(dependency),
        "experiment_manifest": sha256_json(experiment),
    }
    main = RunContext.start(
        command="e02-physical-setup" if action == "setup" else "e02-physical-smc",
        argv=[experiment_id, suffix],
        cfg=None,
        paths=paths,
        parent_run_ids=parent_ids,
        raw_input_hashes=raw_inputs,
        schema_versions={
            "physical": PHYSICAL_EXPERIMENT_SCHEMA if action == "setup" else PHYSICAL_SMC_RUN_SCHEMA
        },
    )

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command,
            argv=[],
            cfg=None,
            paths=paths,
            parent_run_ids=parent_run_ids,
            raw_input_hashes=raw_inputs,
        )

    try:
        with PersistentJuliaWorker(
            _julia(), ROOT / "julia", session, P1_LOOP_PROFILE, paths=paths, probe=probe
        ) as worker:
            if action == "setup":
                ref = prepare_physical_experiment(
                    ctx=main,
                    experiment_id=experiment_id,
                    design_id=str(experiment["design_id"]),
                    truth_seed=int(cast(int | str, experiment["truth_seed"])),
                    worker=worker,
                    ledger=ledger,
                    run_factory=run_factory,
                )
                payload = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
                passed = payload["status"] == "PASS"
            else:
                assert parent_ref is not None
                config = InferenceConfig(
                    n_particles=particles,
                    cess_fraction=0.8,
                    resample_fraction=0.5,
                    max_beta_steps=24,
                    moves_per_level=2,
                    seed=inference_seed,
                    rw_scale=0.25,
                    pcn_scale=0.2,
                )
                ref = run_physical_smc(
                    ctx=main,
                    experiment_ref=parent_ref,
                    worker=worker,
                    ledger=ledger,
                    run_factory=run_factory,
                    config=config,
                    resume_checkpoint_ref=resume_ref,
                )
                payload = json.loads(paths.resolve(ref.path).read_text(encoding="utf-8"))
                passed = payload["algorithm_status"] == "COMPLETE" and payload["beta"] == 1.0
        ledger_ref = register_artifact(
            ledger.path,
            paths,
            media_type="application/json",
            schema_version=ledger.record.schema_version,
            producer_run_id=main.run_id,
            now=datetime.now(UTC),
        )
        main.finish("PASS" if passed else "FAIL", outputs={"session_ledger": ledger_ref})
        assert passed, payload
    except BaseException as exc:
        if main.record.status == "RUNNING":
            main.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
        raise
