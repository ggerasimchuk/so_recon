"""One separately budgeted native SMC start against the converged reduced reference."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from so_recon.config.inference import InferenceConfig
from so_recon.config.resources import P1_LOOP_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.inference.contracts import ObservationBundle
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.reduced_inverse import ReducedDesign
from so_recon.validation.e01_dependency import require_e01_ow
from so_recon.validation.reduced_smc import run_reduced_smc
from so_recon.validation.reference_inverse import reference_artifact_from_path

ROOT = Path(__file__).resolve().parents[2]


def _bounded_probe(session: Path) -> Callable[[], ResourceSnapshot]:
    def probe() -> ResourceSnapshot:
        return probe_resources(None, session).model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "swap_used_bytes": 0,
            }
        )

    return probe


def _required_path(name: str, paths: ProjectPaths) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"{name} is required for the native reduced SMC gate")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else paths.root / candidate


def _required_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"{name} is required for the native reduced SMC gate")
    try:
        return int(raw)
    except ValueError as exc:
        pytest.fail(f"{name} must be an integer: {exc}")


def _julia() -> Path:
    try:
        return find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"native reduced SMC requires Julia: {exc}")


@pytest.mark.julia
def test_reduced_native_smc_reaches_beta_one() -> None:
    paths = ProjectPaths.default(ROOT)
    paths.ensure_dirs()
    dependency = require_e01_ow(_required_path("E02_E01_REPORT", paths), paths)
    reference_path = _required_path("E02_REDUCED_REFERENCE", paths)
    reference_ref = reference_artifact_from_path(reference_path, paths)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference.get("schema_version") != "e02-reduced-reference-3"
        or reference.get("diagnostic_protocol") != "e02-reduced-reference-diagnostic-v1"
        or reference.get("status") != "CONVERGED"
    ):
        pytest.fail("E02_REDUCED_REFERENCE must be the converged diagnostic-v1 authority")
    design = ReducedDesign.model_validate(reference["design"])
    observations = ObservationBundle.model_validate(reference["observations"])
    n_particles = _required_int("E02_REDUCED_SMC_PARTICLES")
    seed = _required_int("E02_REDUCED_SMC_SEED")
    if n_particles not in {32, 64} or seed not in {11, 12}:
        pytest.fail("the registered reduced SMC matrix is N=32/64 and seed=11/12")
    session_id = os.environ.get(
        "E02_REDUCED_SMC_SESSION_ID", f"e02-reduced-smc-n{n_particles}-s{seed}"
    )
    session = paths.artifacts / session_id
    probe = _bounded_probe(session)
    ledger = BudgetLedger.start(
        profile=P1_LOOP_PROFILE,
        path=session / "ledger.json",
        session_id=session_id,
        probe=probe,
    )
    config = InferenceConfig(
        n_particles=n_particles,
        cess_fraction=0.8,
        resample_fraction=0.5,
        max_beta_steps=24,
        moves_per_level=2,
        seed=seed,
        rw_scale=0.25,
        pcn_scale=0.2,
    )
    raw_inputs = {
        "e01_dependency": sha256_json(dependency),
        "reduced_reference": reference_ref.sha256,
    }
    main = RunContext.start(
        command="e02-reduced-smc",
        argv=[],
        cfg=None,
        paths=paths,
        parent_run_ids=(reference_ref.producer_run_id,),
        raw_input_hashes=raw_inputs,
        schema_versions={"reduced_smc": "e02-reduced-smc-run-1"},
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
            summary_ref = run_reduced_smc(
                ctx=main,
                reference_ref=reference_ref,
                design=design,
                observations=observations,
                worker=worker,
                ledger=ledger,
                run_factory=run_factory,
                config=config,
            )
        summary = json.loads(paths.resolve(summary_ref.path).read_text(encoding="utf-8"))
        complete = summary["algorithm_status"] == "COMPLETE" and summary["beta"] == 1.0
        main.finish("PASS" if complete else "FAIL")
        assert complete, summary
    except BaseException as exc:
        if main.record.status == "RUNNING":
            main.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
        raise
