"""Two-hundred-replicate noise recovery using one immutable physical prediction bank."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from so_recon.inference.contracts import ObservationBundle
from so_recon.observation.bins import rounding_grid
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext, RunRecord
from so_recon.validation.noise_recovery import publish_noise_recovery
from so_recon.validation.reference_inverse import reference_artifact_from_path

ROOT = Path(__file__).resolve().parents[2]


def _required_path(name: str, paths: ProjectPaths) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        pytest.fail(f"{name} is required for the noise recovery gate")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else paths.root / candidate


def _output_ref(path: Path, key: str, paths: ProjectPaths) -> ArtifactRef:
    run = RunRecord.model_validate_json((path.parent / "run.json").read_text(encoding="utf-8"))
    try:
        ref = run.outputs[key]
    except KeyError as exc:
        raise ValueError(f"producer run has no {key!r} output") from exc
    if (
        paths.resolve(ref.path) != path
        or sha256_file(path) != ref.sha256
        or path.stat().st_size != ref.size_bytes
    ):
        raise ValueError(f"{path} failed immutable output identity")
    return ref


def test_noise_recovery_on_fixed_native_prediction_bank() -> None:
    paths = ProjectPaths.default(ROOT)
    reference_path = _required_path("E02_REDUCED_REFERENCE", paths)
    comparison_path = _required_path("E02_REDUCED_SMC_COMPARISON", paths)
    design_path = _required_path("E02_REDUCED_DESIGN_CHECK", paths)
    truth_result_path = _required_path("E02_REDUCED_TRUTH_RESULT", paths)
    reference_ref = reference_artifact_from_path(reference_path, paths)
    comparison_ref = _output_ref(comparison_path, "reduced_smc_comparison", paths)
    design_ref = _output_ref(design_path, "design_check", paths)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (
        comparison.get("status") != "PASS"
        or comparison.get("diagnostic_protocol") != "e02-reduced-smc-diagnostic-n128-v1"
    ):
        pytest.fail("noise recovery requires the passing N128 reduced SMC diagnostic")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    observations = ObservationBundle.model_validate(reference["observations"])
    if (
        design.get("status") != "INFORMATIVE"
        or design.get("observations") != reference["observations"]
    ):
        pytest.fail("design check and converged reference do not name the same observations")
    values = tuple(float(value) for value in design["watercut"])
    if len(values) != 12:
        pytest.fail("fixed physical prediction bank must contain all 12 calendar months")
    prediction = {("P1", month): value for month, value in enumerate(values)}
    ctx = RunContext.start(
        command="e02-noise-recovery",
        argv=[],
        cfg=None,
        paths=paths,
        parent_run_ids=(
            reference_ref.producer_run_id,
            comparison_ref.producer_run_id,
            design_ref.producer_run_id,
        ),
        raw_input_hashes={"truth_result": sha256_file(truth_result_path)},
        schema_versions={"noise_recovery": "e02-noise-recovery-1"},
    )
    try:
        result_ref = publish_noise_recovery(
            ctx=ctx,
            template=observations.history,
            prediction=prediction,
            grid=rounding_grid(0.01),
            truth_result_path=truth_result_path,
            paths=paths,
            parent_refs=(reference_ref, comparison_ref, design_ref),
        )
        ctx.finish("PASS", outputs={"noise_recovery": result_ref})
        payload = json.loads(paths.resolve(result_ref.path).read_text(encoding="utf-8"))
        assert payload["status"] == "PASS", payload
        assert payload["repetitions"] == 200
        assert payload["seeds"] == [8000, 8199]
        assert payload["prediction_bank"]["new_physical_forwards"] == 0
    except BaseException as exc:
        if ctx.record.status == "RUNNING":
            ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
        raise
