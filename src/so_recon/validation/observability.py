"""Pre-registered observation-space and state-space ambiguity diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np

from so_recon.inference.contracts import F64, ModelObservations
from so_recon.registry.artifact import ArtifactRef, write_json_artifact
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.worker import PersistentJuliaWorker


class RunFactory(Protocol):
    def __call__(self, command: str, parent_run_ids: tuple[str, ...]) -> RunContext: ...


def ambiguity_metrics(fw_a: F64, fw_b: F64, so_a: F64, so_b: F64) -> dict[str, float]:
    """Report closeness in observed watercut beside separation in physical saturation."""
    first_fw, second_fw, first_so, second_so = (
        np.asarray(values, dtype=np.float64) for values in (fw_a, fw_b, so_a, so_b)
    )
    if first_fw.shape != second_fw.shape or first_so.shape != second_so.shape:
        raise ValueError("each ambiguity pair must share observation and state shapes")
    if first_fw.size == 0 or first_so.size == 0:
        raise ValueError("ambiguity metrics need non-empty observation and state support")
    if not all(np.isfinite(values).all() for values in (first_fw, second_fw, first_so, second_so)):
        raise ValueError("ambiguity metrics do not turn missing values into zero gaps")
    return {
        "mean_absolute_fw_gap": float(np.mean(np.abs(first_fw - second_fw))),
        "max_absolute_so_gap": float(np.max(np.abs(first_so - second_so))),
    }


def compare_pair(
    prediction_a: ModelObservations,
    prediction_b: ModelObservations,
    states_a: F64,
    states_b: F64,
    support: F64,
) -> dict[str, float | int | str]:
    """Compare actual F outputs on mutually observed rows and common report support."""
    keys = sorted(set(prediction_a.fw) & set(prediction_b.fw))
    common = [
        key for key in keys if prediction_a.fw[key] is not None and prediction_b.fw[key] is not None
    ]
    if not common:
        return {"status": "NO_COMMON_DATA", "n_common_fw": 0}
    matrix = np.asarray(support, dtype=np.float64)
    first_state = np.asarray(states_a, dtype=np.float64)
    second_state = np.asarray(states_b, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1:] != first_state.shape
        or first_state.shape != second_state.shape
    ):
        raise ValueError(
            f"support/state shape mismatch: {matrix.shape}, {first_state.shape}, "
            f"{second_state.shape}"
        )
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("report support must be finite and non-negative")
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("every compared report zone must carry positive support")
    zone_a = matrix @ first_state / totals
    zone_b = matrix @ second_state / totals
    metrics = ambiguity_metrics(
        np.asarray([prediction_a.fw[key] for key in common], dtype=np.float64),
        np.asarray([prediction_b.fw[key] for key in common], dtype=np.float64),
        zone_a,
        zone_b,
    )
    demonstrated = (
        metrics["mean_absolute_fw_gap"] <= 0.01 + 1.0e-12
        and metrics["max_absolute_so_gap"] >= 0.05 - 1.0e-12
    )
    return {
        "status": "AMBIGUITY_DEMONSTRATED" if demonstrated else "AMBIGUITY_NOT_DEMONSTRATED",
        "n_common_fw": len(common),
        **metrics,
    }


def run_observability(
    experiment: dict[str, Any],
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
) -> ArtifactRef:
    """Persist metrics for a pair of already completed, explicitly identified F outputs.

    Native pair execution is intentionally not inferred from arbitrary JSON.  Callers must
    supply the two registered predictions and state arrays produced by their bounded jobs;
    this function only evaluates and records them on a common support.
    """
    del ledger
    required = {"experiment_id", "prediction_a", "prediction_b", "states_a", "states_b", "support"}
    missing = sorted(required - experiment.keys())
    if missing:
        raise ValueError(f"observability experiment is missing registered pair fields {missing}")
    prediction_a = ModelObservations.model_validate(experiment["prediction_a"])
    prediction_b = ModelObservations.model_validate(experiment["prediction_b"])
    metrics = compare_pair(
        prediction_a,
        prediction_b,
        np.asarray(experiment["states_a"], dtype=np.float64),
        np.asarray(experiment["states_b"], dtype=np.float64),
        np.asarray(experiment["support"], dtype=np.float64),
    )
    ctx = run_factory("e02-observability", ())
    ref = write_json_artifact(
        ctx.run_dir / "observability.json",
        {
            "schema_version": "e02-observability-1",
            "experiment_id": experiment["experiment_id"],
            "model_hashes": [prediction_a.model_hash, prediction_b.model_hash],
            **metrics,
        },
        worker.paths,
        schema_version="e02-observability-1",
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )
    ctx.finish(
        "PASS" if metrics["status"] == "AMBIGUITY_DEMONSTRATED" else "FAIL",
        outputs={"observability": ref},
    )
    return ref


__all__ = ["ambiguity_metrics", "compare_pair", "run_observability"]
