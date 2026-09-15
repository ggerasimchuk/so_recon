"""Native SMC execution and independent-reference comparison for reduced-v2."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from so_recon.config.inference import InferenceConfig
from so_recon.inference.checkpoint import save_state
from so_recon.inference.contracts import ObservationBundle, PosteriorBundle, SMCState
from so_recon.inference.smc import infer
from so_recon.inference.target import ReducedPhysicalTarget, RunFactory
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import (
    ArtifactRef,
    register_artifact,
    write_json_artifact,
)
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.reduced_inverse import ReducedDesign, ReducedGaussianPrior
from so_recon.validation.inverse_metrics import weighted_quantile

REDUCED_SMC_RUN_SCHEMA = "e02-reduced-smc-run-1"
REDUCED_SMC_COMPARISON_SCHEMA = "e02-reduced-smc-comparison-1"


def _finite_number(value: object, *, label: str) -> float:
    number = float(cast(float, value))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {number}")
    return number


def _mc_se(values: np.ndarray) -> float:
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("across-start MC SE requires exactly two finite run statistics")
    return float(np.std(values, ddof=1) / math.sqrt(2.0))


def compare_reduced_smc_summaries(
    summaries: Sequence[Mapping[str, object]],
    reference: Mapping[str, object],
) -> dict[str, object]:
    """Apply the pre-registered two-seed N32/N64 reference criteria."""
    reference_mean = _finite_number(reference["mean"], label="reference mean")
    reference_quantiles = np.asarray(reference["quantiles"], dtype=np.float64)
    reference_logz = _finite_number(reference["logz"], label="reference logz")
    if reference_quantiles.shape != (3,) or not np.isfinite(reference_quantiles).all():
        raise ValueError("reference quantiles must contain finite q05/q50/q95")

    by_particles: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for n_particles in (32, 64):
        rows = sorted(
            (row for row in summaries if int(cast(int, row["n_particles"])) == n_particles),
            key=lambda row: int(cast(int, row["seed"])),
        )
        seeds = [int(cast(int, row["seed"])) for row in rows]
        if seeds != [11, 12]:
            raise ValueError(
                f"N{n_particles} comparison requires exactly seeds 11/12, got {seeds}"
            )
        complete = all(
            row.get("algorithm_status") == "COMPLETE"
            and _finite_number(row["beta"], label="beta") == 1.0
            for row in rows
        )
        means = np.asarray([row["mean"] for row in rows], dtype=np.float64)
        quantiles = np.asarray([row["quantiles"] for row in rows], dtype=np.float64)
        log_evidence = np.asarray([row["log_evidence"] for row in rows], dtype=np.float64)
        if (
            means.shape != (2,)
            or quantiles.shape != (2, 3)
            or log_evidence.shape != (2,)
            or not np.isfinite(means).all()
            or not np.isfinite(quantiles).all()
            or not np.isfinite(log_evidence).all()
        ):
            raise ValueError(f"N{n_particles} summaries have invalid statistic shapes or values")
        pooled_mean = float(np.mean(means))
        pooled_quantiles = np.mean(quantiles, axis=0)
        mean_se = _mc_se(means)
        quantile_se = np.asarray([_mc_se(quantiles[:, index]) for index in range(3)])
        mean_tolerance = max(0.05, 3.0 * mean_se)
        quantile_tolerances = np.maximum(0.08, 3.0 * quantile_se)
        mean_pass = abs(pooled_mean - reference_mean) <= mean_tolerance
        quantile_pass = np.abs(pooled_quantiles - reference_quantiles) <= quantile_tolerances
        passed = bool(complete and mean_pass and np.all(quantile_pass))
        checks[f"N{n_particles}"] = passed
        by_particles[str(n_particles)] = {
            "seeds": seeds,
            "algorithm_complete_at_beta_one": complete,
            "pooled_mean": pooled_mean,
            "reference_mean": reference_mean,
            "mean_difference": abs(pooled_mean - reference_mean),
            "mean_mc_se": mean_se,
            "mean_tolerance": mean_tolerance,
            "mean_pass": bool(mean_pass),
            "pooled_quantiles": pooled_quantiles.tolist(),
            "reference_quantiles": reference_quantiles.tolist(),
            "quantile_differences": np.abs(pooled_quantiles - reference_quantiles).tolist(),
            "quantile_mc_se": quantile_se.tolist(),
            "quantile_tolerances": quantile_tolerances.tolist(),
            "quantile_pass": quantile_pass.tolist(),
            "mean_log_evidence": float(np.mean(log_evidence)),
            "log_evidence_mc_se": _mc_se(log_evidence),
            "reference_log_evidence": reference_logz,
            "minimum_unique_ancestors": min(
                int(cast(int, row["unique_ancestors"])) for row in rows
            ),
        }
    return {
        "schema_version": REDUCED_SMC_COMPARISON_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "by_particles": by_particles,
    }


def _histogram_modes(values: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    edges = np.linspace(-7.0, 7.0, 57, dtype=np.float64)
    mass, _ = np.histogram(values, bins=edges, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    local = [
        index
        for index, value in enumerate(mass)
        if value > 0.0
        and value >= (mass[index - 1] if index else 0.0)
        and value >= (mass[index + 1] if index + 1 < mass.size else 0.0)
    ]
    ordered = sorted(local, key=lambda index: (-mass[index], index))
    return {
        "histogram_edges": edges.tolist(),
        "local_mode_centers": [float(centers[index]) for index in ordered],
        "local_mode_masses": [float(mass[index]) for index in ordered],
        "mass_below_reference_grid": float(weights[values < -7.0].sum()),
        "mass_above_reference_grid": float(weights[values > 7.0].sum()),
        "family_probabilities": {"0": 1.0},
    }


def summarize_reduced_smc(
    state: SMCState,
    *,
    config: InferenceConfig,
    ledger: BudgetLedger,
) -> dict[str, object]:
    """Create one immutable, finite diagnostic summary from an SMC state."""
    weights = np.exp(np.asarray(state.log_weights, dtype=np.float64))
    values = np.asarray(
        [particle.evaluation.theta.v[0] for particle in state.particles], dtype=np.float64
    )
    if values.shape != (config.n_particles,) or weights.shape != values.shape:
        raise ValueError("SMC state particle and weight counts do not match its configuration")
    mean = float(weights @ values)
    moves = cast(list[dict[str, object]], state.diagnostics.get("moves", []))
    kernels = sorted({str(move["kernel"]) for move in moves})
    move_diagnostics = {
        kernel: {
            "attempted": sum(str(move["kernel"]) == kernel for move in moves),
            "accepted": sum(
                str(move["kernel"]) == kernel and bool(move["accepted"]) for move in moves
            ),
        }
        for kernel in kernels
    }
    return {
        "schema_version": REDUCED_SMC_RUN_SCHEMA,
        "algorithm_status": state.algorithm_status,
        "beta": state.beta,
        "n_particles": config.n_particles,
        "seed": config.seed,
        "mean": mean,
        "variance": float(weights @ ((values - mean) ** 2)),
        "quantiles": weighted_quantile(
            values, weights, np.asarray([0.05, 0.5, 0.95], dtype=np.float64)
        ).tolist(),
        "log_evidence": state.log_evidence,
        "evidence": math.exp(state.log_evidence),
        "unique_ancestors": len({particle.ancestor_id for particle in state.particles}),
        "beta_history": list(state.diagnostics.get("beta_history", [])),
        "ess_history": list(state.diagnostics.get("ess_history", [])),
        "resampling": list(state.diagnostics.get("resampling", [])),
        "move_diagnostics": move_diagnostics,
        "mode_diagnostics": _histogram_modes(values, weights),
        "budget": {
            "session": ledger.session_totals().model_dump(mode="json"),
            "cumulative": ledger.cumulative_totals().model_dump(mode="json"),
        },
    }


def run_reduced_smc(
    *,
    ctx: RunContext,
    reference_ref: ArtifactRef,
    design: ReducedDesign,
    observations: ObservationBundle,
    worker: PersistentJuliaWorker,
    ledger: BudgetLedger,
    run_factory: RunFactory,
    config: InferenceConfig,
) -> ArtifactRef:
    """Execute and publish one separately budgeted reduced native SMC start."""
    paths = worker.paths
    prior = ReducedGaussianPrior(design)
    target = ReducedPhysicalTarget(
        prior,
        prior,
        design,
        observations,
        worker,
        ledger,
        run_factory,
        parent_run_ids=(ctx.run_id, reference_ref.producer_run_id),
    )
    state = infer(
        target,
        prior,
        prior.schema,
        config,
        ctx.run_dir / "working",
        lambda: False,
    )
    checkpoint_ref = save_state(state, ctx.run_dir / "checkpoint/manifest.json", paths, ctx)
    ledger_ref = register_artifact(
        ledger.path,
        paths,
        schema_version=ledger.record.schema_version,
        producer_run_id=ctx.run_id,
        media_type="application/json",
        now=datetime.now(UTC),
    )
    ctx.add_output("budget_ledger", ledger_ref)
    summary = summarize_reduced_smc(state, config=config, ledger=ledger)
    payload = {
        **summary,
        "reference": reference_ref.model_dump(mode="json"),
        "design": design.model_dump(mode="json"),
        "observations_hash": target.observation_semantic_hash,
        "target_hashes": target.checkpoint_hashes,
        "config": config.model_dump(mode="json"),
        "ledger_sha256": sha256_file(ledger.path),
        "checkpoint": checkpoint_ref.model_dump(mode="json"),
    }
    summary_ref = write_json_artifact(
        ctx.run_dir / "reduced_smc.json",
        payload,
        paths,
        schema_version=REDUCED_SMC_RUN_SCHEMA,
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(reference_ref.artifact_id, checkpoint_ref.artifact_id),
        now=datetime.now(UTC),
    )
    ctx.add_output("reduced_smc", summary_ref)

    manifest = json.loads(paths.resolve(checkpoint_ref.path).read_text(encoding="utf-8"))
    particles_ref = ArtifactRef.model_validate(manifest["shards"]["particles"])
    physical_by_id = {
        ref.artifact_id: ref
        for particle in state.particles
        if (ref := particle.evaluation.forward_ref) is not None
    }
    physical_refs = tuple(physical_by_id[key] for key in sorted(physical_by_id))
    posterior = PosteriorBundle(
        state_ref=checkpoint_ref,
        particles_ref=particles_ref,
        diagnostics_ref=summary_ref,
        ledger_ref=ledger_ref,
        algorithm_status=state.algorithm_status,
        beta=state.beta,
        convergence_status="NOT_ASSESSED",
        physical_state_refs=physical_refs,
        parent_run_ids=(reference_ref.producer_run_id,),
    )
    posterior_ref = write_json_artifact(
        ctx.run_dir / "posterior_bundle.json",
        posterior.model_dump(mode="json"),
        paths,
        schema_version="e02-posterior-bundle-1",
        producer_run_id=ctx.run_id,
        parent_artifact_ids=(
            checkpoint_ref.artifact_id,
            particles_ref.artifact_id,
            summary_ref.artifact_id,
            ledger_ref.artifact_id,
        ),
        now=datetime.now(UTC),
    )
    ctx.add_output("posterior_bundle", posterior_ref)
    return summary_ref


def load_reduced_smc_summary(path: Path, paths: ProjectPaths) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REDUCED_SMC_RUN_SCHEMA:
        raise ValueError(f"{paths.relative(path)} has unsupported reduced SMC schema")
    return cast(dict[str, Any], payload)


__all__ = [
    "REDUCED_SMC_COMPARISON_SCHEMA",
    "REDUCED_SMC_RUN_SCHEMA",
    "compare_reduced_smc_summaries",
    "load_reduced_smc_summary",
    "run_reduced_smc",
    "summarize_reduced_smc",
]
