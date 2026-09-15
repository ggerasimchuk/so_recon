"""Bounded E02 command bodies and status semantics."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from so_recon.config.schema import ProjectConfig
from so_recon.inference.weights import normalize_log_weights
from so_recon.observation.bins import bin_log_probs, rounding_grid
from so_recon.observation.logs import log_date_mixture
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_json_artifact
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.validation.e02_report import (
    STATUS_SCHEMA_VERSION,
    build_e02_report,
    render_e02_report,
)
from so_recon.validation.toy_inverse import (
    bimodal_reference,
    gaussian_reference,
    prior_predictive_calibration,
    toy_suite,
)


def e02_exit_code(algorithm_status: str, convergence_status: str) -> int:
    """Only a completed algorithm with a passing requested convergence gate is zero."""
    return 0 if algorithm_status == "COMPLETE" and convergence_status == "PASS" else 2


def forecast_forward_calls(n_particles: int, max_beta_steps: int, moves_per_level: int) -> int:
    if n_particles < 1 or max_beta_steps < 1 or moves_per_level < 1:
        raise ValueError("forecast dimensions must be positive")
    return n_particles + n_particles * max_beta_steps * moves_per_level


def _target_summary(run: dict[str, object], name: str) -> dict[str, float]:
    return cast(dict[str, float], run[name])


def _math_acceptance() -> tuple[dict[str, bool], dict[str, Any]]:
    """Execute the frozen pure E02 acceptance matrix used by the unit gate."""
    runs64 = [toy_suite(seed, 64) for seed in range(20)]
    runs32 = [toy_suite(seed, 32) for seed in range(5)]
    gaussian = [_target_summary(run, "gaussian") for run in runs64]
    pooled_mean = float(np.mean([row["mean"] for row in gaussian]))
    pooled_variance = float(
        np.mean([row["variance"] + (row["mean"] - pooled_mean) ** 2 for row in gaussian])
    )
    evidence = np.asarray([row["evidence"] for row in gaussian], dtype=np.float64)
    reference_mean, reference_variance, reference_logz = gaussian_reference(0.0, 1.0, 2.0, 1.0)
    reference_evidence = math.exp(reference_logz)
    gaussian_pass = (
        abs(pooled_mean - reference_mean) <= 0.08 * math.sqrt(reference_variance)
        and abs(pooled_variance / reference_variance - 1.0) <= 0.12
        and abs(float(evidence.mean()) / reference_evidence - 1.0) <= 0.10
        and all(
            run["algorithm_status"] == {"gaussian": "COMPLETE", "bimodal": "COMPLETE"}
            for run in runs64
        )
    )
    family = np.asarray(
        [_target_summary(run, "bimodal")["family_one_probability"] for run in runs64]
    )
    expected_family = cast(tuple[float, ...], bimodal_reference()["family_probabilities"])[1]
    missed_mode_pass = (
        abs(float(family.mean()) - expected_family) <= 0.06 and np.count_nonzero(family > 0.0) >= 18
    )
    paired_pass = [run["seed"] for run in runs32] == [run["seed"] for run in runs64[:5]]

    calibration = prior_predictive_calibration(20260915, repetitions=200, n_particles=32)
    smc_interval = cast(tuple[float, float], calibration["smc_coverage_wilson_95"])
    exact_interval = cast(tuple[float, float], calibration["exact_coverage_wilson_95"])
    generator_pass = smc_interval[0] <= 0.9 <= smc_interval[1]
    generator_pass = generator_pass and exact_interval[0] <= 0.9 <= exact_interval[1]

    grid = rounding_grid(0.01)
    bin_logs = bin_log_probs(grid, mu=0.7, sigma=0.03, nu=5.0)
    normalized_pass = abs(float(logsumexp(bin_logs))) < 1.0e-12
    date_pass = math.isclose(
        log_date_mixture(np.log([0.1, 0.8]), np.array([0.25, 0.75])),
        math.log(0.625),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    rng = np.random.default_rng(221)
    particles = rng.normal(3.0, 1.0, size=100_000)
    corrected, _ = normalize_log_weights(
        norm.logpdf(particles) + norm.logpdf(2.0, particles, 1.0) - norm.logpdf(particles, 3.0, 1.0)
    )
    corrected_mean = float(np.exp(corrected) @ particles)
    measure_pass = abs(corrected_mean - 1.0) <= 0.03
    checks = {
        "likelihood_normalized": bool(normalized_pass),
        "generator_calibrated": bool(generator_pass),
        "date_mixture": bool(date_pass),
        "measure_consistent": bool(measure_pass),
        "toy_reference": bool(gaussian_pass and missed_mode_pass and paired_pass),
    }
    diagnostics: dict[str, Any] = {
        "pooled_gaussian_mean": pooled_mean,
        "pooled_gaussian_variance": pooled_variance,
        "mean_evidence": float(evidence.mean()),
        "evidence_mc_se": float(evidence.std(ddof=1) / math.sqrt(evidence.size)),
        "family_one_probability": float(family.mean()),
        "family_nonzero_runs": int(np.count_nonzero(family > 0.0)),
        "smc_coverage": calibration["smc_coverage"],
        "smc_coverage_wilson_95": smc_interval,
        "exact_coverage": calibration["exact_coverage"],
        "exact_coverage_wilson_95": exact_interval,
        "corrected_shifted_proposal_mean": corrected_mean,
        "unique_ancestors": min(
            int(_target_summary(run, "gaussian")["unique_ancestors"]) for run in runs64
        ),
    }
    return checks, diagnostics


def _publish_status(
    ctx: RunContext,
    paths: ProjectPaths,
    *,
    suite: str,
    algorithm_status: str,
    convergence_status: str,
    beta: float,
    checks: dict[str, bool],
    diagnostics: dict[str, Any],
) -> None:
    ref = write_json_artifact(
        ctx.run_dir / "e02_status.json",
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "suite": suite,
            "algorithm_status": algorithm_status,
            "convergence_status": convergence_status,
            "beta": beta,
            "checks": checks,
            "diagnostics": diagnostics,
        },
        paths,
        schema_version=STATUS_SCHEMA_VERSION,
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )
    ctx.add_output("e02_status", ref)


def run_e02(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    suite: str,
    experiment: str | None = None,
    resume: Path | None = None,
    argv: Sequence[str] = (),
) -> RunContext:
    """Run one explicitly bounded suite; currently only the pure math suite is complete."""
    inference = cfg.inference
    if inference is None:
        raise ValueError("E02 commands require the validated inference block")

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        del log
        if suite == "math":
            checks, diagnostics = _math_acceptance()
            passing = all(checks.values())
            _publish_status(
                ctx,
                paths,
                suite="math",
                algorithm_status="COMPLETE" if passing else "EVALUATION_FAILURE",
                convergence_status="PASS" if passing else "FAIL",
                beta=1.0,
                checks=checks,
                diagnostics=diagnostics,
            )
            return ("PASS" if passing else "FAIL"), []
        note = (
            f"suite {suite!r} is not complete in this code state; experiment={experiment!r}, "
            f"resume={str(resume) if resume else None}. Native work requires fresh E01 evidence."
        )
        _publish_status(
            ctx,
            paths,
            suite=suite,
            algorithm_status="INCOMPLETE_BUDGET",
            convergence_status="NOT_RUN",
            beta=0.0,
            checks={},
            diagnostics={},
        )
        return "FAIL", [note]

    return execute_run(
        command="verify-inverse" if resume is None else "inverse-resume",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"e02_status": STATUS_SCHEMA_VERSION},
    )


def _experiments(paths: ProjectPaths) -> dict[str, Any]:
    path = paths.configs / "e02_experiments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "e02-experiments-1":
        raise ValueError(f"{path}: unsupported experiment schema")
    return cast(dict[str, Any], payload)


def run_inverse_budget(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    experiment: str,
    argv: Sequence[str] = (),
) -> RunContext:
    """Publish a read-only call-count forecast; elapsed seconds stay unmeasured."""
    inference = cfg.inference
    if inference is None:
        raise ValueError("inverse-budget requires the inference configuration")

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        del log
        payload = _experiments(paths)
        known = {item["experiment_id"] for item in payload["experiments"]}
        known.add(payload["reduced_experiment"]["experiment_id"])
        if experiment not in known:
            raise ValueError(f"unknown E02 experiment {experiment!r}; available {sorted(known)}")
        rows = []
        notes = []
        for particles in (32, 64):
            calls = forecast_forward_calls(
                particles, inference.max_beta_steps, inference.moves_per_level
            )
            fits = calls <= 2000
            rows.append({"particles": particles, "worst_case_new_forwards": calls, "fits_p1": fits})
            notes.append(
                f"N{particles}: {calls} worst-case new forwards"
                + ("" if fits else " exceeds P1_LOOP max_new_forward=2000")
            )
        ref = write_json_artifact(
            ctx.run_dir / "inverse_budget.json",
            {
                "schema_version": "e02-budget-1",
                "experiment": experiment,
                "status": "BUDGET_UNMEASURED",
                "reason": (
                    "no accepted warm-forward benchmark was supplied to this read-only forecast"
                ),
                "rows": rows,
            },
            paths,
            schema_version="e02-budget-1",
            producer_run_id=ctx.run_id,
            now=datetime.now(UTC),
        )
        ctx.add_output("inverse_budget", ref)
        return "PASS", notes

    return execute_run(
        command="inverse-budget",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"e02_budget": "e02-budget-1"},
    )


def run_e02_report(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    run_dirs: Sequence[str],
    argv: Sequence[str] = (),
) -> RunContext:
    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        del log
        resolved = tuple(paths.resolve(value) for value in run_dirs)
        report = build_e02_report(resolved, paths)
        json_ref = write_json_artifact(
            ctx.run_dir / "E02.json",
            report,
            paths,
            schema_version="e02-stage-report-1",
            producer_run_id=ctx.run_id,
            now=datetime.now(UTC),
        )
        markdown = render_e02_report(report).encode("utf-8")
        from so_recon.registry.artifact import write_artifact

        md_ref = write_artifact(
            paths.reports / "stages/E02.md",
            markdown,
            paths,
            schema_version="e02-stage-report-1",
            producer_run_id=ctx.run_id,
            media_type="text/markdown",
            now=datetime.now(UTC),
        )
        ctx.add_output("e02_report_json", json_ref)
        ctx.add_output("e02_report_markdown", md_ref)
        return ("PASS" if report["status"] == "PASS" else "FAIL"), list(report["reasons"])

    return execute_run(
        command="e02-report",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"e02_stage_report": "e02-stage-report-1"},
    )


__all__ = [
    "e02_exit_code",
    "forecast_forward_calls",
    "run_e02",
    "run_e02_report",
    "run_inverse_budget",
]
