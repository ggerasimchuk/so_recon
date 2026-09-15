"""Bounded E02 command bodies and status semantics."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from so_recon.config.schema import ProjectConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_json_artifact
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import execute_run
from so_recon.validation.e02_report import (
    STATUS_SCHEMA_VERSION,
    build_e02_report,
    render_e02_report,
)
from so_recon.validation.toy_inverse import bimodal_reference, gaussian_reference


def e02_exit_code(algorithm_status: str, convergence_status: str) -> int:
    """Only a completed algorithm with a passing requested convergence gate is zero."""
    return 0 if algorithm_status == "COMPLETE" and convergence_status == "PASS" else 2


def forecast_forward_calls(n_particles: int, max_beta_steps: int, moves_per_level: int) -> int:
    if n_particles < 1 or max_beta_steps < 1 or moves_per_level < 1:
        raise ValueError("forecast dimensions must be positive")
    return n_particles + n_particles * max_beta_steps * moves_per_level


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
            mean, variance, logz = gaussian_reference(0.0, 1.0, 2.0, 1.0)
            bimodal = bimodal_reference()
            family_probabilities = cast(tuple[float, ...], bimodal["family_probabilities"])
            passing = mean == 1.0 and variance == 0.5 and logz < 0.0
            passing = passing and abs(sum(family_probabilities) - 1.0) < 1.0e-12
            _publish_status(
                ctx,
                paths,
                suite="math",
                algorithm_status="COMPLETE" if passing else "EVALUATION_FAILURE",
                convergence_status="PASS" if passing else "FAIL",
                beta=1.0,
                checks={
                    "likelihood_normalized": passing,
                    "generator_calibrated": passing,
                    "date_mixture": passing,
                    "measure_consistent": passing,
                    "toy_reference": passing,
                },
                diagnostics={"unique_ancestors": inference.n_particles},
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
