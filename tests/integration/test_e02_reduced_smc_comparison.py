"""Publish the immutable two-seed N32/N64 reduced SMC comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.validation.reduced_smc import (
    publish_reduced_smc_comparison,
    reduced_smc_artifact_from_path,
)
from so_recon.validation.reference_inverse import reference_artifact_from_path

ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, paths: ProjectPaths) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else paths.root / candidate


def test_publish_reduced_smc_comparison() -> None:
    paths = ProjectPaths.default(ROOT)
    raw_reference = os.environ.get("E02_REDUCED_REFERENCE")
    raw_runs = os.environ.get("E02_REDUCED_SMC_RUNS")
    if raw_reference is None or raw_runs is None:
        pytest.fail("E02_REDUCED_REFERENCE and E02_REDUCED_SMC_RUNS are required")
    reference_ref = reference_artifact_from_path(_project_path(raw_reference, paths), paths)
    run_paths = tuple(
        _project_path(item.strip(), paths) for item in raw_runs.split(",") if item.strip()
    )
    smc_refs = tuple(reduced_smc_artifact_from_path(path, paths) for path in run_paths)
    raw_counts = os.environ.get("E02_REDUCED_SMC_COUNTS", "32,64")
    counts = tuple(int(item) for item in raw_counts.split(","))
    protocol = os.environ.get("E02_REDUCED_SMC_DIAGNOSTIC_PROTOCOL")
    ctx = RunContext.start(
        command="e02-reduced-smc-comparison",
        argv=[],
        cfg=None,
        paths=paths,
        parent_run_ids=(reference_ref.producer_run_id, *(ref.producer_run_id for ref in smc_refs)),
    )
    try:
        comparison_ref = publish_reduced_smc_comparison(
            ctx=ctx,
            smc_refs=smc_refs,
            reference_ref=reference_ref,
            paths=paths,
            particle_counts=counts,
            diagnostic_protocol=protocol,
        )
        payload = json.loads(paths.resolve(comparison_ref.path).read_text(encoding="utf-8"))
        ctx.finish("PASS" if payload["status"] == "PASS" else "FAIL")
        expected = os.environ.get("E02_REDUCED_SMC_EXPECT", "PASS")
        assert payload["status"] == expected, payload
    except BaseException as exc:
        if ctx.record.status == "RUNNING":
            ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
        raise
