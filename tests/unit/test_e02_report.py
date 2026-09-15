"""The E02 report derives its verdict from complete registered evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths
from so_recon.validation.e02_report import build_e02_report, render_e02_report


def _run(root: Path, name: str, payload: dict[str, object]) -> Path:
    run = root / "artifacts/runs" / name
    run.mkdir(parents=True)
    (run / "e02_status.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def test_beta_below_one_cannot_be_reported_as_posterior_pass(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    run = _run(
        root,
        "intermediate",
        {
            "schema_version": "e02-status-1",
            "suite": "reduced",
            "algorithm_status": "COMPLETE",
            "convergence_status": "PASS",
            "beta": 0.9,
            "checks": {},
            "diagnostics": {"unique_ancestors": 32},
        },
    )
    report = build_e02_report((run,), paths)
    assert report["status"] == "FAIL"
    assert "beta" in " ".join(report["reasons"])


def test_missing_native_and_ancestry_evidence_cannot_be_pass(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    run = _run(
        root,
        "math-only",
        {
            "schema_version": "e02-status-1",
            "suite": "math",
            "algorithm_status": "COMPLETE",
            "convergence_status": "PASS",
            "beta": 1.0,
            "checks": {"toy_reference": True},
            "diagnostics": {},
        },
    )
    report = build_e02_report((run,), paths)
    assert report["status"] == "FAIL"
    assert "native_reduced" in report["missing_checks"]
    assert "unique_ancestors" in " ".join(report["reasons"])


def test_report_refuses_unrecognised_status_schema(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    run = _run(root, "bad", {"schema_version": "unknown"})
    with pytest.raises(ValueError, match="schema"):
        build_e02_report((run,), paths)


def test_initial_report_renders_explicit_not_run() -> None:
    markdown = render_e02_report(
        {
            "schema_version": "e02-stage-report-1",
            "status": "NOT_RUN",
            "run_dirs": [],
            "checks": {},
            "missing_checks": [],
            "reasons": ["No verified E02 run artifacts were supplied."],
        }
    )
    assert "# E02" in markdown
    assert "**Status:** NOT_RUN" in markdown
