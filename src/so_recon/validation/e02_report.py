"""Derive the E02 stage boundary from registered suite status artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from so_recon.paths import ProjectPaths

STATUS_SCHEMA_VERSION = "e02-status-1"
STAGE_SCHEMA_VERSION = "e02-stage-report-1"
REQUIRED_CHECKS = frozenset(
    {
        "likelihood_normalized",
        "generator_calibrated",
        "date_mixture",
        "measure_consistent",
        "toy_reference",
        "reduced_reference",
        "native_reduced",
        "p1_t1",
        "p1_t2",
        "p1_t4",
        "artifact_checksums",
        "ancestry",
    }
)


def _load_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "e02_status.json"
    if not path.is_file():
        raise ValueError(f"run {run_dir} has no e02_status.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != STATUS_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported E02 status schema {payload.get('schema_version')!r}")
    required = {"suite", "algorithm_status", "convergence_status", "beta", "checks", "diagnostics"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{path}: status payload is missing {missing}")
    if not isinstance(payload["checks"], dict) or not isinstance(payload["diagnostics"], dict):
        raise ValueError(f"{path}: checks and diagnostics must be mappings")
    return payload


def build_e02_report(run_dirs: tuple[Path, ...], paths: ProjectPaths) -> dict[str, Any]:
    """Combine suite evidence without promoting incomplete mathematics or native work."""
    del paths
    if not run_dirs:
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "status": "NOT_RUN",
            "run_dirs": [],
            "checks": {},
            "missing_checks": sorted(REQUIRED_CHECKS),
            "reasons": ["No verified E02 run artifacts were supplied."],
        }
    statuses = [_load_status(path) for path in run_dirs]
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    for status in statuses:
        checks.update({str(key): bool(value) for key, value in status["checks"].items()})
        beta = float(status["beta"])
        if status["convergence_status"] == "PASS" and beta < 1.0:
            reasons.append(
                f"suite {status['suite']} claims convergence at beta={beta:g}; "
                "beta<1 is intermediate"
            )
        if (
            status["algorithm_status"] == "COMPLETE"
            and "unique_ancestors" not in status["diagnostics"]
        ):
            reasons.append(f"suite {status['suite']} has no unique_ancestors diagnostic")
        if status["algorithm_status"] != "COMPLETE":
            reasons.append(f"suite {status['suite']} algorithm_status={status['algorithm_status']}")
    missing = sorted(name for name in REQUIRED_CHECKS if not checks.get(name, False))
    if missing:
        reasons.append(f"mandatory checks not passing: {missing}")
    stage_status = "PASS" if not reasons and not missing else "FAIL"
    return {
        "schema_version": STAGE_SCHEMA_VERSION,
        "status": stage_status,
        "run_dirs": [str(path) for path in run_dirs],
        "checks": dict(sorted(checks.items())),
        "missing_checks": missing,
        "reasons": reasons,
    }


def render_e02_report(report: dict[str, Any]) -> str:
    """Readable view of the machine verdict; never a second source of status."""
    lines = [
        "# E02 — Probabilistic inverse",
        "",
        f"**Status:** {report['status']}",
        "",
        "This page is a view of registered E02 status artifacts. It is not native evidence.",
        "",
        "## Mandatory checks",
        "",
    ]
    checks = report.get("checks", {})
    if checks:
        lines.extend(
            f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in sorted(checks.items())
        )
    else:
        lines.append("- No checks recorded.")
    lines.extend(["", "## Reasons", ""])
    reasons = report.get("reasons", [])
    lines.extend(f"- {reason}" for reason in reasons or ["No failing reason recorded."])
    return "\n".join(lines) + "\n"


__all__ = [
    "REQUIRED_CHECKS",
    "STAGE_SCHEMA_VERSION",
    "STATUS_SCHEMA_VERSION",
    "build_e02_report",
    "render_e02_report",
]
