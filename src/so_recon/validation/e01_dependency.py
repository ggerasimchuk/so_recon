"""The executable border with E01: prove the accepted oil-water scope, or refuse.

`require_e01_ow` is the only place in E02 that is allowed to say «E01 is accepted». It
reads the machine report `so-recon e01-report` publishes — `E01.json`, the serialised
`so_recon.validation.e01_report.StageReport`, written beside the Markdown page inside the
report command's own run directory — and maps it into the `e01-ow-dependency-1` evidence
protocol of plan Task 1.

Three rules this module exists to hold.

**Nothing is taken on the report's word.** A payload saying `status: PASS` is a string. The
gate re-derives the verdict from the matrix the report actually carries: every mandatory
check of `so_recon.simulator.commands.MANDATORY_CHECKS` present and passing, a restart row
that names its evidence, a benchmark run that exists, and no job stopped by the memory
guard — SPEC §18.4 makes a resource failure NOT_RUN and never a pass.

**Nothing is invented.** Every protocol field is taken from the report or derived from the
repository by a rule stated here; where E01 publishes no source for one, the gate refuses.
`code_tree_hash` is the clearest case: E01 records `git_commit` and `git_dirty`, so the
code state is identified exactly when the tree was clean, and a report whose tree was dirty
gets a refusal instead of a manufactured digest.

**A newer E02 commit is allowed; an unexplained change is not.** The accepted commit must
be an ancestor of the current HEAD, and the physics, configs and locks must still be the
ones the evidence was taken against — compared by content, file by file. A change is not
fatal to E02, but it is fatal to THIS evidence: the remedy is to re-run the affected E01
native gates at the new commit and republish the machine report, never to widen what counts
as unchanged. Adding a Python dependency rewrites `uv.lock` and therefore invalidates the
recorded lineage exactly like any other lock change; that is the intended behaviour of plan
Task 1.3 and not a special case to exempt.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from so_recon.config.schema import StrictModel
from so_recon.inference.contracts import E01DependencyError
from so_recon.paths import PathEscapeError, ProjectPaths
from so_recon.registry.hashing import sha256_file, sha256_json
from so_recon.simulator.commands import BENCHMARK_FILENAME, MANDATORY_CHECKS
from so_recon.simulator.contracts import RelativePath, Sha256
from so_recon.validation.e01_report import STAGE_REPORT_SCHEMA_VERSION, StageReport

E01_OW_EVIDENCE_SCHEMA_VERSION: Literal["e01-ow-dependency-1"] = "e01-ow-dependency-1"

#: A stage verdict E02 may build on. `PASS_WITH_LIMITATIONS` is what an accepted oil-water
#: scope carries while black oil is NOT_RUN, which plan §1 explicitly permits.
ACCEPTED_STAGE_STATUSES = frozenset({"PASS", "PASS_WITH_LIMITATIONS"})

#: Black oil is a separate capability and E02 does not use it, so NOT_RUN and PASS are both
#: acceptable. A FAIL is a known defect of the dependency, and plan §1 has remaining E01
#: defects fixed before a dependent native gate rather than carried into one.
ACCEPTED_BO_STATUSES = frozenset({"NOT_RUN", "PASS"})

#: The parts of the tree whose change invalidates a physical verdict: the Julia project and
#: solver scripts, the Python code that builds, drives and scores a case, and the E01
#: configuration it ran under. The tolerance block and the job plan are pinned by the
#: hashes the report itself records, and so are the locks; adding an unrelated file —
#: E02's own configuration, E02's own modules — is not a change to E01's physics and is
#: deliberately not listed here.
PHYSICS_LINEAGE_PATHS: tuple[str, ...] = (
    "julia",
    "src/so_recon/simulator",
    "src/so_recon/synthetic",
    "src/so_recon/validation/physics.py",
    "src/so_recon/validation/balance.py",
    "configs/e01.yml",
)

#: How long any single git query may take before the gate refuses instead of hanging.
GIT_TIMEOUT_S = 60


class CheckEvidence(StrictModel):
    """One mandatory check, as the dependency protocol records it."""

    check_id: str = Field(min_length=1)
    status: Literal["PASS", "FAIL", "NOT_RUN"]
    metrics: dict[str, float]
    limits: dict[str, float]
    artifact_refs: tuple[RelativePath, ...]


class E01OwEvidence(StrictModel):
    """The `e01-ow-dependency-1` document: what E02 is allowed to assume about E01."""

    schema_version: Literal["e01-ow-dependency-1"] = E01_OW_EVIDENCE_SCHEMA_VERSION
    accepted_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    code_tree_hash: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_hash: Sha256
    ow_status: Literal["PASS"]
    checks: tuple[CheckEvidence, ...]
    benchmark_ref: RelativePath
    restart_ref: RelativePath
    report_ref: RelativePath


# --------------------------------------------------------------------------------------
# reading the machine report
# --------------------------------------------------------------------------------------


def _load_report(evidence_path: Path, paths: ProjectPaths) -> tuple[StageReport, str]:
    """Read and validate E01's machine report, and place it inside the repository."""
    try:
        report_ref = paths.relative(evidence_path)
    except PathEscapeError as exc:
        raise E01DependencyError(
            f"the E01 evidence {evidence_path} is outside the repository root "
            f"{paths.root}: the gate cites artifacts by project-relative path"
        ) from exc
    if not evidence_path.is_file():
        raise E01DependencyError(
            f"E01 evidence not found at {report_ref}: run `so-recon e01-report --runs ...` "
            "and point the gate at the E01.json it writes in its run directory"
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise E01DependencyError(f"{report_ref} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise E01DependencyError(f"{report_ref}: the E01 machine report is a JSON object")
    declared = payload.get("schema_version")
    if declared != STAGE_REPORT_SCHEMA_VERSION:
        raise E01DependencyError(
            f"{report_ref} declares schema_version {declared!r}; this gate reads E01's "
            f"final machine report, schema {STAGE_REPORT_SCHEMA_VERSION!r}"
        )
    try:
        report = StageReport.model_validate(payload)
    except ValidationError as exc:
        raise E01DependencyError(f"{report_ref} is not a valid E01 machine report: {exc}") from exc
    return report, report_ref


# --------------------------------------------------------------------------------------
# the verdict, re-derived from the matrix
# --------------------------------------------------------------------------------------


def _require_ow_matrix(report: StageReport, report_ref: str) -> tuple[CheckEvidence, ...]:
    if report.status not in ACCEPTED_STAGE_STATUSES:
        raise E01DependencyError(
            f"{report_ref}: E01 stage status is {report.status} ({report.status_reason}); "
            f"E02 may build on {sorted(ACCEPTED_STAGE_STATUSES)} only"
        )
    if report.ow_gate != "PASS":
        raise E01DependencyError(f"{report_ref}: the oil-water gate is {report.ow_gate}, not PASS")
    if report.bo_status not in ACCEPTED_BO_STATUSES:
        raise E01DependencyError(
            f"{report_ref}: black oil reports {report.bo_status}. A NOT_RUN capability is "
            "acceptable to E02 and a failed one is a defect of the dependency, fixed "
            "before a dependent gate rather than carried into it"
        )
    by_name = {check.name: check for check in report.checks}
    missing = sorted(MANDATORY_CHECKS - by_name.keys())
    if missing:
        raise E01DependencyError(
            f"{report_ref}: the mandatory oil-water matrix has no row for {missing}; a "
            "matrix with a hole in it is not a passing matrix"
        )
    not_passing = sorted(
        f"{name}={by_name[name].status}"
        for name in MANDATORY_CHECKS
        if by_name[name].status != "PASS"
    )
    if not_passing:
        raise E01DependencyError(f"{report_ref}: mandatory checks that did not pass: {not_passing}")
    stopped = sorted(
        f"{job.job_id}={job.status}" for job in report.jobs if job.status == "RESOURCE_FAILURE"
    )
    if stopped:
        raise E01DependencyError(
            f"{report_ref}: jobs stopped by the resource guard: {stopped}. SPEC 18.4 makes "
            "a resource failure NOT_RUN and never a pass, so this session is not evidence"
        )
    if report.remaining_job_ids:
        raise E01DependencyError(
            f"{report_ref}: planned jobs were never reached: {sorted(report.remaining_job_ids)}"
        )
    try:
        return tuple(
            CheckEvidence(
                check_id=name,
                status=by_name[name].status,
                metrics=dict(by_name[name].metrics),
                limits=dict(by_name[name].thresholds),
                artifact_refs=tuple(by_name[name].evidence_paths),
            )
            for name in sorted(MANDATORY_CHECKS)
        )
    except ValidationError as exc:
        # The commonest cause is an evidence path that is not project-relative: a machine
        # fingerprint in a published record, which E01 itself relativises on the way out.
        raise E01DependencyError(
            f"{report_ref}: a mandatory check cannot be expressed in the "
            f"{E01_OW_EVIDENCE_SCHEMA_VERSION} protocol: {exc}"
        ) from exc


def _restart_ref(report: StageReport, report_ref: str) -> str:
    """The artifact the restart round trip was scored on, from the report's own row."""
    row: Mapping[str, Any] = report.restart_round_trip
    if not row:
        raise E01DependencyError(
            f"{report_ref}: the report carries no restart_round_trip row. A native restart "
            "is part of the accepted scope, and a status that outruns its own restart "
            "evidence is not evidence"
        )
    if row.get("status") != "PASS":
        raise E01DependencyError(
            f"{report_ref}: the restart_round_trip row is {row.get('status')!r}"
        )
    evidence = row.get("evidence_paths") or ()
    if not isinstance(evidence, list | tuple) or not evidence:
        raise E01DependencyError(
            f"{report_ref}: the restart_round_trip row names no evidence artifact"
        )
    return str(evidence[0])


def _benchmark_ref(report: StageReport, report_ref: str) -> str:
    """The measured-cost artifact, from the benchmark run the report itself cites."""
    runs = [record.run_dir for record in report.commands if record.command == "benchmark-forward"]
    if not runs:
        raise E01DependencyError(
            f"{report_ref}: the report cites no `benchmark-forward` run, so E01's measured "
            "forward cost is not part of this evidence"
        )
    return f"{runs[-1]}/{BENCHMARK_FILENAME}"


# --------------------------------------------------------------------------------------
# lineage: the artifacts, the recorded hashes and the commit
# --------------------------------------------------------------------------------------


def _require_artifact(relative: str, paths: ProjectPaths, *, role: str) -> Path:
    try:
        resolved = paths.resolve(relative)
    except (ValueError, PathEscapeError) as exc:
        raise E01DependencyError(f"{role} path {relative!r} is not usable: {exc}") from exc
    if not resolved.is_file():
        raise E01DependencyError(
            f"{role} {relative} is named by the E01 report but is not on disk; the "
            "evidence cannot be checked against an artifact that is gone"
        )
    return resolved


def _require_recorded_hashes(report: StageReport, paths: ProjectPaths, report_ref: str) -> str:
    """Re-hash everything the report pinned by content, and return the lock digest.

    Refusing on a mismatch is the point: the physical verdict was produced against THESE
    bytes, and a component that has moved since needs its native gate run again.
    """
    recorded: dict[str, str] = {
        report.tolerances_path: report.tolerances_sha256,
        report.job_plan_path: report.job_plan_sha256,
    }
    recorded.update(report.lock_hashes)
    # The published source manifests are keyed by bare filename in the report; the
    # directory they live in comes from ProjectPaths rather than from a literal prefix.
    recorded.update(
        {
            paths.relative(paths.manifests / name): digest
            for name, digest in report.source_manifest.items()
        }
    )
    for relative, digest in sorted(recorded.items()):
        if len(digest) != 64:
            raise E01DependencyError(
                f"{report_ref}: {relative} is recorded as {digest!r} rather than a SHA-256, "
                "so the evidence does not identify the bytes it was taken against"
            )
        actual = sha256_file(_require_artifact(relative, paths, role="pinned input"))
        if actual != digest:
            raise E01DependencyError(
                f"{report_ref}: {relative} now hashes to {actual}, but the E01 evidence was "
                f"taken against {digest}. Re-run the affected E01 native gates at this "
                "commit and republish the machine report"
            )
    return sha256_json(dict(sorted(report.lock_hashes.items())))


def _git(paths: ProjectPaths, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=paths.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise E01DependencyError(
            f"git {' '.join(args)} could not be run in {paths.root}: {exc}. The gate proves "
            "lineage from the repository and refuses rather than assuming it"
        ) from exc


def _require_identified_commit(report: StageReport, report_ref: str) -> str:
    if report.git_dirty:
        raise E01DependencyError(
            f"{report_ref}: the working tree was dirty when E01 ran, so no commit "
            "identifies the code that produced these results. Commit the tree, re-run the "
            "affected native gates and republish the report"
        )
    commit = report.git_commit
    if not commit or len(commit) != 40:
        raise E01DependencyError(
            f"{report_ref}: the report records git_commit {commit!r}; without a commit "
            "there is no code state to prove ancestry against"
        )
    return commit


def _require_ancestry(commit: str, paths: ProjectPaths, report_ref: str) -> str:
    """Prove the E02 checkout descends from the evidence, and return its code tree hash."""
    tree = _git(paths, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if tree.returncode != 0:
        raise E01DependencyError(
            f"{report_ref}: commit {commit} is not in this repository, so its code tree "
            f"cannot be resolved: {tree.stderr.strip()}"
        )
    ancestry = _git(paths, "merge-base", "--is-ancestor", commit, "HEAD")
    if ancestry.returncode != 0:
        raise E01DependencyError(
            f"{report_ref}: {commit} is not an ancestor of HEAD. E02 may run ahead of the "
            "E01 commit, but only on a history that contains it"
        )
    return tree.stdout.strip()


def _require_unchanged_physics(commit: str, paths: ProjectPaths, report_ref: str) -> None:
    """Refuse any difference in the physics tree between the evidence and this checkout.

    Both halves are needed: `git diff` names what moved in tracked files, committed or
    not, and `ls-files --others` names an untracked file that was added beside them.
    """
    changed = _git(paths, "diff", "--name-only", commit, "--", *PHYSICS_LINEAGE_PATHS)
    if changed.returncode != 0:
        raise E01DependencyError(
            f"{report_ref}: the physics tree could not be compared with {commit}: "
            f"{changed.stderr.strip()}"
        )
    added = _git(paths, "ls-files", "--others", "--exclude-standard", "--", *PHYSICS_LINEAGE_PATHS)
    if added.returncode != 0:
        raise E01DependencyError(
            f"{report_ref}: untracked physics files could not be listed: {added.stderr.strip()}"
        )
    names = sorted({*changed.stdout.split(), *added.stdout.split()})
    if names:
        raise E01DependencyError(
            f"{report_ref}: the physics, configs or solver code changed since {commit}: "
            f"{names}. Re-run the affected E01 native gates at this commit and republish "
            "the machine report; the evidence lineage is not widened to cover them"
        )


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------


def require_e01_ow(evidence_path: Path, paths: ProjectPaths) -> dict[str, object]:
    """Prove E01's accepted oil-water scope, or raise `E01DependencyError`.

    Returns the `e01-ow-dependency-1` document as a plain mapping, so a caller can record
    the exact evidence its run stood on. Pure mathematics — the toy and reduced inverse
    tests — never calls this: it is the border for work that claims a PHYSICAL dependency.
    """
    report, report_ref = _load_report(evidence_path, paths)
    checks = _require_ow_matrix(report, report_ref)
    restart_ref = _restart_ref(report, report_ref)
    benchmark_ref = _benchmark_ref(report, report_ref)
    _require_artifact(restart_ref, paths, role="restart evidence")
    _require_artifact(benchmark_ref, paths, role="benchmark evidence")
    for check in checks:
        for artifact in check.artifact_refs:
            _require_artifact(artifact, paths, role=f"{check.check_id} evidence")
    lock_hash = _require_recorded_hashes(report, paths, report_ref)
    commit = _require_identified_commit(report, report_ref)
    code_tree_hash = _require_ancestry(commit, paths, report_ref)
    _require_unchanged_physics(commit, paths, report_ref)
    try:
        evidence = E01OwEvidence(
            accepted_commit=commit,
            code_tree_hash=code_tree_hash,
            lock_hash=lock_hash,
            ow_status="PASS",
            checks=checks,
            benchmark_ref=benchmark_ref,
            restart_ref=restart_ref,
            report_ref=report_ref,
        )
    except ValidationError as exc:
        raise E01DependencyError(
            f"{report_ref}: the evidence does not form a valid "
            f"{E01_OW_EVIDENCE_SCHEMA_VERSION} document: {exc}"
        ) from exc
    return evidence.model_dump(mode="json")


__all__ = [
    "ACCEPTED_BO_STATUSES",
    "ACCEPTED_STAGE_STATUSES",
    "E01_OW_EVIDENCE_SCHEMA_VERSION",
    "PHYSICS_LINEAGE_PATHS",
    "CheckEvidence",
    "E01OwEvidence",
    "require_e01_ow",
]
