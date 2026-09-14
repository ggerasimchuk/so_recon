"""E01.12.9 — the stage validators. They READ a published suite; they never run one.

Every test here opens `e01_suite.json` and the artifacts it names, under the newest run
directory `so-recon verify-physics` left behind. Plan 12.9 is explicit about why: the stage
gate has to cite numbers that a real session published, so a metric that exists only inside
a test process is a metric the gate cannot use — and re-running the suite here would score
a second set of results against the first's conclusions.

They are marked `e01_physics` and SKIP without `--run-e01-physics`, because a plain `pytest`
cycle must not depend on a session nobody asked for. `scripts/e01_gate.sh` runs the suites
and then runs these with the option, on the artifacts those suites saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from so_recon.paths import ProjectPaths, find_repo_root
from so_recon.simulator.commands import MANDATORY_CHECKS, SUITE_REPORT_FILENAME, SuiteReport

pytestmark = pytest.mark.e01_physics

ROOT = find_repo_root(Path(__file__).resolve().parents[2])


def _suite_reports() -> dict[str, tuple[Path, SuiteReport]]:
    """The newest published report of every suite, keyed by suite name."""
    paths = ProjectPaths.default(ROOT)
    found: dict[str, tuple[Path, SuiteReport]] = {}
    if not paths.runs.is_dir():
        return found
    for run_dir in sorted(paths.runs.iterdir()):
        path = run_dir / SUITE_REPORT_FILENAME
        if not path.is_file():
            continue
        report = SuiteReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        found[report.suite] = (run_dir, report)
    return found


@pytest.fixture(scope="module")
def suites() -> dict[str, tuple[Path, SuiteReport]]:
    reports = _suite_reports()
    if not reports:
        pytest.skip(
            "no published E01 suite under artifacts/runs; run `so-recon --config configs/e01.yml "
            "verify-physics --suite p0` first"
        )
    return reports


def test_the_published_suite_is_about_the_tolerances_in_force(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """A verdict scored against a different gate is a verdict about a different question."""
    paths = ProjectPaths.default(ROOT)
    from so_recon.registry.hashing import sha256_file

    for name, (_run_dir, report) in suites.items():
        assert report.tolerances_sha256 == sha256_file(paths.resolve(report.tolerances_path)), (
            f"{name}: the published report was scored against a tolerance block that has "
            "changed since; rerun the suite rather than reading its old numbers"
        )
        assert report.job_plan_sha256 == sha256_file(paths.resolve(report.job_plan_path)), name


def test_every_mandatory_check_of_a_passing_suite_is_present_and_green(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """Plan 12.9: a stage matrix with a hole in it is not a passing matrix."""
    for name, (_run_dir, report) in suites.items():
        if report.exit_code != 0:
            continue
        expected = _expected_mandatory(report)
        present = {c.name for c in report.checks}
        assert expected <= present, f"{name}: missing mandatory checks {sorted(expected - present)}"
        for check in report.checks:
            if check.name in MANDATORY_CHECKS:
                assert check.status == "PASS", f"{name}/{check.name}: {check.reason}"


def _expected_mandatory(report: SuiteReport) -> set[str]:
    plan = json.loads((ROOT / report.job_plan_path).read_text(encoding="utf-8"))
    return {
        job["scored_as"]
        for job in plan["suites"][report.suite]["jobs"]
        if job["scored_as"] in MANDATORY_CHECKS and job.get("deferred_to") is None
    }


def test_every_scored_metric_came_from_a_file_that_is_still_there(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """The evidence a check names is a repository path, and it exists."""
    for name, (_run_dir, report) in suites.items():
        for check in report.checks:
            for evidence in check.evidence_paths:
                assert not Path(evidence).is_absolute(), (
                    f"{name}/{check.name}: evidence path {evidence!r} is absolute; a persisted "
                    "record must not carry a machine path"
                )
                assert (ROOT / evidence).exists(), f"{name}/{check.name}: {evidence} is gone"


def test_a_published_balance_table_carries_both_statements(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """`balances.parquet` holds TWO balances with three leading label columns.

    A validator that assumed one row per component would score half the table, which is
    exactly the mistake the label columns exist to prevent — so the selection is made on the
    `balance` column here rather than on row order.
    """
    checked = 0
    for _name, (_run_dir, report) in suites.items():
        for check in report.checks:
            for evidence in check.evidence_paths:
                path = ROOT / evidence
                if path.name != "balances.parquet" or not path.is_file():
                    continue
                rows = pq.read_table(path).to_pylist()
                labels = {row["balance"] for row in rows}
                assert labels == {"full_system_surface", "reservoir_connections"}, evidence
                for label in labels:
                    components = {r["component"] for r in rows if r["balance"] == label}
                    assert components == {"water", "oil"}, (evidence, label)
                checked += 1
    if checked == 0:
        pytest.skip("no published balance table among the evidence of these suites")


def test_the_planned_jobs_and_the_jobs_that_ran_agree(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """Every declared job either ran, is named as remaining, or is named as deferred."""
    for name, (_run_dir, report) in suites.items():
        ran = {job.job_id for job in report.jobs}
        accounted = ran | set(report.remaining_job_ids) | set(report.deferred_job_ids)
        missing = set(report.planned_job_ids) - accounted
        assert not missing, f"{name}: planned jobs neither run nor named: {sorted(missing)}"


def test_no_forward_was_charged_twice(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """Launcher forwards and ledger forwards are disjoint, and their sum is the whole cost."""
    for name, (_run_dir, report) in suites.items():
        ran = [job for job in report.jobs if job.status != "NOT_RUN"]
        launcher = [job for job in ran if job.accounting == "launcher"]
        ledger = [job for job in ran if job.accounting == "ledger"]
        assert len(launcher) + len(ledger) == len(ran), name
        assert len({job.job_id for job in ran}) == len(ran), f"{name}: a job id was recorded twice"


def test_the_ledger_records_every_ledger_forward(
    suites: dict[str, tuple[Path, SuiteReport]],
) -> None:
    """Every attempt that reached the driver was reserved and resolved on a ledger.

    A job that never reached it — a worker the memory guard stopped before the reservation,
    or a worker that was already terminated — is deliberately NOT counted: it is not an
    attempt, it spent nothing, and `NOTHING_RAN_COST` exists precisely because a zero that
    means "nothing ran" is a different zero from a measured one.
    """
    for name, (run_dir, report) in suites.items():
        ledger_paths = sorted((run_dir / "ledgers").glob("*.json"))
        if not ledger_paths:
            continue
        entries = [
            entry
            for path in ledger_paths
            for entry in json.loads(path.read_text(encoding="utf-8")).get("entries", [])
        ]
        expected = sum(
            1
            for job in report.jobs
            if job.accounting == "ledger" and job.status not in ("NOT_RUN", "RESOURCE_FAILURE")
        )
        assert len(entries) >= expected, (
            f"{name}: {expected} ledger-accounted forwards ran and the session's "
            f"{len(ledger_paths)} ledger(s) hold {len(entries)} entries; SPEC 3.3 counts "
            "every attempt"
        )
