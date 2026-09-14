"""E01.12.9 — a gate that greenlights a skip is not a gate.

12.9 is normative: «Stage gate требует присутствия всех mandatory checks и нуля
skipped/unrun в его собственной matrix; обычный pytest PASS с skips недостаточен.» Plain
`pytest -q` exits 0 with skips, so `scripts/e01_gate.sh` reported PASS while
`test_a_published_balance_table_carries_both_statements` had skipped itself.

`--fail-on-skip` is the option that closes it, and the first test below runs a real pytest
under it rather than asserting anything about the conftest's source.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "e01_gate.sh"

SKIPPING_TEST = """
import pytest


def test_it_passes() -> None:
    assert True


def test_it_skips() -> None:
    pytest.skip("nothing to read")
"""


def _pytest(target: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-p", "tests.conftest", "-q", *extra],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_skip_is_a_pass_without_the_option(tmp_path: Path) -> None:
    target = tmp_path / "test_sample.py"
    target.write_text(SKIPPING_TEST, encoding="utf-8")
    result = _pytest(target)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_skip_is_a_failure_under_fail_on_skip(tmp_path: Path) -> None:
    target = tmp_path / "test_sample.py"
    target.write_text(SKIPPING_TEST, encoding="utf-8")
    result = _pytest(target, "--fail-on-skip")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed" in result.stdout and "1 passed" in result.stdout
    assert "--fail-on-skip" in result.stdout


# --------------------------------------------------------------------------------------
# what the gate script itself must do
# --------------------------------------------------------------------------------------


def _gate_text() -> str:
    return GATE.read_text(encoding="utf-8")


def test_the_gate_runs_the_stage_validators_with_fail_on_skip() -> None:
    lines = _gate_text().splitlines()
    line = next(x for x in lines if "test_e01_stage.py" in x and "pytest" in x)
    assert "--fail-on-skip" in line, line


def test_the_gate_checks_the_frozen_inputs_even_when_a_step_failed() -> None:
    """A frozen input mutated by a FAILING step is exactly the one worth reporting."""
    text = _gate_text()
    assert "trap" in text, "check_frozen only ran when every earlier step had passed"
    assert "check_frozen" in text


def test_the_gate_rebuilds_the_stage_report() -> None:
    """Otherwise the committed `reports/stages/E01.md` can drift from the artifacts."""
    assert "e01-report" in _gate_text()
