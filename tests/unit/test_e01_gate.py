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
import re
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
    """The validator step names the stage validators and refuses to count a skip as a pass.

    The file list moved into a `validators` array when Task 13 made the black-oil validators
    conditional on the black-oil suite having run, so the assertion is on the array and on
    the `pytest` line that expands it rather than on one line carrying both.
    """
    lines = _gate_text().splitlines()
    declared = next(x for x in lines if "validators=(" in x and "test_e01_stage.py" in x)
    assert "tests/integration/test_e01_stage.py" in declared, declared
    invocation = next(x for x in lines if "pytest" in x and "validators[@]" in x and "run" in x)
    assert "--fail-on-skip" in invocation, invocation
    assert "--run-e01-physics" in invocation, invocation


def test_the_black_oil_validators_run_only_when_the_black_oil_suite_did() -> None:
    """Plan 13.5: an oil-water gate must not fail for want of a black-oil session.

    `tests/integration/test_e01_blackoil.py` reads a published `bo` session and
    `--fail-on-skip` turns a missing one into a failure, so listing it unconditionally would
    couple the two gates the capability is kept separate from.
    """
    text = _gate_text()
    assert "test_e01_blackoil.py" in text
    guard = next(
        x for x in text.splitlines() if "test_e01_blackoil.py" in x and "validators+=" in x
    )
    assert '*" bo "*)' in guard, guard


def test_the_gate_names_check_frozen_at_all() -> None:
    """The cheap half of the frozen-input rule; the executed half is below."""
    assert "check_frozen" in _gate_text()


def _code_lines() -> list[str]:
    """The script with its comments stripped, so prose about `trap` is not code."""
    return [re.sub(r"#.*", "", line) for line in _gate_text().splitlines()]


def test_the_gate_installs_no_trap() -> None:
    """The header's rule 2 says there is no `trap`, and bounds its guarantee on that.

    A trap re-introduced here would silently make that paragraph false again — and on this
    host an EXIT trap in a pipeline's subshell does not fire, so it would also be useless.
    Matched against the code with comments removed, because rule 2 discusses `trap` at
    length and a substring test over the whole file would pass on the prose.
    """
    offending = [line for line in _code_lines() if re.search(r"(^|[;&|\s])trap\s", line)]
    assert not offending, offending


def test_the_gate_rebuilds_the_stage_report() -> None:
    """Otherwise the committed `reports/stages/E01.md` can drift from the artifacts."""
    assert "e01-report" in _gate_text()


# --------------------------------------------------------------------------------------
# the gate, executed rather than grepped
# --------------------------------------------------------------------------------------
#
# The three tests above assert on SUBSTRINGS of the script, and all three passed against a
# `step()` that always returned 0: `if "$@"; then return 0; fi` leaves `$?` as the `if`
# statement's own status, which is 0 when the condition failed and there is no `else`. So
# every `step ... || return` in `main` was inert and a broken environment no longer aborted
# before the P0/P1 physics suites were launched. The two tests below run the real thing.


def _step_function() -> str:
    """The `step()` definition, lifted verbatim out of the committed script."""
    text = _gate_text()
    start = text.index("step() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


def test_step_returns_the_command_s_real_exit_code(tmp_path: Path) -> None:
    """`step` is the gate's only failure detector. It has to detect one."""
    script = tmp_path / "probe.sh"
    script.write_text(
        "FAILURES=()\n" + _step_function() + "\nstep 'one' bash -c 'exit 3'\n"
        "rc=$?\n"
        'echo "rc=${rc}"\n'
        'echo "n=${#FAILURES[@]}"\n'
        'echo "first=${FAILURES[0]:-none}"\n',
        encoding="utf-8",
    )
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=False).stdout
    assert "rc=3" in out, out
    assert "n=1" in out, out
    assert "exit 3" in out, out


def test_a_failing_early_step_aborts_before_the_physics_suites(tmp_path: Path) -> None:
    """Fail fast is the promise at the top of the script; this is it being kept.

    The gate is copied into a scratch root so nothing here touches the repository, and `uv`
    is replaced by a stub that fails. Nothing below step 1 may run: launching the P0 and P1
    suites after a broken environment is exactly the expensive outcome the script's own
    header says it avoids.
    """
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "e01_gate.sh").write_text(_gate_text(), encoding="utf-8")
    stub = root / "stub-uv"
    stub.write_text('#!/usr/bin/env bash\necho "stub-uv $*" >&2\nexit 3\n', encoding="utf-8")
    stub.chmod(0o755)

    result = subprocess.run(
        ["bash", str(root / "scripts" / "e01_gate.sh")],
        env=dict(os.environ, UV=str(stub)),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined
    assert "exited 3" in combined, "the stub's real exit code never reached the log"
    assert "registered E01 suites" not in combined, (
        "the gate went on to launch the physics suites after the environment step failed"
    )
    assert "verify-physics" not in combined, combined
    # ...and the frozen inputs were still checked, after the failure rather than instead of
    # it. This used to be a `trap ... EXIT` set inside a function that ran as the left side
    # of a pipeline, which on bash 3.2.57 never fires at all.
    assert "frozen inputs (checked whatever happened above" in combined, combined
    assert "e01 gate FAIL" in combined, combined
    assert "failed step: 1/8" in combined, combined
