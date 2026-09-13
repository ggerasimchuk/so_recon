"""Guard: no personal absolute paths in any tracked file.

The rule is phrased over the *tracked* set, so the file list comes from ``git ls-files``
rather than from a filesystem walk. A walk needs a hand-maintained directory list, and such
a list is wrong in both directions: tracked files outside it escape the guard, while
untracked generated output inside it (later stages write into ``reports/``) would fail a
rule that never applied to it.

Consequence: this test needs a working git checkout. When ``git ls-files`` cannot be run it
skips with the reason rather than passing vacuously.

Two tracked files contain the pattern on purpose and are exempt by exact path: this file,
which defines the regex, and the E00 implementation plan, which quotes both the regex and a
``grep -n "/Users/"`` command.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"(/Users/|/home/|[Cc]:\\Users|/Volumes/)")

# Binary formats: never prose, and a chance byte sequence must not fail the guard.
SKIP_SUFFIXES = {".pyc", ".parquet", ".png"}

EXEMPT = frozenset(
    {
        "tests/test_no_absolute_paths.py",
        "docs/superpowers/plans/2026-09-13-e00-foundation.md",
    }
)


def _tracked_files() -> list[str]:
    """Repo-relative paths of every tracked file, or skip if git cannot tell us."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"cannot enumerate tracked files: git ls-files failed ({exc})")
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        pytest.skip(
            f"cannot enumerate tracked files: git ls-files exited {proc.returncode}: {detail}"
        )
    # -z keeps paths as raw bytes; surrogateescape round-trips a non-UTF-8 name intact.
    listing = proc.stdout.decode("utf-8", errors="surrogateescape")
    return [name for name in listing.split("\0") if name]


def test_no_personal_absolute_paths_in_tracked_files() -> None:
    offenders: list[str] = []
    for name in _tracked_files():
        if name in EXEMPT or Path(name).suffix in SKIP_SUFFIXES:
            continue
        path = ROOT / name
        if not path.is_file():
            continue  # staged deletion: in the index, gone from the working tree
        # Read bytes and decode defensively: a non-UTF-8 tracked file must not crash
        # the guard, it must simply be scanned.
        text = path.read_bytes().decode("utf-8", errors="replace")
        for m in FORBIDDEN.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{name}:{line_no}")
    assert offenders == [], f"personal absolute paths found: {offenders}"
