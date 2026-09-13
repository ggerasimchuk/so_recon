"""Git provenance helpers (SPEC 19.12: git commit in every output)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def git_commit(root: Path) -> str | None:
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_is_dirty(root: Path) -> bool | None:
    out = _git(root, "status", "--porcelain")
    if out is None:
        return None
    return bool(out.strip())
