"""Repository root discovery, path containment and canonical project directories."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from so_recon.config.schema import PathsConfig

ROOT_ENV_VAR = "SO_RECON_ROOT"

# Windows drive paths (including "C:x") and UNC are invalid configuration paths.
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:|\\\\)")


class RepoRootNotFoundError(RuntimeError):
    """Raised when no directory with pyproject.toml + src/so_recon is found."""


class PathEscapeError(ValueError):
    """Raised when a path would resolve outside the repository root."""


def validate_relative_path(value: str) -> str:
    """Accept only a plain relative POSIX path that stays inside the repository.

    Rejects absolute POSIX paths, Windows drive and UNC paths, '~' expansion,
    backslash separators, '.' and '..' segments, and NUL bytes (invariant I4).
    """
    if not value or value != value.strip():
        raise ValueError(f"path must be a non-empty trimmed string, got {value!r}")
    if "\x00" in value:
        raise ValueError(f"path must not contain NUL bytes, got {value!r}")
    if value.startswith("~"):
        raise ValueError(f"path must not use '~' expansion, got {value!r}")
    if "\\" in value:
        raise ValueError(f"path must use '/' separators and must not be a UNC path, got {value!r}")
    if _WINDOWS_ABSOLUTE.match(value):
        raise ValueError(f"path must not be a Windows absolute/UNC path, got {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute():
        raise ValueError(f"path must be relative to repository root, got {value!r}")
    # Split the raw string rather than using pure.parts: PurePosixPath silently
    # collapses single "." segments while parsing, so a check on pure.parts would
    # never see them and "data/./raw" would wrongly be accepted.
    if any(segment in ("..", ".") for segment in value.split("/")):
        raise ValueError(f"path must not contain '.' or '..' segments, got {value!r}")
    return value


def resolve_within_root(root: Path, relative: str) -> Path:
    """Validate the form, then resolve and prove the result stays inside the root.

    resolve() expands symlinks, so a symlink pointing outside the repository is rejected.
    """
    validate_relative_path(relative)
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and not candidate.is_relative_to(root_resolved):
        raise PathEscapeError(f"{relative!r} resolves to {candidate}, outside {root_resolved}")
    return candidate


def _is_root(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "src" / "so_recon").is_dir()


def _walk_up(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _is_root(candidate):
            return candidate
    return None


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root: env override, then an ancestor walk.

    When `start` is given explicitly, it is authoritative: only its own ancestry is
    searched. When `start` is omitted, the default lookup walks up from the current
    working directory and, if that fails, falls back to walking up from this module's
    own location. Silently deferring to the module's own path when a caller-specified
    `start` finds nothing would be a surprising, lax fallback for a function that gates
    path containment.
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        # Checked with the same marker the ancestor walk uses. Returning it unvalidated
        # would let a stale export redirect every artifact of the run into an unrelated
        # tree, silently and with no note anywhere in the record (invariant I4).
        candidate = Path(env).resolve()
        if not _is_root(candidate):
            raise RepoRootNotFoundError(
                f"{ROOT_ENV_VAR} is set to {env!r}, which resolves to {candidate}, "
                "but that is not a repository root (no pyproject.toml + src/so_recon); "
                f"unset {ROOT_ENV_VAR} or point it at the repository"
            )
        return candidate
    origins = (start,) if start is not None else (Path.cwd(), Path(__file__))
    for origin in origins:
        found = _walk_up(origin)
        if found is not None:
            return found
    raise RepoRootNotFoundError(
        f"cannot locate repository root (pyproject.toml + src/so_recon); set {ROOT_ENV_VAR}"
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path
    configs: Path
    julia: Path

    @classmethod
    def default(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            raw=root / "data" / "raw",
            interim=root / "data" / "interim",
            processed=root / "data" / "processed",
            artifacts=root / "artifacts",
            reports=root / "reports",
            configs=root / "configs",
            julia=root / "julia",
        )

    @classmethod
    def from_config(cls, root: Path, cfg: PathsConfig) -> ProjectPaths:
        root = root.resolve()
        return cls(
            root=root,
            raw=resolve_within_root(root, cfg.raw),
            interim=resolve_within_root(root, cfg.interim),
            processed=resolve_within_root(root, cfg.processed),
            artifacts=resolve_within_root(root, cfg.artifacts),
            reports=resolve_within_root(root, cfg.reports),
            configs=resolve_within_root(root, cfg.configs),
            julia=resolve_within_root(root, cfg.julia),
        )

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs"

    @property
    def manifests(self) -> Path:
        return self.reports / "manifests"

    @property
    def stages(self) -> Path:
        return self.reports / "stages"

    def resolve(self, relative: str) -> Path:
        return resolve_within_root(self.root, relative)

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.root.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise PathEscapeError(f"{path} is outside the repository root {root}")
        return resolved.relative_to(root).as_posix()

    def ensure_dirs(self) -> None:
        """Create runtime directories only. The source directory is never created:
        a missing source must surface as an explicit error (SPEC 7.1, invariant I1)."""
        runtime = (self.interim, self.processed, self.runs, self.manifests, self.stages)
        # Validate every derived directory before the first mkdir. Checking only the
        # configured parent misses symlinks such as artifacts/runs -> outside.
        raw = self.raw.resolve()
        for p in (*runtime, self.artifacts, self.reports, self.configs):
            resolved = p.resolve()
            if resolved.is_relative_to(raw) or raw.is_relative_to(resolved):
                raise PathEscapeError(f"output directory {p} overlaps raw sources {self.raw}")
        for p in runtime:
            self.relative(p)
        for p in runtime:
            p.mkdir(parents=True, exist_ok=True)

    def validate_run_destination(self) -> None:
        """Check the record destination before claiming a run, including fallback paths."""
        self.relative(self.runs)
        raw = self.raw.resolve()
        if self.runs.resolve().is_relative_to(raw) or raw.is_relative_to(self.runs.resolve()):
            raise PathEscapeError(f"run directory {self.runs} overlaps raw sources {self.raw}")
