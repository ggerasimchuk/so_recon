"""Repository root discovery, path containment and canonical project directories."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT_ENV_VAR = "SO_RECON_ROOT"

# "C:/x", "C:\x" and "\\server\share" must never be accepted from configuration.
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


class RepoRootNotFoundError(RuntimeError):
    """Raised when no directory with pyproject.toml + src/so_recon is found."""


class PathEscapeError(ValueError):
    """Raised when a path would resolve outside the repository root."""


def validate_relative_path(value: str) -> str:
    """Accept only a plain relative POSIX path that stays inside the repository.

    Rejects absolute POSIX paths, Windows drive-absolute and UNC paths, '~' expansion,
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
        return Path(env).resolve()
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

    # ProjectPaths.from_config is added in Task 4, together with the PathsConfig schema
    # it depends on. so_recon.config.schema imports this module (paths) at runtime, so
    # Task 4 will import PathsConfig only under TYPE_CHECKING to avoid a paths <-> config
    # import cycle.

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
        for p in (self.interim, self.processed, self.runs, self.manifests, self.stages):
            p.mkdir(parents=True, exist_ok=True)
