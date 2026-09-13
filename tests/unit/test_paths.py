from pathlib import Path

import pytest

from so_recon.paths import (
    PathEscapeError,
    ProjectPaths,
    RepoRootNotFoundError,
    find_repo_root,
    resolve_within_root,
    validate_relative_path,
)


def _make_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    return tmp_path


def test_find_repo_root_walks_up_from_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    root = _make_root(tmp_path)
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == root.resolve()


def test_find_repo_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root(tmp_path)
    monkeypatch.setenv("SO_RECON_ROOT", str(root))
    assert find_repo_root(Path("/")) == root.resolve()


def test_find_repo_root_rejects_a_stale_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale SO_RECON_ROOT must fail loudly, not redirect artifacts into another tree."""
    stale = tmp_path / "not-the-repo"
    stale.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.setenv("SO_RECON_ROOT", str(stale))
    with pytest.raises(RepoRootNotFoundError) as exc:
        find_repo_root(_make_root(real))
    message = str(exc.value)
    assert "SO_RECON_ROOT" in message
    assert str(stale) in message


def test_find_repo_root_raises_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    with pytest.raises(RepoRootNotFoundError):
        find_repo_root(tmp_path)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " data/raw",
        "data/raw ",
        "~/data",
        "/abs/data",
        "//server/share",
        "\\\\server\\share",
        "C:/data",
        "C:\\data",
        "c:/data",
        "data\\raw",
        "../outside",
        "data/../../outside",
        "data/./raw",
        "..",
        "data/raw\x00",
    ],
)
def test_validate_relative_path_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_path(bad)


@pytest.mark.parametrize(
    "good", ["data/raw", "configs", "julia/smoke/smoke_case.jl", "data/Ромашка_сырые"]
)
def test_validate_relative_path_accepts(good: str) -> None:
    assert validate_relative_path(good) == good


def test_resolve_within_root_returns_absolute_path(tmp_path: Path) -> None:
    assert resolve_within_root(tmp_path, "data/raw") == (tmp_path.resolve() / "data" / "raw")


def test_resolve_within_root_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathEscapeError):
        resolve_within_root(root, "escape/secrets.csv")


def test_resolve_within_root_allows_symlink_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "real").mkdir(parents=True)
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    assert resolve_within_root(root, "link/f.csv") == (root.resolve() / "real" / "f.csv")


def test_default_paths_and_relative(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    assert paths.raw == tmp_path / "data" / "raw"
    assert paths.runs == tmp_path / "artifacts" / "runs"
    assert paths.manifests == tmp_path / "reports" / "manifests"
    assert paths.stages == tmp_path / "reports" / "stages"
    assert paths.relative(paths.raw / "mer.csv") == "data/raw/mer.csv"


def test_relative_rejects_path_outside_root(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path / "repo")
    with pytest.raises(PathEscapeError):
        paths.relative(tmp_path / "elsewhere" / "f.csv")


def test_ensure_dirs_creates_runtime_dirs_but_not_sources(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    for p in (paths.interim, paths.processed, paths.runs, paths.manifests, paths.stages):
        assert p.is_dir()
    assert not paths.raw.exists(), "source directory must never be created by the code"
