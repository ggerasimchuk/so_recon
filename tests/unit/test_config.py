from pathlib import Path

import pytest
from pydantic import ValidationError

from so_recon.config.load import ConfigError, config_hash, load_project_config, resolved_config_dict
from so_recon.config.schema import PathsConfig, ProjectConfig, SourceFileSpec, SourcesConfig
from so_recon.paths import ProjectPaths

MINIMAL_YAML = """
spec_version: "3.0"
config_version: "test.1"
paths: {}
sources:
  files:
    - {name: coords, path: data/raw/coords.csv, encoding: utf-8, delimiter: ",", decimal: "."}
smoke: {}
julia: {}
"""


def _spec(**over: object) -> SourceFileSpec:
    base: dict[str, object] = {
        "name": "x",
        "path": "data/raw/x.csv",
        "encoding": "utf-8",
        "delimiter": ";",
        "decimal": ",",
    }
    base.update(over)
    return SourceFileSpec(**base)  # type: ignore[arg-type]


def test_load_minimal_config(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert cfg.spec_version == "3.0"
    assert cfg.paths.raw == "data/raw"
    assert cfg.sources.files[0].encoding == "utf-8"
    assert cfg.sources.files[0].header_lines == 1
    assert cfg.smoke.seed == 20260913
    assert cfg.julia.smoke_script == "julia/smoke/smoke_case.jl"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML + "unexpected: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(p)


def test_wrong_spec_version_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML.replace('"3.0"', '"2.9"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(p)


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_project_config(tmp_path / "absent.yml")


@pytest.mark.parametrize(
    "bad",
    [
        "/tmp/raw",
        "../raw",
        "~/raw",
        "C:/raw",
        "C:\\raw",
        "\\\\server\\share",
        "//server/share",
        "data\\raw",
    ],
)
def test_dangerous_paths_config_values_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        PathsConfig(raw=bad)


@pytest.mark.parametrize("bad", ["/abs/x.csv", "../x.csv", "data/../../x.csv", "C:\\x.csv"])
def test_dangerous_source_paths_are_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        _spec(path=bad)


def test_duplicate_source_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SourcesConfig(files=[_spec(name="a"), _spec(name="a", path="data/raw/y.csv")])


def test_negative_header_lines_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(header_lines=-1)


def test_config_hash_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    cfg = load_project_config(p)
    assert config_hash(cfg) == config_hash(load_project_config(p))
    assert resolved_config_dict(cfg)["smoke"]["seed"] == 20260913


def test_config_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "project.yml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    q = tmp_path / "other.yml"
    q.write_text(MINIMAL_YAML.replace("test.1", "test.2"), encoding="utf-8")
    assert config_hash(load_project_config(p)) != config_hash(load_project_config(q))


def test_project_paths_from_config(tmp_path: Path) -> None:
    cfg = ProjectConfig(
        spec_version="3.0",
        config_version="t",
        paths=PathsConfig(raw="custom/raw"),
        sources=SourcesConfig(files=[]),
    )
    paths = ProjectPaths.from_config(tmp_path, cfg.paths)
    assert paths.raw == tmp_path.resolve() / "custom" / "raw"
    assert paths.reports == tmp_path.resolve() / "reports"


def test_repo_config_file_is_valid_and_matches_audited_contracts() -> None:
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "project.yml"
    cfg = load_project_config(repo_cfg)
    by_name = {f.name: f for f in cfg.sources.files}
    assert list(by_name) == ["coords", "gis", "mer", "perf", "plastoper"]
    # Contracts verified byte-wise against the raw files (DATA_AUDIT.md section 3).
    assert (by_name["coords"].encoding, by_name["coords"].delimiter) == ("utf-8", ",")
    assert (by_name["gis"].encoding, by_name["gis"].decimal) == ("cp1251", ",")
    assert (by_name["mer"].encoding, by_name["mer"].decimal) == ("utf-8-sig", ",")
    assert (by_name["perf"].encoding, by_name["perf"].decimal) == ("cp1251", ",")
    # plastoper.csv stores "269,7600098": the decimal separator is a comma, not a dot.
    assert (by_name["plastoper"].encoding, by_name["plastoper"].decimal) == ("utf-8-sig", ",")
    assert all(f.header_lines == 1 for f in cfg.sources.files)
    # Sources stay where they are: E00 never moves or rewrites them.
    assert all(f.path.startswith("data/Ромашка_сырые/") for f in cfg.sources.files)
    assert cfg.paths.raw == "data/Ромашка_сырые"
