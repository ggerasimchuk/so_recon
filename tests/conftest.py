import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

PROJECT_YAML = """
spec_version: "3.0"
config_version: "test.1"
paths:
  raw: data/sources
sources:
  files:
    - {name: a, path: data/sources/a.csv, encoding: utf-8, delimiter: ",", decimal: "."}
    - {name: b, path: data/sources/b.csv, encoding: cp1251, delimiter: ";", decimal: ","}
smoke: {seed: 11, n_wells: 2, n_months: 2, nx: 5, n_steps: 2, rel_tol: 1.0e-6}
julia: {}
"""

JULIA_MANIFEST = """
julia_version = "1.12.7"
manifest_format = "2.0"

[[deps.JSON]]
version = "1.1.2"

[[deps.Jutul]]
version = "0.4.40"

[[deps.JutulDarcy]]
version = "0.3.11"
"""


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal repository layout: marker files, config, two fake immutable sources."""
    monkeypatch.delenv("SO_RECON_ROOT", raising=False)
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "src" / "so_recon").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "project.yml").write_text(PROJECT_YAML, encoding="utf-8")
    (tmp_path / "data" / "sources").mkdir(parents=True)
    (tmp_path / "data" / "sources" / "a.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "data" / "sources" / "b.csv").write_bytes(b"x;y\r\n1;2\r\n")
    (tmp_path / "julia" / "smoke").mkdir(parents=True)
    (tmp_path / "julia" / "smoke" / "smoke_case.jl").write_text("# fake\n")
    (tmp_path / "julia" / "Manifest.toml").write_text(JULIA_MANIFEST, encoding="utf-8")
    (tmp_path / "julia" / ".julia-version").write_text("1.12.7\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n")
    return tmp_path


@pytest.fixture
def fake_launcher_factory() -> Callable[..., Any]:
    """Build a launcher that echoes a fixed payload, optionally hashing the real case file."""

    class FakeLauncher:
        def __init__(self, payload: dict[str, Any], *, echo_input_sha: bool = True) -> None:
            self.payload = payload
            self.echo_input_sha = echo_input_sha
            self.calls = 0
            self.cases: list[Path] = []

        def launch(self, script: Path, args: list[str], out_path: Path) -> None:
            self.calls += 1
            case_path = Path(args[args.index("--case") + 1])
            self.cases.append(case_path)
            payload = dict(self.payload)
            if self.echo_input_sha and "input_sha256" not in payload:
                import hashlib

                payload["input_sha256"] = hashlib.sha256(case_path.read_bytes()).hexdigest()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload), encoding="utf-8")

    return FakeLauncher
