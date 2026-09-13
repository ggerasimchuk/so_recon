import json
from pathlib import Path

import pytest

from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    JuliaRunError,
    SubprocessJuliaLauncher,
    find_julia,
    run_julia_smoke,
)

CASE_BYTES = b'{"schema_version":"1"}'

OK_PAYLOAD: dict[str, object] = {
    "status": "ok",
    "case_schema_version": "1",
    "julia_version": "1.12.7",
    "jutuldarcy_version": "0.3.11",
    "jutul_version": "0.4.40",
    "nx": 20,
    "n_steps": 12,
    "cumulative_oil_m3": 1234.5,
    "cumulative_water_injected_m3": 2000.0,
    "mean_so_final": 0.55,
    "wall_time_s": 3.2,
}


class FakeLauncher:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[Path, list[str], Path]] = []

    def launch(self, script: Path, args: list[str], out_path: Path) -> None:
        self.calls.append((script, args, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.payload), encoding="utf-8")


def _case(tmp_path: Path) -> tuple[Path, str]:
    import hashlib

    p = tmp_path / "case.json"
    p.write_bytes(CASE_BYTES)
    return p, hashlib.sha256(CASE_BYTES).hexdigest()


def test_run_julia_smoke_parses_result_and_passes_the_case(tmp_path: Path) -> None:
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({**OK_PAYLOAD, "input_sha256": sha})
    res = run_julia_smoke(
        launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
    )
    assert res.cumulative_oil_m3 == 1234.5
    assert res.input_sha256 == sha
    assert launcher.calls[0][1] == ["--case", str(case_path)]


def test_run_julia_smoke_rejects_a_mismatched_input_hash(tmp_path: Path) -> None:
    """End-to-end guard: Julia must prove it read exactly the case Python wrote."""
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({**OK_PAYLOAD, "input_sha256": "f" * 64})
    with pytest.raises(JuliaRunError, match="input_sha256"):
        run_julia_smoke(
            launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
        )


def test_run_julia_smoke_raises_on_error_status(tmp_path: Path) -> None:
    case_path, sha = _case(tmp_path)
    launcher = FakeLauncher({"status": "error", "message": "boom", "wall_time_s": 0.1})
    with pytest.raises(JuliaRunError, match="boom"):
        run_julia_smoke(
            launcher, tmp_path / "s.jl", case_path, tmp_path / "out.json", expected_input_sha256=sha
        )


def test_find_julia_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "julia"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("SO_RECON_JULIA", str(exe))
    assert find_julia() == exe


def test_find_julia_raises_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SO_RECON_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(JuliaNotFoundError):
        find_julia()


def test_subprocess_launcher_builds_command_and_reports_failure(tmp_path: Path) -> None:
    fake = tmp_path / "fake_julia.sh"
    fake.write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/argv.txt"\necho fatal >&2\nexit 3\n')
    fake.chmod(0o755)
    launcher = SubprocessJuliaLauncher(fake, tmp_path / "proj", timeout_s=30)
    with pytest.raises(JuliaRunError, match="fatal"):
        launcher.launch(tmp_path / "s.jl", ["--case", "c.json"], tmp_path / "o.json")
    argv = (tmp_path / "argv.txt").read_text().split()
    assert argv[0] == f"--project={tmp_path / 'proj'}"
    assert argv[1] == "--startup-file=no"
    assert argv[-2:] == ["--out", str(tmp_path / "o.json")]
