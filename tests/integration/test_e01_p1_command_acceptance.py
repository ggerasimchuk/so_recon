"""The real P1 command must publish scored evidence, including a closed preflight."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from so_recon.config.load import load_project_config
from so_recon.config.resources import P1_LOOP_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.paths import ProjectPaths
from so_recon.simulator import commands, suites
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.julia
def test_p1_command_scores_preflight_and_signal_before_accepting(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        executable = find_julia()
    except JuliaNotFoundError:
        pytest.skip("Julia executable is unavailable")
    shutil.copytree(ROOT / "julia", tmp_project / "julia", dirs_exist_ok=True)
    shutil.copyfile(ROOT / "uv.lock", tmp_project / "uv.lock")
    shutil.copyfile(ROOT / "configs/e01_tolerances.yml", tmp_project / "configs/e01_tolerances.yml")
    paths = ProjectPaths.default(tmp_project)
    paths.ensure_dirs()
    cfg = load_project_config(ROOT / "configs/e01.yml").model_copy(
        update={"resources": P1_LOOP_PROFILE}
    )

    def probe(pid: int | None, session_dir: Path) -> ResourceSnapshot:
        # Keep process measurements real; unrelated host applications cannot veto this test.
        return probe_resources(pid, session_dir).model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "swap_used_bytes": 0,
            }
        )

    # `probe_machine` moved to `simulator/suites.py` when Task 12 took the group runners out
    # of the command module; patching it on `commands` raised AttributeError.
    monkeypatch.setattr(suites, "probe_machine", probe)
    outcome = commands.run_synthetic_p1(cfg, paths, seeds=(41,), julia=str(executable))
    assert outcome.exit_code == 0
    suite = json.loads((paths.reports / "p1_suite_manifest.json").read_text())
    assert len(suite["rows"]) == 1
    # A successful subset is still not the complete five-parent suite.
    assert suite["fully_accepted"] is False
    row = suite["rows"][0]
    assert row["accepted"] is True
    gates = row["gates"]
    assert gates["equilibrium_preflight"]["status"] == "COMPLETE"
    assert gates["equilibrium_preflight"]["pressure_relative_drift"] < 1e-6
    assert gates["watercut_signal"]["passed"] is True
