"""E02 command statuses and forecasts never call a checkpoint a completed posterior."""

from __future__ import annotations

import json
from pathlib import Path

from so_recon.cli import main
from so_recon.inference.commands import e02_exit_code, forecast_forward_calls

ROOT = Path(__file__).resolve().parents[2]


def test_checkpoint_is_not_completed_inference() -> None:
    assert e02_exit_code("INCOMPLETE_BUDGET", "NOT_RUN") == 2
    assert e02_exit_code("COMPLETE", "FAIL") == 2
    assert e02_exit_code("COMPLETE", "PASS") == 0


def test_worst_case_forecast_keeps_initial_and_every_move() -> None:
    assert forecast_forward_calls(32, 24, 2) == 1568
    assert forecast_forward_calls(64, 24, 2) == 3136


def test_math_cli_runs_without_starting_julia(tmp_project: Path, monkeypatch: object) -> None:
    config = tmp_project / "configs/e02.yml"
    config.write_text((ROOT / "configs/e02.yml").read_text(encoding="utf-8"), encoding="utf-8")
    experiments = tmp_project / "configs/e02_experiments.json"
    experiments.write_text(
        (ROOT / "configs/e02_experiments.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert (
        main(
            [
                "--root",
                str(tmp_project),
                "--config",
                str(config),
                "verify-inverse",
                "--suite",
                "math",
            ]
        )
        == 0
    )
    run_dirs = list((tmp_project / "artifacts/runs").iterdir())
    assert len(run_dirs) == 1
    status = json.loads((run_dirs[0] / "e02_status.json").read_text(encoding="utf-8"))
    assert status["suite"] == "math"
    assert status["algorithm_status"] == "COMPLETE"
    assert status["checks"]["toy_reference"] is True


def test_budget_cli_reports_n64_worst_case_as_over_p1_cap(
    tmp_project: Path, capsys: object
) -> None:
    config = tmp_project / "configs/e02.yml"
    config.write_text((ROOT / "configs/e02.yml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_project / "configs/e02_experiments.json").write_text(
        (ROOT / "configs/e02_experiments.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert (
        main(
            [
                "--root",
                str(tmp_project),
                "--config",
                str(config),
                "inverse-budget",
                "--experiment",
                "e02-t1-v1-s141",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "N64" in output
    assert "3136" in output
    assert "exceeds P1_LOOP max_new_forward=2000" in output
