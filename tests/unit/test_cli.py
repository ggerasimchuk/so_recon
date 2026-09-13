import json
from pathlib import Path
from typing import Any

import pytest

from so_recon.cli import main

OK: dict[str, Any] = {
    "status": "ok",
    "case_schema_version": "1",
    "julia_version": "1.12.7",
    "jutuldarcy_version": "0.3.11",
    "jutul_version": "0.4.40",
    "nx": 5,
    "n_steps": 2,
    "cumulative_oil_m3": 100.0,
    "cumulative_water_injected_m3": 150.0,
    "mean_so_final": 0.6,
    "wall_time_s": 1.0,
}


def _only_run(tmp_project: Path) -> dict[str, Any]:
    runs = sorted((tmp_project / "artifacts" / "runs").iterdir())
    assert len(runs) == 1
    return json.loads((runs[0] / "run.json").read_text(encoding="utf-8"))


def test_manifest_command_writes_manifest_and_run_record(tmp_project: Path) -> None:
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    manifest = json.loads(
        (tmp_project / "reports" / "manifests" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert [s["name"] for s in manifest["sources"]] == ["a", "b"]
    assert manifest["sources"][0]["physical_line_count"] == 2
    assert manifest["sources"][0]["data_rows"] == 1
    record = _only_run(tmp_project)
    assert record["command"] == "manifest"
    assert record["status"] == "PASS"
    assert set(record["raw_input_hashes"]) == {"a", "b"}
    assert record["outputs"]["source_manifest"]["schema_version"] == "2"


def test_manifest_is_byte_identical_on_a_second_run(tmp_project: Path) -> None:
    """Amendment 10: repeated runs must not change the committed manifest."""
    published = tmp_project / "reports" / "manifests" / "source_manifest.json"
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    first = published.read_bytes()
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    assert published.read_bytes() == first


def test_manifest_command_fails_on_missing_source(tmp_project: Path) -> None:
    (tmp_project / "data" / "sources" / "b.csv").unlink()
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert any("data/sources/b.csv" in n for n in record["notes"])


def test_broken_config_still_produces_a_fail_run_record(tmp_project: Path) -> None:
    """Amendment 5: a configuration error must not leave the run unrecorded."""
    (tmp_project / "configs" / "project.yml").write_text('spec_version: "9.9"\n', encoding="utf-8")
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert record["config_version"] == "unavailable"
    assert any("config" in n.lower() for n in record["notes"])


def test_env_report_command(tmp_project: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "env-report"]) == 0
    assert (tmp_project / "reports" / "environment_report.md").is_file()
    env = json.loads(
        (tmp_project / "reports" / "manifests" / "environment.json").read_text(encoding="utf-8")
    )
    assert env["uv_lock_sha256"] is not None
    assert env["julia_pinned_version"] == "1.12.7"
    assert "created_at" not in env


def test_smoke_command_freeze_then_pass(
    tmp_project: Path, fake_launcher_factory: Any, capsys: Any
) -> None:
    launcher = fake_launcher_factory(OK)
    factory = lambda paths, cfg, julia: launcher  # noqa: E731
    assert (
        main(["--root", str(tmp_project), "smoke", "--freeze-expected"], launcher_factory=factory)
        == 0
    )
    assert main(["--root", str(tmp_project), "smoke"], launcher_factory=factory) == 0
    out = capsys.readouterr().out
    assert "status=PASS" in out
    assert "artifacts/runs/" in out


def test_smoke_command_returns_1_on_fail(tmp_project: Path, fake_launcher_factory: Any) -> None:
    launcher = fake_launcher_factory(OK)
    assert (
        main(["--root", str(tmp_project), "smoke"], launcher_factory=lambda p, c, j: launcher) == 1
    )


def test_smoke_command_without_julia_returns_1_and_records_fail(
    tmp_project: Path, monkeypatch: Any
) -> None:
    # PATH and HOME are redirected so the developer's real julia cannot be discovered:
    # this test is about the FAIL record, not about running a simulation.
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "smoke", "--julia", str(tmp_project / "absent")]) == 1
    record = _only_run(tmp_project)
    assert record["status"] == "FAIL"
    assert any("julia" in n.lower() for n in record["notes"])


def test_unknown_command_returns_2(tmp_project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_project), "nope"])
    assert exc.value.code == 2
