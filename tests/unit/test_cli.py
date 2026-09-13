import hashlib
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


def _only_run_dir(tmp_project: Path) -> Path:
    runs = sorted((tmp_project / "artifacts" / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


def _only_run(tmp_project: Path) -> dict[str, Any]:
    return json.loads((_only_run_dir(tmp_project) / "run.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_run_scoped_stamp(stamp: dict[str, Any], run_dir: Path) -> None:
    """The amendment-10 contract: run-scoped facts live here, not in the committed file."""
    assert stamp["run_id"] == run_dir.name
    assert stamp["created_at"].startswith("20")
    assert "git_commit" in stamp  # None outside a git checkout, but always recorded
    assert "git_dirty" in stamp


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
    assert record["schema_versions"] == {"source_manifest": "2", "run_record": "1"}


def test_manifest_command_writes_a_run_scoped_stamp(tmp_project: Path) -> None:
    """Amendment 10: the run-scoped facts the deterministic manifest must NOT carry.

    Without this test both stamp writers could be deleted outright and the suite would
    still be green, so the deliverable that keeps `run_id`/`created_at`/`git_commit` out
    of the committed manifest would be unguarded.
    """
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    run_dir = _only_run_dir(tmp_project)
    stamp_path = run_dir / "source_manifest_stamp.json"
    assert stamp_path.is_file()
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    _assert_run_scoped_stamp(stamp, run_dir)
    published = tmp_project / "reports" / "manifests" / "source_manifest.json"
    assert stamp["manifest_sha256"] == _sha256(published)
    assert stamp["manifest_version"] == "2"
    # The stamp is the ONLY place these live: the published manifest stays deterministic.
    committed = json.loads(published.read_text(encoding="utf-8"))
    for run_scoped in ("run_id", "created_at", "git_commit", "git_dirty"):
        assert run_scoped not in committed


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


def test_env_report_command_writes_a_run_scoped_stamp(tmp_project: Path, monkeypatch: Any) -> None:
    """The second half of the amendment-10 deliverable, guarded the same way."""
    monkeypatch.setenv("PATH", str(tmp_project))
    monkeypatch.setenv("HOME", str(tmp_project))
    assert main(["--root", str(tmp_project), "env-report"]) == 0
    run_dir = _only_run_dir(tmp_project)
    stamp_path = run_dir / "environment_stamp.json"
    assert stamp_path.is_file()
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    _assert_run_scoped_stamp(stamp, run_dir)
    published = tmp_project / "reports" / "manifests" / "environment.json"
    assert stamp["report_sha256"] == _sha256(published)
    # Important 3: the host fingerprint moved here, and none of it was lost on the way.
    assert stamp["os"] and stamp["arch"]
    for moved in ("os", "arch", "uv_version", "julia_executable_version"):
        assert moved in stamp
        assert moved not in json.loads(published.read_text(encoding="utf-8"))


def test_smoke_command_freeze_then_pass(
    tmp_project: Path, fake_launcher_factory: Any, capsys: Any
) -> None:
    launcher = fake_launcher_factory(OK)

    def factory(paths: Any, cfg: Any, julia: Any) -> Any:
        return launcher

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


def test_cli_exits_cleanly_when_the_runs_directory_is_unusable(
    tmp_project: Path, capsys: Any
) -> None:
    """No traceback may reach the user: the failure is explained and the exit code is 1."""
    (tmp_project / "artifacts").write_text("not a directory\n")
    assert main(["--root", str(tmp_project), "manifest"]) == 1
    err = capsys.readouterr().err
    assert "cannot record this run" in err
    assert "Traceback" not in err


def test_cli_explains_a_bad_config_even_when_no_record_can_be_written(
    tmp_project: Path, capsys: Any
) -> None:
    """Both failures at once. The startup-failure path calls execute_run too, so it needs the
    same guard: the config error must still be explained and no traceback may escape."""
    (tmp_project / "artifacts").write_text("not a directory\n")
    bad = tmp_project / "configs" / "bad.yml"
    bad.write_text("paths: [this is not a mapping\n", encoding="utf-8")
    assert main(["--root", str(tmp_project), "--config", str(bad), "manifest"]) == 1
    err = capsys.readouterr().err
    assert "config error" in err
    assert "cannot record this run" in err
    assert "Traceback" not in err


def test_unknown_command_returns_2(tmp_project: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_project), "nope"])
    assert exc.value.code == 2
