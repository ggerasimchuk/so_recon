import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.simulator.julia_bridge import JuliaNotFoundError, JuliaSmokeResult
from so_recon.smoke import EXPECTED_FILENAME, SmokeExpectation, compare_with_expected, run_smoke

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


def _setup(tmp_project: Path) -> tuple[Any, ProjectPaths]:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    return cfg, ProjectPaths.from_config(tmp_project, cfg.paths)


def _expectation(**over: Any) -> SmokeExpectation:
    base: dict[str, Any] = {
        "schema_version": "2",
        "fixture_content_hash": "h",
        "case_sha256": "c",
        "nx": 5,
        "n_steps": 2,
        "cumulative_oil_m3": 100.0,
        "cumulative_water_injected_m3": 150.0,
        "mean_so_final": 0.6,
        "julia_version": "1.12.7",
        "jutul_version": "0.4.40",
        "jutuldarcy_version": "0.3.11",
    }
    base.update(over)
    return SmokeExpectation(**base)


def _result(**over: Any) -> JuliaSmokeResult:
    return JuliaSmokeResult.model_validate({**OK, "input_sha256": "i", **over})


def test_compare_with_expected_tolerances() -> None:
    exp = _expectation()
    assert (
        compare_with_expected(
            exp,
            fixture_hash="h",
            case_sha256="c",
            result=_result(cumulative_oil_m3=100.0 + 5e-5),
            rel_tol=1e-6,
        )
        == []
    )
    mism = compare_with_expected(
        exp,
        fixture_hash="h",
        case_sha256="c",
        result=_result(cumulative_oil_m3=101.0),
        rel_tol=1e-6,
    )
    assert any("cumulative_oil_m3" in m for m in mism)


def test_compare_detects_input_drift() -> None:
    exp = _expectation()
    assert any(
        "fixture_content_hash" in m
        for m in compare_with_expected(
            exp, fixture_hash="x", case_sha256="c", result=_result(), rel_tol=1e-6
        )
    )
    assert any(
        "case_sha256" in m
        for m in compare_with_expected(
            exp, fixture_hash="h", case_sha256="x", result=_result(), rel_tol=1e-6
        )
    )


def test_version_drift_is_a_mismatch_not_a_note() -> None:
    """Amendment 9: a JutulDarcy bump must fail the smoke, not be logged as a note."""
    exp = _expectation()
    mism = compare_with_expected(
        exp,
        fixture_hash="h",
        case_sha256="c",
        result=_result(jutuldarcy_version="0.3.12"),
        rel_tol=1e-6,
    )
    assert any("jutuldarcy_version" in m for m in mism)


def test_run_smoke_fails_without_expected_then_freezes_then_passes(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    cfg, paths = _setup(tmp_project)
    launcher = fake_launcher_factory(OK)

    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: launcher)
    assert ctx.record.status == "FAIL"
    assert any("freeze-expected" in n for n in ctx.record.notes)

    ctx2 = run_smoke(
        cfg=cfg,
        paths=paths,
        argv=["smoke"],
        launcher_factory=lambda: launcher,
        freeze_expected=True,
    )
    assert ctx2.record.status == "PASS"
    expected_path = paths.configs / EXPECTED_FILENAME
    frozen = json.loads(expected_path.read_text(encoding="utf-8"))
    assert frozen["nx"] == 5 and frozen["cumulative_oil_m3"] == 100.0
    assert "frozen_at" not in frozen, "the committed expectation must carry no timestamp"

    ctx3 = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: launcher)
    assert ctx3.record.status == "PASS"
    assert ctx3.record.julia_version == "1.12.7"
    assert ctx3.record.jutuldarcy_version == "0.3.11"
    assert (ctx3.run_dir / "fixture" / "wells.parquet").is_file()
    assert (ctx3.run_dir / "case.json").is_file()
    assert (ctx3.run_dir / "julia_smoke.json").is_file()
    assert (ctx3.run_dir / "run.log").is_file()
    assert ctx3.record.outputs["case"].path.endswith("case.json")
    assert ctx3.record.outputs["case"].sha256 == launcher_case_sha(ctx3.run_dir)


def launcher_case_sha(run_dir: Path) -> str:
    import hashlib

    return hashlib.sha256((run_dir / "case.json").read_bytes()).hexdigest()


def test_run_smoke_detects_numeric_drift(tmp_project: Path, fake_launcher_factory: Any) -> None:
    cfg, paths = _setup(tmp_project)
    run_smoke(
        cfg=cfg,
        paths=paths,
        argv=["smoke"],
        launcher_factory=lambda: fake_launcher_factory(OK),
        freeze_expected=True,
    )
    drifted = fake_launcher_factory({**OK, "mean_so_final": 0.61})
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: drifted)
    assert ctx.record.status == "FAIL"
    assert any("mean_so_final" in n for n in ctx.record.notes)


def test_run_smoke_fails_when_versions_drift_from_the_lock(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    """Amendment 9: running JutulDarcy differs from julia/Manifest.toml -> FAIL."""
    cfg, paths = _setup(tmp_project)
    run_smoke(
        cfg=cfg,
        paths=paths,
        argv=["smoke"],
        launcher_factory=lambda: fake_launcher_factory(OK),
        freeze_expected=True,
    )
    drifted = fake_launcher_factory({**OK, "jutuldarcy_version": "0.3.99"})
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: drifted)
    assert ctx.record.status == "FAIL"
    assert any("JutulDarcy" in n or "jutuldarcy" in n for n in ctx.record.notes)


def test_run_smoke_records_fail_when_julia_is_missing(tmp_project: Path) -> None:
    """Amendment 5: a missing Julia executable must still produce a FAIL run record."""
    cfg, paths = _setup(tmp_project)

    def factory() -> Any:
        raise JuliaNotFoundError("julia executable not found")

    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=factory)
    assert ctx.record.status == "FAIL"
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "FAIL"
    assert any("julia executable not found" in n for n in record["notes"])


def test_run_smoke_records_fail_when_julia_reports_an_error(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    cfg, paths = _setup(tmp_project)
    broken = fake_launcher_factory(
        {"status": "error", "message": "solver blew up", "wall_time_s": 0.0}, echo_input_sha=False
    )
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: broken)
    assert ctx.record.status == "FAIL"
    assert any("solver blew up" in n for n in ctx.record.notes)


def test_run_smoke_detects_a_tampered_case_file(
    tmp_project: Path, fake_launcher_factory: Any
) -> None:
    """End-to-end guard: Julia must return the hash of the bytes it actually read."""
    cfg, paths = _setup(tmp_project)
    liar = fake_launcher_factory({**OK, "input_sha256": "f" * 64}, echo_input_sha=False)
    ctx = run_smoke(cfg=cfg, paths=paths, argv=["smoke"], launcher_factory=lambda: liar)
    assert ctx.record.status == "FAIL"
    assert any("input_sha256" in n for n in ctx.record.notes)


def test_freezing_outside_the_repository_is_refused(
    tmp_project: Path, tmp_path_factory: Any, fake_launcher_factory: Any
) -> None:
    """Invariant I4: the third publish site must prove containment like the other two.

    write_bytes_atomic validates nothing, so a configs/ that points outside the root must
    be refused before any byte is written — the run still ends with a FAIL record (I6).
    """
    cfg, paths = _setup(tmp_project)
    # A sibling of the repository root, not a subdirectory of it: the point is escape.
    outside = Path(tmp_path_factory.mktemp("elsewhere"))
    escaped = replace(paths, configs=outside)
    launcher = fake_launcher_factory(OK)
    ctx = run_smoke(
        cfg=cfg,
        paths=escaped,
        argv=["smoke"],
        launcher_factory=lambda: launcher,
        freeze_expected=True,
    )
    assert ctx.record.status == "FAIL"
    assert any("PathEscapeError" in n for n in ctx.record.notes)
    assert not (outside / EXPECTED_FILENAME).exists()
