from pathlib import Path

import pytest

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_bytes
from so_recon.simulator.julia_bridge import JuliaNotFoundError, default_launcher, run_julia_smoke
from so_recon.synthetic.fixture import build_smoke_case, build_smoke_fixture, write_smoke_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.julia
def test_real_julia_smoke_case_runs_end_to_end(tmp_path: Path) -> None:
    cfg = load_project_config(ROOT / "configs" / "project.yml")
    paths = ProjectPaths.from_config(ROOT, cfg.paths)
    if not (paths.julia / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        launcher = default_launcher(paths, cfg.julia)
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")

    fixture = build_smoke_fixture(cfg.smoke)
    case = build_smoke_case(cfg.smoke, fixture)
    case_path = tmp_path / "case.json"
    payload = write_smoke_case(case, case_path)

    res = run_julia_smoke(
        launcher,
        paths.root / cfg.julia.smoke_script,
        case_path,
        tmp_path / "smoke.json",
        expected_input_sha256=sha256_bytes(payload),
    )
    assert res.status == "ok"
    assert res.input_sha256 == sha256_bytes(payload)
    assert res.case_schema_version == case["schema_version"]
    assert res.nx == cfg.smoke.nx and res.n_steps == cfg.smoke.n_steps
    assert res.cumulative_oil_m3 > 0
    assert res.cumulative_water_injected_m3 > 0
    assert 0.0 < res.mean_so_final < 0.8
    assert res.jutuldarcy_version.startswith("0.3.")
