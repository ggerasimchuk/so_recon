import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.registry.hashing import canonical_json
from so_recon.synthetic.fixture import (
    CASE_SCHEMA_VERSION,
    build_smoke_case,
    build_smoke_fixture,
    case_bytes,
    fixture_content_hash,
    write_smoke_case,
    write_smoke_fixture,
)


def test_fixture_is_deterministic_for_same_seed() -> None:
    a = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    b = build_smoke_fixture(SmokeFixtureConfig(seed=7, n_wells=4, n_months=3))
    assert a.content_hash == b.content_hash
    assert a.wells.to_pylist() == b.wells.to_pylist()


def test_fixture_changes_with_seed() -> None:
    assert (
        build_smoke_fixture(SmokeFixtureConfig(seed=7)).content_hash
        != build_smoke_fixture(SmokeFixtureConfig(seed=8)).content_hash
    )


def test_fixture_shape_and_invariants() -> None:
    fx = build_smoke_fixture(SmokeFixtureConfig(seed=1, n_wells=5, n_months=4))
    assert fx.wells.num_rows == 5
    assert fx.well_month.num_rows == 20
    assert fx.wells.column_names == ["well_id", "role", "x_m", "y_m"]
    assert fx.well_month.column_names == ["well_id", "month", "liquid_m3", "injection_m3"]
    roles = dict(zip(fx.wells["well_id"].to_pylist(), fx.wells["role"].to_pylist(), strict=True))
    for row in fx.well_month.to_pylist():
        assert row["liquid_m3"] >= 0 and row["injection_m3"] >= 0
        if roles[row["well_id"]] == "injector":
            assert row["liquid_m3"] == 0.0
        else:
            assert row["injection_m3"] == 0.0
    assert sorted(set(fx.well_month["month"].to_pylist())) == [
        "2020-01-01",
        "2020-02-01",
        "2020-03-01",
        "2020-04-01",
    ]


def test_write_fixture_roundtrips_and_records_hash(tmp_path: Path) -> None:
    fx = build_smoke_fixture(SmokeFixtureConfig(seed=3, n_wells=2, n_months=2))
    files = write_smoke_fixture(fx, tmp_path / "fixture")
    assert (
        fixture_content_hash(pq.read_table(files["wells"]), pq.read_table(files["well_month"]))
        == fx.content_hash
    )
    meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    assert meta["content_hash"] == fx.content_hash
    assert meta["seed"] == 3


def test_case_is_deterministic_and_derived_from_the_fixture() -> None:
    cfg = SmokeFixtureConfig(seed=5, n_wells=4, n_months=3, nx=7, n_steps=4)
    fx = build_smoke_fixture(cfg)
    case = build_smoke_case(cfg, fx)
    assert case["schema_version"] == CASE_SCHEMA_VERSION
    assert case["fixture_content_hash"] == fx.content_hash
    assert case["grid"]["nx"] == 7
    assert case["schedule"]["n_steps"] == 4
    frac = case["controls"]["injected_pore_volume_fraction"]
    assert 0.25 < frac <= 0.75
    assert build_smoke_case(cfg, build_smoke_fixture(cfg)) == case


def test_case_changes_when_the_fixture_changes() -> None:
    a_cfg = SmokeFixtureConfig(seed=5, n_wells=4, n_months=3)
    b_cfg = SmokeFixtureConfig(seed=6, n_wells=4, n_months=3)
    a = build_smoke_case(a_cfg, build_smoke_fixture(a_cfg))
    b = build_smoke_case(b_cfg, build_smoke_fixture(b_cfg))
    assert case_bytes(a) != case_bytes(b)


def test_case_bytes_are_canonical_and_hashable(tmp_path: Path) -> None:
    cfg = SmokeFixtureConfig(seed=5, n_wells=2, n_months=2)
    case = build_smoke_case(cfg, build_smoke_fixture(cfg))
    path = tmp_path / "case.json"
    written = write_smoke_case(case, path)
    assert written == path.read_bytes()
    assert written == canonical_json(case).encode("utf-8")
    # This is the exact digest Julia is required to return as input_sha256.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(written).hexdigest()


def test_case_rejects_a_fixture_with_no_volumes() -> None:
    cfg = SmokeFixtureConfig(seed=1, n_wells=2, n_months=1)
    fx = build_smoke_fixture(cfg)
    import pyarrow as pa

    empty = pa.table(
        {
            "well_id": pa.array([], pa.string()),
            "month": pa.array([], pa.string()),
            "liquid_m3": pa.array([], pa.float64()),
            "injection_m3": pa.array([], pa.float64()),
        }
    )
    with pytest.raises(ValueError):
        build_smoke_case(cfg, type(fx)(wells=fx.wells, well_month=empty, content_hash="h", seed=1))
