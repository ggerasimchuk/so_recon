"""Deterministic synthetic fixture and the JutulDarcy case description for the E00 smoke.

Not a physical model: two tiny tables (wells, well_month) generated from a seeded PCG64
stream, hashed by canonical JSON content so the hash is independent of Parquet encoding.
The fixture then determines case.json, which is the only input Julia reads — that makes
the smoke end-to-end: Julia returns the SHA-256 of the bytes it actually read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so_recon.config.schema import SmokeFixtureConfig
from so_recon.registry.artifact import check_artifact_content
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import canonical_json, sha256_json

FIXTURE_SCHEMA_VERSION = "1"
CASE_SCHEMA_VERSION = "1"

_START_YEAR = 2020
_START_MONTH = 1

# Fixed case geometry and fluid properties. Only the injection fraction is data-derived.
_CELL_DX_M = 50.0
_CELL_DY_M = 50.0
_CELL_DZ_M = 10.0
_PERMEABILITY_DARCY = 0.1
_POROSITY = 0.25
_WATER_DENSITY = 1000.0
_OIL_DENSITY = 850.0
_INITIAL_PRESSURE_BAR = 150.0
_INITIAL_SW = 0.2
_PRODUCER_BHP_BAR = 100.0
_STEP_DAYS = 30.0


def _month_label(index: int) -> str:
    total = (_START_YEAR * 12 + (_START_MONTH - 1)) + index
    return f"{total // 12:04d}-{total % 12 + 1:02d}-01"


def fixture_content_hash(wells: pa.Table, well_month: pa.Table) -> str:
    return sha256_json({"wells": wells.to_pylist(), "well_month": well_month.to_pylist()})


@dataclass(frozen=True)
class SmokeFixture:
    wells: pa.Table
    well_month: pa.Table
    content_hash: str
    seed: int


def build_smoke_fixture(cfg: SmokeFixtureConfig) -> SmokeFixture:
    rng = np.random.default_rng(cfg.seed)
    well_ids = [f"W{i + 1:03d}" for i in range(cfg.n_wells)]
    roles = ["injector" if i % 2 == 0 else "producer" for i in range(cfg.n_wells)]
    xs = np.round(rng.uniform(0.0, 1000.0, size=cfg.n_wells), 3)
    ys = np.round(rng.uniform(0.0, 1000.0, size=cfg.n_wells), 3)
    wells = pa.table(
        {
            "well_id": pa.array(well_ids, pa.string()),
            "role": pa.array(roles, pa.string()),
            "x_m": pa.array(xs.tolist(), pa.float64()),
            "y_m": pa.array(ys.tolist(), pa.float64()),
        }
    )

    wm_ids: list[str] = []
    wm_month: list[str] = []
    wm_liq: list[float] = []
    wm_inj: list[float] = []
    for wid, role in zip(well_ids, roles, strict=True):
        volumes = np.round(rng.uniform(10.0, 100.0, size=cfg.n_months), 6)
        for m in range(cfg.n_months):
            wm_ids.append(wid)
            wm_month.append(_month_label(m))
            v = float(volumes[m])
            wm_liq.append(0.0 if role == "injector" else v)
            wm_inj.append(v if role == "injector" else 0.0)
    well_month = pa.table(
        {
            "well_id": pa.array(wm_ids, pa.string()),
            "month": pa.array(wm_month, pa.string()),
            "liquid_m3": pa.array(wm_liq, pa.float64()),
            "injection_m3": pa.array(wm_inj, pa.float64()),
        }
    )
    return SmokeFixture(
        wells=wells,
        well_month=well_month,
        content_hash=fixture_content_hash(wells, well_month),
        seed=cfg.seed,
    )


def write_smoke_fixture(fixture: SmokeFixture, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wells_path = out_dir / "wells.parquet"
    wm_path = out_dir / "well_month.parquet"
    meta_path = out_dir / "fixture_meta.json"
    payloads: dict[Path, bytes] = {}
    for path, table in ((wells_path, fixture.wells), (wm_path, fixture.well_month)):
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer)
        payloads[path] = buffer.getvalue().to_pybytes()
    payloads[meta_path] = (
        json.dumps(
            {
                "schema_version": FIXTURE_SCHEMA_VERSION,
                "seed": fixture.seed,
                "content_hash": fixture.content_hash,
                "n_rows": {
                    "wells": fixture.wells.num_rows,
                    "well_month": fixture.well_month.num_rows,
                },
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    # Check the whole set before writing, so a conflict in the last file cannot
    # replace the first two. Identical reuse preserves the original files.
    for path, data in payloads.items():
        check_artifact_content(path, data)
    for path, data in payloads.items():
        if not path.exists():
            write_bytes_atomic(path, data)
    return {"wells": wells_path, "well_month": wm_path, "meta": meta_path}


def build_smoke_case(cfg: SmokeFixtureConfig, fixture: SmokeFixture) -> dict[str, Any]:
    """Derive the JutulDarcy case description from the configuration and the fixture."""
    total_injection = float(sum(fixture.well_month["injection_m3"].to_pylist()))
    total_liquid = float(sum(fixture.well_month["liquid_m3"].to_pylist()))
    total = total_injection + total_liquid
    if total <= 0.0:
        raise ValueError("fixture carries no volumes; cannot derive an injection target")
    fraction = round(0.25 + 0.5 * (total_injection / total), 9)
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "seed": cfg.seed,
        "fixture_content_hash": fixture.content_hash,
        "grid": {"nx": cfg.nx, "dx_m": _CELL_DX_M, "dy_m": _CELL_DY_M, "dz_m": _CELL_DZ_M},
        "rock": {"permeability_darcy": _PERMEABILITY_DARCY, "porosity": _POROSITY},
        "fluids": {
            "water_density_kg_m3": _WATER_DENSITY,
            "oil_density_kg_m3": _OIL_DENSITY,
        },
        "initial": {
            "pressure_bar": _INITIAL_PRESSURE_BAR,
            "water_saturation": _INITIAL_SW,
            "oil_saturation": round(1.0 - _INITIAL_SW, 9),
        },
        "schedule": {"n_steps": cfg.n_steps, "dt_days": _STEP_DAYS},
        "controls": {
            "injected_pore_volume_fraction": fraction,
            "producer_bhp_bar": _PRODUCER_BHP_BAR,
        },
    }


def case_bytes(case: dict[str, Any]) -> bytes:
    """The exact bytes Julia reads and hashes back as input_sha256."""
    return canonical_json(case).encode("utf-8")


def write_smoke_case(case: dict[str, Any], path: Path) -> bytes:
    payload = case_bytes(case)
    check_artifact_content(path, payload)
    if not path.exists():
        write_bytes_atomic(path, payload)
    return payload
