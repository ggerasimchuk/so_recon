"""E01.13.1 — the capability assertions that need no solver: the record and the refusals.

Task 13 builds a SMALL educational black-oil capability beside the oil-water one, and 13.1
is explicit that a BO test is not marked passed on one successful constructor. This module
holds the half of 13.1 that is about the contract — what a black-oil case may declare, what
it may not, and that an unknown physics class is `INVALID_INPUT` rather than a default —
and `tests/integration/test_e01_blackoil.py` holds the half that reads what a real session
published.

Nothing here launches Julia. The three-phase saturation identity, the gas that appears below
the bubble point and the component gas balance are measured on real native results, in the
integration module, because a saturation identity a test wrote down itself is not evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from so_recon.paths import ProjectPaths, find_repo_root
from so_recon.simulator.case_io import (
    CaseIntegrityError,
    compute_model_hash,
    decode_case_payload,
)
from so_recon.simulator.contracts import (
    CASE_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION_BO,
    BlackOilFluidSpec,
    CaseBundle,
    FluidSpec,
)

ROOT = find_repo_root(Path(__file__).resolve().parents[2])

DIGEST = "b" * 64


def bo_fluids(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "BO",
        "pvt_source": "JutulDarcy.blackoil_bench_pvt(:spe1)",
        "pvt_table_hashes": {name: DIGEST for name in ("pvtw", "pvto", "pvdg", "relperm")},
        "density_sc_kg_m3": [1037.84, 786.507, 0.969758],
        "initial_rs_m3_m3": 50.0,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------------------
# 13.1 — what the black-oil fluid record may say
# --------------------------------------------------------------------------------------


def test_the_capability_fluid_declares_its_pvt_by_the_numbers_it_was_built_from() -> None:
    """A PVT named only by the function that returned it is not identified by its numbers."""
    fluid = BlackOilFluidSpec.model_validate(bo_fluids())
    assert fluid.pvt_source == "JutulDarcy.blackoil_bench_pvt(:spe1)"
    assert set(fluid.pvt_table_hashes) >= {"pvtw", "pvto", "pvdg", "relperm"}
    assert fluid.academic_benchmark is True, "the SPE1 tables are a benchmark, not a field PVT"
    assert fluid.hysteresis == "none"
    assert fluid.rv == 0.0

    with pytest.raises(ValidationError, match="pvt_table_hashes must name every table"):
        BlackOilFluidSpec.model_validate(
            bo_fluids(pvt_table_hashes={"pvtw": DIGEST, "pvto": DIGEST})
        )


def test_the_three_phase_relperm_is_an_explicit_declared_approximation() -> None:
    """Independent Corey curves, exponents (2,2,2), residuals (0.1,0.1,0.0), endpoints (1,1,1)."""
    fluid = BlackOilFluidSpec.model_validate(bo_fluids())
    assert fluid.corey_exponents == (2.0, 2.0, 2.0)
    assert fluid.residual_saturations == (0.1, 0.1, 0.0)
    assert fluid.kr_endpoints == (1.0, 1.0, 1.0)
    assert "BrooksCorey" in fluid.relperm_definition


def test_the_residual_saturations_have_to_leave_the_three_phases_a_state_to_share() -> None:
    """Swc + Sorw + Sgc < 1, and endpoints positive. Refused, never clipped."""
    with pytest.raises(ValidationError, match="must be below 1"):
        BlackOilFluidSpec.model_validate(bo_fluids(residual_saturations=[0.5, 0.4, 0.2]))
    with pytest.raises(ValidationError, match="must be positive"):
        BlackOilFluidSpec.model_validate(bo_fluids(kr_endpoints=[1.0, 0.0, 1.0]))
    with pytest.raises(ValidationError, match="must not exceed 1"):
        BlackOilFluidSpec.model_validate(bo_fluids(kr_endpoints=[1.2, 1.0, 1.0]))


def test_a_vaporised_oil_term_is_refused_because_the_constructor_is_given_no_pvtg() -> None:
    with pytest.raises(ValidationError, match="disgas-only"):
        BlackOilFluidSpec.model_validate(bo_fluids(rv=1e-4))


def test_the_gas_reference_density_is_a_phase_and_has_to_be_positive() -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        BlackOilFluidSpec.model_validate(bo_fluids(density_sc_kg_m3=[1037.84, 786.507, 0.0]))


# --------------------------------------------------------------------------------------
# 13.1 — a JSON / unknown physics class is INVALID_INPUT, never a default
# --------------------------------------------------------------------------------------


def _case_payload(fluids: dict[str, Any], schema_version: str) -> dict[str, Any]:
    paths = ProjectPaths.default(ROOT)
    del paths
    return {
        "schema_version": schema_version,
        "spec_version": "4.0",
        "case_id": "unit-bo",
        "world_id": "unit",
        "information_mode": "synthetic_forward",
        "start_date": "2020-01-01",
        "cutoff": "2020-03-01",
        "report_edges_s": [0.0, 86400.0],
        "grid": _grid(),
        "rock": _rock(),
        "fluids": fluids,
        "wells": [],
        "controls": [],
        "initial": {"kind": "equilibrium", "meaning": "synthetic_initial"},
        "boundary": {"kind": "closed", "cells": []},
        "observations": {
            "dynamic_channels": [],
            "pressure_available": False,
            "truth_access": "forbidden",
        },
        "gravity_m_s2": 9.80665,
        "renderer_version": "unit",
        "units": {"control_rate": "m3_sc/day"},
        "seeds": {},
        "source_hashes": {},
        "model_hash": "c" * 64,
    }


def _ref(dataset: str, shape: tuple[int, ...], unit: str, axes: tuple[str, ...]) -> dict[str, Any]:
    return {
        "path": "artifacts/arrays/unit.h5",
        "dataset": dataset,
        "sha256": DIGEST,
        "shape": list(shape),
        "dtype": "float64",
        "unit": unit,
        "axis_order": list(axes),
    }


def _grid() -> dict[str, Any]:
    return {
        "shape": [1, 1, 1],
        "extent_m": [10.0, 10.0, 10.0],
        "cell_centers_m": _ref("cell_centers_m", (1, 3), "m", ("cell", "dim")),
        "cell_volume_m3": _ref("cell_volume_m3", (1,), "m3", ("cell",)),
        "neighbors": dict(_ref("neighbors", (0, 2), "1", ("face", "side")), dtype="int64"),
        "z_positive": "down",
        "crs": None,
    }


def _rock() -> dict[str, Any]:
    return {
        "porosity": _ref("porosity", (1,), "1", ("cell",)),
        "permeability_m2": _ref("permeability_m2", (3, 1), "m2", ("dim", "cell")),
        "rock_compressibility_pa_inv": 0.0,
    }


def test_an_unknown_physics_class_is_refused_and_never_defaulted_to_oil_water() -> None:
    """`fluids.kind = 'XYZ'` names no model this build has. That is INVALID_INPUT."""
    payload = _case_payload({"kind": "XYZ"}, CASE_SCHEMA_VERSION)
    with pytest.raises(ValidationError):
        CaseBundle.model_validate(payload)
    with pytest.raises(CaseIntegrityError):
        decode_case_payload(payload, where="unit")


def test_a_case_that_is_not_json_at_all_is_refused_by_the_decoder() -> None:
    with pytest.raises(CaseIntegrityError, match="not one this build reads"):
        decode_case_payload({"schema_version": "case-9"}, where="unit")


def test_a_case_1_manifest_may_not_smuggle_a_black_oil_fluid_through_the_legacy_decoder() -> None:
    """The legacy decoder reads case-1 as what it is; it never invents a class for it."""
    with pytest.raises(CaseIntegrityError, match="oil-water case by construction"):
        decode_case_payload(_case_payload(bo_fluids(), CASE_SCHEMA_VERSION), where="unit")


def test_the_declared_schema_and_the_declared_physics_class_agree() -> None:
    with pytest.raises(ValidationError, match="declares schema_version 'case-2'"):
        CaseBundle.model_validate(_case_payload(bo_fluids(), CASE_SCHEMA_VERSION))
    with pytest.raises(ValidationError, match="declares schema_version 'case-1'"):
        CaseBundle.model_validate(_case_payload(FluidSpec().model_dump(), CASE_SCHEMA_VERSION_BO))


# --------------------------------------------------------------------------------------
# 13.3 / the interface note — case-2 reads case-1, and no OW manifest is rewritten
# --------------------------------------------------------------------------------------


def test_every_published_oil_water_case_still_reads_and_still_hashes_to_its_own_model() -> None:
    """The legacy decoder is proved on the manifests that are really on disk, not on a mock.

    `schema_version` is the first field of `MODEL_HASH_FIELDS`. If `case-2` had been an
    upgrade in place rather than a second member of a union, every one of these would now
    hash to something else — which is exactly the damage this test exists to detect.
    """
    manifests = sorted((ROOT / "artifacts").rglob("case-*.json"))
    if not manifests:
        pytest.skip("no published case manifests in this checkout")
    read = 0
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "schema_version" not in payload:
            continue
        case = decode_case_payload(payload, where=str(path.name))
        assert case.schema_version == (
            CASE_SCHEMA_VERSION_BO if case.fluids.kind == "BO" else CASE_SCHEMA_VERSION
        )
        assert compute_model_hash(case) == case.model_hash, path.name
        read += 1
    assert read > 0, "no case manifest was actually decoded"


# --------------------------------------------------------------------------------------
# 13.1 — the state assertion itself, as the brief spells it
# --------------------------------------------------------------------------------------


def assert_bo_state(
    sw: np.ndarray,
    so: np.ndarray,
    sg: np.ndarray,
    rs: np.ndarray,
    bo: np.ndarray,
    bg: np.ndarray,
) -> None:
    """Plan 13.1, verbatim. Imported by the integration module and applied to real states."""
    np.testing.assert_allclose(sw + so + sg, 1.0, atol=1e-10)
    assert np.all(sg >= -1e-8)
    assert np.all(rs >= 0)
    assert np.all(bo > 0) and np.all(bg > 0)


def test_the_state_assertion_refuses_a_state_that_does_not_close() -> None:
    """A check that cannot fail is not a check."""
    ones = np.ones(3)
    assert_bo_state(0.2 * ones, 0.7 * ones, 0.1 * ones, 50.0 * ones, 1.2 * ones, 0.01 * ones)
    with pytest.raises(AssertionError):
        assert_bo_state(0.2 * ones, 0.7 * ones, 0.2 * ones, 50.0 * ones, 1.2 * ones, 0.01 * ones)
    with pytest.raises(AssertionError):
        assert_bo_state(0.2 * ones, 0.7 * ones, 0.1 * ones, -1.0 * ones, 1.2 * ones, 0.01 * ones)
    with pytest.raises(AssertionError):
        assert_bo_state(0.2 * ones, 0.7 * ones, 0.1 * ones, 50.0 * ones, 1.2 * ones, 0.0 * ones)
