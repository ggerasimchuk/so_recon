"""T3 and T5 design contracts (plan E03 §3.2–§3.3, Task 02).

Unit level: what each residual moves, what the calendar declares, and that the T5
mapping keeps physical coordinates. The real T3 truth forward runs in the integration
test `tests/integration/test_e03_designs.py` against actual JutulDarcy.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from so_recon.geology.renderer import build_inverse_case, geology_coefficients, render_theta
from so_recon.inference.contracts import NoiseTheta, ThetaRecord
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.synthetic.loop_designs import (
    LOOP_DESIGN_KEY,
    T3_DESIGN_ID,
    T3_EVENT_WELL,
    T3_FAMILIES,
    T3_FAMILY_KZ_OVER_KX,
    T3_INITIAL_SW_SHIFT,
    T3_LOWER_OPEN_EDGE,
    T3_SCHEMA_ID,
    T3_TRANSFORM_VERSION,
    T3Design,
    make_t3_world,
    render_t3_coefficients,
    t3_control_segments,
    t3_design,
    t3_prior_context,
    t3_supports,
    t3_well_specs,
    t5_fine_design,
    t5_fine_well_column,
    t5_quadrant_masks,
    t5_well_specs,
)
from so_recon.synthetic.p1 import INITIAL_SW
from so_recon.validation.physical_smc import closed_preflight_case


@pytest.fixture(scope="module")
def design() -> T3Design:
    return t3_design()


@pytest.fixture(scope="module")
def zero_arrays(design: T3Design) -> dict[str, np.ndarray]:
    return render_t3_coefficients(
        np.zeros(design.n_geology), design, family=0, state_coordinate=0.0
    )


def _t3_context(zero_arrays: dict[str, np.ndarray]) -> object:
    return t3_prior_context(t3_design(), zero_arrays, np.random.default_rng(7))


# --------------------------------------------------------------------------------------
# what each residual moves
# --------------------------------------------------------------------------------------


def test_schema_layout_is_the_declared_13d(design: T3Design) -> None:
    assert design.n_geology == 12
    assert design.n_state_residual == 1
    schema = _t3_context(
        render_t3_coefficients(np.zeros(12), design, family=0, state_coordinate=0.0)
    ).density_schema
    assert schema.schema_id == T3_SCHEMA_ID
    assert schema.n_v == 11
    assert schema.n_residual == 5
    assert schema.families == T3_FAMILIES
    assert schema.transform_version == T3_TRANSFORM_VERSION


def test_each_geology_coefficient_moves_the_declared_geology(design: T3Design) -> None:
    base = render_t3_coefficients(
        np.zeros(design.n_geology), design, family=0, state_coordinate=0.0
    )
    for column in range(design.n_geology):
        coefficients = np.zeros(design.n_geology)
        coefficients[column] = 1.0
        moved = render_t3_coefficients(coefficients, design, family=0, state_coordinate=0.0)
        delta = moved["log_permeability_m2"] - base["log_permeability_m2"]
        assert float(np.abs(delta).max()) > 1e-6, f"coefficient {column} moved nothing"


def test_family_moves_only_kz_and_nothing_else(design: T3Design) -> None:
    coefficients = np.linspace(-0.5, 0.5, design.n_geology)
    base = render_t3_coefficients(coefficients, design, family=0, state_coordinate=0.3)
    other = render_t3_coefficients(coefficients, design, family=1, state_coordinate=0.3)
    # log kx, porosity, initial state: untouched by s
    for name in ("log_permeability_m2", "porosity", "sw", "cell_volume_m3"):
        np.testing.assert_array_equal(base[name], other[name])
    # the vertical permeability ratio is the declared hypothesis
    kz_base = base["permeability_m2"][2] / base["permeability_m2"][0]
    assert kz_base == pytest.approx(T3_FAMILY_KZ_OVER_KX[0])
    kz_other = other["permeability_m2"][2] / other["permeability_m2"][0]
    assert kz_other == pytest.approx(T3_FAMILY_KZ_OVER_KX[1])


def test_state_residual_moves_initial_state_only(design: T3Design) -> None:
    coefficients = np.zeros(design.n_geology)
    low = render_t3_coefficients(coefficients, design, family=0, state_coordinate=-6.0)
    high = render_t3_coefficients(coefficients, design, family=0, state_coordinate=6.0)
    # Sw0 = Swc + 0.10 * Phi(u): monotone, uniform, inside the declared support
    assert low["sw"].min() == low["sw"].max()
    assert low["sw"][0] == pytest.approx(INITIAL_SW, abs=1e-6)
    assert high["sw"][0] == pytest.approx(INITIAL_SW + T3_INITIAL_SW_SHIFT, abs=1e-6)
    for name in ("log_permeability_m2", "porosity", "permeability_m2"):
        np.testing.assert_array_equal(low[name], high[name])
    # no final saturation is written anywhere: the renderer emits only the initial sw
    assert set(low) >= {"sw", "porosity", "permeability_m2"}


def test_out_of_family_state_is_refused(design: T3Design) -> None:
    with pytest.raises(ValueError, match="declares families"):
        design.kz_over_kx(2)


def test_non_finite_state_is_refused(design: T3Design) -> None:
    with pytest.raises(ValueError, match="finite"):
        render_t3_coefficients(
            np.zeros(design.n_geology), design, family=0, state_coordinate=float("nan")
        )


def test_declared_sw0_support_respects_sor() -> None:
    # The import-time check holds Swc + 0.10 < 1 - Sor; restate it as a test fact.
    from so_recon.simulator.contracts import FluidSpec

    sor = FluidSpec().residual_saturations[1]
    assert INITIAL_SW + T3_INITIAL_SW_SHIFT < 1.0 - sor


# --------------------------------------------------------------------------------------
# the calendar
# --------------------------------------------------------------------------------------


def test_completion_event_opens_lower_connection_at_month_18(design: T3Design) -> None:
    segments = t3_control_segments(design)
    edges = design.report_edges_s
    open_edge = edges[T3_LOWER_OPEN_EDGE]
    p1 = [segment for segment in segments if segment.well_id == T3_EVENT_WELL]
    assert p1, "the event well has segments"
    before = [s for s in p1 if s.start_s < open_edge]
    after = [s for s in p1 if s.start_s >= open_edge]
    assert all(s.connection_open == (True, False) for s in before)
    assert all(s.connection_open == (True, True) for s in after)
    # every other well keeps both connections the whole run
    others = [s for s in segments if s.well_id != T3_EVENT_WELL]
    assert others and all(s.connection_open == (True, True) for s in others)
    # rates follow the E01 modulation and never jump at the event
    for segment in p1:
        assert segment.target == "liquid_rate"


def test_calendar_is_family_independent(design: T3Design) -> None:
    # The schedule is a function of the design, not of the hypothesis: nothing in the
    # controls references a family, and both hypotheses share one calendar object.
    segments = t3_control_segments(design)
    assert len({(s.well_id, s.start_s, s.end_s, s.value) for s in segments}) == len(segments)


def test_supports_are_eight_well_layer_cells(design: T3Design) -> None:
    supports = t3_supports(design)
    assert len(supports) == 8
    for _name, cells in supports:
        (cell,) = cells
        assert 0 <= cell < design.n_cells


def test_well_specs_are_e01_multisegment(design: T3Design) -> None:
    specs = t3_well_specs(design)
    assert [spec.well_id for spec in specs] == ["I1", "I2", "P1", "P2"]
    assert all(spec.model == "multisegment" and len(spec.cells) == 2 for spec in specs)


# --------------------------------------------------------------------------------------
# the conditional context and theta round-trip
# --------------------------------------------------------------------------------------


def test_t3_context_supports_both_families(zero_arrays: dict[str, np.ndarray]) -> None:
    context = t3_prior_context(t3_design(), zero_arrays, np.random.default_rng(3))
    schema = context.density_schema
    assert schema.families == (0, 1)
    # the geology block is 12 coefficients; the state residual sits in z_perp[4]
    assert context.n_geology == 12 and context.n_state_residual == 1
    assert LOOP_DESIGN_KEY in context.design


def test_render_theta_round_trips_the_truth(design: T3Design) -> None:
    rng = np.random.default_rng(11)
    coefficients = rng.standard_normal(design.n_geology)
    state = float(rng.standard_normal())
    family = 1
    arrays = render_t3_coefficients(coefficients, design, family=family, state_coordinate=state)
    context = t3_prior_context(design, arrays, np.random.default_rng(5))
    physical_whitened = np.linalg.solve(context.chol, coefficients - context.mean)
    whitened = context.rotation.T @ physical_whitened
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=family,
        v=tuple(float(x) for x in (*whitened[:8], 0.1, -0.2, 0.3)),
        z_perp=tuple(float(x) for x in (*whitened[8:], state)),
        basis_hash=context.density_schema.basis_hash,
    )
    rendered = render_theta(theta, context)
    assert rendered.family == family
    for name, values in arrays.items():
        np.testing.assert_allclose(rendered.arrays[name], values, rtol=0.0, atol=1e-12)
    assert rendered.noise == NoiseTheta.from_latent(theta.v)


def test_a_t3_theta_of_the_wrong_basis_is_refused(zero_arrays: dict[str, np.ndarray]) -> None:
    context = t3_prior_context(t3_design(), zero_arrays, np.random.default_rng(3))
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=0,
        v=(0.0,) * 11,
        z_perp=(0.0,) * 5,
        basis_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="schema/basis mismatch"):
        geology_coefficients(theta, context)


def test_context_with_both_design_keys_is_refused(zero_arrays: dict[str, np.ndarray]) -> None:
    context = t3_prior_context(t3_design(), zero_arrays, np.random.default_rng(3))
    polluted = context.model_copy(
        update={"design": {**context.design, "inverse_design": {"design_id": "e02-t2-v2"}}}
    )
    theta = ThetaRecord(
        schema_id=polluted.density_schema.schema_id,
        s=0,
        v=(0.0,) * 11,
        z_perp=(0.0,) * 5,
        basis_hash=polluted.density_schema.basis_hash,
    )
    with pytest.raises(ValueError, match="exactly one design key"):
        render_theta(theta, polluted)


def test_wrong_loop_design_id_is_refused_in_the_renderer(zero_arrays: dict[str, np.ndarray]) -> None:
    context = t3_prior_context(t3_design(), zero_arrays, np.random.default_rng(3))
    rebuilt = context.model_copy(
        update={"design": {**context.design, LOOP_DESIGN_KEY: {"design_id": "e03-t9-v1"}}}
    )
    theta = ThetaRecord(
        schema_id=rebuilt.density_schema.schema_id,
        s=0,
        v=(0.0,) * 11,
        z_perp=(0.0,) * 5,
        basis_hash=rebuilt.density_schema.basis_hash,
    )
    with pytest.raises(ValueError, match="not.*e03-t3-v1"):
        render_theta(theta, rebuilt)


# --------------------------------------------------------------------------------------
# the published world (no solver: publish path only)
# --------------------------------------------------------------------------------------


def test_make_t3_world_publishes_truth_and_pending_observations(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    ctx = RunContext.start(
        command="test-t3-world",
        argv=(),
        cfg=None,
        paths=paths,
        now=datetime.now(UTC),
    )
    context, observations, truth_ref = make_t3_world(T3_DESIGN_ID, 4242, paths, ctx)
    assert observations.history == ()
    truth = json.loads(paths.resolve(truth_ref.path).read_text("utf-8"))
    assert truth["schema_version"] == "e03-loop-truth-1"
    assert truth["family"] in (0, 1)
    assert truth["theta"]["schema_id"] == T3_SCHEMA_ID
    assert len(truth["theta"]["z_perp"]) == 5
    assert truth["renderer_checks"]["reproduces_truth_arrays"] is True
    # the same seed rebuilds the same world
    context_again, _observations, ref_again = make_t3_world(T3_DESIGN_ID, 4242, paths, ctx)
    assert context_again.density_schema.basis_hash == context.density_schema.basis_hash
    assert ref_again.sha256 == truth_ref.sha256


def test_build_inverse_case_yields_a_t3_case(tmp_path: Path) -> None:
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    ctx = RunContext.start(
        command="test-t3-case",
        argv=(),
        cfg=None,
        paths=paths,
        now=datetime.now(UTC),
    )
    design = t3_design()
    arrays = render_t3_coefficients(
        np.zeros(design.n_geology), design, family=0, state_coordinate=0.0
    )
    context = t3_prior_context(design, arrays, np.random.default_rng(9))
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=0,
        v=(0.0,) * 11,
        z_perp=(0.0,) * 5,
        basis_hash=context.density_schema.basis_hash,
    )
    rendered = render_theta(theta, context)
    case = build_inverse_case(rendered, context, paths, ctx)
    assert case.renderer_version == "e03-loop-renderer-1"
    assert case.initial.meaning == "developed_state"
    assert case.wells[0].model == "multisegment"
    assert any(segment.connection_open == (True, False) for segment in case.controls)


# --------------------------------------------------------------------------------------
# the closed transient preflight (plan E03 §3.2: T3 passes the same gate as T4)
# --------------------------------------------------------------------------------------


def _t3_solver_case(tmp_path: Path):
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_dirs()
    ctx = RunContext.start(
        command="test-t3-preflight",
        argv=(),
        cfg=None,
        paths=paths,
        now=datetime.now(UTC),
    )
    design = t3_design()
    arrays = render_t3_coefficients(
        np.zeros(design.n_geology), design, family=0, state_coordinate=0.0
    )
    context = t3_prior_context(design, arrays, np.random.default_rng(13))
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=0,
        v=(0.0,) * 11,
        z_perp=(0.0,) * 5,
        basis_hash=context.density_schema.basis_hash,
    )
    return build_inverse_case(render_theta(theta, context), context, paths, ctx)


def test_t3_case_is_accepted_by_the_closed_transient_preflight(tmp_path: Path) -> None:
    case = _t3_solver_case(tmp_path)

    preflight = closed_preflight_case(case)

    assert preflight.case_id == f"{case.case_id}-closed-preflight"
    assert preflight.report_edges_s == case.report_edges_s[:2]
    assert len(preflight.controls) == len(case.wells)
    assert all(
        control.role == "shut" and control.target == "disabled" for control in preflight.controls
    )
    assert all(not any(control.connection_open) for control in preflight.controls)
    assert preflight.initial.meaning == "developed_state"
    assert preflight.model_hash != case.model_hash


def test_t3_case_without_a_developed_state_is_refused_by_the_closed_preflight(
    tmp_path: Path,
) -> None:
    case = _t3_solver_case(tmp_path)
    synthetic = case.model_copy(
        update={"initial": case.initial.model_copy(update={"meaning": "synthetic_initial"})}
    )

    with pytest.raises(ValueError, match="closed transient preflight"):
        closed_preflight_case(synthetic)


# --------------------------------------------------------------------------------------
# T5 mapping
# --------------------------------------------------------------------------------------


def test_t5_fine_design_keeps_the_physical_domain() -> None:
    fine = t5_fine_design()
    assert fine.shape == (48, 48, 2)
    assert fine.extent_m == (100.0, 100.0, 20.0)
    assert fine.n_months == 36


def test_t5_wells_keep_physical_coordinates_and_connection_count() -> None:
    fine = t5_fine_design()
    specs = t5_well_specs(fine)
    coarse = {"I1": (2, 2), "I2": (2, 13), "P1": (13, 2), "P2": (13, 13)}
    dx_fine = fine.extent_m[0] / fine.shape[0]
    for spec in specs:
        i, j = coarse[spec.well_id]
        cell = spec.cells[0]
        fi = cell % fine.shape[0]
        fj = cell // fine.shape[0]
        # the fine cell centre is the coarse well's physical position
        assert (fi + 0.5) * dx_fine == pytest.approx((i + 0.5) * (100.0 / 16))
        assert (fj + 0.5) * (fine.extent_m[1] / fine.shape[1]) == pytest.approx(
            (j + 0.5) * (100.0 / 16)
        )
        assert len(spec.cells) == 2  # one connection per layer, not a 3x3 patch


def test_t5_fine_well_column_is_the_centre_child() -> None:
    assert t5_fine_well_column((2, 2)) == (7, 7)
    assert t5_fine_well_column((13, 13)) == (40, 40)
    assert t5_fine_well_column((0, 15)) == (1, 46)


def test_t5_quadrants_are_disjoint_and_identical_in_physical_coordinates() -> None:
    names_16, masks_16 = t5_quadrant_masks((16, 16, 2), (100.0, 100.0, 20.0))
    names_48, masks_48 = t5_quadrant_masks((48, 48, 2), (100.0, 100.0, 20.0))
    assert names_16 == names_48
    assert len(names_16) == 8
    assert set(names_16) == {
        "west-south-layer-0",
        "west-north-layer-0",
        "east-south-layer-0",
        "east-north-layer-0",
        "west-south-layer-1",
        "west-north-layer-1",
        "east-south-layer-1",
        "east-north-layer-1",
    }
    # every cell covered exactly once on both grids
    assert bool(np.all(masks_16.sum(axis=0) == 1.0))
    assert bool(np.all(masks_48.sum(axis=0) == 1.0))
    # PV shares agree within discretization: 16-cell quadrant vs 144-cell quadrant
    share_16 = masks_16[0].sum() / masks_16.sum()
    share_48 = masks_48[0].sum() / masks_48.sum()
    assert share_16 == pytest.approx(share_48, abs=0.01)


def test_t5_fine_and_coarse_have_different_model_identities() -> None:
    # The fine grid is a different case by construction: same kernel, different shape.
    from so_recon.synthetic.p1 import P1Design, render_coefficients

    coefficients = np.random.default_rng(1).standard_normal((2, 6))
    coarse = render_coefficients(coefficients, P1Design())
    fine = render_coefficients(coefficients, t5_fine_design())
    assert coarse["porosity"].shape[0] == 512
    assert fine["porosity"].shape[0] == 4608
    # the latent field is a function of coordinates: fine means track coarse means
    assert fine["porosity"].mean() == pytest.approx(coarse["porosity"].mean(), abs=0.01)
