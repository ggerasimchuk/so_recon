"""E02.2 — the latent renderer: a `ThetaRecord` to the physical coefficient arrays.

Two separate claims are under test here.

The first is that extracting `render_coefficients` out of E01's `render_p1` changed
nothing. The golden digests below were taken from the generator BEFORE the extraction, so
they are a drift detector and not a restatement of the current code: a world that renders
one bit differently stops matching them.

The second is that the renderer is the honest image of the latent measure. It keeps the
residual coordinates (SPEC §7.5), it adds no Jacobian to the latent density (the measure is
`counting_x_latent_lebesgue`), it writes nothing and reads no truth, and a geology E01
refuses to publish comes back as `RendererNumericalError` rather than as a zero prior
density: the conditional prior is not silently truncated to the region that renders nicely.
"""

from __future__ import annotations

import builtins
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from so_recon.geology.conditional import (
    FAMILY_BY_S,
    LOG_K_QUANTITY,
    p1_prior_context,
)
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.geology.renderer import (
    RenderedParameters,
    geology_coefficients,
    render_theta,
    renderer_hash,
    theta_hash,
)
from so_recon.inference.contracts import (
    N_P1_GEOLOGY_IN_V,
    NoiseTheta,
    PriorContext,
    RendererNumericalError,
    ThetaRecord,
)
from so_recon.synthetic.p1 import (
    FAMILY_KZ_OVER_KX,
    N_LAYERS,
    N_MODES,
    P1_PARENTS,
    P1Design,
    observation_supports,
    render_coefficients,
    render_p1,
    streams,
    support_mean,
)

#: SHA-256 of every scientific array of the five P1 parents, recorded from `render_p1`
#: before `render_coefficients` was extracted from it. E01's world outputs are unchanged if
#: and only if these still match.
GOLDEN_WORLD_ARRAY_SHA256: dict[tuple[int, str], str] = {
    (41, "base"): "9c5cb80fecfa17d6e9cc3e4f2b37ff3d726fe62c96a92190d44806ba2697b287",
    (42, "base"): "6c42a637c405d527fc947f952de8f842d736f68327b7c88414f8643d54635a23",
    (43, "base"): "417270053df33b53b1556b78222d5ff2a48b3b78c7fb9bd7c8ba8468abd3b840",
    (44, "low_vertical"): "944e0cb60b22682d6fdb237a70f0927d07aafa28d82e44c9fc7e8ddccf6d0f83",
    (45, "high_contrast"): "681e5c788c8875a883409ffff39ea334595cd7d0a410668a846bb876d8ca2ab1",
}


def _array_digest(arrays: Mapping[str, NDArray[np.float64]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name], dtype=np.float64)
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _context(seed: int = 41, design: P1Design | None = None) -> PriorContext:
    design = design or P1Design()
    observations = {
        str(row["observation_id"]): float(row["value"])
        for row in render_p1(seed, design).static_observations.to_pylist()
        if row["quantity"] == LOG_K_QUANTITY
    }
    return p1_prior_context(P1Design(), observations)


def _theta(context: PriorContext, coordinates: NDArray[np.float64], s: int = 0) -> ThetaRecord:
    schema = context.density_schema
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=s,
        v=tuple(coordinates[: schema.n_v]),
        z_perp=tuple(coordinates[schema.n_v :]),
        basis_hash=schema.basis_hash,
    )


def _draw(context: PriorContext, seed: int, s: int = 0) -> ThetaRecord:
    n = context.density_schema.n_v + context.density_schema.n_residual
    return _theta(context, np.random.default_rng(seed).standard_normal(n), s)


# --------------------------------------------------------------------------------------
# the E01 extraction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "family"), P1_PARENTS)
def test_the_world_arrays_are_the_ones_e01_published(seed: int, family: str) -> None:
    world = render_p1(seed, P1Design(family=family))
    assert _array_digest(world.arrays) == GOLDEN_WORLD_ARRAY_SHA256[(seed, family)]


def test_render_p1_renders_its_coefficients_through_the_extracted_renderer() -> None:
    design = P1Design()
    world = render_p1(41, design)
    coefficients = np.random.default_rng(streams(41).geology).standard_normal((N_LAYERS, N_MODES))
    np.testing.assert_array_equal(
        coefficients, np.asarray(world.theta["coefficients"], dtype=np.float64)
    )
    arrays = render_coefficients(coefficients, design)
    assert sorted(arrays) == sorted(world.arrays)
    for name, values in arrays.items():
        assert values.tobytes() == world.arrays[name].tobytes(), name


def test_render_coefficients_is_deterministic() -> None:
    design = P1Design()
    coefficients = np.random.default_rng(2).standard_normal((N_LAYERS, N_MODES))
    first = render_coefficients(coefficients, design)
    second = render_coefficients(coefficients, design)
    for name, values in first.items():
        assert values.tobytes() == second[name].tobytes(), name


@pytest.mark.parametrize("shape", [(12,), (3, 4), (2, 5), (1, 6)])
def test_render_coefficients_refuses_a_shape_it_cannot_read(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="coefficients"):
        render_coefficients(np.zeros(shape), P1Design())


# --------------------------------------------------------------------------------------
# the latent renderer
# --------------------------------------------------------------------------------------


def test_the_renderer_reproduces_an_e01_world_from_its_own_coefficients() -> None:
    """The two paths meet: the latent coordinates that map to E01's draw render E01's rock."""
    design = P1Design()
    world = render_p1(41, design)
    context = _context()
    target = np.asarray(world.theta["coefficients"], dtype=np.float64).ravel()
    # A general solve: the factor is the symmetric root of the covariance, not a
    # triangular one, and inverting it as though it were would silently give a
    # different point.
    whitened = context.rotation.T @ np.linalg.solve(context.chol, target - context.mean)
    nuisance = [0.3, -0.4, 0.5]
    coordinates = np.concatenate(
        [whitened[:N_P1_GEOLOGY_IN_V], nuisance, whitened[N_P1_GEOLOGY_IN_V:]]
    )
    theta = _theta(context, coordinates)
    np.testing.assert_allclose(geology_coefficients(theta, context), target, atol=1e-9)
    rendered = render_theta(theta, context)
    assert sorted(rendered.arrays) == sorted(world.arrays)
    for name, values in world.arrays.items():
        np.testing.assert_allclose(rendered.arrays[name], values, rtol=1e-9, atol=1e-12)


def test_the_rendered_parameters_carry_the_family_the_noise_and_a_hash() -> None:
    context = _context()
    theta = _draw(context, 3, s=1)
    rendered = render_theta(theta, context)
    assert isinstance(rendered, RenderedParameters)
    assert rendered.family == 1
    assert rendered.noise == NoiseTheta.from_latent(theta.v)
    assert rendered.renderer_hash == renderer_hash(context)
    assert len(rendered.renderer_hash) == 64


def test_rendering_the_same_theta_twice_gives_the_same_bytes() -> None:
    context = _context()
    theta = _draw(context, 5)
    first, second = render_theta(theta, context), render_theta(theta, context)
    for name, values in first.arrays.items():
        assert values.tobytes() == second.arrays[name].tobytes(), name


def test_the_renderer_reads_no_file_and_writes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context()
    theta = _draw(context, 7)
    render_theta(theta, context)  # warm every lazy import before the file system is closed

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the latent renderer touched the file system")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    assert render_theta(theta, context).arrays


# --------------------------------------------------------------------------------------
# the residual is part of the measure
# --------------------------------------------------------------------------------------


def test_the_residual_coordinates_move_the_rock() -> None:
    """SPEC §7.5: the remaining modes are kept. A renderer that dropped them would not move."""
    context = _context()
    base = _draw(context, 11)
    moved = base.model_copy(update={"z_perp": tuple(-x - 1.0 for x in base.z_perp)})
    first, second = render_theta(base, context), render_theta(moved, context)
    for name in ("permeability_m2", "porosity", "log_permeability_m2"):
        assert not np.array_equal(first.arrays[name], second.arrays[name]), name


def test_the_residual_moves_the_rock_without_moving_the_measured_logs() -> None:
    """What the residual block MEANS, read off the rendered arrays themselves.

    `z_perp` spans the directions the sparse logs left unconstrained, so a move in it must
    change the field everywhere and leave the eight support averages the prior was
    conditioned on exactly where they were. That is a statement about `chol @ rotation` —
    the map the renderer applies — and not about `rotation` alone.
    """
    context = _context()
    design = P1Design()
    supports = observation_supports(design)
    base = _draw(context, 101)
    moved_residual = base.model_copy(update={"z_perp": tuple(x + 1.5 for x in base.z_perp)})
    moved_informed = base.model_copy(
        update={"v": (base.v[0] + 1.5, *base.v[1:])},
    )

    def measured(theta: ThetaRecord) -> list[float]:
        field = render_theta(theta, context).arrays["log_permeability_m2"]
        return [support_mean(field, support.cell_ids) for support in supports]

    reference = measured(base)
    np.testing.assert_allclose(measured(moved_residual), reference, atol=1e-10)
    assert not np.array_equal(
        render_theta(moved_residual, context).arrays["log_permeability_m2"],
        render_theta(base, context).arrays["log_permeability_m2"],
    )
    assert float(np.max(np.abs(np.array(measured(moved_informed)) - reference))) > 0.1


def test_two_prior_draws_differ_in_permeability_and_porosity() -> None:
    context = _context()
    first, second = GaussianConditionalPrior(context).sample(2, np.random.default_rng(13))
    assert first.z_perp != second.z_perp
    a, b = render_theta(first, context), render_theta(second, context)
    assert not np.array_equal(a.arrays["permeability_m2"], b.arrays["permeability_m2"])
    assert not np.array_equal(a.arrays["porosity"], b.arrays["porosity"])


def test_all_twelve_latent_directions_reach_the_rock() -> None:
    context = _context()
    n = context.density_schema.n_v + context.density_schema.n_residual
    zero = np.zeros(n)
    reference = render_theta(_theta(context, zero), context).arrays["log_permeability_m2"]
    geology = [*range(N_P1_GEOLOGY_IN_V), *range(context.density_schema.n_v, n)]
    for index in geology:
        moved = zero.copy()
        moved[index] = 1.0
        rendered = render_theta(_theta(context, moved), context).arrays["log_permeability_m2"]
        assert not np.array_equal(rendered, reference), index


def test_the_hypothesis_moves_the_vertical_permeability_and_nothing_else() -> None:
    context = _context()
    theta = _draw(context, 17)
    first = render_theta(theta, context)
    second = render_theta(theta.model_copy(update={"s": 1}), context)
    for name in ("porosity", "log_permeability_m2", "pressure_pa", "sw"):
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name], err_msg=name)
    for rendered in (first, second):
        permeability = rendered.arrays["permeability_m2"]
        np.testing.assert_array_equal(permeability[0], permeability[1])
    np.testing.assert_array_equal(
        first.arrays["permeability_m2"][0], second.arrays["permeability_m2"][0]
    )
    for rendered, family in ((first, FAMILY_BY_S[0]), (second, FAMILY_BY_S[1])):
        permeability = rendered.arrays["permeability_m2"]
        np.testing.assert_allclose(
            permeability[2] / permeability[0], FAMILY_KZ_OVER_KX[family], rtol=1e-12
        )


# --------------------------------------------------------------------------------------
# the measure the renderer does NOT change
# --------------------------------------------------------------------------------------


def test_the_latent_density_carries_no_renderer_jacobian() -> None:
    """`k = exp(z)` is a physical transform; the latent measure stays Lebesgue in `z`.

    The plan fixes `measure='counting_x_latent_lebesgue'`. Folding the renderer's Jacobian
    in would add `sum(log k)` — some four figures here — to every latent log density and
    would make the prior a different, unnormalised measure.
    """
    context = _context()
    prior = GaussianConditionalPrior(context)
    theta = _draw(context, 19)
    coordinates = np.array([*theta.v, *theta.z_perp])
    expected = float(np.sum(stats.norm.logpdf(coordinates))) + math.log(0.5)
    assert prior.log_prob(theta) == pytest.approx(expected, abs=1e-12)

    rendered = render_theta(theta, context)
    log_k = rendered.arrays["log_permeability_m2"]
    np.testing.assert_allclose(np.log(rendered.arrays["permeability_m2"][0]), log_k, rtol=1e-12)
    jacobian = float(np.sum(log_k))
    assert abs(prior.log_prob(theta) - (expected - jacobian)) > 1.0
    assert context.density_schema.measure == "counting_x_latent_lebesgue"


def test_an_unrenderable_geology_is_a_failure_and_not_a_zero_prior_density() -> None:
    """E01 refuses an extreme permeability. That refusal is not a truncation of the prior.

    The prior stays the normalised Gaussian it declared, so `log_prob` is finite here. The
    physical evaluation stops instead, and it says which theta it stopped on.
    """
    context = _context()
    n = context.density_schema.n_v + context.density_schema.n_residual
    theta = _theta(context, np.full(n, 500.0))
    with pytest.raises(RendererNumericalError) as raised:
        render_theta(theta, context)
    assert theta_hash(theta) in str(raised.value)
    assert theta_hash(theta) in raised.value.reason
    assert math.isfinite(GaussianConditionalPrior(context).log_prob(theta))


def test_the_theta_hash_names_the_point_and_nothing_else() -> None:
    context = _context()
    theta = _draw(context, 23)
    assert theta_hash(theta) == theta_hash(theta.model_copy())
    assert theta_hash(theta) != theta_hash(theta.model_copy(update={"s": 1}))
    assert len(theta_hash(theta)) == 64


def test_the_renderer_hash_names_the_map_and_not_the_draw() -> None:
    context = _context()
    first, second = _draw(context, 29), _draw(context, 31, s=1)
    assert render_theta(first, context).renderer_hash == render_theta(second, context).renderer_hash
    assert renderer_hash(_context(42)) != renderer_hash(context)


@pytest.mark.parametrize("fault", ["schema_id", "basis_hash", "family", "n_v", "n_residual"])
def test_the_renderer_refuses_a_theta_from_another_density(fault: str) -> None:
    context = _context()
    theta = _draw(context, 37)
    updates: dict[str, dict[str, object]] = {
        "schema_id": {"schema_id": "toy-2"},
        "basis_hash": {"basis_hash": _context(43).density_schema.basis_hash},
        "family": {"s": 4},
        "n_v": {"v": theta.v[:-1]},
        "n_residual": {"z_perp": (*theta.z_perp, 0.5)},
    }
    with pytest.raises(ValueError):
        render_theta(theta.model_copy(update=updates[fault]), context)
