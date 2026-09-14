"""E02.2 — the conditional Gaussian prior over E01's twelve geology coefficients.

The subject of this module is a NORMALISED measure, so the tests are about normalisation
and about what the measure keeps:

* The conditioning algebra is checked against a second, differently-shaped derivation of
  the same Gaussian conditional, and on a scalar case whose answer is known by hand.
* The design matrix is checked against E01's OWN rendered field rather than against a
  re-derivation of the cosine basis: a matrix that merely looked like the generator's would
  condition on something nobody measured.
* The density is checked to integrate to one by an importance-sampling estimate under a
  proposal it never uses, and pointwise against `scipy.stats`, not against the helper it is
  built from.
* The residual block is checked to keep its FULL unit prior variance. SPEC §7.5 forbids
  zeroing the remaining modes, and a rotation that quietly shrank them would do exactly
  that while still looking like a Gaussian.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, stats

from so_recon.geology.conditional import (
    EIGENVALUE_TIE_RTOL,
    FAMILY_BY_S,
    LOG_K_QUANTITY,
    N_GEOLOGY,
    OBSERVATION_KEY,
    P1_SCHEMA_ID,
    SIGMA_KEY,
    TRANSFORM_VERSION,
    WHITENING_KEY,
    cell_log_permeability_operator,
    check_informed_split,
    condition_gaussian,
    information_matrix,
    log_k_observation_ids,
    p1_design_for,
    p1_prior_context,
    support_log_permeability_operator,
    tie_blocks,
    whitening_rotation,
)
from so_recon.geology.density import Density, GaussianConditionalPrior
from so_recon.geology.renderer import geology_coefficients
from so_recon.inference.contracts import N_P1_GEOLOGY_IN_V, PriorContext, ThetaRecord
from so_recon.simulator.contracts import MILLIDARCY_M2
from so_recon.synthetic.p1 import (
    LAYER_BASE_PERMEABILITY_MD,
    SIGMA_LOG_PERMEABILITY,
    P1Design,
    observation_supports,
    render_p1,
    support_mean,
)

#: The number of cheap latent draws the moment checks run on, and the tolerances the plan's
#: task brief fixes for them. At 20 000 draws each bound below is more than five standard
#: errors wide, so a pass is a statement about the sampler and not about the seed.
N_DRAWS = 20_000
MEAN_TOLERANCE = 0.04
VARIANCE_TOLERANCE = 0.05
FAMILY_TOLERANCE = 0.02


def _observations(seed: int = 41, design: P1Design | None = None) -> dict[str, float]:
    """The sparse noisy `log k` of one E01 world: the only information E02 conditions on."""
    design = design or P1Design()
    return {
        str(row["observation_id"]): float(row["value"])
        for row in render_p1(seed, design).static_observations.to_pylist()
        if row["quantity"] == LOG_K_QUANTITY
    }


def _context(seed: int = 41) -> PriorContext:
    return p1_prior_context(P1Design(), _observations(seed))


def _theta(context: PriorContext, coordinates: np.ndarray, s: int = 0) -> ThetaRecord:
    schema = context.density_schema
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=s,
        v=tuple(coordinates[: schema.n_v]),
        z_perp=tuple(coordinates[schema.n_v :]),
        basis_hash=schema.basis_hash,
    )


# --------------------------------------------------------------------------------------
# the conditioning algebra
# --------------------------------------------------------------------------------------


def test_gaussian_conditioning_has_known_mean_and_variance() -> None:
    mean, cov = condition_gaussian(np.array([[1.0]]), np.array([2.0]), np.array([1.0]))
    np.testing.assert_allclose(mean, [1.0], atol=1e-12)
    np.testing.assert_allclose(cov, [[0.5]], atol=1e-12)


def test_conditioning_matches_the_joint_gain_formula() -> None:
    """The same conditional by the other algebra: `Sigma A^T (A Sigma A^T + R)^-1` with Sigma=I.

    The implementation inverts a 12x12 precision; this inverts an 8x8 innovation covariance.
    Agreement is evidence about the mathematics rather than about one factorisation.
    """
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 3))
    y = rng.standard_normal(5)
    variance = rng.uniform(0.2, 2.0, size=5)
    mean, cov = condition_gaussian(a, y, variance)
    gain = a.T @ np.linalg.inv(a @ a.T + np.diag(variance))
    np.testing.assert_allclose(mean, gain @ y, atol=1e-10)
    np.testing.assert_allclose(cov, np.eye(3) - gain @ a, atol=1e-10)


def test_conditioning_returns_a_symmetric_positive_definite_covariance() -> None:
    rng = np.random.default_rng(11)
    a = rng.standard_normal((6, 4))
    _, cov = condition_gaussian(a, rng.standard_normal(6), np.full(6, 0.04))
    np.testing.assert_array_equal(cov, cov.T)
    assert float(np.min(np.linalg.eigvalsh(cov))) > 0.0


@pytest.mark.parametrize(
    ("a", "y", "variance"),
    [
        (np.ones((2, 2)), np.ones(3), np.ones(3)),
        (np.ones((2, 2)), np.ones(2), np.ones(3)),
        (np.ones(2), np.ones(2), np.ones(2)),
        (np.ones((2, 2)), np.array([np.nan, 1.0]), np.ones(2)),
        (np.ones((2, 2)), np.array([np.inf, 1.0]), np.ones(2)),
        (np.ones((2, 2)), np.ones(2), np.array([0.0, 1.0])),
        (np.ones((2, 2)), np.ones(2), np.array([-1.0, 1.0])),
    ],
)
def test_conditioning_refuses_what_it_cannot_condition(
    a: np.ndarray, y: np.ndarray, variance: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        condition_gaussian(a, y, variance)


# --------------------------------------------------------------------------------------
# the operator: E01's own basis, weights and support averaging
# --------------------------------------------------------------------------------------


def test_the_design_matrix_is_e01s_own_cosine_field() -> None:
    """`basis @ a + intercept` must BE the log permeability E01 rendered for those `a`."""
    design = P1Design()
    basis, intercept = cell_log_permeability_operator(design)
    world = render_p1(41, design)
    coefficients = np.asarray(world.theta["coefficients"], dtype=np.float64).ravel()
    np.testing.assert_allclose(
        basis @ coefficients + intercept, world.arrays["log_permeability_m2"], atol=1e-11
    )


def test_the_rows_of_a_are_the_support_averages_e01_measures() -> None:
    design = P1Design()
    basis, intercept = cell_log_permeability_operator(design)
    a, b = support_log_permeability_operator(design)
    supports = observation_supports(design)
    assert a.shape == (len(supports), N_GEOLOGY)
    assert b.shape == (len(supports),)
    for row, support in enumerate(supports):
        for column in range(N_GEOLOGY):
            assert a[row, column] == support_mean(basis[:, column], support.cell_ids)
        assert b[row] == support_mean(intercept, support.cell_ids)


def test_the_intercept_is_the_declared_base_permeability_in_square_metres() -> None:
    design = P1Design()
    _, b = support_log_permeability_operator(design)
    for row, support in enumerate(observation_supports(design)):
        base = LAYER_BASE_PERMEABILITY_MD[support.layer_index]
        assert b[row] == pytest.approx(math.log(base * MILLIDARCY_M2), abs=1e-12)


def test_each_support_reads_only_the_layer_it_sits_in() -> None:
    design = P1Design()
    a, _ = support_log_permeability_operator(design)
    for row, support in enumerate(observation_supports(design)):
        other = 1 - support.layer_index
        block = slice(other * (N_GEOLOGY // 2), (other + 1) * (N_GEOLOGY // 2))
        np.testing.assert_array_equal(a[row, block], np.zeros(N_GEOLOGY // 2))


def test_the_observation_ids_are_the_ones_e01_publishes() -> None:
    """The conditioning set is named, not positional: eight numbers in an unknown order
    would condition the wrong coefficients on the wrong support."""
    design = P1Design()
    published = [
        str(row["observation_id"])
        for row in render_p1(41, design).static_observations.to_pylist()
        if row["quantity"] == LOG_K_QUANTITY
    ]
    assert sorted(log_k_observation_ids(design)) == sorted(published)


def test_the_two_families_carry_the_same_information_about_g() -> None:
    """`s` moves `kz/kx` and nothing else, so `G` cannot tell the families apart."""
    a0, b0 = support_log_permeability_operator(P1Design(family=FAMILY_BY_S[0]))
    a1, b1 = support_log_permeability_operator(P1Design(family=FAMILY_BY_S[1]))
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_array_equal(b0, b1)


# --------------------------------------------------------------------------------------
# the whitening rotation
# --------------------------------------------------------------------------------------


def test_the_rotation_is_orthonormal_and_ordered_by_information() -> None:
    a, _ = support_log_permeability_operator(P1Design())
    variance = np.full(a.shape[0], SIGMA_LOG_PERMEABILITY**2)
    rotation, eigenvalues = whitening_rotation(information_matrix(a, variance))
    assert rotation.shape == (N_GEOLOGY, N_GEOLOGY)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(N_GEOLOGY), atol=1e-12)
    assert np.all(np.diff(eigenvalues) <= 1e-9)


def test_every_column_of_the_rotation_has_a_positive_leading_entry() -> None:
    a, _ = support_log_permeability_operator(P1Design())
    rotation, _ = whitening_rotation(information_matrix(a, np.full(a.shape[0], 0.04)))
    for column in range(N_GEOLOGY):
        magnitudes = np.abs(rotation[:, column])
        peak = float(np.max(magnitudes))
        leading = int(np.flatnonzero(magnitudes >= peak * (1.0 - 1e-9))[0])
        assert rotation[leading, column] > 0.0


def test_a_sign_tie_is_broken_by_the_lowest_coordinate_index() -> None:
    """Half this design's columns hold two entries of magnitude `1/sqrt(2)`.

    Which of them `argmax` calls the largest is decided by the last bit, so the sign of a
    stored latent coordinate would depend on the run. The tie goes to the lower index.
    """
    rotation, eigenvalues = whitening_rotation(
        information_matrix(np.array([[1.0, -1.0]]), np.ones(1))
    )
    root = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(rotation, [[root, root], [-root, root]], atol=1e-12)
    np.testing.assert_allclose(eigenvalues, [2.0, 0.0], atol=1e-12)


def test_a_degenerate_block_is_canonicalised_and_not_left_to_the_eigensolver() -> None:
    """A tied subspace has no preferred eigenvector, so the basis is DECLARED.

    `a` spans the first two coordinates through an arbitrary rotation, so the information
    matrix is `diag(1, 1, 0)`: one tied block on the plane and one on its normal. The
    canonical basis projects the standard unit vectors in coordinate order, so it is the
    identity regardless of which mixture LAPACK happened to return.
    """
    angle = 0.7
    a = np.array(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0]]
    )
    rotation, eigenvalues = whitening_rotation(information_matrix(a, np.ones(2)))
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(eigenvalues, [1.0, 1.0, 0.0], atol=1e-12)


def test_the_canonical_basis_survives_a_different_route_to_the_same_information() -> None:
    """Rotating the observations leaves `A^T R^-1 A` unchanged in exact arithmetic.

    In floating point it does not, and the degenerate blocks of this design span eight of
    the twelve directions. A basis that came straight out of `eigh` would move; this one
    must not, because a moved basis silently re-labels every stored latent coordinate.
    """
    a, _ = support_log_permeability_operator(P1Design())
    variance = np.full(a.shape[0], SIGMA_LOG_PERMEABILITY**2)
    unitary, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((a.shape[0], a.shape[0])))
    first, _ = whitening_rotation(information_matrix(a, variance))
    second, _ = whitening_rotation(information_matrix(unitary @ a, variance))
    np.testing.assert_allclose(second, first, atol=1e-9)


def test_tie_blocks_group_by_the_scale_of_the_spectrum() -> None:
    eigenvalues = np.array([10.0, 10.0 + 1e-13, 4.0, 1e-16, -1e-16])
    assert tie_blocks(eigenvalues) == ((0, 1), (2,), (3, 4))


def test_a_split_inside_a_degenerate_block_is_refused() -> None:
    """Eight coordinates go to `v` and four to `z_perp`; the cut may not halve a tie.

    Inside a tied block the individual directions are a convention, so a cut through one
    would make the meaning of `v` and `z_perp` depend on the eigensolver.
    """
    check_informed_split(np.array([9.0, 9.0, 2.0, 0.0]), 2)
    with pytest.raises(ValueError, match="degenerate"):
        check_informed_split(np.array([9.0, 9.0, 2.0, 0.0]), 1)


def test_the_p1_split_falls_on_the_information_boundary() -> None:
    a, _ = support_log_permeability_operator(P1Design())
    _, eigenvalues = whitening_rotation(
        information_matrix(a, np.full(a.shape[0], SIGMA_LOG_PERMEABILITY**2))
    )
    scale = float(np.max(eigenvalues))
    assert np.all(eigenvalues[:N_P1_GEOLOGY_IN_V] > EIGENVALUE_TIE_RTOL * scale)
    np.testing.assert_allclose(
        eigenvalues[N_P1_GEOLOGY_IN_V:], 0.0, atol=EIGENVALUE_TIE_RTOL * scale
    )


# --------------------------------------------------------------------------------------
# the prior context
# --------------------------------------------------------------------------------------


def test_the_context_carries_the_exact_conditional_law() -> None:
    design = P1Design()
    observations = _observations()
    context = p1_prior_context(design, observations)
    a, b = support_log_permeability_operator(design)
    y = np.array([observations[name] for name in log_k_observation_ids(design)]) - b
    mean, cov = condition_gaussian(a, y, np.full(len(y), SIGMA_LOG_PERMEABILITY**2))
    np.testing.assert_allclose(context.mean, mean, atol=1e-12)
    np.testing.assert_allclose(context.chol @ context.chol.T, cov, atol=1e-12)


def test_the_context_declares_the_p1_schema_the_plan_fixes() -> None:
    context = _context()
    schema = context.density_schema
    assert schema.schema_id == P1_SCHEMA_ID
    assert schema.transform_version == TRANSFORM_VERSION
    assert (schema.n_v, schema.n_residual) == (11, 4)
    assert schema.families == (0, 1)
    assert schema.measure == "counting_x_latent_lebesgue"
    assert (context.n_geology, context.n_state_residual) == (12, 0)


def test_the_context_persists_the_decomposition_a_resume_must_not_redo() -> None:
    """Rotation, mean and factor are arrays on the context; the spectrum travels beside them."""
    design = P1Design()
    context = p1_prior_context(design, _observations())
    a, _ = support_log_permeability_operator(design)
    _, eigenvalues = whitening_rotation(
        information_matrix(a, np.full(a.shape[0], SIGMA_LOG_PERMEABILITY**2))
    )
    stored = context.design[WHITENING_KEY]
    np.testing.assert_array_equal(stored["eigenvalues"], eigenvalues)
    assert stored["n_informed"] == N_P1_GEOLOGY_IN_V
    assert context.design[SIGMA_KEY] == SIGMA_LOG_PERMEABILITY
    assert context.design[OBSERVATION_KEY] == list(log_k_observation_ids(design))


def test_the_rotation_diagonalises_the_covariance_and_keeps_the_residual_variance() -> None:
    """SPEC §7.5: the remaining modes are kept. Here that is a number, not a promise."""
    context = _context()
    cov = context.chol @ context.chol.T
    rotated = context.rotation.T @ cov @ context.rotation
    diagonal = np.diag(rotated)
    np.testing.assert_allclose(rotated - np.diag(diagonal), 0.0, atol=1e-12)
    np.testing.assert_allclose(diagonal[N_P1_GEOLOGY_IN_V:], 1.0, atol=1e-12)
    assert np.all(diagonal[:N_P1_GEOLOGY_IN_V] < 1.0)


def test_the_residual_directions_are_the_ones_the_logs_say_nothing_about() -> None:
    a, _ = support_log_permeability_operator(P1Design())
    context = _context()
    np.testing.assert_allclose(a @ context.rotation[:, N_P1_GEOLOGY_IN_V:], 0.0, atol=1e-10)


def test_conditioning_moves_the_mean_towards_the_measured_logs() -> None:
    """A noiseless `G` from known coefficients must be recovered up to prior shrinkage."""
    design = P1Design()
    a, b = support_log_permeability_operator(design)
    truth = np.asarray(render_p1(41, design).theta["coefficients"], dtype=np.float64).ravel()
    exact = dict(zip(log_k_observation_ids(design), a @ truth + b, strict=True))
    context = p1_prior_context(design, exact)
    predicted = a @ context.mean + b
    measured = np.array([exact[name] for name in log_k_observation_ids(design)])
    assert float(np.max(np.abs(predicted - measured))) < SIGMA_LOG_PERMEABILITY
    assert float(np.linalg.norm(context.mean)) < float(np.linalg.norm(truth))


@pytest.mark.parametrize(
    "fault", ["missing", "unknown", "nonfinite", "family", "zero_sigma", "negative_sigma"]
)
def test_the_context_refuses_an_information_set_it_cannot_condition_on(fault: str) -> None:
    design = P1Design()
    observations = _observations()
    names = log_k_observation_ids(design)
    kwargs: dict[str, float] = {}
    if fault == "missing":
        observations.pop(names[0])
    elif fault == "unknown":
        observations["P9-L0-log_permeability_m2"] = 1.0
    elif fault == "nonfinite":
        observations[names[0]] = math.nan
    elif fault == "family":
        design = P1Design(family="high_contrast")
    elif fault == "zero_sigma":
        kwargs["sigma"] = 0.0
    elif fault == "negative_sigma":
        kwargs["sigma"] = -SIGMA_LOG_PERMEABILITY
    with pytest.raises(ValueError):
        p1_prior_context(design, observations, **kwargs)


def test_the_context_names_the_family_each_hypothesis_renders() -> None:
    context = _context()
    assert [p1_design_for(context, s).family for s in (0, 1)] == list(FAMILY_BY_S)
    assert p1_design_for(context, 0).shape == P1Design().shape
    with pytest.raises(ValueError):
        p1_design_for(context, 2)


def test_a_different_g_is_a_different_basis_and_a_different_information_set() -> None:
    first, second = _context(41), _context(42)
    assert first.g_hash != second.g_hash
    assert first.information_hash != second.information_hash
    assert first.density_schema.basis_hash != second.density_schema.basis_hash
    np.testing.assert_array_equal(first.rotation, second.rotation)


# --------------------------------------------------------------------------------------
# the density
# --------------------------------------------------------------------------------------


def test_the_conditional_prior_is_the_density_seam_smc_will_hold() -> None:
    """E03 swaps this object for a trained flow, so the seam is three members wide."""
    assert isinstance(GaussianConditionalPrior(_context()), Density)


def test_the_prior_is_standard_normal_in_the_whitened_coordinates() -> None:
    """Checked against `scipy.stats`, which knows nothing about this implementation."""
    context = _context()
    prior = GaussianConditionalPrior(context)
    rng = np.random.default_rng(5)
    for _ in range(8):
        coordinates = rng.standard_normal(context.density_schema.n_v + 4)
        for s in (0, 1):
            expected = float(np.sum(stats.norm.logpdf(coordinates))) + math.log(0.5)
            assert prior.log_prob(_theta(context, coordinates, s)) == pytest.approx(
                expected, abs=1e-12
            )


def test_both_families_are_equally_likely_under_the_conditional_prior() -> None:
    """`G` has the same mean and covariance under either `s`, so `p(s|G) = 0.5`."""
    context = _context()
    prior = GaussianConditionalPrior(context)
    coordinates = np.random.default_rng(6).standard_normal(15)
    assert prior.log_prob(_theta(context, coordinates, 0)) == prior.log_prob(
        _theta(context, coordinates, 1)
    )


def test_the_prior_integrates_to_one() -> None:
    """An importance-sampling estimate under a proposal the prior never draws from.

    The estimator `E_q[p/q]` with `q = N(0, 1.1^2 I)` is one exactly when `p` is normalised,
    and it uses neither `sample` nor the normalising constant the implementation wrote down.
    The one-dimensional quadrature beside it pins the constant itself.
    """
    context = _context()
    prior = GaussianConditionalPrior(context)
    dimension = context.density_schema.n_v + context.density_schema.n_residual
    scale = 1.1
    rng = np.random.default_rng(17)
    draws = scale * rng.standard_normal((N_DRAWS, dimension))
    log_q = np.sum(stats.norm.logpdf(draws, scale=scale), axis=1)
    mass = np.array(
        [sum(math.exp(prior.log_prob(_theta(context, row, s))) for s in (0, 1)) for row in draws]
    )
    assert float(np.mean(mass / np.exp(log_q))) == pytest.approx(1.0, abs=0.02)
    area, _ = integrate.quad(lambda x: float(stats.norm.pdf(x)), -np.inf, np.inf)
    assert area == pytest.approx(1.0, abs=1e-10)


def test_cheap_prior_draws_carry_the_declared_moments() -> None:
    context = _context()
    prior = GaussianConditionalPrior(context)
    draws = prior.sample(N_DRAWS, np.random.default_rng(23))
    assert len(draws) == N_DRAWS
    coordinates = np.array([[*theta.v, *theta.z_perp] for theta in draws])
    families = np.array([theta.s for theta in draws])
    assert float(np.max(np.abs(coordinates.mean(axis=0)))) < MEAN_TOLERANCE
    assert float(np.max(np.abs(coordinates.var(axis=0) - 1.0))) < VARIANCE_TOLERANCE
    assert abs(float(np.mean(families)) - 0.5) < FAMILY_TOLERANCE


def test_the_drawn_coefficients_carry_the_conditioned_mean_and_covariance() -> None:
    context = _context()
    prior = GaussianConditionalPrior(context)
    draws = prior.sample(N_DRAWS, np.random.default_rng(29))
    coefficients = np.array([geology_coefficients(theta, context) for theta in draws])
    np.testing.assert_allclose(coefficients.mean(axis=0), context.mean, atol=0.05)
    np.testing.assert_allclose(
        np.cov(coefficients, rowvar=False), context.chol @ context.chol.T, atol=0.05
    )


def test_every_draw_belongs_to_the_schema_it_names() -> None:
    context = _context()
    schema = context.density_schema
    for theta in GaussianConditionalPrior(context).sample(4, np.random.default_rng(31)):
        schema.validate_theta(theta)
        assert theta.basis_hash == schema.basis_hash


def test_the_sampler_repeats_for_the_same_seed_and_moves_for_another() -> None:
    prior = GaussianConditionalPrior(_context())
    first = prior.sample(16, np.random.default_rng(37))
    assert first == prior.sample(16, np.random.default_rng(37))
    assert first != prior.sample(16, np.random.default_rng(38))


def test_a_theta_from_another_basis_is_not_scored_against_this_prior() -> None:
    context = _context()
    prior = GaussianConditionalPrior(context)
    foreign = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=0,
        v=(0.0,) * context.density_schema.n_v,
        z_perp=(0.0,) * context.density_schema.n_residual,
        basis_hash=_context(42).density_schema.basis_hash,
    )
    with pytest.raises(ValueError, match="basis"):
        prior.log_prob(foreign)


def test_the_fingerprint_names_the_information_and_not_the_draw() -> None:
    context = _context()
    prior = GaussianConditionalPrior(context)
    assert prior.fingerprint == GaussianConditionalPrior(_context()).fingerprint
    assert prior.fingerprint != GaussianConditionalPrior(_context(42)).fingerprint
    prior.sample(4, np.random.default_rng(41))
    assert prior.fingerprint == GaussianConditionalPrior(context).fingerprint


@pytest.mark.parametrize("n", [0, -1])
def test_a_sample_of_no_particles_is_refused(n: int) -> None:
    with pytest.raises(ValueError):
        GaussianConditionalPrior(_context()).sample(n, np.random.default_rng(43))
