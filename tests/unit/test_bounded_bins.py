"""E02.3 — the bounded-bin Student-t kernel every E02 observation is scored through.

The subject is a NORMALISED partition of the unit interval, so the tests are about
normalisation, about the two half-bins at its ends, and about the tail where a subtraction
of two nearly equal numbers would have silently lost the answer:

* The partition is checked to sum to one in probability over a DECLARED numerical grid
  that includes locations twenty g-space units outside the support, where every bin
  probability is a ratio of two tail masses.
* Every bin probability is checked against an INDEPENDENT integration: `scipy.integrate.
  quad` over `t.logpdf` in g-space, against the production path's analytic CDF and SF
  differences. A quadrature reference that called the production helper would prove
  nothing, so it calls only `scipy`.
* The half-bins at `f = 0` and `f = 1` are checked to be half a step wide and to be the
  bins that a report of exactly zero and exactly one fall in.
* The refusals are checked to be refusals: a report outside its own support, a scale that
  is not positive, a grid that does not tile `[0, 1]`, and an integral the quadrature
  could not resolve are all errors at their source, never a clipped value and never a
  floored probability.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import t

from so_recon.inference.contracts import LikelihoodNumericalError

# `_scaled_pdf_quadrature` is a numerical safeguard the public path cannot be steered into:
# an interval wide enough to hide the density's peak always has a resolvable CDF difference
# and never reaches the fallback. It is imported by name so that the peak it must not miss
# can be tested directly.
from so_recon.observation.bins import (
    G_MAX,
    BinGrid,
    _scaled_pdf_quadrature,
    bin_index,
    bin_log_probs,
    log_interval_t,
    rounding_grid,
    to_g_space,
)

NU = 5.0

#: The declared numerical test grid: the `(mu, sigma)` pairs in g-space over which this
#: kernel's accuracy is CLAIMED. The first four are the plan's own; the rest add a
#: location inside the support and three scales small enough that the far bins carry
#: probabilities near `1e-14`. Outside this envelope the tail bins of a bounded grid stop
#: being resolvable at double precision and the kernel says so instead of guessing.
NUMERICAL_TEST_GRID = (
    (0.0, 0.03),
    (G_MAX, 0.03),
    (-20.0, 0.001),
    (20.0, 0.001),
    (0.6, 0.02),
    (0.0, 0.002),
    (G_MAX, 0.002),
    (0.7, 0.002),
)

#: Eleven bins of the percent grid: both half-bins, both of their neighbours, and the
#: interior in between.
REFERENCE_BINS = (0, 1, 2, 10, 25, 50, 75, 90, 98, 99, 100)


def reference_log_interval(lo: float, hi: float, *, mu: float, sigma: float) -> float:
    """`log P(lo < X < hi)` by quadrature over the Student-t density in g-space.

    This is the independent half of every accuracy claim below, and it shares no step with
    the production kernel: the production kernel differences the CDF analytically, this
    one integrates the PDF numerically. It is anchored at the density's largest value on
    `[lo, hi]` so that a bin twenty thousand scales out in the tail — where the unscaled
    density is `1e-90` and would underflow to a hard zero — is integrated at full relative
    precision, and it is broken at the location and the first scales of the density so
    that a spike narrow against `[0, pi/2]` cannot fall between the quadrature's nodes.
    """
    peak = min(max(mu, lo), hi)
    anchor = float(t.logpdf((peak - mu) / sigma, NU))

    def scaled_density(x: float) -> float:
        return float(np.exp(t.logpdf((x - mu) / sigma, NU) - anchor)) / sigma

    breaks = [x for k in (-10.0, -1.0, 0.0, 1.0, 10.0) if lo < (x := mu + k * sigma) < hi]
    value, _abserr = quad(
        scaled_density,
        lo,
        hi,
        epsabs=1e-13,
        epsrel=1e-12,
        points=breaks or None,
        full_output=1,
    )[:2]
    assert value > 0.0, f"the reference quadrature underflowed on [{lo}, {hi}]"
    return anchor + math.log(value)


def reference_log_bin_prob(grid: BinGrid, index: int, *, mu: float, sigma: float) -> float:
    """The reference bin probability: its mass over the mass of the whole `[0, pi/2]`."""
    edges = to_g_space(grid.edges)
    inside = reference_log_interval(
        float(edges[index]), float(edges[index + 1]), mu=mu, sigma=sigma
    )
    total = reference_log_interval(0.0, G_MAX, mu=mu, sigma=sigma)
    return inside - total


# ---------------------------------------------------------------------------------------
# the partition
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mu,sigma", [(0.0, 0.03), (np.pi / 2, 0.03), (-20.0, 0.001), (20.0, 0.001)]
)
def test_bins_form_a_probability_partition(mu: float, sigma: float) -> None:
    grid = rounding_grid(0.01)
    lp = bin_log_probs(grid, mu=mu, sigma=sigma, nu=5.0)
    assert np.isfinite(lp).all()
    assert abs(logsumexp(lp)) < 1e-10
    assert bin_index(0.0, grid) == 0
    assert bin_index(1.0, grid) == 100
    assert grid.edges[1] == 0.005


@pytest.mark.parametrize("mu,sigma", NUMERICAL_TEST_GRID)
def test_the_partition_normalises_over_the_declared_numerical_grid(mu: float, sigma: float) -> None:
    """The whole declared envelope, not only the four locations the plan names."""
    grid = rounding_grid(0.01)
    lp = bin_log_probs(grid, mu=mu, sigma=sigma, nu=NU)
    assert np.isfinite(lp).all()
    assert abs(logsumexp(lp)) < 1e-10


def test_a_finer_grid_still_normalises() -> None:
    """A tenth of a percent is ten times as many bins and ten times as narrow a tail bin."""
    grid = rounding_grid(0.001)
    assert grid.edges.size == 1002
    lp = bin_log_probs(grid, mu=0.4, sigma=0.02, nu=NU)
    assert np.isfinite(lp).all()
    assert abs(logsumexp(lp)) < 1e-10


# ---------------------------------------------------------------------------------------
# against an independent integration
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("mu,sigma", NUMERICAL_TEST_GRID)
def test_bins_match_an_independent_pdf_quadrature(mu: float, sigma: float) -> None:
    """Eleven bins against `quad` over `t.pdf`, to 1e-8 in log probability."""
    grid = rounding_grid(0.01)
    lp = bin_log_probs(grid, mu=mu, sigma=sigma, nu=NU)
    for index in REFERENCE_BINS:
        expected = reference_log_bin_prob(grid, index, mu=mu, sigma=sigma)
        assert abs(float(lp[index]) - expected) < 1e-8, f"bin {index} at mu={mu}, sigma={sigma}"


def test_the_compared_bins_actually_reach_the_narrow_tail() -> None:
    """A guard on the guard: the comparison above must not quietly stop covering the tail.

    The whole point of the CDF/SF split is the bin whose probability is `1e-14`. If the
    declared grid ever drifted to locations where every compared bin was of order one, the
    accuracy test would still pass and would no longer be testing anything.
    """
    grid = rounding_grid(0.01)
    deepest = min(
        float(bin_log_probs(grid, mu=mu, sigma=sigma, nu=NU)[index])
        for mu, sigma in NUMERICAL_TEST_GRID
        for index in REFERENCE_BINS
    )
    assert deepest < -30.0


# ---------------------------------------------------------------------------------------
# the grid itself
# ---------------------------------------------------------------------------------------


def test_the_rounding_grid_has_half_bins_at_both_ends() -> None:
    grid = rounding_grid(0.01)
    assert grid.centers.size == 101
    assert grid.edges.size == 102
    assert grid.n_bins == 101
    assert grid.edges[0] == 0.0
    assert grid.edges[-1] == 1.0
    assert grid.centers[0] == 0.0
    assert grid.centers[-1] == 1.0

    widths = np.diff(grid.edges)
    assert widths[0] == pytest.approx(0.005, abs=1e-15)
    assert widths[-1] == pytest.approx(0.005, abs=1e-15)
    assert widths[1:-1] == pytest.approx(0.01, abs=1e-15)
    # every interior edge is the midpoint of the two report centres it separates
    assert grid.edges[1:-1] == pytest.approx(0.5 * (grid.centers[:-1] + grid.centers[1:]), abs=0.0)


@pytest.mark.parametrize("step", [0.03, 0.0, -0.01, 1.5, float("nan"), float("inf")])
def test_a_step_that_does_not_divide_the_unit_interval_is_refused(step: float) -> None:
    with pytest.raises(ValueError):
        rounding_grid(step)


def test_a_grid_whose_edges_repeat_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BinGrid(
            centers=np.array([0.0, 0.5, 1.0]),
            edges=np.array([0.0, 0.25, 0.25, 1.0]),
        )


def test_a_grid_whose_edges_run_backwards_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BinGrid(centers=np.array([0.0, 0.5]), edges=np.array([0.0, 0.75, 0.5]))


@pytest.mark.parametrize(
    "edges", [[0.0, 0.5, 0.9], [0.1, 0.5, 1.0], [0.0, 0.5, 1.5], [-0.1, 0.5, 1.0]]
)
def test_a_grid_that_leaves_a_gap_in_the_unit_interval_is_refused(edges: list[float]) -> None:
    """A partition whose outer edges are not exactly zero and one cannot sum to one."""
    with pytest.raises(ValueError, match="exactly 0"):
        BinGrid(centers=np.array([0.2, 0.8]), edges=np.array(edges))


def test_a_grid_whose_centres_do_not_belong_to_their_bins_is_refused() -> None:
    with pytest.raises(ValueError, match="bin"):
        BinGrid(centers=np.array([0.9, 0.1]), edges=np.array([0.0, 0.5, 1.0]))


def test_a_grid_with_the_wrong_number_of_centres_is_refused() -> None:
    with pytest.raises(ValueError, match="centers"):
        BinGrid(centers=np.array([0.25, 0.5, 0.75]), edges=np.array([0.0, 0.5, 1.0]))


def test_a_grid_with_fewer_than_two_bins_is_refused() -> None:
    with pytest.raises(ValueError, match="two bins"):
        BinGrid(centers=np.array([0.5]), edges=np.array([0.0, 1.0]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_grid_with_a_non_finite_edge_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BinGrid(centers=np.array([0.2, 0.8]), edges=np.array([0.0, bad, 1.0]))


def test_a_published_grid_cannot_be_mutated_behind_the_rows_that_resolved_against_it() -> None:
    grid = rounding_grid(0.01)
    with pytest.raises(ValueError):
        grid.edges[3] = 0.5


# ---------------------------------------------------------------------------------------
# which bin a report falls in
# ---------------------------------------------------------------------------------------


def test_boundary_equality_belongs_to_the_bin_on_the_right() -> None:
    grid = rounding_grid(0.01)
    for k in range(1, 101):
        assert bin_index(float(grid.edges[k]), grid) == k, f"edge {k}"
    # f = 1 is the exception: the last edge belongs to the last bin, not to a bin after it
    assert bin_index(float(grid.edges[101]), grid) == 100
    assert bin_index(0.0, grid) == 0


def test_a_report_just_below_an_edge_belongs_to_the_bin_on_the_left() -> None:
    grid = rounding_grid(0.01)
    for k in (1, 2, 50, 100, 101):
        below = float(np.nextafter(grid.edges[k], -np.inf))
        assert bin_index(below, grid) == k - 1, f"edge {k}"


def test_reported_centres_fall_in_their_own_bins() -> None:
    grid = rounding_grid(0.01)
    for k in range(101):
        assert bin_index(float(grid.centers[k]), grid) == k


@pytest.mark.parametrize("raw", [-1e-12, -0.5, 1.0 + 1e-12, 2.0, float("nan"), float("inf")])
def test_a_report_outside_the_unit_interval_is_refused_not_clipped(raw: float) -> None:
    grid = rounding_grid(0.01)
    with pytest.raises(ValueError):
        bin_index(raw, grid)


# ---------------------------------------------------------------------------------------
# the interval kernel
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.0, -0.01, float("nan"), float("inf")])
def test_a_scale_that_is_not_positive_is_refused(sigma: float) -> None:
    grid = rounding_grid(0.01)
    with pytest.raises(ValueError, match="sigma"):
        bin_log_probs(grid, mu=0.4, sigma=sigma, nu=NU)
    with pytest.raises(ValueError, match="sigma"):
        log_interval_t(0.1, 0.2, mu=0.4, sigma=sigma, nu=NU)


@pytest.mark.parametrize("nu", [0.0, -5.0, float("nan"), float("inf")])
def test_a_degrees_of_freedom_that_is_not_positive_is_refused(nu: float) -> None:
    grid = rounding_grid(0.01)
    with pytest.raises(ValueError, match="nu"):
        bin_log_probs(grid, mu=0.4, sigma=0.02, nu=nu)
    with pytest.raises(ValueError, match="nu"):
        log_interval_t(0.1, 0.2, mu=0.4, sigma=0.02, nu=nu)


@pytest.mark.parametrize("lo,hi", [(0.3, 0.3), (0.3, 0.2)])
def test_an_empty_interval_is_refused(lo: float, hi: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        log_interval_t(lo, hi, mu=0.4, sigma=0.02, nu=NU)


@pytest.mark.parametrize("mu", [float("nan"), float("inf")])
def test_a_location_that_is_not_finite_is_refused(mu: float) -> None:
    with pytest.raises(ValueError, match="mu"):
        log_interval_t(0.1, 0.2, mu=mu, sigma=0.02, nu=NU)


def test_the_interval_is_the_difference_of_two_tail_masses() -> None:
    """Against `scipy`'s own CDF at a location where a linear difference is safe."""
    lo, hi = 0.2, 0.9
    mu, sigma = 0.5, 0.1
    got = log_interval_t(lo, hi, mu=mu, sigma=sigma, nu=NU)
    expected = math.log(float(t.cdf((hi - mu) / sigma, NU)) - float(t.cdf((lo - mu) / sigma, NU)))
    assert got == pytest.approx(expected, rel=1e-12)


def test_no_extra_log_sigma_is_added_to_a_bin_probability() -> None:
    """The scale is normalised through the CDF already.

    A stray `-log sigma` on each bin — the density's Jacobian, which belongs to a density
    over g and not to a probability over bins — would leave the partition summing to
    `1/sigma` instead of to one. Halving sigma would then shift every log probability by
    `log 2`; the ratio of two bins under one law is the shape of the law and must not move
    when sigma is rescaled by a factor the bins also feel.
    """
    grid = rounding_grid(0.01)
    for sigma in (0.02, 0.04):
        lp = bin_log_probs(grid, mu=0.6, sigma=sigma, nu=NU)
        assert abs(logsumexp(lp)) < 1e-10
        assert float(lp.max()) < 0.0


# ---------------------------------------------------------------------------------------
# the numerical guard
# ---------------------------------------------------------------------------------------


def test_an_unresolvable_cdf_difference_falls_back_to_quadrature() -> None:
    """One ulp apart in the far tail, the two tail masses are the SAME double.

    Differencing them gives zero and `log 0` is `-inf`: a likelihood that would refuse a
    perfectly ordinary particle. The quadrature answers instead, and over an interval one
    ulp wide the answer is the density times the width to full precision.
    """
    lo = 1.0e6
    hi = float(np.nextafter(lo, np.inf))
    assert float(t.logsf(lo, NU)) == float(t.logsf(hi, NU))

    got = log_interval_t(lo, hi, mu=0.0, sigma=1.0, nu=NU)
    expected = float(t.logpdf(lo, NU)) + math.log(hi - lo)
    assert got == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("half_width", [1e3, 1e6, 1e12])
def test_the_quadrature_covers_the_peak_its_interval_spans(half_width: float) -> None:
    """A spike of width one inside a range of `1e12` is what an adaptive rule loses.

    Every node of a Gauss-Kronrod panel over `[-1e12, 1e12]` lands in the tail, the peak
    integrates to zero and the rule reports NO error: the wrong answer arrives looking
    converged. The expected value here comes from the survival function in closed form,
    not from the kernel.
    """
    got = _scaled_pdf_quadrature(-half_width, half_width, NU)
    expected = math.log1p(-2.0 * float(t.sf(half_width, NU)))
    assert got == pytest.approx(expected, abs=1e-8)


def test_the_quadrature_refuses_instead_of_flooring_a_probability() -> None:
    """An integral that came out at zero is a failure to evaluate, not a probability.

    `1e-300` in its place would be a number nobody computed, and it would propagate into a
    weight and then into a posterior as though it had been measured.
    """
    with pytest.raises(LikelihoodNumericalError):
        _scaled_pdf_quadrature(5.0, 5.0, NU)


def test_a_scale_small_enough_to_overflow_the_standardisation_is_refused() -> None:
    """`(lo - mu) / sigma` is not a probability statement once it stops being finite."""
    with pytest.raises(LikelihoodNumericalError):
        log_interval_t(0.1, 0.2, mu=0.0, sigma=5e-324, nu=NU)


@pytest.mark.parametrize("lo,hi", [(1e20, 1.5e20), (1e30, 3e30), (1e60, 2e60)])
def test_the_far_tail_is_still_the_student_t_power_law(
    lo: float,
    hi: float,
) -> None:
    """Sixty orders of magnitude out, the interval still follows the power-law tail.

    A Student-t tail is `sf(x) -> K nu^((nu+1)/2) x^-nu / nu` in closed form, which is what
    the answer is checked against.  Whether SciPy reaches this through its log-survival
    asymptotic or the production fallback is version-dependent, so the test fixes the
    mathematical result instead of an implementation detail of the installed SciPy.
    """
    got = log_interval_t(lo, hi, mu=0.0, sigma=1.0, nu=NU)

    log_k = float(gammaln((NU + 1.0) / 2.0) - 0.5 * math.log(NU * math.pi) - gammaln(NU / 2.0))
    log_c = log_k + ((NU + 1.0) / 2.0) * math.log(NU) - math.log(NU)
    expected = log_c - NU * math.log(lo) + math.log1p(-((lo / hi) ** NU))
    assert got == pytest.approx(expected, abs=1e-9)


def test_an_interval_beyond_the_density_itself_is_refused_quietly() -> None:
    """Past `1e154` scales the density's own `x * x` overflows, and there is no answer.

    The refusal is raised before the quadrature starts, so the failure arrives once, as a
    named reason, and not as several hundred overflow warnings from inside the rule.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(LikelihoodNumericalError, match="not representable"):
            log_interval_t(1e300, 2e300, mu=0.0, sigma=1.0, nu=NU)


# ---------------------------------------------------------------------------------------
# g-space
# ---------------------------------------------------------------------------------------


def test_g_space_maps_the_unit_interval_onto_zero_to_half_pi() -> None:
    f = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    g = to_g_space(f)
    assert g[0] == 0.0
    assert g[-1] == G_MAX == math.pi / 2
    assert np.all(np.diff(g) > 0.0)
    assert g[2] == pytest.approx(math.pi / 4, abs=1e-15)


@pytest.mark.parametrize("bad", [1.0 + 1e-9, -1e-9, 2.0, float("nan")])
def test_a_value_outside_the_unit_interval_has_no_g_coordinate(bad: float) -> None:
    """`arcsin` above one is a silent NaN, and a NaN mu would score every bin as NaN."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_g_space(np.array([0.5, bad]))
