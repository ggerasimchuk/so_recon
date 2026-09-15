"""E02.3 — the probability of a ROUNDED report on a bounded interval, in log space.

A reported water cut of `0.97` is not the statement `f = 0.97`. It is the statement that
the true value was rounded to `0.97`, which is the bin `[0.965, 0.975]`; and `0.00` and
`1.00` are half-bins, because a fraction cannot be rounded down from below zero or up from
above one. The likelihood of a report is therefore the probability of an INTERVAL, and the
interval is bounded.

The law over that interval is a Student-t with `nu = 5` in the variance-stabilising
coordinate `g(f) = arcsin(sqrt(f))`, truncated to `[0, g(1)]` and renormalised there. Two
consequences are the whole content of this module.

**The tail is where the arithmetic breaks, so nothing is subtracted in linear space.** For
a location twenty g-space units outside the support — a model that says «dry» about a well
that reports water — every bin of the grid is a sliver of one tail, and every bin
probability is a ratio of two numbers near `1e-22`. Subtracting two CDF values near one to
get such a sliver loses every significant digit it has. So the negative side is evaluated
through `logcdf`, the positive side through `logsf`, and the difference is taken in log
space with `log1p`/`expm1`, where the cancellation is exact.

**A probability that could not be evaluated is not a small probability.** When two tail
masses collapse onto the same double, the interval is re-evaluated by an independent
anchored quadrature of the density. If that cannot resolve it either, the call raises
`LikelihoodNumericalError`. It never returns a floor: `1e-300` is a number nobody computed,
and once it is in a weight it is indistinguishable from evidence.

The scale is normalised THROUGH the CDF. `sigma` enters by standardising the interval, so
a bin probability carries no `-log sigma` of its own; that Jacobian belongs to a density
over `g` and adding it here would leave the partition summing to `1/sigma`.

**Declared numerical envelope.** `tests/unit/test_bounded_bins.py` fixes the grid of
`(mu, sigma)` this kernel's accuracy is claimed on — locations up to twenty units outside
the support, scales down to `0.001`, bin probabilities down to `1e-14` — and checks the
partition to `1e-10` and every bin to `1e-8` in log probability against an independent
integration. Far outside that envelope (`|mu| / sigma` beyond about `1e8`) the neighbouring
tail masses of a bounded grid differ by less than the resolution of their own logarithms,
and the answer degrades before the guard trips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import t

from so_recon.inference.contracts import F64, LikelihoodNumericalError

#: `g(1)`: the top of the bounded support in g-space. `g(0)` is zero.
G_MAX = math.pi / 2

#: The fallback quadrature's own accuracy contract. `epsabs` is the scaled integrand's, so
#: it is an absolute tolerance on a quantity whose peak is exactly one; the refusal is on
#: the RELATIVE error, which is what a log probability is sensitive to.
QUADRATURE_EPSABS = 1e-12
QUADRATURE_EPSREL = 1e-10
QUADRATURE_MAX_RELATIVE_ERROR = 1e-8


def to_g_space(f: F64) -> F64:
    """`g(f) = arcsin(sqrt(f))`, elementwise: the coordinate every `lo`, `hi` and `mu` is in.

    The variance of a bounded fraction collapses at both ends, so a noise law of constant
    scale is a statement about `g`, not about `f`. `g` maps `[0, 1]` onto `[0, G_MAX]` and
    is defined here once, so that no caller can score against a second transform.

    A value outside `[0, 1]` is refused rather than mapped: `arcsin` of an argument above
    one is a silent NaN, and a fraction outside the unit interval — a simulated water cut
    of `1.0000001`, a saturation of `-1e-9` — is a broken number at its source.
    """
    values = np.asarray(f, dtype=np.float64)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError(
            f"g is the coordinate of a fraction and takes [0, 1], got values in "
            f"[{np.min(values)!r}, {np.max(values)!r}]"
        )
    g: F64 = np.arcsin(np.sqrt(values))
    return g


@dataclass(frozen=True)
class BinGrid:
    """A partition of the reported unit interval: the report centres and their edges.

    Both arrays are in f-space, the space the report is written in. The outer edges are
    exactly `0` and `1` — a partition that stopped short of either could not sum to one —
    the interior edges are the midpoints of adjacent report centres, and the bins at `0`
    and `1` are consequently half-bins.

    The arrays are read-only. Rows resolve their `bin_index` against a grid and store the
    integer; a grid mutated afterwards would silently redefine what those integers meant.
    """

    centers: F64
    edges: F64

    def __post_init__(self) -> None:
        centers = _read_only(self.centers, label="centers")
        edges = _read_only(self.edges, label="edges")
        if edges.size < 3:
            raise ValueError(
                f"a bounded-bin partition needs at least two bins, so at least three "
                f"edges, got {edges.size}"
            )
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError(f"bin edges must be strictly increasing, got {edges!r}")
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError(
                f"the outer edges of a reported partition are exactly 0 and 1, got "
                f"{edges[0]!r} and {edges[-1]!r}: a grid that leaves a gap at either end "
                f"cannot sum to one over the reports it accepts"
            )
        if centers.size != edges.size - 1:
            raise ValueError(
                f"{centers.size} centers for {edges.size - 1} bins: every bin is one "
                f"reported value and every reported value is one bin"
            )
        outside = np.flatnonzero((centers < edges[:-1]) | (centers > edges[1:]))
        if outside.size:
            index = int(outside[0])
            raise ValueError(
                f"center {centers[index]!r} is outside bin {index} "
                f"[{edges[index]!r}, {edges[index + 1]!r}]: the edges do not describe "
                f"these centers"
            )
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "edges", edges)

    @property
    def n_bins(self) -> int:
        """How many bins the grid has; one per reported value."""
        return int(self.edges.size) - 1


def _read_only(values: F64, *, label: str) -> F64:
    array = np.array(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} is a one-dimensional array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite, got {array!r}")
    array.setflags(write=False)
    return array


def rounding_grid(step: float) -> BinGrid:
    """The partition a report rounded to `step` induces on `[0, 1]`.

    The centres are the values a report can take — `0`, `step`, ..., `1` — so `step` must
    divide the unit interval exactly; a step that does not is a reporting convention whose
    `1.0` is not a reportable value, and this kernel does not know what its last bin is.
    """
    if not math.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError(f"a reporting step lies in (0, 1], got {step!r}")
    n_steps = round(1.0 / step)
    if not math.isclose(n_steps * step, 1.0, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(
            f"a reporting step divides the unit interval, and {step!r} does not: "
            f"{n_steps} steps span {n_steps * step!r}, so 1.0 is not a reportable value"
        )
    centers = np.arange(n_steps + 1, dtype=np.float64) / n_steps
    edges = np.empty(n_steps + 2, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    return BinGrid(centers=centers, edges=edges)


def bin_index(raw: float, grid: BinGrid) -> int:
    """The bin a raw report falls in.

    A value sitting exactly on an edge belongs to the bin on its RIGHT, so that the
    convention does not depend on which of two adjacent rounding operations produced it.
    The single exception is the top edge `f = 1`, which has no bin to its right and
    belongs to the last bin.

    A report outside `[0, 1]` is refused. Clipping it would turn a broken meter into an
    extreme but admissible observation, and the likelihood would then score it.
    """
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"a report must be finite, got {value!r}")
    low = float(grid.edges[0])
    high = float(grid.edges[-1])
    if value < low or value > high:
        raise ValueError(
            f"report {value!r} is outside the reported support [{low!r}, {high!r}]: a "
            f"value outside its own support is an error at its source, not something to "
            f"clip into the nearest bin"
        )
    index = int(np.searchsorted(grid.edges, value, side="right")) - 1
    return min(index, grid.n_bins - 1)


def log_interval_t(lo: float, hi: float, *, mu: float, sigma: float, nu: float) -> float:
    """`log P(lo < X < hi)` for `X ~ mu + sigma * t(nu)`, with `lo`, `hi` and `mu` in g-space.

    This is the UNNORMALISED interval mass on the whole line. The bounded probability of a
    bin is this over `log_interval_t(0, G_MAX, ...)`, which is what `bin_log_probs` does.
    """
    _check_law(mu=mu, sigma=sigma, nu=nu)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"an interval has finite ends, got [{lo!r}, {hi!r}]")
    if not lo < hi:
        raise ValueError(f"an interval runs from lo to hi with lo < hi, got [{lo!r}, {hi!r}]")
    z_lo = (lo - mu) / sigma
    z_hi = (hi - mu) / sigma
    if not math.isfinite(z_lo) or not math.isfinite(z_hi):
        raise LikelihoodNumericalError(
            f"standardising [{lo!r}, {hi!r}] by mu={mu!r}, sigma={sigma!r} overflowed to "
            f"[{z_lo!r}, {z_hi!r}]; no probability was evaluated"
        )
    return _standardized_log_interval(z_lo, z_hi, nu)


def bin_log_probs(grid: BinGrid, *, mu: float, sigma: float, nu: float) -> F64:
    """The log probability of every bin of `grid`, summing to one over the grid.

    `mu` and `sigma` are in g-space. The first bin starts at exactly `g(0) = 0` and the
    last ends at exactly `g(1) = G_MAX`, the same two ends the normaliser uses, so the
    partition telescopes onto its own normaliser instead of nearly doing so.
    """
    _check_law(mu=mu, sigma=sigma, nu=nu)
    g_edges = to_g_space(grid.edges)
    log_total = log_interval_t(0.0, G_MAX, mu=mu, sigma=sigma, nu=nu)
    out = np.empty(grid.n_bins, dtype=np.float64)
    for index in range(grid.n_bins):
        out[index] = (
            log_interval_t(
                float(g_edges[index]), float(g_edges[index + 1]), mu=mu, sigma=sigma, nu=nu
            )
            - log_total
        )
    return out


# ---------------------------------------------------------------------------------------
# the standardized kernel
# ---------------------------------------------------------------------------------------


def _check_law(*, mu: float, sigma: float, nu: float) -> None:
    if not math.isfinite(mu):
        raise ValueError(f"mu is a finite location in g-space, got {mu!r}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma is a finite positive scale in g-space, got {sigma!r}")
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError(f"nu is a finite positive degrees of freedom, got {nu!r}")


def _log_diff_exp(a: float, b: float) -> float:
    """`log(exp(a) - exp(b))` without ever forming either exponential.

    `b < a` is the caller's claim that the interval has positive mass. When the two tail
    masses have collapsed onto one double the claim is false, and this raises rather than
    returning `log 0`.
    """
    if not b < a:
        raise FloatingPointError("unresolved CDF interval")
    return float(a + np.log(-np.expm1(b - a)))


def _standardized_log_interval(lo: float, hi: float, nu: float) -> float:
    """`log P(lo < Z < hi)` for a standard Student-t, on the side that keeps its digits.

    An interval wholly in the positive half is differenced from the SURVIVAL function and
    everything else from the CDF, so the quantity being differenced is never the part of a
    probability near one that carries no information about the interval.
    """
    try:
        if lo >= 0.0:
            return _log_diff_exp(float(t.logsf(lo, nu)), float(t.logsf(hi, nu)))
        return _log_diff_exp(float(t.logcdf(hi, nu)), float(t.logcdf(lo, nu)))
    except FloatingPointError:
        return _scaled_pdf_quadrature(lo, hi, nu)


def _scaled_pdf_quadrature(lo: float, hi: float, nu: float) -> float:
    """The independent second route: integrate the density itself, anchored at its peak.

    The integrand is `exp(logpdf(x) - logpdf(peak))`, where `peak` is the point of `[lo,
    hi]` closest to zero. It is therefore at most one everywhere and exactly one at the
    peak, whatever the interval's depth in the tail: an interval whose true mass is `1e-90`
    is integrated at the same relative precision as one near the mode, instead of
    underflowing to a hard zero.

    An error estimate the rule believes is not the same thing as an error. A Student-t on a
    wide interval is a spike of width one inside a range of millions, and a Gauss-Kronrod
    panel whose every node lands in the tail integrates the spike to zero AND reports no
    error. `_scale_breakpoints` therefore splits the interval where the density changes
    scale, so that no panel can hide the peak.
    """
    peak = min(max(0.0, lo), hi)
    # Beyond about 1e154 standard scales the density's own `x * x / nu` overflows. numpy
    # says so once here instead of once per quadrature node, and the refusal is below.
    with np.errstate(over="ignore", invalid="ignore"):
        anchor = float(t.logpdf(peak, nu))
    if not math.isfinite(anchor):
        raise LikelihoodNumericalError(
            f"the density at {peak!r} with nu={nu!r} is not representable, so the interval "
            f"[{lo!r}, {hi!r}] has no scale left to be integrated at; no probability was "
            f"evaluated"
        )

    def scaled_density(x: float) -> float:
        return float(np.exp(t.logpdf(x, nu) - anchor))

    breakpoints = _scale_breakpoints(lo, hi)
    value, abserr = quad(
        scaled_density,
        lo,
        hi,
        epsabs=QUADRATURE_EPSABS,
        epsrel=QUADRATURE_EPSREL,
        points=breakpoints or None,
        # QUADPACK needs one subdivision slot per break point before it may adapt at all.
        limit=max(50, 4 * len(breakpoints)),
        # Reported rather than warned about: the verdict below is this function's own.
        full_output=1,
    )[:2]
    if not math.isfinite(value) or value <= 0.0:
        raise LikelihoodNumericalError(
            f"the density over [{lo!r}, {hi!r}] with nu={nu!r} integrated to {value!r}; no "
            f"probability was evaluated, and a floor in its place would be a number nobody "
            f"computed"
        )
    if abserr > QUADRATURE_MAX_RELATIVE_ERROR * value:
        raise LikelihoodNumericalError(
            f"the density over [{lo!r}, {hi!r}] with nu={nu!r} integrated to {value!r} with "
            f"an estimated error of {abserr!r}, above the {QUADRATURE_MAX_RELATIVE_ERROR} "
            f"relative accuracy a log probability is read at"
        )
    return anchor + math.log(value)


def _scale_breakpoints(lo: float, hi: float) -> list[float]:
    """Where the Student-t changes scale on `[lo, hi]`: its peak, and each decade out.

    The density is flat within one unit of zero and falls like `|x|^-(nu+1)` outside it, so
    a decade of `x` is `nu + 1` decades of density. One panel per decade keeps every panel
    at a dynamic range the quadrature rule integrates accurately, and the breakpoint at
    zero is the peak itself, which is the one a wide interval would otherwise lose.
    """
    points = [0.0] if lo < 0.0 < hi else []
    reach = max(abs(lo), abs(hi))
    decade = 1.0
    while decade < reach:
        points.extend(x for x in (-decade, decade) if lo < x < hi)
        decade *= 10.0
    return sorted(points)


__all__ = [
    "G_MAX",
    "QUADRATURE_EPSABS",
    "QUADRATURE_EPSREL",
    "QUADRATURE_MAX_RELATIVE_ERROR",
    "BinGrid",
    "bin_index",
    "bin_log_probs",
    "log_interval_t",
    "rounding_grid",
    "to_g_space",
]
