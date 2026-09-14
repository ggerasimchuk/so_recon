"""E02.2 — the conditional Gaussian law of E01's twelve geology coefficients, given G.

E01 draws twelve standard normal coefficients and renders a rock. E02 is handed the sparse
`log k` those coefficients produced — eight well-layer support averages with `sigma = 0.2
ln(m²)` — and needs the law of the coefficients GIVEN that information. The map from
coefficients to a support-averaged `log k` is linear, so the conditional is Gaussian and
exact: with `a ~ N(0, I12)`, `G - b = A a + e` and `e ~ N(0, R)`,

    cov  = (I + A^T R^-1 A)^-1,    mean = cov A^T R^-1 (G - b).

Three things this module refuses to do.

**It does not re-derive the basis.** Every column of `A` comes out of `layer_fields`, the
E01 kernel itself, evaluated on a one-hot coefficient vector, and every row is
`support_mean` over the cells the E01 support declares. A matrix that merely resembled the
generator's cosine modes would condition on something nobody measured.

**It does not drop the unconstrained directions.** Eight observations inform eight
directions of the twelve; the other four are the null space of `A` and keep their full unit
prior variance. SPEC §7.5 forbids zeroing them, so they are rotated into named coordinates
and carried, not discarded.

**It does not let the eigensolver choose a basis.** The four wells of the P1 design sit in a
symmetric pattern, so the information matrix is massively degenerate: its spectrum is four
tied blocks of sizes 2, 4, 2 and 4. Inside a tied block no eigenvector is preferred and
LAPACK's answer is an accident of rounding. The basis of every tied block is therefore
DECLARED here — projected standard unit vectors in coordinate order, orthonormalised — so
that a stored latent coordinate keeps its meaning across machines and across resumes.

The hypothesis `s` selects `kz/kx ∈ {0.05, 0.001}` and nothing else. Both families share one
geology kernel, so `A` and `b` are identical under either and `G` cannot tell them apart:
`p(s | G) = 0.5` by construction, not by assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from so_recon.inference.contracts import (
    F64,
    N_P1_GEOLOGY_IN_V,
    DensitySchema,
    PriorContext,
)
from so_recon.registry.hashing import sha256_json
from so_recon.simulator.contracts import MILLIDARCY_M2
from so_recon.synthetic.p1 import (
    FAMILY_CONTRAST,
    GENERATOR_VERSION,
    LAYER_BASE_PERMEABILITY_MD,
    N_LAYERS,
    N_MODES,
    SIGMA_LOG_PERMEABILITY,
    Family,
    P1Design,
    layer_fields,
    observation_supports,
    support_mean,
)

#: The twelve coefficients of the E01 generator: six cosine modes on each of two layers.
N_GEOLOGY = N_LAYERS * N_MODES

#: The three nuisance coordinates of the noise law that follow the geology block in `v`.
N_P1_NUISANCE = 3

#: The one static channel E02 conditions on, named as E01 names it. Porosity is held out
#: (`use='heldout_diagnostic'`): its likelihood is not Gaussian in these coefficients and a
#: linear conditional on it would be a different, unstated model.
LOG_K_QUANTITY = "log_permeability_m2"

#: This transform, versioned. A change to the operator, to the rotation convention or to the
#: coordinate order is a new version: the same numbers would otherwise mean something else.
TRANSFORM_VERSION = "e02-geology-conditional-1"
P1_SCHEMA_ID = "p1-conditional-12"

#: `s` names an E01 family, and only the two that share a geology kernel. `high_contrast`
#: multiplies the latent field by 1.8, which would change the mean and covariance of `G`
#: and make the hypothesis identifiable from the static logs alone.
FAMILY_BY_S: tuple[Family, ...] = ("base", "low_vertical")

#: Two eigenvalues are tied when they differ by no more than this fraction of the largest
#: one. Relative to the SPECTRUM, not to the pair: the null block's entries are rounding
#: noise about zero, and a pairwise-relative test would never group them.
EIGENVALUE_TIE_RTOL = 1e-12

#: A projected unit vector shorter than this carries no direction the block does not
#: already hold, and is skipped rather than normalised into numerical noise.
CANONICAL_BASIS_FLOOR = 1e-10

#: A column's sign is fixed by its largest absolute entry — and several of this design's
#: columns carry two entries of magnitude `1/sqrt(2)` that differ only in the last bits.
#: Entries within this relative distance of the peak count as equally large and the LOWEST
#: coordinate index wins, so a one-ulp reordering cannot flip a stored coordinate's sign.
SIGN_TIE_RTOL = 1e-9

#: Where the context keeps what the renderer needs to rebuild a design, which supports
#: were conditioned on (SPEC §5: they are not multiplied in again as a likelihood) and the
#: spectrum the whitening was built from. The rotation, mean and factor are persisted as
#: arrays on the context itself, so a resume reads this decomposition instead of redoing it.
DESIGN_KEY = "p1_design"
FAMILY_KEY = "family_by_s"
SIGMA_KEY = "log_k_sigma"
OBSERVATION_KEY = "log_k_observation_ids"
WHITENING_KEY = "whitening"


# --------------------------------------------------------------------------------------
# the exact Gaussian conditional
# --------------------------------------------------------------------------------------


def condition_gaussian(A: F64, y: F64, variance: F64) -> tuple[F64, F64]:  # noqa: N803
    """Exact conditioning of `a ~ N(0, I)` on `y = A a + e`, `e ~ N(0, diag(variance))`.

    Returns the conditional mean and covariance. The precision `I + A^T R^-1 A` is formed
    and factored rather than the covariance inverted: a Cholesky of the precision is the
    stable route and its failure is a genuine refusal, not a silently huge number.
    """
    A, y, variance = (np.asarray(x, dtype=np.float64) for x in (A, y, variance))  # noqa: N806
    if A.ndim != 2 or y.shape != (A.shape[0],) or variance.shape != y.shape:
        raise ValueError("conditioning shape mismatch")
    if not all(np.isfinite(x).all() for x in (A, y, variance)) or np.any(variance <= 0):
        raise ValueError("invalid conditioning inputs")
    precision = np.eye(A.shape[1]) + A.T @ (A / variance[:, None])
    factor = cho_factor(precision, lower=True)
    cov = cho_solve(factor, np.eye(A.shape[1]))
    mean = cho_solve(factor, A.T @ (y / variance))
    return mean, (cov + cov.T) / 2


# --------------------------------------------------------------------------------------
# the operator: E01's own basis, weights and support averaging
# --------------------------------------------------------------------------------------


def cell_log_permeability_operator(design: P1Design) -> tuple[F64, F64]:
    """The affine map from the twelve coefficients to `log k` in every cell, `ln(m²)`.

    Column `(layer, mode)` is `layer_fields` evaluated on a one-hot coefficient vector and
    stripped of its base permeability — which is exactly `contrast * basis_mode`, because
    E01 applies the family contrast inside the kernel. Taking the column from the kernel's
    own output is what keeps this operator from drifting away from the generator.
    """
    nx, ny, _ = design.shape
    cells_per_layer = design.cells_per_layer
    basis = np.zeros((design.n_cells, N_GEOLOGY), dtype=np.float64)
    intercept = np.zeros(design.n_cells, dtype=np.float64)
    for layer in range(N_LAYERS):
        base_md = math.log(LAYER_BASE_PERMEABILITY_MD[layer])
        low, high = layer * cells_per_layer, (layer + 1) * cells_per_layer
        # `log k` is published in m²: the millidarcy conversion belongs to the intercept.
        intercept[low:high] = base_md + math.log(MILLIDARCY_M2)
        for mode in range(N_MODES):
            unit = np.zeros(N_MODES, dtype=np.float64)
            unit[mode] = 1.0
            k_md, _ = layer_fields(unit, nx, ny, layer, design.family)
            basis[low:high, layer * N_MODES + mode] = np.log(k_md).ravel(order="F") - base_md
    return basis, intercept


def support_log_permeability_operator(design: P1Design) -> tuple[F64, F64]:
    """`A` and `b` of `G = A a + b + e` for the eight well-layer supports of the design.

    A row is the SUPPORT AVERAGE of the cell operator, computed with E01's own
    `support_mean`, so a support that grows past one cell keeps meaning the same thing here
    as it does in the generator.
    """
    basis, intercept = cell_log_permeability_operator(design)
    supports = observation_supports(design)
    a = np.array(
        [
            [support_mean(basis[:, column], support.cell_ids) for column in range(N_GEOLOGY)]
            for support in supports
        ],
        dtype=np.float64,
    )
    b = np.array(
        [support_mean(intercept, support.cell_ids) for support in supports], dtype=np.float64
    )
    return a, b


def log_k_observation_ids(design: P1Design) -> tuple[str, ...]:
    """The E01 observation ids of the sparse `log k`, in the order `A`'s rows are built.

    The identity is what keeps a permuted set of eight numbers from being conditioned on as
    though it were the measured one.
    """
    return tuple(
        f"{support.well_id}-L{support.layer_index}-{LOG_K_QUANTITY}"
        for support in observation_supports(design)
    )


# --------------------------------------------------------------------------------------
# the whitening rotation
# --------------------------------------------------------------------------------------


def information_matrix(a: F64, variance: F64) -> F64:
    """`A^T R^-1 A`, symmetrised. This is what orders the latent directions."""
    matrix = np.asarray(a, dtype=np.float64)
    spread = np.asarray(variance, dtype=np.float64)
    if matrix.ndim != 2 or spread.shape != (matrix.shape[0],):
        raise ValueError(
            f"information shape mismatch: A is {matrix.shape} and variance is {spread.shape}"
        )
    if not (np.isfinite(matrix).all() and np.isfinite(spread).all()) or np.any(spread <= 0.0):
        raise ValueError("an information matrix needs finite rows and positive variances")
    information: F64 = matrix.T @ (matrix / spread[:, None])
    return (information + information.T) / 2


def tie_blocks(
    eigenvalues: Sequence[float] | F64, *, rtol: float = EIGENVALUE_TIE_RTOL
) -> tuple[tuple[int, ...], ...]:
    """Group a descending spectrum into blocks of eigenvalues that are the same number.

    The tolerance is relative to the largest absolute eigenvalue. The null block of this
    design sits at `1e-15` against a spectrum of order `1e2`, so a pairwise-relative test
    would call its entries distinct and hand four coordinates a basis made of rounding.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"a spectrum is a non-empty vector, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("a spectrum must be finite")
    scale = float(np.max(np.abs(values)))
    tolerance = rtol * scale if scale > 0.0 else rtol
    blocks: list[tuple[int, ...]] = []
    current = [0]
    for index in range(1, values.size):
        if abs(float(values[index] - values[current[0]])) <= tolerance:
            current.append(index)
        else:
            blocks.append(tuple(current))
            current = [index]
    blocks.append(tuple(current))
    return tuple(blocks)


def _canonical_block(columns: F64) -> F64:
    """The DECLARED orthonormal basis of one tied eigenspace.

    Inside a tied block every orthonormal basis diagonalises the matrix equally well, so
    the one that is kept has to be chosen by a rule rather than by the solver. The rule:
    project the standard unit vectors, in increasing coordinate index, onto the block and
    orthonormalise them by modified Gram–Schmidt, skipping any whose remainder is shorter
    than `CANONICAL_BASIS_FLOOR`. The projector is a property of the subspace and not of
    the eigenvectors that happened to span it, which is what makes the result stable.
    """
    projector = columns @ columns.T
    accepted: list[F64] = []
    for index in range(projector.shape[0]):
        candidate: F64 = projector[:, index].copy()
        for vector in accepted:
            candidate = candidate - float(vector @ candidate) * vector
        norm = float(np.linalg.norm(candidate))
        if norm < CANONICAL_BASIS_FLOOR:
            continue
        accepted.append(candidate / norm)
        if len(accepted) == columns.shape[1]:
            return np.column_stack(accepted)
    raise ValueError(
        f"a tied eigenspace of dimension {columns.shape[1]} yielded only {len(accepted)} "
        "canonical directions; the projected unit vectors do not span it"
    )


def whitening_rotation(information: F64) -> tuple[F64, F64]:
    """An orthonormal `Q` ordered by information, and the descending eigenvalues.

    `Q` re-labels the whitened latent coordinates; it is orthonormal in that coordinate
    space. The cosine field basis of E01 is NOT orthonormal and is not claimed to be.
    """
    matrix = np.asarray(information, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"an information matrix is square, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("an information matrix must be finite")
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10 * scale):
        raise ValueError("an information matrix is symmetric; this one is not")
    raw_eigenvalues, raw_vectors = np.linalg.eigh(matrix)
    eigenvalues: F64 = raw_eigenvalues[::-1].copy()
    vectors: F64 = raw_vectors[:, ::-1].copy()
    if float(np.min(eigenvalues)) < -EIGENVALUE_TIE_RTOL * scale:
        raise ValueError(
            f"A^T R^-1 A is positive semidefinite by construction; its smallest eigenvalue "
            f"here is {float(np.min(eigenvalues)):g}, which is a failure and not a rounding"
        )
    rotation = np.empty_like(vectors)
    for block in tie_blocks(eigenvalues):
        rotation[:, list(block)] = _canonical_block(vectors[:, list(block)])
    for column in range(rotation.shape[1]):
        magnitudes = np.abs(rotation[:, column])
        peak = float(np.max(magnitudes))
        leading = int(np.flatnonzero(magnitudes >= peak * (1.0 - SIGN_TIE_RTOL))[0])
        if rotation[leading, column] < 0.0:
            rotation[:, column] = -rotation[:, column]
    return rotation, eigenvalues


def check_informed_split(
    eigenvalues: Sequence[float] | F64, n_informed: int, *, rtol: float = EIGENVALUE_TIE_RTOL
) -> None:
    """Refuse a split of the coordinates that cuts a degenerate eigenspace in half.

    `v` takes the first `n_informed` whitened coordinates and `z_perp` the rest. Inside a
    tied block the individual directions are a convention of this module, so a cut through
    one would make the contents of `v` and `z_perp` an accident of the eigensolver.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if not 0 < n_informed < values.size:
        raise ValueError(
            f"the informed split is strictly inside the spectrum: got {n_informed} of "
            f"{values.size} directions"
        )
    starts = {block[0] for block in tie_blocks(values, rtol=rtol)}
    if n_informed not in starts:
        raise ValueError(
            f"the split at {n_informed} falls inside a degenerate eigenspace of "
            f"{tie_blocks(values, rtol=rtol)}; a tied block is kept whole or not at all"
        )


def conditional_square_root(rotation: F64, eigenvalues: F64) -> F64:
    """The PRINCIPAL SYMMETRIC square root of `cov = (I + A^T R^-1 A)^-1`.

    `cov` shares its eigenvectors with the information matrix, so in the canonical basis it
    is `Q diag(1/(1+lambda)) Q^T` and its principal root is `S = Q diag((1+lambda)^-1/2) Q^T`.

    The symmetry is the whole point. The renderer composes `S @ Q`, and only a root that
    commutes with `Q` leaves that composition equal to `Q diag((1+lambda)^-1/2)` — the
    columns stay the eigen-directions, in the order the eigenvalues put them, so the four
    trailing columns are still the null space of `A` and a move in `z_perp` does not touch
    the log permeabilities the prior was conditioned on. A lower-triangular Cholesky factor
    is an equally valid square root of `cov` and gives exactly the same law, but it re-mixes
    the columns and destroys the ordering the whole eigendecomposition existed to produce.
    """
    q = np.asarray(rotation, dtype=np.float64)
    spectrum = np.asarray(eigenvalues, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or spectrum.shape != (q.shape[0],):
        raise ValueError(
            f"a square root needs a square rotation and one eigenvalue per column, got "
            f"{q.shape} and {spectrum.shape}"
        )
    if not (np.isfinite(q).all() and np.isfinite(spectrum).all()):
        raise ValueError("the rotation and its spectrum must be finite")
    if float(np.min(spectrum)) <= -1.0:
        raise ValueError(
            f"the posterior precision I + A^T R^-1 A is positive definite, so every "
            f"eigenvalue exceeds -1; the smallest here is {float(np.min(spectrum)):g}"
        )
    root: F64 = (q * np.sqrt(1.0 / (1.0 + spectrum))) @ q.T
    return (root + root.T) / 2


# --------------------------------------------------------------------------------------
# the P1 conditional prior context
# --------------------------------------------------------------------------------------


def p1_prior_context(
    design: P1Design,
    observations: Mapping[str, float],
    *,
    sigma: float = SIGMA_LOG_PERMEABILITY,
) -> PriorContext:
    """Condition the twelve-coefficient prior on one set of sparse `log k` measurements.

    `observations` is keyed by E01 observation id and must name every support exactly once:
    a missing or unknown key is a refusal, because a silently smaller information set would
    produce a wider prior that still called itself conditional.

    The returned context carries the mean, the Cholesky factor of the covariance and the
    canonical rotation, so a resume re-reads them instead of re-deriving them.
    """
    if design.family != FAMILY_BY_S[0]:
        raise ValueError(
            f"the family is the hypothesis s, not a property of the design: build the "
            f"context on {FAMILY_BY_S[0]!r} and let s choose among {FAMILY_BY_S}, got "
            f"{design.family!r}"
        )
    contrasts = {FAMILY_CONTRAST[family] for family in FAMILY_BY_S}
    if len(contrasts) != 1:
        raise ValueError(
            f"the two hypotheses must share one geology kernel for p(s|G) to be 0.5; "
            f"{FAMILY_BY_S} declare contrasts {sorted(contrasts)}"
        )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"the log-permeability noise is a positive scale, got {sigma}")

    names = log_k_observation_ids(design)
    missing = [name for name in names if name not in observations]
    unknown = [name for name in observations if name not in set(names)]
    if missing or unknown:
        raise ValueError(
            f"the conditioning set is exactly the {len(names)} sparse log k supports: "
            f"missing {missing}, unknown {unknown}"
        )
    values = np.array([float(observations[name]) for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"every conditioning measurement is finite, got {values.tolist()}")

    a, b = support_log_permeability_operator(design)
    variance = np.full(values.size, sigma * sigma, dtype=np.float64)
    mean, cov = condition_gaussian(a, values - b, variance)
    rotation, eigenvalues = whitening_rotation(information_matrix(a, variance))
    check_informed_split(eigenvalues, N_P1_GEOLOGY_IN_V)
    # `chol` is the plan's name for the factor; the SYMMETRIC root is the one that keeps
    # `rotation`'s ordering through the renderer's composition. See the function.
    chol = conditional_square_root(rotation, eigenvalues)
    if not np.allclose(chol @ chol.T, cov, rtol=0.0, atol=1e-10):
        raise ValueError(
            "the square root taken from the spectrum does not reproduce the covariance the "
            "conditioning returned; the rotation and the conditional disagree"
        )

    supports = observation_supports(design)
    g_hash = sha256_json(
        {
            "quantity": LOG_K_QUANTITY,
            "unit": "ln(m2)",
            "sigma": sigma,
            "observations": [
                {
                    "observation_id": name,
                    "support_cell_ids": list(support.cell_ids),
                    "value": float(value),
                }
                for name, support, value in zip(names, supports, values, strict=True)
            ],
        }
    )
    information_hash = sha256_json(
        {
            "role": "condition_prior",
            "g_hash": g_hash,
            "generator_version": GENERATOR_VERSION,
            "transform_version": TRANSFORM_VERSION,
            "design": design.payload(),
        }
    )
    basis_hash = sha256_json(
        {
            "transform_version": TRANSFORM_VERSION,
            "information_hash": information_hash,
            "n_geology": N_GEOLOGY,
            "n_informed": N_P1_GEOLOGY_IN_V,
            "eigenvalues": eigenvalues.tolist(),
            "rotation": rotation.tolist(),
            "mean": mean.tolist(),
            "chol": chol.tolist(),
        }
    )
    schema = DensitySchema(
        schema_id=P1_SCHEMA_ID,
        n_v=N_P1_GEOLOGY_IN_V + N_P1_NUISANCE,
        n_residual=N_GEOLOGY - N_P1_GEOLOGY_IN_V,
        families=tuple(range(len(FAMILY_BY_S))),
        basis_hash=basis_hash,
        transform_version=TRANSFORM_VERSION,
    )
    return PriorContext(
        density_schema=schema,
        n_geology=N_GEOLOGY,
        n_state_residual=0,
        mean=mean,
        chol=chol,
        rotation=rotation,
        design={
            DESIGN_KEY: design.model_dump(mode="json"),
            FAMILY_KEY: list(FAMILY_BY_S),
            SIGMA_KEY: sigma,
            OBSERVATION_KEY: list(names),
            WHITENING_KEY: {
                "transform_version": TRANSFORM_VERSION,
                "n_informed": N_P1_GEOLOGY_IN_V,
                "eigenvalues": eigenvalues.tolist(),
            },
        },
        g_hash=g_hash,
        information_hash=information_hash,
    )


def p1_design_for(context: PriorContext, s: int) -> P1Design:
    """The E01 design one hypothesis renders: the stored design with `s`'s family."""
    families: list[str] = list(context.design[FAMILY_KEY])
    if not 0 <= s < len(families):
        raise ValueError(f"family index outside the declared support: s={s} of {families}")
    payload = {**context.design[DESIGN_KEY], "family": families[s]}
    return P1Design.model_validate(payload)


__all__ = [
    "CANONICAL_BASIS_FLOOR",
    "DESIGN_KEY",
    "EIGENVALUE_TIE_RTOL",
    "FAMILY_BY_S",
    "FAMILY_KEY",
    "LOG_K_QUANTITY",
    "N_GEOLOGY",
    "N_P1_NUISANCE",
    "OBSERVATION_KEY",
    "P1_SCHEMA_ID",
    "SIGMA_KEY",
    "SIGN_TIE_RTOL",
    "TRANSFORM_VERSION",
    "WHITENING_KEY",
    "cell_log_permeability_operator",
    "check_informed_split",
    "condition_gaussian",
    "conditional_square_root",
    "information_matrix",
    "log_k_observation_ids",
    "p1_design_for",
    "p1_prior_context",
    "support_log_permeability_operator",
    "tie_blocks",
    "whitening_rotation",
]
