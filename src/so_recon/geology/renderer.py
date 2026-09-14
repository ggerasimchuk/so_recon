"""E02.2 — the latent renderer: one `ThetaRecord` to the physical coefficient arrays.

`render_theta` is the deterministic map `theta -> (arrays, noise law, family)`. It is the
only place the latent coordinates become rock, and it is PURE: it opens no file, reads no
truth world and consults no clock. Publishing a rendered case is Task 6's job, deliberately
kept out of here so that this function can be called inside an SMC inner loop without a
side effect.

The geology block of the latent point is assembled as `concat(v[:8], z_perp[:4])` — all
twelve whitened coordinates, none dropped (SPEC §7.5) — and mapped to the E01 coefficients
by `a = mean + chol @ rotation @ w`, where `chol` is the PRINCIPAL SYMMETRIC square root of
the conditional covariance (`conditional_square_root`). Only a root that commutes with the
rotation leaves the composition's columns in eigenvalue order, and therefore leaves the
four trailing ones in the null space of the log-permeability operator: with a triangular
factor a unit move in `z_perp` shifts the very logs the prior was conditioned on. The
hypothesis `s` selects the family, which moves `kz/kx` and nothing else; the three nuisance
coordinates of `v` give the noise law through the map `NoiseTheta` already declares.

**An unrenderable geology is a failure, not a zero density.** E01's `layer_geology` refuses
a permeability outside `[1e-3, 1e6]` mD rather than clipping it. That refusal is NOT a
normalised truncation of the Gaussian prior: no draw is rejected and retried until it looks
physical, and no `-inf` is returned. The physical evaluation stops with
`RendererNumericalError` naming the theta it stopped on. Giving the prior a finite physical
support would require a new, explicitly versioned prior with its own sampler and
normaliser tests — not a silent clip inside a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so_recon.geology.conditional import (
    N_GEOLOGY,
    N_P1_NUISANCE,
    TRANSFORM_VERSION,
    p1_design_for,
)
from so_recon.inference.contracts import (
    F64,
    N_P1_GEOLOGY_IN_V,
    NoiseTheta,
    PriorContext,
    RendererNumericalError,
    ThetaRecord,
)
from so_recon.registry.hashing import sha256_json
from so_recon.synthetic.p1 import (
    GENERATOR_VERSION,
    N_LAYERS,
    N_MODES,
    render_coefficients,
)
from so_recon.synthetic.p1 import RENDERER_VERSION as E01_RENDERER_VERSION

#: This map, versioned. The coordinate order, the assembly of the geology block and the
#: family map are all part of it: changing any of them changes what a stored theta means.
RENDERER_VERSION = "e02-latent-renderer-1"

#: How many of the residual coordinates are geology. The rest, if a schema declares any,
#: are independent initial-state coordinates and belong to another renderer.
N_GEOLOGY_IN_RESIDUAL = N_GEOLOGY - N_P1_GEOLOGY_IN_V


@dataclass(frozen=True)
class RenderedParameters:
    """What one latent point is, physically: the arrays, the noise law and the family."""

    arrays: dict[str, F64]
    noise: NoiseTheta
    family: int
    renderer_hash: str


def theta_hash(theta: ThetaRecord) -> str:
    """The identity of a latent point, so a failure can name the theta that caused it."""
    return sha256_json(theta.model_dump(mode="json"))


def renderer_hash(context: PriorContext) -> str:
    """The identity of the MAP, not of a draw: two thetas of one context share it.

    It covers the renderer's own version, the E01 generator it renders through, the design
    and the basis the latent coordinates are expressed in. A cache keyed by this and by
    `theta_hash` cannot return one context's rock for another context's coordinates.
    """
    return sha256_json(
        {
            "renderer_version": RENDERER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "e01_renderer_version": E01_RENDERER_VERSION,
            "transform_version": TRANSFORM_VERSION,
            "schema_id": context.density_schema.schema_id,
            "basis_hash": context.density_schema.basis_hash,
            "g_hash": context.g_hash,
            "design": context.design,
        }
    )


def _check_context(context: PriorContext) -> None:
    schema = context.density_schema
    if context.n_geology != N_GEOLOGY or schema.transform_version != TRANSFORM_VERSION:
        raise ValueError(
            f"this renderer is the {N_GEOLOGY}-coefficient P1 map of {TRANSFORM_VERSION}; "
            f"the context declares {context.n_geology} coefficients under "
            f"{schema.transform_version!r}. A reduced or toy schema has its own renderer"
        )
    if schema.n_v < N_P1_GEOLOGY_IN_V + N_P1_NUISANCE:
        raise ValueError(
            f"v holds {N_P1_GEOLOGY_IN_V} geology and {N_P1_NUISANCE} nuisance "
            f"coordinates, so it needs at least {N_P1_GEOLOGY_IN_V + N_P1_NUISANCE}, "
            f"got {schema.n_v}"
        )
    if schema.n_residual != N_GEOLOGY_IN_RESIDUAL + context.n_state_residual:
        raise ValueError(
            f"z_perp holds the remaining {N_GEOLOGY_IN_RESIDUAL} geology coordinates and "
            f"{context.n_state_residual} initial-state coordinates, so it needs "
            f"{N_GEOLOGY_IN_RESIDUAL + context.n_state_residual}, got {schema.n_residual}"
        )


def geology_coefficients(theta: ThetaRecord, context: PriorContext) -> F64:
    """The twelve E01 coefficients of a latent point: `mean + chol @ rotation @ w`.

    `w` is `concat(v[:8], z_perp[:4])` — every whitened direction, including the four the
    sparse logs left unconstrained, which the symmetric factor keeps unconstrained.
    """
    _check_context(context)
    context.density_schema.validate_theta(theta)
    whitened = np.concatenate(
        [
            np.asarray(theta.v[:N_P1_GEOLOGY_IN_V], dtype=np.float64),
            np.asarray(theta.z_perp[:N_GEOLOGY_IN_RESIDUAL], dtype=np.float64),
        ]
    )
    coefficients: F64 = context.mean + context.chol @ (context.rotation @ whitened)
    return coefficients


def render_theta(theta: ThetaRecord, context: PriorContext) -> RenderedParameters:
    """Render one latent point. No file is opened and no truth world is read."""
    coefficients = geology_coefficients(theta, context).reshape(N_LAYERS, N_MODES)
    design = p1_design_for(context, theta.s)
    try:
        arrays = render_coefficients(coefficients, design)
    except ValueError as error:
        # E01 refused this rock. That is a failed physical evaluation carrying the theta
        # that produced it, and never a statement about the prior's density there.
        raise RendererNumericalError(
            f"theta {theta_hash(theta)} renders no physical geology in family "
            f"{design.family!r}: {error}"
        ) from error
    return RenderedParameters(
        arrays=arrays,
        noise=NoiseTheta.from_latent(theta.v),
        family=theta.s,
        renderer_hash=renderer_hash(context),
    )


__all__ = [
    "N_GEOLOGY_IN_RESIDUAL",
    "RENDERER_VERSION",
    "RenderedParameters",
    "geology_coefficients",
    "render_theta",
    "renderer_hash",
    "theta_hash",
]
