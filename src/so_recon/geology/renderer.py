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

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import (
    cartesian_neighbors,
    compute_model_hash,
    validate_case,
    write_arrays,
)
from so_recon.simulator.contracts import (
    CELL_AXES,
    CELL_DIM_AXES,
    DIM_CELL_AXES,
    FACE_AXES,
    BoundarySpec,
    CaseBundle,
    FluidSpec,
    GridSpec,
    InitialStateSpec,
    ObservationSpec,
    RockSpec,
)
from so_recon.synthetic.inverse_designs import (
    INVERSE_DESIGN_KEY,
    INVERSE_RENDERER_VERSION,
    InversePhysicalDesign,
    inverse_control_segments,
    inverse_well_specs,
    render_inverse_coefficients,
)
from so_recon.synthetic.p1 import (
    DYNAMIC_CHANNELS,
    GENERATOR_VERSION,
    N_LAYERS,
    N_MODES,
    OBSERVATION_SCHEMA,
    PRESSURE_AVAILABLE,
    P1Design,
    control_segments,
    render_coefficients,
    well_specs,
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
            "inverse_renderer_version": INVERSE_RENDERER_VERSION,
            "transform_version": TRANSFORM_VERSION,
            "schema_id": context.density_schema.schema_id,
            "basis_hash": context.density_schema.basis_hash,
            "g_hash": context.g_hash,
            "design": context.design,
        }
    )


def _check_context(context: PriorContext) -> None:
    schema = context.density_schema
    custom = INVERSE_DESIGN_KEY in context.design
    if not custom and (
        context.n_geology != N_GEOLOGY or schema.transform_version != TRANSFORM_VERSION
    ):
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
    geology_residual = context.n_geology - N_P1_GEOLOGY_IN_V
    if geology_residual < 0:
        raise ValueError(
            f"the renderer needs at least {N_P1_GEOLOGY_IN_V} geology coordinates, "
            f"got {context.n_geology}"
        )
    if schema.n_residual != geology_residual + context.n_state_residual:
        raise ValueError(
            f"z_perp holds the remaining {geology_residual} geology coordinates and "
            f"{context.n_state_residual} initial-state coordinates, so it needs "
            f"{geology_residual + context.n_state_residual}, got {schema.n_residual}"
        )


def _design_for(context: PriorContext, family: int) -> P1Design | InversePhysicalDesign:
    payload = context.design.get(INVERSE_DESIGN_KEY)
    if payload is None:
        return p1_design_for(context, family)
    if family != 0:
        raise ValueError(f"the fixed {payload['design_id']} renderer has only family s=0")
    return InversePhysicalDesign.model_validate(payload)


def geology_coefficients(theta: ThetaRecord, context: PriorContext) -> F64:
    """The twelve E01 coefficients of a latent point: `mean + chol @ rotation @ w`.

    `w` is `concat(v[:8], z_perp[:4])` — every whitened direction, including the four the
    sparse logs left unconstrained, which the symmetric factor keeps unconstrained.
    """
    _check_context(context)
    context.density_schema.validate_theta(theta)
    geology_residual = context.n_geology - N_P1_GEOLOGY_IN_V
    whitened = np.concatenate(
        [
            np.asarray(theta.v[:N_P1_GEOLOGY_IN_V], dtype=np.float64),
            np.asarray(theta.z_perp[:geology_residual], dtype=np.float64),
        ]
    )
    coefficients: F64 = context.mean + context.chol @ (context.rotation @ whitened)
    return coefficients


def render_theta(theta: ThetaRecord, context: PriorContext) -> RenderedParameters:
    """Render one latent point. No file is opened and no truth world is read."""
    coefficients = geology_coefficients(theta, context)
    design = _design_for(context, theta.s)
    try:
        if isinstance(design, InversePhysicalDesign):
            state_index = context.n_geology - N_P1_GEOLOGY_IN_V
            state_coordinate = float(theta.z_perp[state_index]) if context.n_state_residual else 0.0
            arrays = render_inverse_coefficients(
                coefficients,
                design,
                state_coordinate=state_coordinate,
            )
        else:
            arrays = render_coefficients(coefficients.reshape(N_LAYERS, N_MODES), design)
    except ValueError as error:
        # E01 refused this rock. That is a failed physical evaluation carrying the theta
        # that produced it, and never a statement about the prior's density there.
        raise RendererNumericalError(
            f"theta {theta_hash(theta)} renders no physical geology in design "
            f"{design.design_id!r}: {error}"
        ) from error
    return RenderedParameters(
        arrays=arrays,
        noise=NoiseTheta.from_latent(theta.v),
        family=theta.s,
        renderer_hash=renderer_hash(context),
    )


def _theta_for_coefficients(
    theta: ThetaRecord, context: PriorContext, coefficients: F64
) -> ThetaRecord:
    """Express physical coefficients in the context's stored conditional coordinates."""
    physical_whitened = np.linalg.solve(context.chol, coefficients - context.mean)
    whitened = context.rotation.T @ physical_whitened
    geology_residual = context.n_geology - N_P1_GEOLOGY_IN_V
    residual = [float(value) for value in whitened[N_P1_GEOLOGY_IN_V:]]
    residual.extend(float(value) for value in theta.z_perp[geology_residual:])
    updated = theta.model_copy(
        update={
            "v": tuple(
                [float(value) for value in whitened[:N_P1_GEOLOGY_IN_V]]
                + list(theta.v[N_P1_GEOLOGY_IN_V:])
            ),
            "z_perp": tuple(residual),
        }
    )
    context.density_schema.validate_theta(updated)
    return updated


def swap_t2_layers(theta: ThetaRecord, context: PriorContext) -> ThetaRecord:
    """Create T2's physical layer-exchange pair without permuting latent labels by hand."""
    design = _design_for(context, theta.s)
    if not isinstance(design, InversePhysicalDesign) or not design.is_t2:
        raise ValueError("layer exchange is defined only for E02 T2 designs")
    coefficients = geology_coefficients(theta, context)
    swapped = np.concatenate([coefficients[N_MODES:], coefficients[:N_MODES]])
    return _theta_for_coefficients(theta, context, swapped)


def with_t4_remote_state(
    theta: ThetaRecord, context: PriorContext, state_coordinate: float
) -> ThetaRecord:
    """Hold all T4 coordinates fixed except the declared remote initial-state residual."""
    design = _design_for(context, theta.s)
    if not isinstance(design, InversePhysicalDesign) or not design.is_t4:
        raise ValueError("remote initial-state pairs are defined only for E02 T4 designs")
    if not np.isfinite(state_coordinate):
        raise ValueError("remote state coordinate must be finite")
    residual = (*theta.z_perp[:-1], float(state_coordinate))
    updated = theta.model_copy(update={"z_perp": residual})
    context.density_schema.validate_theta(updated)
    return updated


def _array_identity(arrays: dict[str, F64]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name], dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _empty_observations_bytes() -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([], schema=OBSERVATION_SCHEMA), sink, compression="snappy")
    payload: bytes = sink.getvalue().to_pybytes()
    return payload


def build_inverse_case(
    rendered: RenderedParameters,
    context: PriorContext,
    paths: ProjectPaths,
    ctx: RunContext,
) -> CaseBundle:
    """Publish a rendered theta as a solver case without consulting a truth world.

    The deterministic case location is derived from the renderer and physical array bytes.
    An empty observation table satisfies the forward exchange contract; observations are
    not an input the solver reads and the actual inverse bundle stays on the Python side.
    """
    design = _design_for(context, rendered.family)
    arrays = rendered.arrays
    identity = sha256_json(
        {
            "renderer_hash": rendered.renderer_hash,
            "array_identity": _array_identity(arrays),
            "design": design.payload(),
        }
    )
    root = paths.artifacts / "inverse_cases" / identity
    grid = write_arrays(
        root / "grid.h5",
        {
            "cell_centers_m": (arrays["cell_centers_m"], "m", CELL_DIM_AXES),
            "cell_volume_m3": (arrays["cell_volume_m3"], "m3", CELL_AXES),
            "neighbors": (cartesian_neighbors(design.shape), "1", FACE_AXES),
        },
        paths=paths,
    )
    geology = write_arrays(
        root / "geology.h5",
        {
            "porosity": (arrays["porosity"], "1", CELL_AXES),
            "permeability_m2": (arrays["permeability_m2"], "m2", DIM_CELL_AXES),
        },
        paths=paths,
    )
    initial = write_arrays(
        root / "initial.h5",
        {
            "pressure_pa": (arrays["pressure_pa"], "Pa", CELL_AXES),
            "sw": (arrays["sw"], "1", CELL_AXES),
        },
        paths=paths,
    )
    observation_ref = write_artifact(
        root / "observations.parquet",
        _empty_observations_bytes(),
        paths,
        schema_version="e02-forward-observations-1",
        producer_run_id=ctx.run_id,
        media_type="application/vnd.apache.parquet",
        now=datetime.now(UTC),
    )
    ctx.add_output(f"inverse_case.{identity}.observations", observation_ref)

    case = CaseBundle(
        case_id=f"inverse-{identity[:16]}",
        world_id=f"inverse-{identity[:16]}",
        start_date=design.start_date,
        cutoff=design.cutoff,
        report_edges_s=design.report_edges_s,
        grid=GridSpec(
            shape=design.shape,
            extent_m=design.extent_m,
            cell_centers_m=grid["cell_centers_m"],
            cell_volume_m3=grid["cell_volume_m3"],
            neighbors=grid["neighbors"],
        ),
        rock=RockSpec(porosity=geology["porosity"], permeability_m2=geology["permeability_m2"]),
        fluids=FluidSpec(),
        wells=(
            inverse_well_specs(design)
            if isinstance(design, InversePhysicalDesign)
            else well_specs(design)
        ),
        controls=(
            inverse_control_segments(design)
            if isinstance(design, InversePhysicalDesign)
            else control_segments(design)
        ),
        initial=InitialStateSpec(
            kind="explicit",
            pressure_pa=initial["pressure_pa"],
            sw=initial["sw"],
            meaning=(
                "developed_state"
                if isinstance(design, InversePhysicalDesign) and design.is_t4
                else "synthetic_initial"
            ),
        ),
        boundary=BoundarySpec(kind="closed", cells=()),
        observations=ObservationSpec(
            table_path=observation_ref.path,
            sha256=observation_ref.sha256,
            dynamic_channels=DYNAMIC_CHANNELS,
            pressure_available=PRESSURE_AVAILABLE,
        ),
        renderer_version=(
            INVERSE_RENDERER_VERSION
            if isinstance(design, InversePhysicalDesign)
            else RENDERER_VERSION
        ),
        units={
            "pressure": "Pa",
            "permeability": "m2",
            "time": "s",
            "length": "m",
            "control_rate": "m3_sc/day",
        },
        seeds={"fixture": 0},
        source_hashes={
            "conditional_information": context.information_hash,
            "renderer": rendered.renderer_hash,
            "physical_arrays": _array_identity(arrays),
        },
        model_hash="e" * 64,
    )
    case = case.model_copy(update={"model_hash": compute_model_hash(case)})
    report = validate_case(case, paths)
    if not report.valid:
        raise ValueError(
            f"rendered inverse case {identity} is invalid: " + "; ".join(report.errors)
        )
    return case


__all__ = [
    "N_GEOLOGY_IN_RESIDUAL",
    "RENDERER_VERSION",
    "RenderedParameters",
    "build_inverse_case",
    "geology_coefficients",
    "render_theta",
    "renderer_hash",
    "swap_t2_layers",
    "theta_hash",
    "with_t4_remote_state",
]
