"""Leakage-safe construction of the first E02 exploratory inverse world."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np

from so_recon.geology.conditional import (
    DESIGN_KEY,
    LOG_K_QUANTITY,
    condition_gaussian,
    conditional_square_root,
    information_matrix,
    p1_prior_context,
    whitening_rotation,
)
from so_recon.geology.renderer import build_inverse_case, render_theta
from so_recon.inference.contracts import (
    FIXED_NOISE_THETA,
    DensitySchema,
    HistoryRow,
    ModelObservations,
    NoiseTheta,
    ObservationBundle,
    PriorContext,
    ThetaRecord,
)
from so_recon.observation.bins import rounding_grid
from so_recon.observation.noise import draw_history
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, write_json_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.synthetic.inverse_designs import (
    INVERSE_DESIGN_KEY,
    INVERSE_RENDERER_VERSION,
    InversePhysicalDesign,
    inverse_design,
    inverse_supports,
    permeability_multiplier,
    render_inverse_coefficients,
)
from so_recon.synthetic.p1 import P1Design, render_p1
from so_recon.synthetic.world_io import write_world

INFERENCE_INPUT_VERSION = "e02-inference-input-1"
INFERENCE_ALLOWLIST = frozenset({"context", "G", "U", "observations", "density_schema", "basis"})
FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_ref",
        "theta",
        "theta_ref",
        "seed",
        "parent_seed",
        "initial_pressure_pa",
        "full_permeability",
        "full_porosity",
        "full_state",
    }
)
HISTORY_GENERATOR_VERSION = "e02-dynamic-history-1"
STATIC_LOG_K_SIGMA = 0.2


def _refuse_hidden_state(value: object, *, location: str = "input") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"{location}.{key}: truth or hidden generator state is forbidden")
            _refuse_hidden_state(item, location=f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _refuse_hidden_state(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and any(
        marker in value.lower().replace("\\", "/")
        for marker in ("/truth/", "/theta.json", "/states.h5")
    ):
        raise ValueError(f"{location}: truth artifact path is forbidden")


def inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete encoder/inference input against a recursive allowlist."""
    _refuse_hidden_state(payload)
    unexpected = sorted(set(payload) - INFERENCE_ALLOWLIST)
    if unexpected:
        raise ValueError(f"unexpected inference input keys: {unexpected}")
    return payload


def _support_mean(values: np.ndarray, cell_ids: tuple[int, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)[list(cell_ids)]))


def _custom_prior_context(
    design: InversePhysicalDesign,
    truth_arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> PriorContext:
    """Condition a T2/T4 Gaussian using that design's declared symmetric/western G."""
    supports = inverse_supports(design)
    zero = render_inverse_coefficients(np.zeros(design.n_geology), design)
    intercept = np.asarray(
        [_support_mean(zero["log_permeability_m2"], support.cell_ids) for support in supports],
        dtype=np.float64,
    )
    operator = np.empty((len(supports), design.n_geology), dtype=np.float64)
    for column in range(design.n_geology):
        unit = np.zeros(design.n_geology, dtype=np.float64)
        unit[column] = 1.0
        rendered = render_inverse_coefficients(unit, design)
        operator[:, column] = [
            _support_mean(rendered["log_permeability_m2"], support.cell_ids) - base
            for support, base in zip(supports, intercept, strict=True)
        ]
    latent = np.asarray(
        [
            _support_mean(truth_arrays["log_permeability_m2"], support.cell_ids)
            for support in supports
        ],
        dtype=np.float64,
    )
    measured = latent + STATIC_LOG_K_SIGMA * rng.standard_normal(len(supports))
    variance = np.full(len(supports), STATIC_LOG_K_SIGMA**2, dtype=np.float64)
    mean, covariance = condition_gaussian(operator, measured - intercept, variance)
    rotation, eigenvalues = whitening_rotation(information_matrix(operator, variance))
    root = conditional_square_root(rotation, eigenvalues)
    if not np.allclose(root @ root.T, covariance, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{design.design_id}: conditional root does not reproduce covariance")

    g_rows = [
        {
            "observation_id": support.observation_id,
            "support_cell_ids": list(support.cell_ids),
            "value": float(value),
            "sigma": STATIC_LOG_K_SIGMA,
        }
        for support, value in zip(supports, measured, strict=True)
    ]
    g_hash = sha256_json({"quantity": LOG_K_QUANTITY, "unit": "ln(m2)", "observations": g_rows})
    information_hash = sha256_json(
        {
            "role": "condition_prior",
            "g_hash": g_hash,
            "renderer": INVERSE_RENDERER_VERSION,
            "design": design.payload(),
        }
    )
    basis_hash = sha256_json(
        {
            "transform_version": design.transform_version,
            "information_hash": information_hash,
            "eigenvalues": eigenvalues.tolist(),
            "rotation": rotation.tolist(),
            "mean": mean.tolist(),
            "root": root.tolist(),
        }
    )
    schema = DensitySchema(
        schema_id=design.schema_id,
        n_v=11,
        n_residual=design.n_geology - 8 + design.n_state_residual,
        families=(0,),
        basis_hash=basis_hash,
        transform_version=design.transform_version,
    )
    return PriorContext(
        density_schema=schema,
        n_geology=design.n_geology,
        n_state_residual=design.n_state_residual,
        mean=mean,
        chol=root,
        rotation=rotation,
        design={
            INVERSE_DESIGN_KEY: design.model_dump(mode="json"),
            "log_k_sigma": STATIC_LOG_K_SIGMA,
            "log_k_observations": g_rows,
            "whitening": {
                "transform_version": design.transform_version,
                "eigenvalues": eigenvalues.tolist(),
                "effective_rank": int(np.count_nonzero(eigenvalues > 1.0e-10)),
            },
        },
        g_hash=g_hash,
        information_hash=information_hash,
    )


def _truth_theta(
    coefficients: np.ndarray,
    nuisance: np.ndarray,
    state_coordinate: float,
    context: PriorContext,
) -> ThetaRecord:
    physical_whitened = np.linalg.solve(context.chol, coefficients - context.mean)
    whitened = context.rotation.T @ physical_whitened
    residual = list(whitened[8:])
    if context.n_state_residual:
        residual.append(float(state_coordinate))
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=0,
        v=tuple(float(value) for value in (*whitened[:8], *nuisance)),
        z_perp=tuple(float(value) for value in residual),
        basis_hash=context.density_schema.basis_hash,
    )
    context.density_schema.validate_theta(theta)
    return theta


def _pending_observations(
    context: PriorContext, cutoff_s: float, design_id: str
) -> ObservationBundle:
    grid = rounding_grid(0.01)
    return ObservationBundle(
        history=(),
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(value) for value in grid.edges)},
        cutoff_s=cutoff_s,
        information_hash=context.information_hash,
        observation_hash=sha256_json(
            {
                "status": "DYNAMIC_HISTORY_PENDING_TRUTH_FORWARD",
                "design_id": design_id,
                "information_hash": context.information_hash,
            }
        ),
    )


def _make_custom_inverse_world(
    design_id: str,
    seed: int,
    paths: ProjectPaths,
    ctx: RunContext,
) -> tuple[PriorContext, ObservationBundle, ArtifactRef]:
    design = inverse_design(design_id)
    truth_stream, static_stream, _history_stream, _schedule_stream = np.random.SeedSequence(
        seed
    ).spawn(4)
    truth_rng = np.random.default_rng(truth_stream)
    coefficients = truth_rng.standard_normal(design.n_geology)
    state_coordinate = float(truth_rng.standard_normal()) if design.n_state_residual else 0.0
    nuisance = truth_rng.standard_normal(3)
    truth_arrays = render_inverse_coefficients(
        coefficients,
        design,
        state_coordinate=state_coordinate,
    )
    context = _custom_prior_context(design, truth_arrays, np.random.default_rng(static_stream))
    theta = _truth_theta(coefficients, nuisance, state_coordinate, context)
    rendered = render_theta(theta, context)
    reproduces = all(
        np.allclose(rendered.arrays[name], values, rtol=0.0, atol=1.0e-12)
        for name, values in truth_arrays.items()
    )
    if not reproduces:
        raise ValueError(f"{design_id}: the conditional theta does not reproduce its truth arrays")
    case = build_inverse_case(rendered, context, paths, ctx)
    multiplier = permeability_multiplier(design)
    truth_ref = write_json_artifact(
        paths.artifacts / "inverse_worlds" / f"{design_id}-s{seed}" / "truth.json",
        {
            "schema_version": "e02-inverse-truth-1",
            "warning": "synthetic truth; evaluator-only; never an inference payload",
            "design_id": design_id,
            "seed": seed,
            "model_hash": case.model_hash,
            "physical_design": design.payload(),
            "theta": theta.model_dump(mode="json"),
            "case": case.model_dump(mode="json"),
            "renderer_checks": {
                "reproduces_truth_arrays": reproduces,
                "remote_k_multiplier": float(multiplier.min()),
                "initial_sw_range": float(np.ptp(truth_arrays["sw"])),
            },
        },
        paths,
        schema_version="e02-inverse-truth-1",
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )
    ctx.add_output(f"inverse_world.{design_id}.truth", truth_ref)
    return context, _pending_observations(context, design.report_edges_s[-1], design_id), truth_ref


def make_inverse_world(
    design_id: str,
    seed: int,
    paths: ProjectPaths,
    ctx: RunContext,
) -> tuple[PriorContext, ObservationBundle, ArtifactRef]:
    """Publish T1 generator truth while returning only conditional context and observations.

    Dynamic history is generated only after the truth forward runs; the bundle returned here
    is therefore explicitly empty rather than carrying guessed watercut.  The truth artifact
    is a separate evaluator output and is never embedded in that bundle.
    """
    if design_id in {"e02-t2-v1", "e02-t2-v2", "e02-t4-v1"}:
        return _make_custom_inverse_world(design_id, seed, paths, ctx)
    if design_id != "e02-t1-v1":
        raise ValueError(
            f"design {design_id!r} has no registered renderer; available generators are "
            "'e02-t1-v1', 'e02-t2-v1', 'e02-t2-v2', and 'e02-t4-v1'"
        )
    design = P1Design(family="base")
    world = render_p1(seed, design)
    log_k = {
        str(row["observation_id"]): float(row["value"])
        for row in world.static_observations.to_pylist()
        if row["quantity"] == LOG_K_QUANTITY
    }
    context = p1_prior_context(design, log_k)
    observations = _pending_observations(context, design.report_edges_s[-1], design_id)
    truth_ref = write_world(world, None, paths, ctx)
    return context, observations, truth_ref


def generate_dynamic_history(
    context: PriorContext,
    prediction: ModelObservations,
    *,
    seed: int,
    noise: NoiseTheta = FIXED_NOISE_THETA,
) -> ObservationBundle:
    """Turn a completed truth forward into the rounded recursive history seen by inference.

    The mask is physical: a dry model month remains an invalid row.  Noise is drawn only
    after F is available, from its own seed, and the exact same recursive discrete kernel
    scores the resulting rows later.
    """
    if seed < 0:
        raise ValueError(f"history seed must be non-negative, got {seed}")
    grid = rounding_grid(0.01)
    templates = tuple(
        HistoryRow(
            well_id=well_id,
            month_index=month,
            raw_value=0.0 if value is not None else None,
            bin_index=0 if value is not None else None,
            quality_group="watercut-0.01",
            observed_valid=value is not None,
            reset=False,
        )
        for (well_id, month), value in sorted(prediction.fw.items())
    )
    rows = draw_history(
        templates,
        prediction.fw,
        noise,
        {"watercut-0.01": grid},
        np.random.default_rng(seed),
    )
    custom = context.design.get(INVERSE_DESIGN_KEY)
    loop = context.design.get("loop_design")
    if custom is not None:
        cutoff_s = inverse_design(str(custom["design_id"])).report_edges_s[-1]
    elif loop is not None:
        from so_recon.synthetic.loop_designs import T3Design

        cutoff_s = T3Design.model_validate(loop).report_edges_s[-1]
    else:
        cutoff_s = P1Design.model_validate(context.design[DESIGN_KEY]).report_edges_s[-1]
    observation_hash = sha256_json(
        {
            "generator_version": HISTORY_GENERATOR_VERSION,
            "information_hash": context.information_hash,
            "model_hash": prediction.model_hash,
            "noise": noise.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in rows],
        }
    )
    return ObservationBundle(
        history=rows,
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(value) for value in grid.edges)},
        cutoff_s=cutoff_s,
        information_hash=context.information_hash,
        observation_hash=observation_hash,
    )


__all__ = [
    "INFERENCE_ALLOWLIST",
    "INFERENCE_INPUT_VERSION",
    "HISTORY_GENERATOR_VERSION",
    "generate_dynamic_history",
    "inference_payload",
    "make_inverse_world",
]
