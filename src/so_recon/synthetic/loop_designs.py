"""E03 loop designs: T3 (unknown initial state on a completion calendar) and T5 mapping.

The historical E02 designs stay untouched in `inverse_designs.py`; this module declares
the E03 additions as their own versioned objects (plan §3.2–§3.3):

**T3 (`e03-t3-v1`)** is the E01 geology and geometry with two declared inter-layer
connection hypotheses `s in {0,1}` (the E01 `base`/`low_vertical` kz-over-kx pair) and
ONE independent initial-state residual `u ~ N(0,1)`:

    Sw0 = Swc + 0.10 * Phi(u)      (uniform over the box, no mask)

`0.10` is a project setting of the educational T3 world, not a field value. The bound
`Swc + 0.10 < 1 - Sor` is checked against the actual `FluidSpec` constants at import.
The selected producer's LOWER connection opens on the month-18 report edge; the calendar
is part of the design and known to every method, while the true `s` is not an encoder
input. Nothing clips a final saturation: `u` moves the initial state only.

**T5** is not a new world: it is the physical mapping of a registered T1 parent onto a
48x48x2 refinement for the paired coarse/fine truth diagnostic. Wells keep ONE
connection per layer at the same physical coordinates, support zones are declared in
physical coordinates, and the fine case differs from the coarse one by its own model
hash.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from scipy.special import ndtr

from so_recon.config.schema import StrictModel
from so_recon.simulator.contracts import (
    MILLIDARCY_M2,
    ControlSegment,
    FluidSpec,
    WellSpec,
)
from so_recon.simulator.schedule import month_edges, month_edges_s
from so_recon.synthetic.p1 import (
    BASE_RATE_M3_SC_DAY,
    DATUM_DEPTH_M,
    INJECTOR_MAX_BHP_PA,
    INITIAL_SW,
    MAX_PERMEABILITY_MD,
    MIN_PERMEABILITY_MD,
    MODES,
    N_LAYERS,
    N_MODES,
    PRODUCER_MIN_BHP_PA,
    RATE_MODULATION,
    WELL_RADIUS_M,
    P1Design,
    oil_connected_hydrostatic_pa,
)

#: The context-design key that routes the latent renderer to the E03 loop designs, next
#: to E02's `inverse_design` key. A context carries exactly one of the two.
LOOP_DESIGN_KEY = "loop_design"

T3_DESIGN_ID = "e03-t3-v1"
T3_SCHEMA_ID = "e03-t3-13d"
T3_TRANSFORM_VERSION = "e03-t3-v1-conditional-1"
T3_RENDERER_VERSION = "e03-loop-renderer-1"
T3_TRUTH_SCHEMA = "e03-loop-truth-1"

#: The two declared inter-layer connection hypotheses. These are the E01 family values
#: (`base`, `low_vertical`) verbatim: the hypothesis moves `kz/kx` and nothing else, so
#: the log-permeability supports — and therefore the conditioning G — are shared.
T3_FAMILIES: tuple[int, ...] = (0, 1)
T3_FAMILY_NAMES: tuple[str, str] = ("base", "low_vertical")
T3_FAMILY_KZ_OVER_KX: tuple[float, float] = (0.05, 0.001)

#: The initial-state residual (plan §3.2). `Sw0 = Swc + T3_INITIAL_SW_SHIFT * Phi(u)`.
T3_INITIAL_SW_SHIFT = 0.10

#: The completion event: the selected producer's lower connection opens on this
#: zero-based report edge (the boundary of month 18) and stays open to the cutoff.
T3_EVENT_WELL = "P1"
T3_LOWER_OPEN_EDGE = 18


def _check_declared_sw0_support() -> None:
    """`Swc + 0.10 < 1 - Sor` against the actual fluid constants (plan §3.2)."""
    fluids = FluidSpec()
    swc, _sor_oil = fluids.residual_saturations
    sor = _sor_oil
    if not INITIAL_SW + T3_INITIAL_SW_SHIFT < 1.0 - sor:
        raise AssertionError(
            f"T3 declares Sw0 = Swc + {T3_INITIAL_SW_SHIFT} with Swc={INITIAL_SW}, "
            f"Sor={sor}: the support check Swc + shift < 1 - Sor fails"
        )
    if not math.isclose(INITIAL_SW, swc, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError(
            f"the educational connate water moved: p1 INITIAL_SW={INITIAL_SW}, "
            f"FluidSpec Swc={swc}. T3's declared support must be re-derived, not guessed"
        )


_check_declared_sw0_support()


class T3Design(StrictModel):
    """The E03 T3 physical definition: E01 rock hypotheses, unknown uniform Sw0."""

    design_id: Literal[T3_DESIGN_ID] = T3_DESIGN_ID
    shape: tuple[int, int, int] = (16, 16, 2)
    extent_m: tuple[float, float, float] = (100.0, 100.0, 20.0)
    n_months: int = 36
    start_date: str = "2000-01-01"

    @model_validator(mode="after")
    def _fixed_geometry(self) -> T3Design:
        if self.shape != (16, 16, 2) or self.extent_m != (100.0, 100.0, 20.0):
            raise ValueError("T3 v1 fixes the E01 16x16x2, 100x100x20 m box")
        if self.n_months != 36 or date.fromisoformat(self.start_date).day != 1:
            raise ValueError("T3 v1 fixes 36 calendar months from a month edge")
        return self

    @property
    def n_cells(self) -> int:
        return math.prod(self.shape)

    @property
    def cells_per_layer(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def cell_size_m(self) -> tuple[float, float, float]:
        return tuple(
            extent / count for extent, count in zip(self.extent_m, self.shape, strict=True)
        )  # type: ignore[return-value]

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def cutoff(self) -> str:
        return month_edges(self.start, self.n_months)[-1].isoformat()

    @property
    def report_edges_s(self) -> tuple[float, ...]:
        return month_edges_s(self.start, self.n_months)

    @property
    def n_geology(self) -> int:
        return N_LAYERS * N_MODES

    @property
    def n_state_residual(self) -> int:
        return 1

    @property
    def schema_id(self) -> str:
        return T3_SCHEMA_ID

    @property
    def transform_version(self) -> str:
        return T3_TRANSFORM_VERSION

    @property
    def layer_base_permeability_md(self) -> tuple[float, float]:
        return (120.0, 40.0)

    @property
    def well_columns(self) -> tuple[tuple[str, tuple[int, int], str], ...]:
        # The E01/T1 layout verbatim: two injectors west, two producers east.
        return (
            ("I1", (2, 2), "injector"),
            ("I2", (2, 13), "injector"),
            ("P1", (13, 2), "producer"),
            ("P2", (13, 13), "producer"),
        )

    def kz_over_kx(self, family: int) -> float:
        """The declared inter-layer connection hypothesis of `family`."""
        if family not in T3_FAMILIES:
            raise ValueError(f"T3 declares families {T3_FAMILIES}, got s={family}")
        return T3_FAMILY_KZ_OVER_KX[family]

    def payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "n_geology": self.n_geology,
            "n_state_residual": self.n_state_residual,
            "schema_id": self.schema_id,
            "transform_version": self.transform_version,
            "layer_base_permeability_md": list(self.layer_base_permeability_md),
            "family_kz_over_kx": list(T3_FAMILY_KZ_OVER_KX),
            "family_names": list(T3_FAMILY_NAMES),
            "well_columns": [
                {"well_id": name, "column": list(column), "role": role}
                for name, column, role in self.well_columns
            ],
            "completion_event": {
                "well": T3_EVENT_WELL,
                "connection": "lower",
                "open_edge": T3_LOWER_OPEN_EDGE,
                "calendar_known_to": "all-methods",
            },
            "initial_state": {
                "kind": "uniform_sw0_shift",
                "swc": INITIAL_SW,
                "shift": T3_INITIAL_SW_SHIFT,
                "meaning": "synthetic_nonvirgin_initial_state",
            },
            "encoder_visibility": "calendar yes, family s no",
        }


def t3_design() -> T3Design:
    return T3Design()


def render_t3_coefficients(
    coefficients: NDArray[np.float64],
    design: T3Design,
    *,
    family: int,
    state_coordinate: float,
) -> dict[str, NDArray[np.float64]]:
    """Render T3: the E01 rock kernel, the family's kz, and the uniform Sw0 shift.

    The geology block is exactly the E01 kernel (six cosine modes per layer over
    normalized coordinates, no remote mask, no multiplier), so a T3 coefficient vector
    and a T1 one mean the same rock. `state_coordinate` enters the INITIAL state only —
    nothing here touches a final saturation, and no clipping is applied anywhere.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (design.n_geology,):
        raise ValueError(
            f"{design.design_id} expects {design.n_geology} coefficients, got {values.shape}"
        )
    if not np.isfinite(values).all() or not math.isfinite(state_coordinate):
        raise ValueError("T3 coefficients and state coordinate must be finite")

    nx, ny, _ = design.shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fields: list[NDArray[np.float64]] = []
    k_layers: list[NDArray[np.float64]] = []
    phi_layers: list[NDArray[np.float64]] = []
    for layer in range(N_LAYERS):
        layer_coefficients = values[layer * N_MODES : (layer + 1) * N_MODES]
        field = sum(
            coefficient
            * np.cos(np.pi * mode_x * xx)
            * np.cos(np.pi * mode_y * yy)
            / (1 + mode_x + mode_y)
            for coefficient, (mode_x, mode_y) in zip(layer_coefficients, MODES, strict=True)
        )
        fields.append(field.ravel(order="F"))
        k_layers.append(
            np.exp(math.log(design.layer_base_permeability_md[layer]) + field).ravel(order="F")
        )
        phi_layers.append((0.12 + 0.18 / (1.0 + np.exp(-0.7 * field))).ravel(order="F"))

    k_md = np.concatenate(k_layers)
    phi = np.concatenate(phi_layers)
    if not np.isfinite(k_md).all() or not np.isfinite(phi).all():
        raise ValueError("T3 renderer produced non-finite rock")
    if float(k_md.min()) < MIN_PERMEABILITY_MD or float(k_md.max()) > MAX_PERMEABILITY_MD:
        raise ValueError(
            f"T3 permeability spans [{float(k_md.min()):g}, {float(k_md.max()):g}] mD "
            "outside the declared renderer range"
        )
    if not ((phi > 0.0) & (phi < 1.0)).all():
        raise ValueError("T3 porosity lies outside (0,1)")

    sw0 = INITIAL_SW + T3_INITIAL_SW_SHIFT * float(ndtr(state_coordinate))
    if not INITIAL_SW <= sw0 <= INITIAL_SW + T3_INITIAL_SW_SHIFT:
        raise ValueError(
            f"T3 initial Sw {sw0!r} lies outside its declared support "
            f"[{INITIAL_SW}, {INITIAL_SW + T3_INITIAL_SW_SHIFT}]"
        )
    sw = np.full(design.n_cells, sw0, dtype=np.float64)

    centers = _t3_cell_centers(design)
    dx, dy, dz = design.cell_size_m
    kx = k_md * MILLIDARCY_M2
    arrays = {
        "cell_centers_m": centers,
        "cell_volume_m3": np.full(design.n_cells, dx * dy * dz, dtype=np.float64),
        "porosity": phi,
        "permeability_m2": np.vstack([kx, kx, kx * design.kz_over_kx(family)]),
        "log_permeability_m2": np.log(kx),
        "generator_field": np.concatenate(fields),
        "pressure_pa": oil_connected_hydrostatic_pa(
            centers[:, 2], design=P1Design(), fluids=FluidSpec()
        ),
        "sw": sw,
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("T3 renderer produced non-finite arrays")
    return arrays


def _t3_cell_centers(design: T3Design) -> NDArray[np.float64]:
    nx, ny, nz = design.shape
    dx, dy, dz = design.cell_size_m
    return np.asarray(
        [
            ((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz)
            for k in range(nz)
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.float64,
    )


def t3_well_specs(design: T3Design) -> tuple[WellSpec, ...]:
    """The E01 wells: multisegment, one connection per layer, crossflow declared."""
    nx, ny, nz = design.shape
    return tuple(
        WellSpec(
            well_id=name,
            cells=tuple(i + nx * (j + ny * layer) for layer in range(nz)),
            radius_m=WELL_RADIUS_M,
            reference_depth_m=DATUM_DEPTH_M,
            model="multisegment",
            allow_crossflow=True,
        )
        for name, (i, j), _role in design.well_columns
    )


def t3_control_segments(design: T3Design) -> tuple[ControlSegment, ...]:
    """The E01 rate policy plus T3's single completion event.

    The event opens the selected producer's LOWER connection on the month-18 report
    edge; it does not rewrite the rate. The calendar is identical for both families and
    every method — the hypothesis `s` moves rock physics, never the schedule.
    """
    if design.n_months != RATE_MODULATION[-1][1]:
        raise ValueError(
            f"the T3 policy is written for {RATE_MODULATION[-1][1]} months, got {design.n_months}"
        )
    edges = design.report_edges_s
    n_connections = design.shape[2]
    both = (True,) * n_connections
    upper_only = (True,) * (n_connections - 1) + (False,)
    segments: list[ControlSegment] = []
    for well_id, _column, role in design.well_columns:
        injector = role == "injector"
        for first, last, factor in RATE_MODULATION:
            cuts = [first, last]
            if well_id == T3_EVENT_WELL and first < T3_LOWER_OPEN_EDGE < last:
                cuts.append(T3_LOWER_OPEN_EDGE)
            bounds = sorted(set(cuts))
            for start, end in zip(bounds, bounds[1:], strict=False):
                upper = well_id == T3_EVENT_WELL and start < T3_LOWER_OPEN_EDGE
                segments.append(
                    ControlSegment(
                        start_s=edges[start],
                        end_s=edges[end],
                        well_id=well_id,
                        role=role,
                        target="water_rate" if injector else "liquid_rate",
                        value=BASE_RATE_M3_SC_DAY * factor,
                        bhp_limit_pa=(
                            INJECTOR_MAX_BHP_PA if injector else PRODUCER_MIN_BHP_PA
                        ),
                        connection_open=upper_only if upper else both,
                    )
                )
    return tuple(segments)


def t3_supports(design: T3Design) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Eight per-well-layer single-cell log-k supports (the T4 layout, shared by s)."""
    nx, ny, nz = design.shape
    supports: list[tuple[str, tuple[int, ...]]] = []
    for well_id, (i, j), _role in design.well_columns:
        for layer in range(nz):
            cell = i + nx * (j + ny * layer)
            supports.append((f"{well_id}-L{layer}-log_permeability_m2", (cell,)))
    return tuple(supports)


# --------------------------------------------------------------------------------------
# the T3 conditional context and truth world
# --------------------------------------------------------------------------------------


def t3_prior_context(
    design: T3Design,
    truth_arrays: dict[str, NDArray[np.float64]],
    rng: np.random.Generator,
) -> PriorContext:
    """Condition the T3 Gaussian on the design's eight log-k supports.

    The supports read `log_permeability_m2`, which is `kx` — the family hypothesis moves
    only `kz/kx` — so the conditioning G, the operator and the basis are IDENTICAL for
    both families. That is what lets one shared encoder/NSF serve the layout while the
    renderer resolves `s` into different physics.
    """
    from datetime import UTC, datetime

    from so_recon.geology.conditional import (
        condition_gaussian,
        conditional_square_root,
        information_matrix,
        whitening_rotation,
    )
    from so_recon.inference.contracts import DensitySchema, PriorContext
    from so_recon.registry.hashing import sha256_json

    supports = t3_supports(design)

    def _support_mean(values: NDArray[np.float64], cells: tuple[int, ...]) -> float:
        return float(np.mean(values[list(cells)]))

    def _render(coefficients: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        # Any family renders the same log-kx; s=0 keeps the operator strictly about rock.
        return render_t3_coefficients(coefficients, design, family=0, state_coordinate=0.0)

    zero = _render(np.zeros(design.n_geology))
    intercept = np.asarray(
        [_support_mean(zero["log_permeability_m2"], cells) for _name, cells in supports],
        dtype=np.float64,
    )
    operator = np.empty((len(supports), design.n_geology), dtype=np.float64)
    for column in range(design.n_geology):
        unit = np.zeros(design.n_geology, dtype=np.float64)
        unit[column] = 1.0
        rendered = _render(unit)
        operator[:, column] = [
            _support_mean(rendered["log_permeability_m2"], cells) - base
            for (_name, cells), base in zip(supports, intercept, strict=True)
        ]
    latent = np.asarray(
        [
            _support_mean(truth_arrays["log_permeability_m2"], cells)
            for _name, cells in supports
        ],
        dtype=np.float64,
    )
    sigma = 0.2
    measured = latent + sigma * rng.standard_normal(len(supports))
    variance = np.full(len(supports), sigma**2, dtype=np.float64)
    mean, covariance = condition_gaussian(operator, measured - intercept, variance)
    rotation, eigenvalues = whitening_rotation(information_matrix(operator, variance))
    root = conditional_square_root(rotation, eigenvalues)
    if not np.allclose(root @ root.T, covariance, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{design.design_id}: conditional root does not reproduce covariance")

    g_rows = [
        {
            "observation_id": name,
            "support_cell_ids": list(cells),
            "value": float(value),
            "sigma": sigma,
        }
        for (name, cells), value in zip(supports, measured, strict=True)
    ]
    g_hash = sha256_json(
        {"quantity": "log_permeability_m2", "unit": "ln(m2)", "observations": g_rows}
    )
    information_hash = sha256_json(
        {
            "role": "condition_prior",
            "g_hash": g_hash,
            "renderer": T3_RENDERER_VERSION,
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
        families=T3_FAMILIES,
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
            LOOP_DESIGN_KEY: design.model_dump(mode="json"),
            "log_k_sigma": sigma,
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


def make_t3_world(
    design_id: str,
    seed: int,
    paths: Any,
    ctx: RunContext,
) -> tuple[PriorContext, Any, Any]:
    """Publish a T3 truth world: draw s, u, coefficients and nuisance; condition on G.

    The truth-latent stream draws the family FIRST (uniform over the declared
    hypotheses), then the geology, then the state residual and the nuisance block — a
    fixed order, so a recorded seed always rebuilds the same world. The published truth
    artifact is evaluator-only; the returned observation bundle is pending, exactly like
    the E02 custom worlds.
    """
    from datetime import UTC, datetime

    from so_recon.inference.contracts import (
        ObservationBundle,
        ThetaRecord,
    )
    from so_recon.observation.bins import rounding_grid
    from so_recon.registry.artifact import write_json_artifact
    from so_recon.registry.hashing import sha256_json

    if design_id != T3_DESIGN_ID:
        raise ValueError(f"this builder renders {T3_DESIGN_ID!r} only, got {design_id!r}")
    design = t3_design()
    truth_stream, static_stream, _history_stream, _schedule_stream = np.random.SeedSequence(
        seed
    ).spawn(4)
    truth_rng = np.random.default_rng(truth_stream)
    family = int(truth_rng.integers(0, len(T3_FAMILIES)))
    coefficients = truth_rng.standard_normal(design.n_geology)
    state_coordinate = float(truth_rng.standard_normal())
    nuisance = truth_rng.standard_normal(3)
    truth_arrays = render_t3_coefficients(
        coefficients, design, family=family, state_coordinate=state_coordinate
    )
    context = t3_prior_context(design, truth_arrays, np.random.default_rng(static_stream))

    physical_whitened = np.linalg.solve(context.chol, coefficients - context.mean)
    whitened = context.rotation.T @ physical_whitened
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=family,
        v=tuple(float(value) for value in (*whitened[:8], *nuisance)),
        z_perp=tuple(float(value) for value in (*whitened[8:], state_coordinate)),
        basis_hash=context.density_schema.basis_hash,
    )
    context.density_schema.validate_theta(theta)

    from so_recon.geology.renderer import build_inverse_case, render_theta

    rendered = render_theta(theta, context)
    reproduces = all(
        np.allclose(rendered.arrays[name], values, rtol=0.0, atol=1.0e-12)
        for name, values in truth_arrays.items()
    )
    if not reproduces:
        raise ValueError(
            f"{design_id}: the conditional theta does not reproduce its truth arrays"
        )
    case = build_inverse_case(rendered, context, paths, ctx)
    truth_ref = write_json_artifact(
        paths.artifacts / "loop_worlds" / f"{design_id}-s{seed}" / "truth.json",
        {
            "schema_version": T3_TRUTH_SCHEMA,
            "warning": "synthetic truth; evaluator-only; never an inference payload",
            "design_id": design_id,
            "seed": seed,
            "family": family,
            "family_name": T3_FAMILY_NAMES[family],
            "model_hash": case.model_hash,
            "physical_design": design.payload(),
            "theta": theta.model_dump(mode="json"),
            "case": case.model_dump(mode="json"),
            "renderer_checks": {
                "reproduces_truth_arrays": reproduces,
                "initial_sw0": float(truth_arrays["sw"][0]),
            },
        },
        paths,
        schema_version=T3_TRUTH_SCHEMA,
        producer_run_id=ctx.run_id,
        now=datetime.now(UTC),
    )
    ctx.add_output(f"loop_world.{design_id}.truth", truth_ref)
    grid = rounding_grid(0.01)
    pending = ObservationBundle(
        history=(),
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(value) for value in grid.edges)},
        cutoff_s=design.report_edges_s[-1],
        information_hash=context.information_hash,
        observation_hash=sha256_json(
            {
                "status": "DYNAMIC_HISTORY_PENDING_TRUTH_FORWARD",
                "design_id": design_id,
                "information_hash": context.information_hash,
            }
        ),
    )
    return context, pending, truth_ref


# --------------------------------------------------------------------------------------
# T5: the coarse/fine physical mapping
# --------------------------------------------------------------------------------------

#: The first mandatory T5 fine grid (plan §3.3): 48x48x2 — an odd 3x refinement keeps
#: every coarse well at a fine CELL CENTRE in the existing cell-centred setting.
T5_FINE_SHAPE: tuple[int, int, int] = (48, 48, 2)
T5_REFINEMENT = 3


def t5_fine_design() -> P1Design:
    """The T1 world's kernel on the fine grid: same domain, calendar, wells, fluids."""
    design = P1Design(shape=T5_FINE_SHAPE, extent_m=(100.0, 100.0, 20.0))
    if design.n_months != 36:
        raise AssertionError("the T5 fine design keeps the 36-month T1 calendar")
    return design


def t5_fine_well_column(coarse: tuple[int, int]) -> tuple[int, int]:
    """The fine column holding the coarse well's physical position.

    A coarse cell of width 3 fine cells centres the well at `(3i + 1.5) * dfx`, which is
    the centre of fine cell `3i + 1` — one connection per layer, never a row of three.
    """
    i, j = coarse
    return (T5_REFINEMENT * i + 1, T5_REFINEMENT * j + 1)


def t5_well_specs(design: P1Design) -> tuple[WellSpec, ...]:
    """T1's wells placed on the fine grid: one connection per layer, same radius/datum.

    The well index (WI) is recomputed by the solver from the fine cell geometry and the
    unchanged radius; nothing here multiplies a coarse index.
    """
    from so_recon.synthetic.p1 import INJECTOR_COLUMNS, PRODUCER_COLUMNS

    nx, ny, nz = design.shape
    specs: list[WellSpec] = []
    for well_id, coarse_column in (*INJECTOR_COLUMNS, *PRODUCER_COLUMNS):
        i, j = t5_fine_well_column(coarse_column)
        specs.append(
            WellSpec(
                well_id=well_id,
                cells=tuple(i + nx * (j + ny * layer) for layer in range(nz)),
                radius_m=WELL_RADIUS_M,
                reference_depth_m=DATUM_DEPTH_M,
                model="multisegment",
                allow_crossflow=True,
            )
        )
    return tuple(specs)


def t5_quadrant_masks(
    shape: tuple[int, int, int], extent_m: tuple[float, float, float]
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """The eight primary quadrants by layers, declared in PHYSICAL coordinates.

    A quadrant is `x < extent_x / 2` (west/east) crossed with `y < extent_y / 2`
    (south/north) within each layer, so the same definition covers the 16x16x2
    inference grid and the 48x48x2 fine truth without an index translation that could
    drift. The masks are disjoint by construction: every cell belongs to exactly one
    quadrant, which is what the primary PV-weighted score of plan §10.2 requires.
    """
    nx, ny, nz = shape
    fx, fy, _fz = extent_m
    dx, dy = fx / nx, fy / ny
    masks = np.zeros((4 * nz, nx * ny * nz), dtype=np.float64)
    names: list[str] = []
    zone = 0
    for layer in range(nz):
        for y_name, y_low, y_high in (("south", 0.0, fy / 2.0), ("north", fy / 2.0, fy)):
            for x_name, x_low, x_high in (("west", 0.0, fx / 2.0), ("east", fx / 2.0, fx)):
                name = f"{x_name}-{y_name}-layer-{layer}"
                for j in range(ny):
                    y_center = (j + 0.5) * dy
                    if not y_low <= y_center < y_high:
                        continue
                    for i in range(nx):
                        x_center = (i + 0.5) * dx
                        if not x_low <= x_center < x_high:
                            continue
                        cell = i + nx * (j + ny * layer)
                        masks[zone, cell] = 1.0
                names.append(name)
                zone += 1
    if not bool(np.all(masks.sum(axis=0) == 1.0)):
        raise AssertionError("the quadrant masks must cover every cell exactly once")
    return tuple(names), masks


def t5_support_zone_names() -> tuple[str, ...]:
    names, _ = t5_quadrant_masks((16, 16, 2), (100.0, 100.0, 20.0))
    return names


__all__ = [
    "LOOP_DESIGN_KEY",
    "T3_DESIGN_ID",
    "T3_EVENT_WELL",
    "T3_FAMILIES",
    "T3_FAMILY_KZ_OVER_KX",
    "T3_FAMILY_NAMES",
    "T3_INITIAL_SW_SHIFT",
    "T3_LOWER_OPEN_EDGE",
    "T3_RENDERER_VERSION",
    "T3_SCHEMA_ID",
    "T3_TRANSFORM_VERSION",
    "T3_TRUTH_SCHEMA",
    "T3Design",
    "T5_FINE_SHAPE",
    "T5_REFINEMENT",
    "make_t3_world",
    "render_t3_coefficients",
    "t3_control_segments",
    "t3_design",
    "t3_prior_context",
    "t3_supports",
    "t3_well_specs",
    "t5_fine_design",
    "t5_fine_well_column",
    "t5_quadrant_masks",
    "t5_support_zone_names",
    "t5_well_specs",
]
