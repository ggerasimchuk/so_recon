"""E01.11 — the deterministic two-layer P1 generator: geology, initial state, policy, G.

This module RENDERS a world. It reads nothing from disk and writes nothing to it; the
persistence, the manifest and the truth/context boundary are `world_io`'s job.

**The geology is a known low-dimensional joint forward map.** Twelve standard-normal
coefficients — six per layer — and the versioned cosine basis of `layer_fields` determine
BOTH the permeability and the porosity of each layer. They are not two independent draws:
`log k` and `phi` are strictly increasing functions of the same latent field, which is what
makes this a generator with a stated dimension rather than a pair of unrelated textures.

Twelve modes is the DEFINED DIMENSIONALITY OF THIS GENERATOR. It is not the truncation of a
pre-existing high-dimensional prior, and E01 claims no conditional latent density
`p(theta | G)`: `theta.json` here records the parameters of a known forward map, and
reconciling any density against sparse G before an inverse use is E02's obligation. The
artifact says so itself, in `theta['dimensionality_claim']`.

**The policy is fixed in TIME and never in the state.** Every rate, every limit and the one
completion event are functions of the month index alone. Nothing in `control_segments`
reads a saturation, a pressure or a seed — a control that looked at the answer would
prescribe the very thing the inverse problem is supposed to recover.

**G is sparse, noisy and never clipped.** Eight well-layer supports carry porosity and
`log k`, and the same supports carry the initial oil saturation at the start date. Each
measurement is a SUPPORT AVERAGE of the latent field over the cells the support declares —
one cell each here, but the average is what the code computes, not an incidental single
value — plus noise from a stream of its own. A measurement that lands outside the
physically admissible range is STORED with a quality flag; the latent truth stays physical
and the measurement is never moved onto the boundary.

Three independent streams are spawned from the world's seed (`SeedSequence.spawn`):
`geology`, `static_observation` and `control`. The control stream is RESERVED and unused —
11.5's policy is deterministic — and it is spawned anyway so that a later stochastic policy
cannot shift the two streams that are used.

Units are the project's own (plan §3.1): pressure Pa, permeability m², time s, human control
rates m³_sc/day, `cell_id = i + nx*(j + ny*k)` zero-based, z depth positive down.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Literal, NamedTuple

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pydantic import model_validator

from so_recon.config.schema import StrictModel
from so_recon.registry.hashing import sha256_json
from so_recon.simulator.contracts import (
    MILLIDARCY_M2,
    STANDARD_GRAVITY_M_S2,
    BoundarySpec,
    ControlSegment,
    FluidSpec,
    OutputRequest,
    WellSpec,
)
from so_recon.simulator.schedule import month_edges, month_edges_s

#: The generator itself, versioned. A change to the basis, the stream layout or the policy
#: is a new generator version, because it makes a different world out of the same seed.
GENERATOR_VERSION = "p1-generator-1"

#: The design this generator renders, and the renderer label the case carries.
DESIGN_ID: Literal["p1-two-layer-v1"] = "p1-two-layer-v1"
RENDERER_VERSION = "e01.11"

#: The six modes of the versioned cosine basis, per layer, and the two layers: twelve
#: coefficients in total. See the module docstring on what that number is and is not.
MODES: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2))
N_LAYERS = 2
N_MODES = len(MODES)

#: The three streams spawned from a world's seed, in this order.
STREAM_NAMES: tuple[str, ...] = ("geology", "static_observation", "control")

#: The base permeability of the upper and the lower layer, mD, and the family multipliers
#: that `layer_fields` applies to the latent field. These MIRROR the verbatim kernel below
#: and `test_the_kernel_constants_are_the_ones_the_design_declares` pins them to it.
LAYER_BASE_PERMEABILITY_MD: tuple[float, float] = (120.0, 40.0)
FAMILY_CONTRAST: dict[str, float] = {"base": 1.0, "low_vertical": 1.0, "high_contrast": 1.8}
FAMILY_KZ_OVER_KX: dict[str, float] = {"base": 0.05, "low_vertical": 0.001, "high_contrast": 0.05}

#: The initial state: connate water everywhere and an OIL-connected hydrostatic column from
#: a datum of 1.5e7 Pa at z = 0, which is the top face of the box. z is absolute depth.
DATUM_PRESSURE_PA = 1.5e7
DATUM_DEPTH_M = 0.0
INITIAL_SW = 0.2

#: The four wells, as zero-based `(i, j)` columns. Both layers are perforated on every one,
#: which makes them mixed-layer wells and therefore `multisegment` under plan §3.2.
INJECTOR_COLUMNS: tuple[tuple[str, tuple[int, int]], ...] = (("I1", (2, 2)), ("I2", (2, 13)))
PRODUCER_COLUMNS: tuple[tuple[str, tuple[int, int]], ...] = (("P1", (13, 2)), ("P2", (13, 13)))
INJECTOR_IDS: tuple[str, ...] = tuple(name for name, _ in INJECTOR_COLUMNS)
PRODUCER_IDS: tuple[str, ...] = tuple(name for name, _ in PRODUCER_COLUMNS)
WELL_RADIUS_M = 0.1

#: SPEC §9.1: a producer is given a TOTAL standard liquid rate and an injector a standard
#: water rate. Both start at 20 m³_sc/day and are modulated by the calendar alone.
BASE_RATE_M3_SC_DAY = 20.0
RATE_MODULATION: tuple[tuple[int, int, float], ...] = ((0, 12, 1.0), (12, 24, 0.75), (24, 36, 1.25))
PRODUCER_MIN_BHP_PA = 5.0e6
INJECTOR_MAX_BHP_PA = 3.0e7

#: The completion event: P2's LOWER connection closes at the start of month 19 and reopens
#: at the start of month 25. Both are month EDGES, zero-based, so they are report edges too.
COMPLETION_EVENT_WELL = "P2"
COMPLETION_SHUT_EDGE = 18
COMPLETION_REOPEN_EDGE = 24

#: What the world publishes as states, and how the run is chunked. A month per chunk keeps
#: the native checkpoint at monthly granularity, which is the resolution the P1 loop reports.
STATE_EDGES: tuple[int, ...] = (0, 12, 24, 36)
CHUNK_MONTHS = 1

#: How long the closed equilibrium preflight runs. One month is long enough for a column
#: that was not in balance to move visibly and short enough to cost almost nothing.
PREFLIGHT_MONTHS = 1

#: The static observation view. These are properties of the VIEW, not of the truth: a noise
#: variant would be another derived view over the same parent, which is why they take no
#: part in `parent_world_id`.
SIGMA_POROSITY = 0.01
SIGMA_LOG_PERMEABILITY = 0.2
SIGMA_OIL_SATURATION = 0.02

#: The dynamic channels a later inverse context is allowed to condition on. Pressure is NOT
#: among them: the true pressure is kept for diagnostics only.
DYNAMIC_CHANNELS: tuple[str, ...] = ("fw", "controls")
PRESSURE_AVAILABLE = False

#: The first — and only — parent set of this plan: three `base` worlds, one `low_vertical`
#: and one `high_contrast`. Nothing expands it automatically (SPEC §18.4).
P1_PARENTS: tuple[tuple[int, str], ...] = (
    (41, "base"),
    (42, "base"),
    (43, "base"),
    (44, "low_vertical"),
    (45, "high_contrast"),
)

#: Largest permeability this generator will publish, mD. Not a clip: a family whose tail
#: reaches past it is REFUSED with the value it reached, because a silently truncated tail
#: is a different generator wearing the same version number.
MAX_PERMEABILITY_MD = 1.0e6
MIN_PERMEABILITY_MD = 1.0e-3


# --------------------------------------------------------------------------------------
# 11.3 the geology kernel
# --------------------------------------------------------------------------------------


def layer_fields(
    coefficients: Sequence[float] | NDArray[np.float64],
    nx: int,
    ny: int,
    layer: int,
    family: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The plan's own kernel, verbatim: one layer's permeability (mD) and porosity.

    Returned arrays are `(nx, ny)` with `xx[i, j]`; flattening in order `F` gives the
    within-layer index `i + nx*j`, which is `cell_id = i + nx*(j + ny*k)` for that layer.
    """
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    modes = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    field = sum(
        a * np.cos(np.pi * i * xx) * np.cos(np.pi * j * yy) / (1 + i + j)
        for a, (i, j) in zip(coefficients, modes, strict=True)
    )
    contrast = 1.0 if family != "high_contrast" else 1.8
    log_k_md = np.log(120.0 if layer == 0 else 40.0) + contrast * field
    k_md = np.exp(log_k_md)
    phi = 0.12 + 0.18 / (1 + np.exp(-0.7 * field))
    return k_md, phi


# --------------------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------------------

Family = Literal["base", "low_vertical", "high_contrast"]


class P1Design(StrictModel):
    """What the P1 world IS, before any seed is drawn.

    The family selects two things and only two: the multiplier `layer_fields` applies to the
    latent field, and the vertical-to-horizontal permeability ratio. Everything else — the
    box, the calendar, the wells and the policy — is shared by all five parents, which is
    what makes a comparison between them a comparison of geology.
    """

    design_id: Literal["p1-two-layer-v1"] = DESIGN_ID
    family: Family = "base"
    shape: tuple[int, int, int] = (16, 16, 2)
    extent_m: tuple[float, float, float] = (400.0, 400.0, 20.0)
    n_months: int = 36
    start_date: str = "2000-01-01"

    @model_validator(mode="after")
    def _is_the_two_layer_box(self) -> P1Design:
        if self.shape[2] != N_LAYERS:
            raise ValueError(f"the P1 design is a two-layer box, got shape {self.shape}")
        if any(n < 1 for n in self.shape) or any(e <= 0.0 for e in self.extent_m):
            raise ValueError(
                f"the box must be positive: shape {self.shape}, extent {self.extent_m}"
            )
        if self.n_months < 1:
            raise ValueError(f"n_months must be positive, got {self.n_months}")
        if date.fromisoformat(self.start_date).day != 1:
            raise ValueError(f"start_date must be the first of a month, got {self.start_date}")
        return self

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def cells_per_layer(self) -> int:
        nx, ny, _ = self.shape
        return nx * ny

    @property
    def contrast(self) -> float:
        """The multiplier `layer_fields` applies to the latent field for this family."""
        return FAMILY_CONTRAST[self.family]

    @property
    def kz_over_kx(self) -> float:
        return FAMILY_KZ_OVER_KX[self.family]

    @property
    def layer_base_permeability_md(self) -> tuple[float, float]:
        return LAYER_BASE_PERMEABILITY_MD

    @property
    def cell_size_m(self) -> tuple[float, float, float]:
        return tuple(e / n for e, n in zip(self.extent_m, self.shape, strict=True))  # type: ignore[return-value]

    @property
    def start(self) -> date:
        return date.fromisoformat(self.start_date)

    @property
    def cutoff(self) -> str:
        return month_edges(self.start, self.n_months)[-1].isoformat()

    @property
    def report_edges_s(self) -> tuple[float, ...]:
        return month_edges_s(self.start, self.n_months)

    def payload(self) -> dict[str, Any]:
        """The design as data, derived constants included, for theta and the parent id."""
        return {
            **self.model_dump(mode="json"),
            "contrast": self.contrast,
            "kz_over_kx": self.kz_over_kx,
            "layer_base_permeability_md": list(self.layer_base_permeability_md),
            "cutoff": self.cutoff,
        }


# --------------------------------------------------------------------------------------
# the random streams
# --------------------------------------------------------------------------------------


class P1Streams(NamedTuple):
    """The three independent streams of a world, spawned from its single seed."""

    geology: np.random.SeedSequence
    static_observation: np.random.SeedSequence
    control: np.random.SeedSequence


def streams(seed: int) -> P1Streams:
    """Spawn the world's three streams. `control` is reserved; see the module docstring."""
    if seed < 0:
        raise ValueError(f"a world seed is non-negative, got {seed}")
    geology, static, control = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return P1Streams(geology=geology, static_observation=static, control=control)


# --------------------------------------------------------------------------------------
# the validated geology
# --------------------------------------------------------------------------------------


def layer_geology(
    coefficients: Sequence[float] | NDArray[np.float64], design: P1Design, layer: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`layer_fields` with its output CHECKED rather than quietly repaired.

    An overflowed exponential, a NaN coefficient or a tail past `MAX_PERMEABILITY_MD` is a
    refusal that names the value it reached. Nothing here clips: a clipped tail would be a
    different generator publishing the same version number and the same seed.
    """
    nx, ny, _ = design.shape
    with np.errstate(over="ignore", invalid="ignore"):
        k_md, phi = layer_fields(coefficients, nx, ny, layer, design.family)
    for name, values in (("permeability_md", k_md), ("porosity", phi)):
        if not np.isfinite(values).all():
            raise ValueError(
                f"layer {layer}: {name} holds nonfinite values; the cosine field overflowed "
                f"the exponential rather than producing a rock. Coefficients "
                f"{np.asarray(coefficients).tolist()} are refused, not clipped"
            )
    worst = float(np.max(k_md))
    smallest = float(np.min(k_md))
    if worst > MAX_PERMEABILITY_MD or smallest < MIN_PERMEABILITY_MD:
        raise ValueError(
            f"layer {layer}: permeability spans [{smallest:g}, {worst:g}] mD, outside the "
            f"[{MIN_PERMEABILITY_MD:g}, {MAX_PERMEABILITY_MD:g}] mD this generator publishes; "
            "an extreme tail is refused rather than truncated in silence"
        )
    if not ((phi > 0.0) & (phi < 1.0)).all():
        raise ValueError(f"layer {layer}: porosity outside (0, 1), got [{phi.min()}, {phi.max()}]")
    return k_md, phi


# --------------------------------------------------------------------------------------
# 11.4 geometry and the known initial state
# --------------------------------------------------------------------------------------


def cell_centers_m(design: P1Design) -> NDArray[np.float64]:
    """Cell centres of the uniform box, `cell_id = i + nx*(j + ny*k)`, z depth positive down.

    The box's top face is the pressure datum at z = 0, so the centres are absolute depths
    and the Julia adapter origins its mesh at exactly this datum.
    """
    nx, ny, nz = design.shape
    dx, dy, dz = design.cell_size_m
    out = np.zeros((design.n_cells, 3), dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                out[i + nx * (j + ny * k)] = (
                    dx * i + 0.5 * dx,
                    dy * j + 0.5 * dy,
                    DATUM_DEPTH_M + dz * k + 0.5 * dz,
                )
    return out


def oil_connected_hydrostatic_pa(
    depth_m: NDArray[np.float64], *, design: P1Design, fluids: FluidSpec | None = None
) -> NDArray[np.float64]:
    """The exact oil-connected hydrostatic column of the educational PVT (plan §3.1).

    Water is at connate saturation everywhere, so `krw = 0` and the CONTINUOUS phase is oil:
    `dp/dz = rho_o(p) * g` with `rho_o(p) = rho_o_sc * exp(c_o * (p - p_sc))`. That ordinary
    differential equation has a closed form,

        p(z) = p_sc - ln( exp(-c_o * (p0 - p_sc)) - c_o * rho_o_sc * g * (z - z0) ) / c_o,

    which is what is evaluated here rather than a first-order approximation of it. A column
    deep enough for the logarithm's argument to reach zero has no hydrostatic solution at
    all and is refused.
    """
    fluid = fluids or FluidSpec()
    rho_o = fluid.density_sc_kg_m3[1]
    c_o = fluid.compressibility_pa_inv[1]
    z = np.asarray(depth_m, dtype=np.float64) - DATUM_DEPTH_M
    if c_o == 0.0:
        return DATUM_PRESSURE_PA + rho_o * STANDARD_GRAVITY_M_S2 * z
    head = c_o * rho_o * STANDARD_GRAVITY_M_S2 * z
    inner = np.exp(-c_o * (DATUM_PRESSURE_PA - fluid.p_sc_pa)) - head
    if not (inner > 0.0).all():
        raise ValueError(
            f"the oil-connected column has no hydrostatic solution below "
            f"{float(np.max(z)):g} m from the datum: the compressible integral diverges"
        )
    pressure: NDArray[np.float64] = fluid.p_sc_pa - np.log(inner) / c_o
    return pressure


def well_specs(design: P1Design) -> tuple[WellSpec, ...]:
    """The four wells, each perforating BOTH layers of its column.

    A mixed-layer completion is a `multisegment` well under plan §3.2 — a vertical chain of
    nodes is exactly what a multisegment wellbore describes — and a wellbore with more than
    one connection always couples them, so `allow_crossflow` is declared true rather than
    asked for and silently not delivered.
    """
    nx, ny, nz = design.shape
    return tuple(
        WellSpec(
            well_id=well_id,
            cells=tuple(i + nx * (j + ny * k) for k in range(nz)),
            radius_m=WELL_RADIUS_M,
            # The case's own datum, which is the top face of the box and the depth the
            # adapter origins the mesh at.
            reference_depth_m=DATUM_DEPTH_M,
            model="multisegment",
            allow_crossflow=True,
        )
        for well_id, (i, j) in (*INJECTOR_COLUMNS, *PRODUCER_COLUMNS)
    )


# --------------------------------------------------------------------------------------
# 11.5 the controls policy
# --------------------------------------------------------------------------------------


def control_segments(design: P1Design) -> tuple[ControlSegment, ...]:
    """The whole policy, as a function of the CALENDAR and of nothing else.

    Three rate periods — 1.0, then 0.75 on months 13-24 and 1.25 on months 25-36 — applied
    to injectors and producers alike, and one completion event on `P2`. The event closes a
    perforation; it does not rewrite the rate the policy had already fixed for those months,
    because a well that loses a completion still asks for what it was asked to deliver and
    the solver is what decides whether it can.
    """
    edges = design.report_edges_s
    if design.n_months != RATE_MODULATION[-1][1]:
        raise ValueError(
            f"the P1 policy is written for {RATE_MODULATION[-1][1]} months, got {design.n_months}"
        )
    n_connections = design.shape[2]
    both = (True,) * n_connections
    lower_shut = (True,) * (n_connections - 1) + (False,)
    segments: list[ControlSegment] = []
    for well_id in (*INJECTOR_IDS, *PRODUCER_IDS):
        injector = well_id in INJECTOR_IDS
        for first, last, factor in RATE_MODULATION:
            # The completion event splits whichever rate period contains it; the split is
            # in the MASK only, so both halves carry the same rate.
            cuts = [first, last]
            if well_id == COMPLETION_EVENT_WELL:
                cuts += [
                    e for e in (COMPLETION_SHUT_EDGE, COMPLETION_REOPEN_EDGE) if first < e < last
                ]
            bounds = sorted(set(cuts))
            for start, end in zip(bounds, bounds[1:], strict=False):
                shut = (
                    well_id == COMPLETION_EVENT_WELL
                    and COMPLETION_SHUT_EDGE <= start < COMPLETION_REOPEN_EDGE
                )
                segments.append(
                    ControlSegment(
                        start_s=edges[start],
                        end_s=edges[end],
                        well_id=well_id,
                        role="injector" if injector else "producer",
                        target="water_rate" if injector else "liquid_rate",
                        value=BASE_RATE_M3_SC_DAY * factor,
                        bhp_limit_pa=INJECTOR_MAX_BHP_PA if injector else PRODUCER_MIN_BHP_PA,
                        connection_open=lower_shut if shut else both,
                    )
                )
    return tuple(segments)


def output_request(design: P1Design) -> OutputRequest:
    """Start, month 12, month 24 and month 36, with the native final restart kept."""
    edges = design.report_edges_s
    return OutputRequest(
        state_times_s=tuple(edges[index] for index in STATE_EDGES),
        keep_native_restart=True,
        chunk_months=CHUNK_MONTHS,
    )


def preflight_output_request(design: P1Design) -> OutputRequest:
    """Both ends of the closed preflight: the state it started at and the state it reached."""
    edges = month_edges_s(design.start, PREFLIGHT_MONTHS)
    return OutputRequest(
        state_times_s=(edges[0], edges[-1]), keep_native_restart=False, chunk_months=1
    )


# --------------------------------------------------------------------------------------
# 11.6 the sparse static observations
# --------------------------------------------------------------------------------------


class ObservationSupport(NamedTuple):
    """One well-layer support: the cells a measurement of it is averaged over."""

    well_id: str
    layer_index: int
    cell_ids: tuple[int, ...]


#: Each observed quantity, the unit it is stated in and the range outside which a
#: measurement is flagged. `log_permeability_m2` has no admissible range: a log permeability
#: is unbounded, so every value of it is a value the quantity can take.
_QUANTITIES: tuple[tuple[str, str, tuple[float, float] | None, float], ...] = (
    ("porosity", "1", (0.0, 1.0), SIGMA_POROSITY),
    ("log_permeability_m2", "ln(m2)", None, SIGMA_LOG_PERMEABILITY),
    # Corey Swc = Sorw = 0.2 (plan §3.1) makes [Sorw, 1 - Swc] the oil saturations this
    # fluid model admits at all; a measurement outside it is kept and flagged.
    ("oil_saturation", "1", (0.2, 0.8), SIGMA_OIL_SATURATION),
)

OBSERVATION_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("well_id", pa.string()),
        ("layer_index", pa.int64()),
        ("support_cell_ids", pa.list_(pa.int64())),
        ("quantity", pa.string()),
        ("date", pa.string()),
        ("value", pa.float64()),
        ("sigma", pa.float64()),
        ("unit", pa.string()),
        ("quality_flag", pa.string()),
        ("valid_range_low", pa.float64()),
        ("valid_range_high", pa.float64()),
    ]
)

SUPPORT_TRUTH_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("well_id", pa.string()),
        ("layer_index", pa.int64()),
        ("support_cell_ids", pa.list_(pa.int64())),
        ("quantity", pa.string()),
        ("date", pa.string()),
        ("latent_value", pa.float64()),
        ("unit", pa.string()),
    ]
)


def observation_supports(design: P1Design) -> tuple[ObservationSupport, ...]:
    """The eight well-layer supports: four wells, both layers, one cell each here."""
    nx, ny, nz = design.shape
    return tuple(
        ObservationSupport(well_id=well_id, layer_index=k, cell_ids=(i + nx * (j + ny * k),))
        for well_id, (i, j) in (*INJECTOR_COLUMNS, *PRODUCER_COLUMNS)
        for k in range(nz)
    )


def support_mean(values: NDArray[np.float64], cell_ids: Sequence[int]) -> float:
    """The AVERAGE of a cell field over the cells a support declares.

    One cell per support makes this arithmetically trivial in the P1 design, and it is
    written as an average anyway: the observation is a property of the SUPPORT, not of a
    well's flux, and a support that grows to several cells must not change what the number
    means. An empty support is a refusal, never a zero.
    """
    cells = list(cell_ids)
    if not cells:
        raise ValueError("an observation support names at least one cell")
    return float(np.mean(np.asarray(values, dtype=np.float64)[cells]))


def _latent(
    arrays: Mapping[str, NDArray[np.float64]], quantity: str, cells: Sequence[int]
) -> float:
    if quantity == "porosity":
        return support_mean(arrays["porosity"], cells)
    if quantity == "log_permeability_m2":
        return support_mean(arrays["log_permeability_m2"], cells)
    if quantity == "oil_saturation":
        return support_mean(1.0 - arrays["sw"], cells)
    raise ValueError(f"unknown observed quantity {quantity!r}")


def support_truth(arrays: Mapping[str, NDArray[np.float64]], design: P1Design) -> pa.Table:
    """The NOISELESS support averages. This belongs to `truth/` and never to a context."""
    rows: list[dict[str, Any]] = []
    for support in observation_supports(design):
        for quantity, unit, _, _ in _QUANTITIES:
            rows.append(
                {
                    "observation_id": f"{support.well_id}-L{support.layer_index}-{quantity}",
                    "well_id": support.well_id,
                    "layer_index": support.layer_index,
                    "support_cell_ids": list(support.cell_ids),
                    "quantity": quantity,
                    "date": design.start_date,
                    "latent_value": _latent(arrays, quantity, support.cell_ids),
                    "unit": unit,
                }
            )
    return pa.Table.from_pylist(rows, schema=SUPPORT_TRUTH_SCHEMA)


def static_observations(
    arrays: Mapping[str, NDArray[np.float64]],
    design: P1Design,
    noise_stream: np.random.SeedSequence,
) -> pa.Table:
    """The sparse noisy G a later inverse problem is allowed to see.

    The noise is drawn ONCE, for every quantity and every support, before any truth is read,
    so a change in the truth moves the measurement by exactly the change and never by a
    different realisation of the noise. That is what makes the leakage test in
    `tests/unit/test_p1_worlds.py` a test of the boundary rather than of the stream.

    The latent value is NOT a column here: this table is what the context references, and a
    truth column on it would be the leak the whole separation exists to prevent.
    """
    supports = observation_supports(design)
    draw = np.random.default_rng(noise_stream).standard_normal((len(_QUANTITIES), len(supports)))
    rows: list[dict[str, Any]] = []
    for q, (quantity, unit, admissible, sigma) in enumerate(_QUANTITIES):
        for s, support in enumerate(supports):
            value = _latent(arrays, quantity, support.cell_ids) + sigma * float(draw[q, s])
            low: float | None = None
            high: float | None = None
            inside = True
            if admissible is not None:
                low, high = admissible
                inside = low <= value <= high
            rows.append(
                {
                    "observation_id": f"{support.well_id}-L{support.layer_index}-{quantity}",
                    "well_id": support.well_id,
                    "layer_index": support.layer_index,
                    "support_cell_ids": list(support.cell_ids),
                    "quantity": quantity,
                    "date": design.start_date,
                    # Stored exactly as measured. A measurement outside the admissible range
                    # is flagged, never moved onto the boundary: clipping would turn a noise
                    # realisation into a fact nobody observed.
                    "value": value,
                    "sigma": sigma,
                    "unit": unit,
                    "quality_flag": "ok" if inside else "out_of_range",
                    "valid_range_low": low,
                    "valid_range_high": high,
                }
            )
    rows.sort(key=lambda r: (str(r["well_id"]), int(r["layer_index"]), str(r["quantity"])))
    return pa.Table.from_pylist(rows, schema=OBSERVATION_SCHEMA)


def observation_masks(design: P1Design) -> dict[str, Any]:
    """What a context is allowed to know about WHERE and WHAT was observed.

    Supports, quantities, their noise levels and the allowed dates — and the explicit
    statement that pressure is not available. No value of any of them appears here.
    """
    return {
        "static_supports": [
            {
                "well_id": support.well_id,
                "layer_index": support.layer_index,
                "cell_ids": list(support.cell_ids),
                "quantities": [name for name, _, _, _ in _QUANTITIES],
            }
            for support in observation_supports(design)
        ],
        "static_quantities": {
            name: {"unit": unit, "sigma": sigma} for name, unit, _, sigma in _QUANTITIES
        },
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "pressure_available": PRESSURE_AVAILABLE,
        "allowed_dates": {
            "start": design.start_date,
            "cutoff": design.cutoff,
            "static_observation_date": design.start_date,
        },
    }


# --------------------------------------------------------------------------------------
# the rendered world
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedWorld:
    """One world's truth and the case fields it determines, in memory and not yet on disk."""

    parent_world_id: str
    seed: int
    design: P1Design
    theta: dict[str, Any]
    arrays: dict[str, NDArray[np.float64]]
    case_fields: dict[str, Any]
    static_observations: pa.Table


def parent_world_id(seed: int, design: P1Design) -> str:
    """The parent identity: generator, design and seed, and nothing measured.

    A derived view — a prefix of the calendar, a different noise realisation — INHERITS this
    id and differs only in its `derived_view_id`, which is why no observation parameter and
    no machine stamp takes part in it.
    """
    digest = sha256_json(
        {"generator_version": GENERATOR_VERSION, "design": design.payload(), "seed": int(seed)}
    )
    return f"p1-{design.family}-s{seed:04d}-{digest[:12]}"


def _theta(seed: int, design: P1Design, coefficients: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "record_kind": (
            "parameters of a known forward generator; this is not a ThetaRecord and E01 "
            "defines no field prior, no PVT record, no likelihood and no posterior"
        ),
        "dimensionality_claim": (
            "twelve coefficients are the DEFINED dimensionality of this generator, not a "
            "truncation of a pre-existing high-dimensional prior. No conditional latent "
            "density p(theta|G) is claimed or implied here; reconciling any density against "
            "sparse G before an inverse use is E02's obligation"
        ),
        "generator_version": GENERATOR_VERSION,
        "design_id": design.design_id,
        "family": design.family,
        "design": design.payload(),
        "seed": int(seed),
        "bit_generator": "PCG64",
        "seed_sequence": "numpy.random.SeedSequence(seed).spawn(3)",
        "streams": list(STREAM_NAMES),
        "reserved_streams": ["control"],
        "basis": (
            "versioned cosine basis p1-cosine-1: "
            "sum_m a_m cos(pi*i_m*x) cos(pi*j_m*y) / (1 + i_m + j_m), x,y in cell-centre "
            "coordinates (k+0.5)/n"
        ),
        "modes": [list(mode) for mode in MODES],
        "n_layers": N_LAYERS,
        "n_modes_per_layer": N_MODES,
        "n_coefficients": N_LAYERS * N_MODES,
        "coefficients": [[float(v) for v in layer] for layer in coefficients],
        "layer_base_permeability_md": list(LAYER_BASE_PERMEABILITY_MD),
        "contrast": design.contrast,
        "kz_over_kx": design.kz_over_kx,
        "porosity_logistic": {"floor": 0.12, "span": 0.18, "slope": 0.7},
        "initial_state": {
            "kind": "oil_connected_hydrostatic",
            "datum_pressure_pa": DATUM_PRESSURE_PA,
            "datum_depth_m": DATUM_DEPTH_M,
            "sw": INITIAL_SW,
            "meaning": "synthetic_initial",
        },
        "observation_view": {
            "sigma_porosity": SIGMA_POROSITY,
            "sigma_log_permeability": SIGMA_LOG_PERMEABILITY,
            "sigma_oil_saturation": SIGMA_OIL_SATURATION,
        },
    }


def render_p1(seed: int, design: P1Design) -> RenderedWorld:
    """Render one world: its geology, its initial state, its policy and its sparse G.

    Deterministic in `(GENERATOR_VERSION, design, seed)` and in nothing else. No clock, no
    hostname and no measurement enters any array, any identity or `theta`.
    """
    geology_stream, static_stream, _control_stream = streams(seed)
    coefficients = np.random.default_rng(geology_stream).standard_normal((N_LAYERS, N_MODES))

    k_md_layers: list[NDArray[np.float64]] = []
    phi_layers: list[NDArray[np.float64]] = []
    field_layers: list[NDArray[np.float64]] = []
    for layer in range(N_LAYERS):
        k_md, phi = layer_geology(coefficients[layer], design, layer)
        k_md_layers.append(k_md.ravel(order="F"))
        phi_layers.append(phi.ravel(order="F"))
        # The latent field the two rock fields share, recovered from the kernel's own output
        # so it cannot drift from what the kernel actually used.
        base = np.log(LAYER_BASE_PERMEABILITY_MD[layer])
        field_layers.append((np.log(k_md).ravel(order="F") - base) / design.contrast)

    porosity = np.concatenate(phi_layers)
    kx = np.concatenate(k_md_layers) * MILLIDARCY_M2
    permeability = np.vstack([kx, kx, kx * design.kz_over_kx])
    centers = cell_centers_m(design)
    pressure = oil_connected_hydrostatic_pa(centers[:, 2], design=design)
    sw = np.full(design.n_cells, INITIAL_SW, dtype=np.float64)
    dx, dy, dz = design.cell_size_m

    arrays: dict[str, NDArray[np.float64]] = {
        "cell_centers_m": centers,
        "cell_volume_m3": np.full(design.n_cells, dx * dy * dz, dtype=np.float64),
        "porosity": porosity,
        "permeability_m2": permeability,
        "log_permeability_m2": np.log(kx),
        "generator_field": np.concatenate(field_layers),
        "pressure_pa": pressure,
        "sw": sw,
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: the rendered world holds nonfinite values")

    theta = _theta(seed, design, coefficients)
    return RenderedWorld(
        parent_world_id=parent_world_id(seed, design),
        seed=int(seed),
        design=design,
        theta=theta,
        arrays=arrays,
        case_fields=_case_fields(seed, design, theta),
        static_observations=static_observations(arrays, design, static_stream),
    )


def _case_fields(seed: int, design: P1Design, theta: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a `CaseBundle` needs that is not an array reference."""
    world_id = parent_world_id(seed, design)
    return {
        "case_id": f"case-{world_id}",
        "world_id": world_id,
        "start_date": design.start_date,
        "cutoff": design.cutoff,
        "report_edges_s": design.report_edges_s,
        "shape": design.shape,
        "extent_m": design.extent_m,
        "fluids": FluidSpec(),
        "wells": well_specs(design),
        "controls": control_segments(design),
        "boundary": BoundarySpec(kind="closed", cells=()),
        "dynamic_channels": DYNAMIC_CHANNELS,
        "pressure_available": PRESSURE_AVAILABLE,
        "renderer_version": RENDERER_VERSION,
        "units": {
            "pressure": "Pa",
            "permeability": "m2",
            "time": "s",
            "length": "m",
            "control_rate": "m3_sc/day",
        },
        "seeds": {"fixture": int(seed)},
        "source_hashes": {
            "generator": sha256_json(theta),
            "design": sha256_json(design.payload()),
        },
    }


def closed_preflight_world(world: RenderedWorld) -> RenderedWorld:
    """The same world over ONE month with every well shut AND every perforation masked.

    This is the equilibrium preflight of plan 11.4. It runs on the production forward path,
    against the same rock and the same initial state, and it is the evidence that the
    oil-connected hydrostatic column this generator writes is in the NATIVE two-point
    balance the solver forms — not merely in the continuous one the formula satisfies.

    Both halves of "closed" are needed and both are here. A `DisabledControl` alone shuts
    only the surface network: the wellbore stays coupled to every cell it perforates, and a
    two-layer completion under an open wellbore equalises the layers it spans (Task 6, Task
    10). `connection_open` all false applies the native `PerforationMask`, which is what
    actually isolates the well. The reservoir has a closed boundary already, so the result is
    a system with no source of any kind: whatever it does, it does on its own.
    """
    design = world.design
    edges = month_edges_s(design.start, PREFLIGHT_MONTHS)
    closed = (False,) * design.shape[2]
    wells: tuple[WellSpec, ...] = world.case_fields["wells"]
    controls = tuple(
        ControlSegment(
            start_s=edges[0],
            end_s=edges[-1],
            well_id=well.well_id,
            role="shut",
            target="disabled",
            value=0.0,
            bhp_limit_pa=None,
            connection_open=closed,
        )
        for well in wells
    )
    return replace(
        world,
        case_fields={
            **world.case_fields,
            "case_id": f"{world.case_fields['case_id']}-preflight",
            "cutoff": month_edges(design.start, PREFLIGHT_MONTHS)[-1].isoformat(),
            "report_edges_s": edges,
            "controls": controls,
        },
    )
