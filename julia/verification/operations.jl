# E01.10 — the OPERATIONAL fixtures: mixing and crossflow, completions and roles, boundaries.
#
# Task 9 established what the physics of this build is: a closed system that does not move, a
# displacement whose weak solution is a formula, a column in hydrostatic balance. Task 10 is
# the layer above it — what happens when WELLS are operated. A producer with two completions
# in two different layers, a wellbore that carries fluid between them while its surface is
# shut, a completion that is closed and one that is only shut, a well that changes role inside
# a calendar month, a sector supported from outside. None of those are new physics; all of
# them are places where an operator's intention and the model's behaviour can quietly differ,
# which is why each one is a fixture with its own measured number rather than a paragraph.
#
# THREE THINGS THIS FILE IS CAREFUL NOT TO CONFLATE.
#
# 1. CROSSFLOW IS NOT A SWITCH HERE. JutulDarcy 0.3.11 has no crossflow mechanism at all —
#    the word does not occur in either pinned package, and WELSPECS' crossflow item is parsed
#    and never read — so `model.jl:check_crossflow` refuses `allow_crossflow = false` over
#    more than one connection instead of pretending to honour it. What DOES exist natively is
#    a `MultiSegmentWell`'s wellbore: its segments carry mass between its connections whenever
#    their reservoir potentials differ, whatever the surface is doing. That is the mechanism
#    10.4 demonstrates, and it is demonstrated by MEASURING two connection fluxes of opposite
#    sign, not by toggling anything.
#
# 2. A SHUT WELL IS NOT AN ISOLATED WELL. `DisabledControl` sets the SURFACE rate to zero;
#    JutulDarcy's own docstring for it says a well is not isolated unless its perforations are
#    closed by a `PerforationMask`. So 10.4 carries both: one case with a shut surface and open
#    completions, where the connections must carry real, opposed flux, and a second with the
#    same surface condition and both completions closed, where each connection must carry
#    nothing. Declaring those one test would let a well that merely stopped producing pass as
#    a well that was isolated.
#
# 3. A FIXED-PRESSURE BOUNDARY IS NOT AN AQUIFER. `FlowBoundaryCondition` holds its pressure
#    for the whole run whatever crosses it: it is an unbounded store, and its support never
#    weakens. A FINITE store is a real cell with a real pore volume, in the grid, in the
#    inventory and in the balance, whose pressure falls as it gives water up. 10.6 runs both
#    against the same closed benchmark and reports the difference as the measurement it is.
#
# The scoring is not here. Julia measures what only Julia can see — the native connection
# flux through each perforation, the native boundary flux, the operating control a well really
# ran on — and publishes it; `so_recon.validation.physics` and
# `tests/integration/test_e01_physics.py` score it against `configs/e01_tolerances.yml`, which
# was fixed before any of these numbers existed.

using Test
using Dates
using Jutul
using JutulDarcy
using JSON

# `analytic.jl` carries Task 9's shared fixture vocabulary and the adapter handle, and its
# `PROGRAM_FILE` guard makes including it inert. `bl_case` comes with it, which is what the
# 10.8 refinement study in `refinement.jl` drives at a second grid and a halved timestep.
include(joinpath(@__DIR__, "analytic.jl"))

# ======================================================================================
# 10.4 the two-layer sector: mixing, and wellbore crossflow
# ======================================================================================

#: The sector plan 10.4 names: 4x4 cells in each of two layers, 5 m of pay each.
const TWO_LAYER_DIMS = (4, 4, 2)
const TWO_LAYER_EXTENT = (40.0, 40.0, 10.0)

#: Horizontal permeability and porosity, BY LAYER, exactly as 10.4 states them. The contrast
#: is the point: a good layer over a poor one is what makes the two connections of one well
#: deliver different fluid, and what makes their potentials fall apart when it is shut in.
const TWO_LAYER_KH_MD = (200.0, 50.0)
const TWO_LAYER_POROSITY = (0.25, 0.15)

#: A SHALE BARRIER between the two layers: vertical permeability is 1e-4 of the horizontal
#: one, so 0.02 mD over the good layer and 0.005 mD over the poor one.
#:
#: This is load-bearing and it was MEASURED into place. At an ordinary k_v = 0.1 k_h the
#: vertical pressure diffusivity is about 0.05 m²/s, so 5 m of layer equilibrates in roughly
#: eight minutes: the 4 MPa contrast of the crossflow case below collapsed through the ROCK
#: within the first substep whether the well was open or shut, and the contrast between the
#: two cases became vacuous — both fell to about -40 kPa. At 1e-4 the vertical time constant
#: is about six days, so over a one-day horizon the isolated case keeps its potentials apart
#: and the open one does not, and the difference between them is the wellbore. It is also
#: what makes the mixing case a MIXING case: with the layers separated, the two streams meet
#: in the well and nowhere else.
const TWO_LAYER_KV_RATIO = 1.0e-4

#: The initial water saturations 10.4 fixes, by layer. Both are inside the §3.1 mobile range
#: [0.2, 0.8], so both phases move in both layers; at an endpoint one of them could not, and
#: the mixing case would have nothing to mix.
const TWO_LAYER_SW = (0.25, 0.65)

const TWO_LAYER_PRESSURE_PA = 2.0e7

#: The producer's total standard LIQUID rate (SPEC 9.1: never two phase rates), and the water
#: rate each of the two layer injectors carries. Two injectors and one producer is three wells,
#: which is the P0_VERIFY ceiling.
const MIXING_LIQUID_RATE_M3_DAY = 4.0
const MIXING_INJECTION_RATE_M3_DAY = 2.0
const MIXING_EDGES_DAYS = [0.0, 10.0, 20.0, 30.0]

const PRODUCER_BHP_FLOOR_PA = 5.0e6
const INJECTOR_BHP_CEILING_PA = 4.0e7

#: The opposed layer potentials of the crossflow case. The shallow, good layer is put 4 MPa
#: ABOVE the deep one, which is the arrangement no static column produces: the wellbore is
#: then the path from one to the other, and the sign of each connection flux is decided by the
#: physics rather than by the fixture.
const CROSSFLOW_PRESSURE_PA = (2.2e7, 1.8e7)
#: Quarter-day steps over one day. The crossflow is a TRANSIENT — the wellbore equalises what
#: it can reach in minutes and then carries whatever the layers keep feeding it — and the
#: extraction evaluates each connection flux on its substep's own converged state, so a step
#: of days would report the tail of the transient and call it the crossflow.
const CROSSFLOW_EDGES_DAYS = [0.0, 0.25, 0.5, 1.0]

"""Per-cell values of a layered box, in the exchange's zero-based `i + nx*(j + ny*k)` order."""
function by_layer(dims::NTuple{3,Int}, per_layer)
    nx, ny, nz = dims
    length(per_layer) == nz ||
        error("by_layer: $(length(per_layer)) values for $(nz) layers")
    values = Vector{Float64}(undef, prod(dims))
    for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        values[i + nx * (j + ny * k) + 1] = Float64(per_layer[k + 1])
    end
    return values
end

"""The `(3, n_cells)` permeability of a layered box: `k_h` in x and y, `kv_ratio * k_h` in z."""
function layered_permeability(dims::NTuple{3,Int}, kh_md, kv_ratio::Float64)
    kh = by_layer(dims, collect(kh_md) .* MILLIDARCY_M2)
    perm = Matrix{Float64}(undef, 3, length(kh))
    perm[1, :] = kh
    perm[2, :] = kh
    perm[3, :] = kh .* kv_ratio
    return perm
end

"""The zero-based cell id of `(i, j, k)` in a box of `dims`, as the exchange numbers cells."""
function cell_id(dims::NTuple{3,Int}, i::Int, j::Int, k::Int)
    nx, ny, _ = dims
    return i + nx * (j + ny * k)
end

"""The two-layer arrays 10.4's cases share, at whatever pressures the case declares."""
function two_layer_arrays(pressure_per_layer)
    return Dict{String,Any}(
        "cell_centers_m" => uniform_cell_centers(TWO_LAYER_DIMS, TWO_LAYER_EXTENT),
        "porosity" => by_layer(TWO_LAYER_DIMS, TWO_LAYER_POROSITY),
        "permeability_m2" =>
            layered_permeability(TWO_LAYER_DIMS, TWO_LAYER_KH_MD, TWO_LAYER_KV_RATIO),
        "pressure_pa" => by_layer(TWO_LAYER_DIMS, pressure_per_layer),
        "sw" => by_layer(TWO_LAYER_DIMS, TWO_LAYER_SW),
    )
end

"""The producer's two connections: one cell in each layer of the same column."""
two_layer_producer_cells() =
    [cell_id(TWO_LAYER_DIMS, 3, 3, 0), cell_id(TWO_LAYER_DIMS, 3, 3, 1)]

"""
    mixing_case() -> (case, arrays)

10.4's first case: one producer whose two completions sit in layers that hold different
fluid, and one injector in each layer so the sector is not simply depleted.

The producer is given a TOTAL standard liquid rate (SPEC 9.1) and NOT two phase rates, so the
phase split of what comes out is the model's answer and not the fixture's. What the case fixes
is the contrast: the shallow layer is 200 mD at Sw = 0.25, where the Corey pair puts the
fractional flow of water near 0.02, and the deep one is 50 mD at Sw = 0.65, where it is near
0.96. The surface stream has to land between them, weighted by what each connection actually
delivered, and the connection fluxes the extraction publishes are what that is checked against.
"""
function mixing_case()
    edges = copy(MIXING_EDGES_DAYS)
    centers = uniform_cell_centers(TWO_LAYER_DIMS, TWO_LAYER_EXTENT)
    producer_cells = two_layer_producer_cells()
    injector_cells = [cell_id(TWO_LAYER_DIMS, 0, 0, 0), cell_id(TWO_LAYER_DIMS, 0, 0, 1)]
    datum = centers[producer_cells[1] + 1, 3]

    controls = Any[]
    for i in 1:(length(edges) - 1)
        push!(
            controls,
            control_segment(
                start_day = edges[i],
                end_day = edges[i + 1],
                well_id = "PRO1",
                role = "producer",
                target = "liquid_rate",
                value = MIXING_LIQUID_RATE_M3_DAY,
                bhp_limit_pa = PRODUCER_BHP_FLOOR_PA,
                connection_open = Bool[true, true],
            ),
        )
        for name in ("INJ_UPPER", "INJ_LOWER")
            push!(
                controls,
                control_segment(
                    start_day = edges[i],
                    end_day = edges[i + 1],
                    well_id = name,
                    role = "injector",
                    target = "water_rate",
                    value = MIXING_INJECTION_RATE_M3_DAY,
                    bhp_limit_pa = INJECTOR_BHP_CEILING_PA,
                    connection_open = Bool[true],
                ),
            )
        end
    end

    case = analytic_skeleton(
        case_id = "operational-two-layer-mixing",
        dims = TWO_LAYER_DIMS,
        extent = TWO_LAYER_EXTENT,
        fluids = educational_fluids(),
        wells = Any[
            analytic_well("PRO1", producer_cells; reference_depth = datum),
            analytic_well(
                "INJ_UPPER", [injector_cells[1]];
                reference_depth = centers[injector_cells[1] + 1, 3],
            ),
            analytic_well(
                "INJ_LOWER", [injector_cells[2]];
                reference_depth = centers[injector_cells[2] + 1, 3],
            ),
        ],
        controls = controls,
        report_edges_days = edges,
    )
    # The mixing case starts at ONE pressure in both layers: what differs between them is the
    # rock and the fluid they hold, so the surface composition is the answer to that and not
    # to a pressure contrast somebody put in.
    return (case, two_layer_arrays((TWO_LAYER_PRESSURE_PA, TWO_LAYER_PRESSURE_PA)))
end

"""
    crossflow_case(; connections_open) -> (case, arrays)

10.4's second and third cases, which differ in EXACTLY one thing: the completion mask.

The sector is initialised with the shallow layer 4 MPa above the deep one, and the single well
is `role = "shut"` with `target = "disabled"` on every interval — the native zero-net-surface
formulation, which makes its surface rate an exact zero rather than a small one. With
`connections_open = (true, true)` its `MultiSegmentWell` wellbore is the only fast path between
the two layers, so its segments must carry mass from the higher-potential connection to the
lower one: two connection fluxes of opposite sign under a surface that is doing nothing.

With `connections_open = (false, false)` the surface condition is IDENTICAL and the well is
isolated: `apply_perforation_mask!` multiplies every well index by zero, so no connection can
exchange anything with the reservoir and the layers keep their potentials apart. The pair is
what separates "stopped producing" from "isolated"; one case cannot say both.
"""
function crossflow_case(; connections_open::NTuple{2,Bool})
    edges = copy(CROSSFLOW_EDGES_DAYS)
    centers = uniform_cell_centers(TWO_LAYER_DIMS, TWO_LAYER_EXTENT)
    producer_cells = two_layer_producer_cells()
    label = all(connections_open) ? "open" : "closed"
    controls = Any[
        control_segment(
            start_day = edges[i],
            end_day = edges[i + 1],
            well_id = "XF1",
            role = "shut",
            target = "disabled",
            value = 0.0,
            bhp_limit_pa = nothing,
            connection_open = Bool[connections_open...],
        ) for i in 1:(length(edges) - 1)
    ]
    case = analytic_skeleton(
        case_id = "operational-two-layer-crossflow-$(label)",
        dims = TWO_LAYER_DIMS,
        extent = TWO_LAYER_EXTENT,
        fluids = educational_fluids(),
        wells = Any[
            analytic_well(
                "XF1", producer_cells; reference_depth = centers[producer_cells[1] + 1, 3],
            ),
        ],
        controls = controls,
        report_edges_days = edges,
    )
    return (case, two_layer_arrays(CROSSFLOW_PRESSURE_PA))
end

# ======================================================================================
# 10.5 completions, isolation and the change of role
# ======================================================================================

#: The 10.5 sector: 8x4 cells in two layers of 5 m. 64 cells and one well, inside P0_VERIFY.
const ROLES_DIMS = (8, 4, 2)
const ROLES_EXTENT = (80.0, 40.0, 10.0)
const ROLES_POROSITY = 0.2
const ROLES_PERMEABILITY_M2 = 100.0 * MILLIDARCY_M2
const ROLES_PRESSURE_PA = 2.0e7
const ROLES_SW = 0.3

#: January, February and March of 2020, as REAL months: 31 + 29 + 31 = 91 days, because 2020
#: is a leap year. The report edges are those months and nothing else.
const ROLES_START_DATE = "2020-01-01"
const ROLES_MONTH_EDGES_DAYS = [0.0, 31.0, 60.0, 91.0]

#: Where the role changes. Day 45 is 14 February and day 52 is 21 February: BOTH events are
#: inside one calendar month, so February carries production, a shut period and injection, and
#: a schedule that rounded either event to a month boundary would lose it. February is also
#: what makes «separate positive production and injection volumes» a real claim: a single
#: signed number for that month would be 7 m³ of liquid out minus 4 m³ of water in.
const ROLES_EVENT_DAYS = [0.0, 45.0, 52.0, 91.0]

#: The rate the well runs on in both roles, m3_sc/day. Small on purpose: the sector is CLOSED
#: and holds 6400 m³ of pore volume at a total compressibility near 8e-10 /Pa, so 0.5 m³/day
#: for 45 days is about 4.3 MPa of depletion — a real, visible drawdown that still leaves the
#: well 10 MPa clear of its bottom-hole floor, so the control is honoured and the volumes
#: below are the volumes that were asked for.
const ROLES_RATE_M3_DAY = 0.5

#: The two extra 10.5 cases. The feasible one asks for the same modest rate; the infeasible
#: one asks for 100 000 m³_sc/day out of a sector that holds 6400 m³ of pore volume, which no
#: bottom-hole pressure above the floor can deliver.
const BHP_PROBE_EDGES_DAYS = [0.0, 1.0, 2.0]
const BHP_INFEASIBLE_RATE_M3_DAY = 1.0e5

"""The well of the 10.5 sector: two connections in the same column, one per layer."""
roles_well_cells() = [cell_id(ROLES_DIMS, 4, 2, 0), cell_id(ROLES_DIMS, 4, 2, 1)]

"""The arrays of the 10.5 sector — homogeneous, so the case is about the well and not the rock."""
function roles_arrays()
    n_cells = prod(ROLES_DIMS)
    return Dict{String,Any}(
        "cell_centers_m" => uniform_cell_centers(ROLES_DIMS, ROLES_EXTENT),
        "porosity" => fill(ROLES_POROSITY, n_cells),
        "permeability_m2" => fill(ROLES_PERMEABILITY_M2, 3, n_cells),
        "pressure_pa" => fill(ROLES_PRESSURE_PA, n_cells),
        "sw" => fill(ROLES_SW, n_cells),
    )
end

"""
    roles_case() -> (case, arrays)

ONE immutable case in which a well produces, is shut in, and comes back as an injector.

The three control intervals restate the role, the target, the recorded limit and the
completion mask; nothing is inherited. The masks are `[1,1] -> [1,0] -> [1,1]`, and the middle
one is the reason this is a single case rather than three: while the well is shut its LOWER
completion is closed and its upper one is not, so «shut» and «isolated» are visible as
different states of the same well on the same run — the closed connection must carry nothing,
and the open one is free to carry whatever the wellbore and the reservoir agree on. The last
interval's mask is `[1,1]` again, which a mask that had been inherited could never be.
"""
function roles_case()
    events = copy(ROLES_EVENT_DAYS)
    wells_cells = roles_well_cells()
    centers = uniform_cell_centers(ROLES_DIMS, ROLES_EXTENT)
    controls = Any[
        control_segment(
            start_day = events[1],
            end_day = events[2],
            well_id = "OPS1",
            role = "producer",
            target = "liquid_rate",
            value = ROLES_RATE_M3_DAY,
            bhp_limit_pa = PRODUCER_BHP_FLOOR_PA,
            connection_open = Bool[true, true],
        ),
        control_segment(
            start_day = events[2],
            end_day = events[3],
            well_id = "OPS1",
            role = "shut",
            target = "disabled",
            value = 0.0,
            bhp_limit_pa = nothing,
            connection_open = Bool[true, false],
        ),
        control_segment(
            start_day = events[3],
            end_day = events[4],
            well_id = "OPS1",
            role = "injector",
            target = "water_rate",
            value = ROLES_RATE_M3_DAY,
            bhp_limit_pa = INJECTOR_BHP_CEILING_PA,
            connection_open = Bool[true, true],
        ),
    ]
    case = analytic_skeleton(
        case_id = "operational-roles",
        dims = ROLES_DIMS,
        extent = ROLES_EXTENT,
        fluids = educational_fluids(),
        wells = Any[
            analytic_well(
                "OPS1", wells_cells; reference_depth = centers[wells_cells[1] + 1, 3],
            ),
        ],
        controls = controls,
        report_edges_days = copy(ROLES_MONTH_EDGES_DAYS),
        start_date = ROLES_START_DATE,
    )
    return (case, roles_arrays())
end

"""
    bhp_probe_case(; rate_m3_day) -> (case, arrays)

The same sector driven on ONE producer rate for two days, to be asked whether it was reached.

At `ROLES_RATE_M3_DAY` it is: the well operates on `lrat`, delivers what was demanded and
never touches its floor. At `BHP_INFEASIBLE_RATE_M3_DAY` it is not: JutulDarcy switches the
well onto the only limit the case RECORDED and keeps simulating, which is exactly the failure
mode `control_evidence` exists to make visible — the numbers still look like a forward.
`set_default_limits = false` is what makes the switch legible: the bottom-hole pressure the
well ends up on has to be the floor this case wrote down, and not a convenience default
nobody agreed to.
"""
function bhp_probe_case(; rate_m3_day::Float64)
    edges = copy(BHP_PROBE_EDGES_DAYS)
    wells_cells = roles_well_cells()
    centers = uniform_cell_centers(ROLES_DIMS, ROLES_EXTENT)
    controls = Any[
        control_segment(
            start_day = edges[i],
            end_day = edges[i + 1],
            well_id = "OPS1",
            role = "producer",
            target = "liquid_rate",
            value = rate_m3_day,
            bhp_limit_pa = PRODUCER_BHP_FLOOR_PA,
            connection_open = Bool[true, true],
        ) for i in 1:(length(edges) - 1)
    ]
    case = analytic_skeleton(
        case_id = "operational-bhp-probe-$(rate_m3_day)",
        dims = ROLES_DIMS,
        extent = ROLES_EXTENT,
        fluids = educational_fluids(),
        wells = Any[
            analytic_well(
                "OPS1", wells_cells; reference_depth = centers[wells_cells[1] + 1, 3],
            ),
        ],
        controls = controls,
        report_edges_days = edges,
    )
    return (case, roles_arrays())
end

# ======================================================================================
# 10.6 the boundary: a closed benchmark, an infinite pressure support, a finite store
# ======================================================================================

#: The sector all three 10.6 cases share: two 10 m cubes in a row, produced from the inner one.
const SECTOR_EXTENT_CELL_M = 10.0
const SECTOR_POROSITY = 0.2
const SECTOR_PERMEABILITY_M2 = 100.0 * MILLIDARCY_M2
const SECTOR_PRESSURE_PA = 1.5e7
const SECTOR_SW = 0.3
const SECTOR_RATE_M3_DAY = 0.1
const SECTOR_EDGES_DAYS = [0.0, 1.0, 2.0, 3.0]

#: The conductance the outer face of the sector has. For two identical Cartesian cells the
#: native face transmissibility is the harmonic sum of two half-cell terms, `A*k/dx`, and the
#: diagnostic below asserts that against the model's own `Transmissibilities` parameter rather
#: than trusting the formula. The BOUNDARY is given the same number on purpose: a fixed-pressure
#: boundary and an attached buffer cell then differ in STORAGE alone, which is the one thing
#: 10.6 is comparing.
const SECTOR_FACE_TRANSMISSIBILITY =
    SECTOR_EXTENT_CELL_M^2 * SECTOR_PERMEABILITY_M2 / SECTOR_EXTENT_CELL_M

#: The finite store: one more cell of the same size at twice the sector's porosity, so its
#: pore volume is 400 m³ against the sector's 400 m³ — a real reserve, and a bounded one.
const BUFFER_POROSITY = 0.4

#: The buffer is WATER-BEARING at the top of the §3.1 mobile range, where `kro` is zero and the
#: oil in it cannot move. Sw = 1 is not expressible with `Sorw = 0.2`, and the educational
#: Corey pair is not bent to make a fixture prettier: what the buffer holds is 0.8 of its pore
#: volume in mobile water and 0.2 in immobile oil, and the numbers below say so.
const BUFFER_SW = 0.8

"""
    boundary_case(kind::Symbol) -> (case, arrays)

The same produced sector under three treatments of its outer face.

* `:closed` — nothing crosses it. The benchmark: a closed sector of 400 m³ of pore volume
  gives up 0.3 m³_sc over three days by decompression alone, and its pressure falls by about
  9 bar.
* `:pressure_water` — a `FlowBoundaryCondition` on the OUTER cell at the initial pressure. It
  is an INFINITE support: its pressure is a constant of the run, so it can give up any amount
  of water and never weaken. This is the minimal aquifer of plan 10.6 and it is not a
  calibrated one.
* `:finite_buffer` — a third cell attached to the sector, water-bearing, with a pore volume
  that is written down. It supports the sector too, and it runs down while it does: its own
  pressure falls, and the support it gives is therefore strictly weaker than the boundary's at
  the same conductance. It is in the grid, so its contents are in the inventory and its
  exchange with the sector is an ordinary internal flux the balance already covers.
"""
function boundary_case(kind::Symbol)
    kind in (:closed, :pressure_water, :finite_buffer) ||
        error("boundary_case: unknown treatment $(repr(kind))")
    has_buffer = kind === :finite_buffer
    n_cells = has_buffer ? 3 : 2
    dims = (n_cells, 1, 1)
    extent = (SECTOR_EXTENT_CELL_M * n_cells, SECTOR_EXTENT_CELL_M, SECTOR_EXTENT_CELL_M)
    centers = uniform_cell_centers(dims, extent)
    edges = copy(SECTOR_EDGES_DAYS)

    porosity = fill(SECTOR_POROSITY, n_cells)
    sw = fill(SECTOR_SW, n_cells)
    if has_buffer
        porosity[3] = BUFFER_POROSITY
        sw[3] = BUFFER_SW
    end
    boundary = if kind === :pressure_water
        Dict{String,Any}(
            "kind" => "pressure_water",
            "cells" => Any[1],
            "pressure_pa" => SECTOR_PRESSURE_PA,
            "trans_flow" => SECTOR_FACE_TRANSMISSIBILITY,
            "fractional_flow" => [1.0, 0.0],
        )
    else
        Dict{String,Any}("kind" => "closed", "cells" => Any[])
    end

    controls = Any[
        control_segment(
            start_day = edges[i],
            end_day = edges[i + 1],
            well_id = "SEC1",
            role = "producer",
            target = "liquid_rate",
            value = SECTOR_RATE_M3_DAY,
            bhp_limit_pa = PRODUCER_BHP_FLOOR_PA,
            connection_open = Bool[true],
        ) for i in 1:(length(edges) - 1)
    ]
    case = analytic_skeleton(
        case_id = "operational-boundary-$(kind)",
        dims = dims,
        extent = extent,
        fluids = educational_fluids(),
        wells = Any[analytic_well("SEC1", [0]; reference_depth = centers[1, 3])],
        controls = controls,
        report_edges_days = edges,
        boundary = boundary,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => centers,
        "porosity" => porosity,
        "permeability_m2" => fill(SECTOR_PERMEABILITY_M2, 3, n_cells),
        "pressure_pa" => fill(SECTOR_PRESSURE_PA, n_cells),
        "sw" => sw,
    )
    return (case, arrays)
end

# ======================================================================================
# 10.7 the five-spot
# ======================================================================================

#: The five-spot of plan 10.7: a homogeneous square of 400 x 400 x 10 m, one layer.
#: `FIVE_SPOT_BASE_NX = 16` is the 256-cell case the plan names; `refinement.jl` drives the
#: same continuous field at 32 and compares, which is why the geometry below is written as a
#: function of `nx` and the wells are placed by PHYSICAL position rather than by index.
const FIVE_SPOT_BASE_NX = 16
const FIVE_SPOT_EXTENT = (400.0, 400.0, 10.0)
const FIVE_SPOT_POROSITY = 0.2
const FIVE_SPOT_PERMEABILITY_M2 = 100.0 * MILLIDARCY_M2
const FIVE_SPOT_PRESSURE_PA = 1.5e7
const FIVE_SPOT_SW = 0.2
const FIVE_SPOT_MONTHS = 36
const FIVE_SPOT_START_DATE = "2020-01-01"
const FIVE_SPOT_INJECTION_M3_DAY = 10.0
const FIVE_SPOT_PRODUCTION_M3_DAY = 40.0
const FIVE_SPOT_PRODUCER_BHP_FLOOR_PA = 5.0e6
const FIVE_SPOT_INJECTOR_BHP_CEILING_PA = 3.0e7

#: The injector columns on the 16-cell base grid, zero-based, and the producer's four cells.
#: Both sets are symmetric about the grid's own mirror `i -> nx-1-i`: 15-2 = 13 and 15-7 = 8.
#: The producer's four cells are EQUAL BY CONSTRUCTION and none of them is "the centre" — an
#: even grid has no central cell, and naming one would put the well half a cell off axis.
const FIVE_SPOT_BASE_INJECTORS = ((2, 2), (2, 13), (13, 2), (13, 13))
const FIVE_SPOT_BASE_PRODUCER = ((7, 7), (7, 8), (8, 7), (8, 8))

"""
    month_edges_days(start_date, months) -> Vector{Float64}

Cumulative day counts of `months` REAL calendar months from `start_date`. 2020 is a leap year
and February is 29 days; twelve 30-day months would be a different case.
"""
function month_edges_days(start_date::AbstractString, months::Int)
    start = Dates.Date(start_date)
    edges = Float64[0.0]
    for m in 1:months
        push!(edges, Float64(Dates.value(start + Dates.Month(m) - start)))
    end
    return edges
end

"""
    refine_cells(base_cells, nx, base_nx) -> Vector{Int}

Map fixed vertical well trajectories to an odd nested grid refinement.

Each base connection represents one full-height vertical perforation at that cell's
centre. An odd refinement has a centre child at exactly the same coordinates; only
that child is completed. Completing all children would multiply well length and WI.
Even refinement is refused because this cell-centred fixture cannot represent its
original trajectories exactly on that grid. Native WI is recomputed from unchanged
radius and perforated thickness in the refined cell geometry.
"""
function refine_cells(base_cells, nx::Int, base_nx::Int)
    factor, remainder = divrem(nx, base_nx)
    remainder == 0 && factor >= 1 && isodd(factor) ||
        error("refine_cells: nx must be an odd whole refinement to preserve well coordinates")
    offset = factor ÷ 2
    return sort!([(bi * factor + offset) + nx * (bj * factor + offset)
                  for (bi, bj) in base_cells])
end

"""
    five_spot_case(nx) -> (case, arrays)

A confined five-spot: four symmetric injectors and one central producer, 36 calendar months.

GRAVITY IS PRESENT and is not switched off. The case is one layer, so every cell centre is at
the same depth and the native `TwoPointGravityDifference` is zero on every face — the
diagnostic asserts that rather than the fixture asserting the absence of gravity. `Pc = 0` is
the §3.1 educational model's own.

The producer is a `SimpleWell` with four connections at one depth, which is what plan 10.7
requires and what makes the pressure support symmetric: a `SimpleWell` is a single node, so
its four connections are equidistant by construction and no segment ordering can favour one
of them. `run_forward` calls `request_extra_outputs!` first, without which such a well's
`ConnectionPressureDrop` is not stored and its connection flux would be read off a
re-initialised zero.

The initial water saturation is the connate 0.2, where `krw` is zero: the sector starts with
its water immobile, which is what makes the displacement a displacement.
"""
function five_spot_case(nx::Int)
    nx >= FIVE_SPOT_BASE_NX && nx % FIVE_SPOT_BASE_NX == 0 ||
        error("five_spot_case: nx=$(nx) is not a whole refinement of $(FIVE_SPOT_BASE_NX)")
    dims = (nx, nx, 1)
    n_cells = prod(dims)
    centers = uniform_cell_centers(dims, FIVE_SPOT_EXTENT)
    edges = month_edges_days(FIVE_SPOT_START_DATE, FIVE_SPOT_MONTHS)
    datum = centers[1, 3]

    producer_cells = refine_cells(FIVE_SPOT_BASE_PRODUCER, nx, FIVE_SPOT_BASE_NX)
    injector_cells =
        [refine_cells((ij,), nx, FIVE_SPOT_BASE_NX) for ij in FIVE_SPOT_BASE_INJECTORS]
    injector_names = ["INJ_SW", "INJ_NW", "INJ_SE", "INJ_NE"]

    # Each injector keeps one vertical connection and the producer keeps its four
    # fixed vertical connections. A single-node SimpleWell keeps equal-depth
    # connections symmetric; the separate crossflow cases retain segmented wells.
    wells = Any[
        analytic_well("PRO1", producer_cells; reference_depth = datum, model = "simple"),
    ]
    for (name, cells) in zip(injector_names, injector_cells)
        push!(wells, analytic_well(name, cells; reference_depth = datum, model = "simple"))
    end

    controls = Any[]
    for i in 1:(length(edges) - 1)
        push!(
            controls,
            control_segment(
                start_day = edges[i],
                end_day = edges[i + 1],
                well_id = "PRO1",
                role = "producer",
                target = "liquid_rate",
                value = FIVE_SPOT_PRODUCTION_M3_DAY,
                bhp_limit_pa = FIVE_SPOT_PRODUCER_BHP_FLOOR_PA,
                connection_open = fill(true, length(producer_cells)),
            ),
        )
        for (name, cells) in zip(injector_names, injector_cells)
            push!(
                controls,
                control_segment(
                    start_day = edges[i],
                    end_day = edges[i + 1],
                    well_id = name,
                    role = "injector",
                    target = "water_rate",
                    value = FIVE_SPOT_INJECTION_M3_DAY,
                    bhp_limit_pa = FIVE_SPOT_INJECTOR_BHP_CEILING_PA,
                    connection_open = fill(true, length(cells)),
                ),
            )
        end
    end

    case = analytic_skeleton(
        case_id = "operational-five-spot-$(nx)",
        dims = dims,
        extent = FIVE_SPOT_EXTENT,
        fluids = educational_fluids(),
        wells = wells,
        controls = controls,
        report_edges_days = edges,
        start_date = FIVE_SPOT_START_DATE,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => centers,
        # The SAME continuous rock at every resolution: homogeneous porosity and permeability,
        # so the fine grid is a refinement of one field and not a second realisation of it.
        "porosity" => fill(FIVE_SPOT_POROSITY, n_cells),
        "permeability_m2" => fill(FIVE_SPOT_PERMEABILITY_M2, 3, n_cells),
        "pressure_pa" => fill(FIVE_SPOT_PRESSURE_PA, n_cells),
        "sw" => fill(FIVE_SPOT_SW, n_cells),
    )
    return (case, arrays)
end

# ======================================================================================
# the registry and the runner
# ======================================================================================

"""
    operational_case(name::Symbol, options) -> (case, arrays)

One registered operational fixture, fully materialized, in the exchange's own JSON shape.
A name that is not registered is an explicit error: a silently wrong reference case is worse
than a missing one, because every later comparison inherits it.
"""
function operational_case(name::Symbol, options::AbstractDict = Dict{String,Any}())
    if name === :mixing
        return mixing_case()
    elseif name === :crossflow_open
        return crossflow_case(connections_open = (true, true))
    elseif name === :crossflow_closed
        return crossflow_case(connections_open = (false, false))
    elseif name === :roles
        return roles_case()
    elseif name === :bhp_feasible
        return bhp_probe_case(rate_m3_day = ROLES_RATE_M3_DAY)
    elseif name === :bhp_infeasible
        return bhp_probe_case(rate_m3_day = BHP_INFEASIBLE_RATE_M3_DAY)
    elseif name === :boundary_closed
        return boundary_case(:closed)
    elseif name === :boundary_pressure_water
        return boundary_case(:pressure_water)
    elseif name === :boundary_finite_buffer
        return boundary_case(:finite_buffer)
    elseif name === :five_spot
        return five_spot_case(Int(get(options, "nx", FIVE_SPOT_BASE_NX)))
    end
    error(
        "operational_case: fixture $(repr(name)) is not registered; this build has " *
        ":mixing, :crossflow_open, :crossflow_closed, :roles, :bhp_feasible, " *
        ":bhp_infeasible, :boundary_closed, :boundary_pressure_water, " *
        ":boundary_finite_buffer and :five_spot",
    )
end

"""
    run_operational(name::Symbol, options) -> Dict

Build and run one operational fixture through the adapter, and return everything it produced.

The physics goes through `build_ow`, `build_forces` and `run_forward` — the same three the
worker's own `run_job` reaches — so what is verified is the operator the production path uses
and not a copy of it. `options` carries the fixture's own parameters (`nx` for the five-spot)
and any solver keyword `run_forward` forwards to `simulate_reservoir`.

A fixture whose PHYSICS cannot be assembled comes back as a status rather than an exception,
classified by `SOReconAdapter.failure_status` — the one the worker uses.
"""
function run_operational(name::Symbol, options::AbstractDict = Dict{String,Any}())
    case, arrays = operational_case(name, options)
    solver = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in options if k != "nx")
    out = Dict{String,Any}(
        "solver_options" => Dict(String(k) => v for (k, v) in solver),
        "name" => String(name),
        "case" => case,
        "arrays" => serialisable_arrays(arrays),
    )
    physical = try
        ADAPTER.build_ow(case, arrays)
    catch err
        out["status"] = ADAPTER.failure_status(err)
        out["phase"] = "build"
        out["reason"] = sprint(showerror, err)
        out["is_invalid_case_input"] = err isa ADAPTER.InvalidCaseInput
        return out
    end
    out["phase"] = "solve"
    out["native"] = operational_diagnostics(physical)
    out["extraction"] = ADAPTER.run_forward(case, arrays; solver...)
    out["status"] = String(out["extraction"]["status"])
    return out
end

"""
    operational_diagnostics(physical) -> Dict

What only Julia can see about the model that was just built, BEFORE a single timestep: the
native face transmissibilities and gravity differences, the native well indices, and the cell
depths. Python recomputes the transmissibility of a uniform pair and the zero gravity head of
a single layer from geometry and compares, so the two numbers come from different objects.
"""
function operational_diagnostics(physical)
    rmodel = physical.model.models[:Reservoir]
    parameters = physical.parameters
    centroids = rmodel.data_domain[:cell_centroids]
    well_indices = Dict{String,Any}()
    for (name, submodel) in pairs(physical.model.models)
        JutulDarcy.model_or_domain_is_well(submodel) || continue
        well_indices[String(name)] = collect(Float64.(parameters[name][:WellIndices]))
    end
    return Dict{String,Any}(
        "cell_center_depth_m" => collect(Float64.(centroids[3, :])),
        "transmissibilities" => collect(Float64.(parameters[:Reservoir][:Transmissibilities])),
        "two_point_gravity_difference" =>
            collect(Float64.(parameters[:Reservoir][:TwoPointGravityDifference])),
        "pore_volume_m3" =>
            collect(Float64.(JutulDarcy.pore_volume(physical.model, parameters))),
        "well_indices" => well_indices,
    )
end

# ======================================================================================
# measurements the checks below share
# ======================================================================================

"""Every connection row of one well, grouped by connection id, in substep order."""
function connection_series(extraction::AbstractDict, well_id::AbstractString)
    out = Dict{Int,Vector{Dict{String,Any}}}()
    for row in extraction["connections"]
        String(row["well_id"]) == well_id || continue
        push!(get!(out, Int(row["connection_id"]), Dict{String,Any}[]), Dict{String,Any}(row))
    end
    for rows in values(out)
        sort!(rows, by = r -> Int(r["step"]))
    end
    return out
end

"""The largest absolute mass rate one connection carried over the whole run, kg/s."""
function peak_connection_mass_kg_s(extraction::AbstractDict, well_id::AbstractString, id::Int)
    rows = connection_series(extraction, well_id)[id]
    return maximum(abs(Float64(r["total_mass_kg_s"])) for r in rows)
end

"""The surface component mass rate of one well at one substep, kg/s, summed over components."""
function surface_mass_rate_kg_s(extraction::AbstractDict, well_id::AbstractString, step::Int)
    components = extraction["wells"][well_id]["surface_component_mass_kg_s"]
    return sum(Float64(c[step]) for c in components)
end

"""
    wellbore_inventory_change_m3_sc(extraction) -> Vector

How much standard volume the WELLBORES gained over the run, per component, m3_sc.

`inventory_m3_sc` is the reservoir AND every well node; `reservoir_inventory_m3_sc` is the
reservoir alone; the difference between their changes is what the wellbores stored. It is
built from `TotalMasses` on both sides and NO `PerforationMask` ever touches it — which is
exactly why the isolation checks below lean on it. `outputs.jl` multiplies every published
connection flux by the interval's mask, so a closed connection's published flux is zero by
CONSTRUCTION and cannot on its own be evidence that anything was isolated; a wellbore that
did not fill, beside one that did, is a measurement of the same claim that the mask cannot
manufacture.
"""
function wellbore_inventory_change_m3_sc(extraction::AbstractDict)
    total = Float64.(extraction["inventory_m3_sc"][end]) .-
            Float64.(extraction["inventory_m3_sc"][1])
    reservoir = Float64.(extraction["reservoir_inventory_m3_sc"][end]) .-
                Float64.(extraction["reservoir_inventory_m3_sc"][1])
    return total .- reservoir
end

"""How much the RESERVOIR gained over the run, per component, m3_sc. Never masked either."""
function reservoir_inventory_change_m3_sc(extraction::AbstractDict)
    return Float64.(extraction["reservoir_inventory_m3_sc"][end]) .-
           Float64.(extraction["reservoir_inventory_m3_sc"][1])
end

"""Standard-volume inventory of one cell's water and oil at one published state, m3_sc."""
function cell_inventory_m3_sc(states::AbstractDict, time_index::Int, cell::Int)
    pv = Float64(states["pore_volume_m3"][time_index][cell + 1])
    sw = Float64(states["sw"][time_index][cell + 1])
    so = Float64(states["so"][time_index][cell + 1])
    bw = Float64(states["bw"][time_index][cell + 1])
    bo = Float64(states["bo"][time_index][cell + 1])
    return (water = sw * pv / bw, oil = so * pv / bo)
end

# ======================================================================================
# the diagnostic (plan Task 10.9). Only runs as a program.
# ======================================================================================

"""Solver options every P0 operational fixture runs with: the report step, taken whole."""
operational_solver_options() = Dict{String,Any}("timesteps" => :none)

function selftest_operations()
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "gravity_constant" => Jutul.gravity_constant,
    )
    fixtures = Dict{String,Any}()

    @testset "E01 operational fixtures: mixing, crossflow, roles, boundaries" begin
        # --- 10.4 mixing --------------------------------------------------------------
        mixing = run_operational(:mixing, operational_solver_options())
        @test mixing["status"] == "COMPLETE"
        mixing_extraction = mixing["extraction"]
        @test mixing_extraction["control_infeasible_reason"] === nothing
        fixtures["mixing"] = mixing

        # --- 10.4 crossflow, open and closed -------------------------------------------
        for (key, name) in (("crossflow_open", :crossflow_open), ("crossflow_closed", :crossflow_closed))
            run = run_operational(name, operational_solver_options())
            @test run["status"] == "COMPLETE"
            extraction = run["extraction"]
            # The surface is shut on every substep, in BOTH cases, and is an exact zero.
            for step in eachindex(extraction["chunk"]["dt_s"])
                @test extraction["wells"]["XF1"]["operating_target"][step] == "disabled"
                @test surface_mass_rate_kg_s(extraction, "XF1", step) == 0.0
            end
            fixtures[key] = run
        end
        open_run = fixtures["crossflow_open"]["extraction"]
        closed_run = fixtures["crossflow_closed"]["extraction"]
        # OPEN: two connections of one wellbore, carrying mass in OPPOSITE directions.
        upper = connection_series(open_run, "XF1")[0]
        lower = connection_series(open_run, "XF1")[1]
        @test all(r["connection_open"] for r in upper)
        @test upper[1]["total_mass_kg_s"] > 0.0    # out of the high-potential layer
        @test lower[1]["total_mass_kg_s"] < 0.0    # and into the low-potential one
        # CLOSED: the same surface condition, and nothing crosses either completion.
        #
        # The published flux being zero is NOT the evidence. `outputs.jl` multiplies every
        # connection flux by the interval's mask, so on a masked connection that number is
        # zero by construction for any model however broken; it is asserted because the
        # extraction promises it, not because it proves anything. The three claims under it
        # are the ones a broken model could fail, and none of them passes through a mask:
        #
        #  (a) the completions still EXIST and are still conductive — their native well
        #      indices are positive. `PerforationMask` multiplies the assembled cross-term
        #      entries (`JutulDarcy 0.3.11 facility/cross_terms.jl:111-115` ->
        #      `apply_perforation_mask!`, `facility/wells/wells.jl:657-690`) and never the
        #      `WellIndices` parameter that `cross_term_perforation_get_conn` reads
        #      (`cross_terms.jl:46`), so a zero here would mean a well that was not completed
        #      rather than one that was shut off;
        #  (b) the WELLBORE did not fill: its own standard-volume inventory barely moved,
        #      against an open wellbore that took a tenth of a cubic metre of water;
        #  (c) the RESERVOIR did not give anything up through it, and kept its layer
        #      potentials apart (asserted below).
        for id in (0, 1)
            @test all(!r["connection_open"] for r in connection_series(closed_run, "XF1")[id])
            @test peak_connection_mass_kg_s(closed_run, "XF1", id) == 0.0
        end
        closed_wi = fixtures["crossflow_closed"]["native"]["well_indices"]["XF1"]
        @test length(closed_wi) == 2
        @test all(>(0.0), closed_wi)
        # And they are the SAME completions as the open case's: one geometry, two masks.
        @test closed_wi ≈ fixtures["crossflow_open"]["native"]["well_indices"]["XF1"]
        open_stored = wellbore_inventory_change_m3_sc(open_run)
        closed_stored = wellbore_inventory_change_m3_sc(closed_run)
        open_reservoir = reservoir_inventory_change_m3_sc(open_run)
        closed_reservoir = reservoir_inventory_change_m3_sc(closed_run)
        @test maximum(abs.(open_stored)) > 0.1
        @test maximum(abs.(closed_stored)) < 1.0e-3 * maximum(abs.(open_stored))
        @test maximum(abs.(open_reservoir)) > 0.1
        @test maximum(abs.(closed_reservoir)) < 1.0e-3 * maximum(abs.(open_reservoir))
        measured["crossflow_isolation"] = Dict{String,Any}(
            "closed_well_indices" => closed_wi,
            "open_wellbore_change_m3_sc" => open_stored,
            "isolated_wellbore_change_m3_sc" => closed_stored,
            "open_reservoir_change_m3_sc" => open_reservoir,
            "isolated_reservoir_change_m3_sc" => closed_reservoir,
        )
        # The well storage term, on the first substep: what the two connections carry differs
        # by what the wellbore itself is filling with, and that is a small part of either.
        @test abs(upper[1]["total_mass_kg_s"] + lower[1]["total_mass_kg_s"]) <
              0.2 * max(
                  abs(upper[1]["total_mass_kg_s"]), abs(lower[1]["total_mass_kg_s"]),
              )
        # And the two runs are not the same run: the open wellbore really equalised the layer
        # potentials while the isolated one left them where they were put. The comparison is
        # on ABSOLUTE gaps, so a sign flip in an equalised column cannot pass it by accident.
        perforated = two_layer_producer_cells()
        function final_layer_gap_pa(states)
            last = length(states["times_s"])
            pressure = states["pressure_pa"][last]
            return Float64(pressure[perforated[1] + 1]) - Float64(pressure[perforated[2] + 1])
        end
        open_gap = abs(final_layer_gap_pa(open_run["states"]))
        closed_gap = abs(final_layer_gap_pa(closed_run["states"]))
        @test open_gap < 0.05 * closed_gap
        # The isolated case kept most of the 4 MPa it was given: what leaked went through the
        # shale barrier, not through a well that was supposed to be shut off from the rock.
        @test closed_gap > 0.5 * (CROSSFLOW_PRESSURE_PA[1] - CROSSFLOW_PRESSURE_PA[2])
        measured["crossflow_layer_gap_pa"] = Dict{String,Any}(
            "initial" => CROSSFLOW_PRESSURE_PA[1] - CROSSFLOW_PRESSURE_PA[2],
            "open_final" => open_gap,
            "isolated_final" => closed_gap,
        )

        # --- 10.5 roles ----------------------------------------------------------------
        roles = run_operational(:roles, operational_solver_options())
        @test roles["status"] == "COMPLETE"
        @test roles["extraction"]["control_infeasible_reason"] === nothing
        # Both of this well's completions are real and conductive on every interval; what
        # changes between them is the MASK, which never reaches the well index. So the zero
        # the shut interval publishes on connection 1 is a completion that was shut off, and
        # not one that was never drilled.
        @test all(>(0.0), roles["native"]["well_indices"]["OPS1"])
        fixtures["roles"] = roles

        for (key, name) in (("bhp_feasible", :bhp_feasible), ("bhp_infeasible", :bhp_infeasible))
            run = run_operational(name, operational_solver_options())
            @test run["status"] == "COMPLETE"
            fixtures[key] = run
        end
        @test fixtures["bhp_feasible"]["extraction"]["control_infeasible_reason"] === nothing
        @test fixtures["bhp_infeasible"]["extraction"]["control_infeasible_reason"] !== nothing

        # --- 10.6 boundaries ------------------------------------------------------------
        for (key, name) in (
            ("boundary_closed", :boundary_closed),
            ("boundary_pressure_water", :boundary_pressure_water),
            ("boundary_finite_buffer", :boundary_finite_buffer),
        )
            run = run_operational(name, operational_solver_options())
            @test run["status"] == "COMPLETE"
            fixtures[key] = run
        end
        # The conductance the fixture hands the boundary IS the native face transmissibility
        # of the sector, read out of the model's own parameter set.
        native_trans = fixtures["boundary_closed"]["native"]["transmissibilities"]
        @test length(native_trans) == 1
        @test native_trans[1] ≈ SECTOR_FACE_TRANSMISSIBILITY rtol = 1.0e-12
        measured["sector_face_transmissibility"] = Dict{String,Any}(
            "native" => native_trans[1],
            "declared_boundary_trans_flow" => SECTOR_FACE_TRANSMISSIBILITY,
        )

        # --- the refusals a boundary that cannot be built has to name --------------------
        measured["boundary_refusals"] = boundary_refusal_messages()
    end

    measured["fixtures"] = fixtures
    measured["status"] = "ok"
    return measured
end

"""
Every guard between a malformed `pressure_water` boundary and silent wrong physics.

A boundary that quietly defaulted its pressure or its conductance would be an aquifer nobody
specified, supporting a sector by an amount nobody wrote down; these are the refusals that
stop that, collected by running each one and keeping the message.
"""
function boundary_refusal_messages()
    case, arrays = boundary_case(:pressure_water)
    physical = ADAPTER.build_ow(case, arrays)
    controls = ADAPTER.compile_intervals(case).controls[1]
    messages = Dict{String,Any}()
    broken = Dict{String,Any}(
        "unknown_kind" => Dict{String,Any}("kind" => "aquifer", "cells" => Any[1]),
        "no_cells" => merge(Dict{String,Any}(case["boundary"]), Dict("cells" => Any[])),
        "no_pressure" => merge(Dict{String,Any}(case["boundary"]), Dict("pressure_pa" => nothing)),
        "no_trans_flow" => merge(Dict{String,Any}(case["boundary"]), Dict("trans_flow" => nothing)),
        "cell_outside_grid" => merge(Dict{String,Any}(case["boundary"]), Dict("cells" => Any[9])),
        "duplicate_cell" => merge(Dict{String,Any}(case["boundary"]), Dict("cells" => Any[1, 1])),
        "fractional_flow_not_a_split" =>
            merge(Dict{String,Any}(case["boundary"]), Dict("fractional_flow" => [0.5, 0.2])),
    )
    for (label, boundary) in broken
        messages[label] = try
            ADAPTER.build_forces(physical.model, controls, boundary)
            error("boundary_refusal_messages: $(label) was accepted")
        catch err
            err isa ADAPTER.InvalidCaseInput ||
                error("boundary_refusal_messages: $(label) raised $(typeof(err))")
            sprint(showerror, err)
        end
    end
    return messages
end

"""Parse the diagnostic to run and the optional `--out` the project launcher appends."""
function parse_operations_args(args::Vector{String})
    out = nothing
    requested = false
    i = 1
    while i <= length(args)
        if args[i] == "--test-operations"
            requested = true
            i += 1
        elseif args[i] == "--out"
            i + 1 <= length(args) || error("missing value for --out")
            out = args[i + 1]
            i += 2
        else
            error("unexpected argument $(args[i])")
        end
    end
    return (requested, out)
end

function operations_main(args::Vector{String})
    requested, out = parse_operations_args(args)
    requested || error(
        "nothing to do: pass --test-operations to run the operational fixtures of plan " *
        "Task 10.4-10.6",
    )
    payload = try
        selftest_operations()
    catch err
        # The launcher reads --out when stderr is empty, so the cause has to be in the file.
        failure = Dict{String,Any}("status" => "error", "message" => sprint(showerror, err))
        if out !== nothing
            open(out, "w") do io
                JSON.print(io, failure)
            end
        end
        rethrow()
    end
    if out !== nothing
        open(out, "w") do io
            JSON.print(io, payload)
            print(io, "\n")
        end
    end
    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    operations_main(collect(ARGS))
    println("so-recon verification: operations diagnostic passed")
end
