# E01.9 — the analytic verification fixtures, and the diagnostic that runs them.
#
# Plan Task 9 is the first task on this plan that SCORES physics rather than wiring. What it
# needs from Julia is a small family of cases whose answer is known for a reason outside the
# simulator — a closed system that must not move, a Buckley-Leverett displacement whose exact
# weak solution is a formula, a column that must stay in hydrostatic balance, a column whose
# heavy phase must fall — and one run of each of them through the SAME `build_ow`,
# `build_forces` and `run_forward` the worker uses. There is no second solver here and no
# reimplemented flux: SPEC 8.1 leaves JutulDarcy the only operational backend, and a
# verification that used its own numerics would verify the wrong thing.
#
# This file is a PROGRAM, not a definition module. `fixtures.jl` deliberately avoids loading
# the adapter so it can be included from either side of the tree; nothing includes this file,
# so it loads the adapter at the top and is done with it. What it shares with `fixtures.jl` is
# only the constants and the JSON shapes — `SECONDS_PER_DAY`, `DATUM_M`, `educational_fluids`,
# `control_segment`, `serialisable_arrays` — which is why that file is included rather than
# copied.
#
# The scoring itself is NOT here. Julia measures what only Julia can see (the native face
# gravity parameter, the native densities, the exception a broken PVT raises at construction)
# and publishes the extraction; `so_recon.validation.physics` scores the artifacts Python
# publishes from it, against `configs/e01_tolerances.yml`. A fixture that graded itself would
# be a fixture that could not fail.

using Test
using Jutul
using JutulDarcy
using JSON

include(joinpath(@__DIR__, "fixtures.jl"))

#: The adapter under verification. Loaded once, at include time: this file is only ever run
#: as a program, so there is no cycle to avoid and no caller who wants the fixtures alone.
const ADAPTER = include(joinpath(@__DIR__, "..", "adapter", "SOReconAdapter.jl"))

# ======================================================================================
# shared fixture construction
# ======================================================================================

"""
    uniform_cell_centers(dims, extent; datum) -> Matrix (n_cells, 3)

Cell centres of a uniform box of `extent` metres divided into `dims` cells, with its top
face at `datum`. `cell_id = i + nx*(j + ny*k)` is zero-based, as the exchange declares, and
z is depth, positive down and ABSOLUTE — the same convention `fixtures.cartesian_cell_centers`
uses, generalised off the 10 m cube because these fixtures are 100 m long and 50 m deep.
"""
function uniform_cell_centers(dims::NTuple{3,Int}, extent::NTuple{3,Float64}; datum::Float64 = DATUM_M)
    nx, ny, nz = dims
    dx, dy, dz = extent ./ dims
    centers = Matrix{Float64}(undef, prod(dims), 3)
    for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        cell = i + nx * (j + ny * k) + 1
        centers[cell, 1] = dx * i + 0.5 * dx
        centers[cell, 2] = dy * j + 0.5 * dy
        centers[cell, 3] = datum + dz * k + 0.5 * dz
    end
    return centers
end

"""
    limit_fluids(; viscosity_pa_s, compressibility_pa_inv, residual_saturations) -> Dict

The §3.1 educational fluid moved into a DECLARED analytical limit.

`analytical_limit = true` is the flag that says this fluid is a verification fixture's and
not the P1 generator's: `CaseBundle._check_gravity` reads it, and every P1 fluid is the
`FluidSpec()` default, where it is false. Everything the caller does not name — the standard
densities, the reference pressure and temperature, the Corey exponents, the endpoints and
`pc_model = "zero"` — stays exactly what §3.1 fixed, so a fixture departs from the
educational model in the ways it states and in no others.
"""
function limit_fluids(;
    viscosity_pa_s::Vector{Float64},
    compressibility_pa_inv::Vector{Float64},
    residual_saturations::Vector{Float64},
)
    fluid = educational_fluids()
    fluid["viscosity_pa_s"] = copy(viscosity_pa_s)
    fluid["compressibility_pa_inv"] = copy(compressibility_pa_inv)
    fluid["residual_saturations"] = copy(residual_saturations)
    fluid["analytical_limit"] = true
    return fluid
end

"""One well of an analytic fixture, perforating `cells` with its datum at `reference_depth`."""
function analytic_well(
    well_id::AbstractString,
    cells::Vector{Int};
    reference_depth::Float64,
    radius::Float64 = 0.1,
)
    return Dict{String,Any}(
        "well_id" => String(well_id),
        "cells" => cells,
        "radius_m" => radius,
        "reference_depth_m" => reference_depth,
        "model" => "multisegment",
        # The native wellbore always couples its connections (`check_crossflow` in model.jl).
        "allow_crossflow" => true,
    )
end

"""
A shut, fully isolated monitoring well, restated on every interval of `edges_days`.

Every fixture here carries one, and it is not decoration. A case with no well at all cannot
be compiled into a schedule — `compile_schedule` refuses one, in Python and in the adapter
alike — so a source-free fixture that is to be PUBLISHED as a forward result needs a well
that is not a source. A `DisabledControl` over a fully closed completion mask is exactly
that, and it turns `closed_connection_mass_kg_s_max` from a threshold over an empty table
into a real measurement: the connections exist, and they have to carry nothing.
"""
function shut_monitor_controls(well_id::AbstractString, edges_days::Vector{Float64}, n_cells::Int)
    return Any[
        control_segment(
            start_day = edges_days[i],
            end_day = edges_days[i + 1],
            well_id = well_id,
            role = "shut",
            target = "disabled",
            value = 0.0,
            bhp_limit_pa = nothing,
            connection_open = fill(false, n_cells),
        ) for i in 1:(length(edges_days) - 1)
    ]
end

"""The case skeleton every analytic fixture shares, in the exchange's own JSON shape."""
function analytic_skeleton(;
    case_id::AbstractString,
    dims::NTuple{3,Int},
    extent::NTuple{3,Float64},
    fluids::AbstractDict,
    wells::Vector{Any},
    controls::Vector{Any},
    report_edges_days::Vector{Float64},
)
    return Dict{String,Any}(
        "schema_version" => "case-1",
        "case_id" => String(case_id),
        "start_date" => "2020-01-01",
        "grid" => Dict{String,Any}(
            "shape" => collect(dims),
            "extent_m" => collect(extent),
            "z_positive" => "down",
        ),
        "rock" => Dict{String,Any}("rock_compressibility_pa_inv" => 0.0),
        "fluids" => fluids,
        "wells" => wells,
        "controls" => controls,
        "boundary" => Dict{String,Any}("kind" => "closed", "cells" => Any[]),
        "report_edges_s" => report_edges_days .* SECONDS_PER_DAY,
        "units" => Dict{String,Any}("control_rate" => "m3_sc/day"),
        "gravity_m_s2" => STANDARD_GRAVITY_M_S2,
    )
end

# ======================================================================================
# 9.5 the closed / PVT fixtures
# ======================================================================================

#: Three report steps of a day, for both closed fixtures (plan 9.5).
const CLOSED_EDGES_DAYS = [0.0, 1.0, 2.0, 3.0]
const CLOSED_PRESSURE_PA = 1.5e7
const CLOSED_SW = 0.3

"""
    closed_pvt_case(n_cells; broken_density) -> (case, arrays)

A closed system with no source at all, at rest, for `n_cells` cells in ONE layer.

One layer is not an accident. A closed multi-layer box under gravity with a uniform
saturation is NOT at rest — water and oil want different pressure gradients, so it segregates
— and a drift tolerance of 1e-8 applied to it would be measuring real physics and calling it
an error. With a single layer every cell centre is at the same depth, the native face gravity
`gdz` is zero on every face, and a uniform pressure and saturation are an exact steady state:
whatever moves is the solver's own arithmetic, which is the thing 9.5 is about.

`broken_density` makes the standard water density negative. Every field is still present, of
the right type and of the right length — the case is WELL FORMED and its physics does not
exist — which is the distinction `assert_physical_pvt` turns into `PHYSICALLY_INVALID`.
"""
function closed_pvt_case(n_cells::Int; broken_density::Bool = false)
    dims = (n_cells, 1, 1)
    extent = (CELL_EDGE_M * n_cells, CELL_EDGE_M, CELL_EDGE_M)
    fluids = educational_fluids()
    if broken_density
        fluids["density_sc_kg_m3"] = [-1000.0, 800.0]
    end
    case = analytic_skeleton(
        case_id = broken_density ? "analytic-broken-pvt" : "analytic-closed-$(n_cells)-cell",
        dims = dims,
        extent = extent,
        fluids = fluids,
        wells = Any[analytic_well("MON1", [0]; reference_depth = DATUM_M)],
        controls = shut_monitor_controls("MON1", CLOSED_EDGES_DAYS, 1),
        report_edges_days = copy(CLOSED_EDGES_DAYS),
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => uniform_cell_centers(dims, extent),
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, n_cells),
        "pressure_pa" => fill(CLOSED_PRESSURE_PA, n_cells),
        "sw" => fill(CLOSED_SW, n_cells),
    )
    return (case, arrays)
end

# ======================================================================================
# 9.6 the Buckley-Leverett fixture
# ======================================================================================

#: The core the three BL grids share: 100 m long, 1 m² of face, porosity 0.2, 100 mD.
const BL_LENGTH_M = 100.0
const BL_AREA_M2 = 1.0
const BL_POROSITY = 0.2
const BL_PERMEABILITY_M2 = 100.0 * MILLIDARCY_M2
#: Total pore volume, m³: 100 x 1 x 0.2. Injecting 0.2 of it over one day is the reference.
const BL_PORE_VOLUME_M3 = BL_LENGTH_M * BL_AREA_M2 * BL_POROSITY
const BL_INJECTED_PV = 0.2
const BL_INJECTION_M3_DAY = BL_INJECTED_PV * BL_PORE_VOLUME_M3
const BL_HORIZON_DAYS = 1.0
#: Eight report steps over the day. Four and eight are both inside the P0_VERIFY limit of
#: twelve; eight is used on ALL THREE grids so the only thing that differs between them is
#: the cell count, which is what makes the sequence a refinement study.
const BL_REPORT_STEPS = 8
#: The downstream pressure the displacement runs against. In this limit the fluids are
#: incompressible and have equal viscosity, so the saturation solution does not depend on the
#: pressure level at all; it is fixed only so the column has a well-posed outlet.
const BL_DOWNSTREAM_BHP_PA = 1.0e7

"""
    bl_case(nx) -> (case, arrays)

The Buckley-Leverett core flood, in the limit the analytic solution is exact in.

Horizontal, one layer: every cell centre is at the same depth, so the native `gdz` is zero on
every face and gravity plays no part without anybody switching gravity off. `Pc = 0`,
`c = 0`, equal phase viscosity, `Swc = Sor = 0`, Corey `n = 2` and an initially dry core are
the rest of the limit, and `analytical_limit = true` says so on the case itself.

Injection is a water rate at the upstream face and the outlet is a bottom-hole pressure at
the downstream one. The injector records NO bottom-hole ceiling, deliberately: the Darcy drop
this core needs to carry 0.2 PV/day through 100 mD and 1 m² is hundreds of bar, so a ceiling
would be reached and the well would come off its rate — and a reference displacement whose
injected volume is not the injected volume is not a reference at all.
"""
function bl_case(nx::Int)
    dims = (nx, 1, 1)
    extent = (BL_LENGTH_M, BL_AREA_M2, BL_AREA_M2)
    edges_days = collect(range(0.0, BL_HORIZON_DAYS; length = BL_REPORT_STEPS + 1))
    centers = uniform_cell_centers(dims, extent)
    well_depth = centers[1, 3]
    injecting = [
        control_segment(
            start_day = edges_days[i],
            end_day = edges_days[i + 1],
            well_id = "INJ1",
            role = "injector",
            target = "water_rate",
            value = BL_INJECTION_M3_DAY,
            bhp_limit_pa = nothing,
            connection_open = Bool[true],
        ) for i in 1:(length(edges_days) - 1)
    ]
    producing = [
        control_segment(
            start_day = edges_days[i],
            end_day = edges_days[i + 1],
            well_id = "PRO1",
            role = "producer",
            target = "bhp",
            value = BL_DOWNSTREAM_BHP_PA,
            bhp_limit_pa = nothing,
            connection_open = Bool[true],
        ) for i in 1:(length(edges_days) - 1)
    ]
    case = analytic_skeleton(
        case_id = "analytic-bl-$(nx)",
        dims = dims,
        extent = extent,
        fluids = limit_fluids(
            viscosity_pa_s = [1.0e-3, 1.0e-3],
            compressibility_pa_inv = [0.0, 0.0],
            residual_saturations = [0.0, 0.0],
        ),
        wells = Any[
            analytic_well("INJ1", [0]; reference_depth = well_depth),
            analytic_well("PRO1", [nx - 1]; reference_depth = well_depth),
        ],
        controls = Any[injecting..., producing...],
        report_edges_days = edges_days,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => centers,
        "porosity" => fill(BL_POROSITY, nx),
        "permeability_m2" => fill(BL_PERMEABILITY_M2, 3, nx),
        "pressure_pa" => fill(BL_DOWNSTREAM_BHP_PA, nx),
        "sw" => fill(0.0, nx),
    )
    return (case, arrays)
end

#: What a BL grid is simulated with: the report step, and one halving of it and no more.
#: `timesteps = :none` takes the report step whole; Jutul's `pick_cut_timestep` divides by
#: `decrease_factor` = 2 on a failure, and `max_timestep_cuts = 1` allows exactly one, so the
#: accepted substeps are the report dt and its half and nothing else (plan 9.6).
function bl_solver_options()
    dt = BL_HORIZON_DAYS * SECONDS_PER_DAY / BL_REPORT_STEPS
    return Dict{String,Any}(
        "timesteps" => :none,
        "initial_dt" => dt,
        "max_timestep" => dt,
        "max_timestep_cuts" => 1,
        # A NEWTON budget, not a timestep one. The first step of this fixture starts from a
        # completely dry core, where `krw(0) = 0` makes the water equation degenerate until
        # the injection cell takes its first water, and JutulDarcy's default budget of 15
        # iterations does not get through it on the 64- and 128-cell grids: measured, both
        # abort in report step 1 and both complete with 30. Raising the iteration count
        # changes what the solver is allowed to SPEND, never what it is allowed to accept —
        # the converged answer still satisfies the same `tol_cnv`/`tol_mb` — whereas letting
        # the timestep cut further would change the substeps the reference is compared at.
        "max_nonlinear_iterations" => 30,
    )
end

# ======================================================================================
# 9.7 the hydrostatic column and the gravity segregation case
# ======================================================================================

#: The vertical column both 9.7 fixtures use: 100 m of height in two 50 m cells, whose top
#: face is at the 1000 m fixture datum, so the centres are at 1025 m and 1075 m. Absolute
#: depth is load-bearing: a column origined at zero would let a grid-origin bug through.
const COLUMN_DIMS = (1, 1, 2)
const COLUMN_EXTENT = (10.0, 10.0, 100.0)
const COLUMN_TOP_PRESSURE_PA = 1.5e7

"""
    hydrostatic_pressure(z, z_ref, p_ref, rho_sc, compressibility) -> Float64

The EXACT hydrostatic pressure of the same `rho(p)` the model is built from.

`dp/dz = rho(p) g` with `rho(p) = rho_sc exp(c (p - p_sc))` integrates in closed form. Let
`u = exp(-c (p - p_sc))`, so `rho = rho_sc / u`; then `du/dz = -c u dp/dz = -c g rho_sc`, `u`
is linear in depth, and

    p(z) = p_sc - ln( u(z_ref) - c g rho_sc (z - z_ref) ) / c.

This is a continuous solution, not the discrete one the two-point scheme satisfies, and that
is the point: initialising from the discrete balance would make the face residual below zero
by construction and prove nothing about the model. `c = 0` degenerates to the incompressible
`p_ref + rho_sc g (z - z_ref)`.
"""
function hydrostatic_pressure(
    z::Float64, z_ref::Float64, p_ref::Float64, rho_sc::Float64, compressibility::Float64,
    p_sc::Float64,
)
    g = STANDARD_GRAVITY_M_S2
    compressibility == 0.0 && return p_ref + rho_sc * g * (z - z_ref)
    u_ref = exp(-compressibility * (p_ref - p_sc))
    u = u_ref - compressibility * g * rho_sc * (z - z_ref)
    u > 0.0 || error("hydrostatic_pressure: the column is too deep for this compressibility")
    return p_sc - log(u) / compressibility
end

"""
    hydrostatic_case() -> (case, arrays)

A closed vertical column of water alone, initialised at hydrostatic equilibrium.

`Sw = 1` is the single-phase limit of the oil-water system, and it is expressible only with
`Swc = Sor = 0` — the mobile range of the §3.1 Corey pair is [0.2, 0.8], and a case outside
it is refused rather than clipped. Everything else is §3.1: the densities, the reference
pressure, the phase compressibilities and the viscosities are untouched, so the `rho_w(p)`
the analytic pressure below integrates is the one the model's own
`ConstantCompressibilityDensities` evaluates.

The column is closed and carries a shut monitoring well. It must stay where it is put.

MEASURED, on the pinned JutulDarcy 0.3.11: at EXACTLY `So = 0` the two-phase Newton update
produces a non-finite pressure increment — `Jutul.check_increment`
(`Jutul/src/utils.jl:94`) reports `Pressure: 2 non-finite values`, and the simulator's own
`failure_cuts_timestep` recovers by halving the step. With the oil phase exactly absent its
conservation equation is `0 = 0`: `So * drho_o/dp` and `dkr_o/dSo` both vanish for the
quadratic Corey pair, so the oil row carries no pressure sensitivity at all. It happens with
the default CPR solver and with a direct one alike, and with and without the well, so it is
the degenerate two-phase Jacobian and not the linear solver or the wellbore. The column
still completes in equilibrium — measured drift below — and the alternative would be to move
the fixture off the single-phase limit it is here to test, so the behaviour is recorded
rather than designed around.
"""
function hydrostatic_case()
    fluids = limit_fluids(
        viscosity_pa_s = [1.0e-3, 3.0e-3],
        compressibility_pa_inv = [4.0e-10, 1.0e-9],
        residual_saturations = [0.0, 0.0],
    )
    centers = uniform_cell_centers(COLUMN_DIMS, COLUMN_EXTENT)
    z_ref = centers[1, 3]
    rho_w_sc = Float64(fluids["density_sc_kg_m3"][1])
    c_w = Float64(fluids["compressibility_pa_inv"][1])
    p_sc = Float64(fluids["p_sc_pa"])
    pressure = Float64[
        hydrostatic_pressure(centers[c, 3], z_ref, COLUMN_TOP_PRESSURE_PA, rho_w_sc, c_w, p_sc)
        for c in 1:size(centers, 1)
    ]
    case = analytic_skeleton(
        case_id = "analytic-hydrostatic",
        dims = COLUMN_DIMS,
        extent = COLUMN_EXTENT,
        fluids = fluids,
        wells = Any[analytic_well("MON1", [0]; reference_depth = DATUM_M)],
        controls = shut_monitor_controls("MON1", CLOSED_EDGES_DAYS, 1),
        report_edges_days = copy(CLOSED_EDGES_DAYS),
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => centers,
        "porosity" => fill(0.2, prod(COLUMN_DIMS)),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, prod(COLUMN_DIMS)),
        "pressure_pa" => pressure,
        "sw" => fill(1.0, prod(COLUMN_DIMS)),
    )
    return (case, arrays)
end

#: What the single-phase limit costs, MEASURED on the pinned Jutul 0.4.31 / JutulDarcy
#: 0.3.11 and asserted exactly. Three report steps of a day become four accepted substeps
#: because the non-finite Newton increment `hydrostatic_case` describes is caught twice by
#: `failure_cuts_timestep` and each catch halves the step; the recovery is bounded, not
#: open-ended, and if either number moves the defect has changed and must be looked at again
#: rather than re-documented. An exhausted cut budget is `dt = NaN` and a hard failure, so
#: these counters growing is the only quiet way this could rot.
const SINGLE_PHASE_CUT_STEPS = 2
const SINGLE_PHASE_ACCEPTED_STEPS = 4

#: The segregation column: heavy water ABOVE light oil, which is the unstable arrangement.
#: Both saturations are inside the §3.1 mobile range [0.2, 0.8] so that both phases can move;
#: at the endpoints one of them could not, and the case would demonstrate nothing.
const SEGREGATION_SW = [0.7, 0.3]
#: A darcy of permeability and thirty days, so the movement is a plain fact rather than a
#: number one has to squint at: the counter-current gravity flux moves of the order of ten
#: percent of the column's water across the face in that time.
const SEGREGATION_PERMEABILITY_M2 = 1000.0 * MILLIDARCY_M2
const SEGREGATION_EDGES_DAYS = [0.0, 10.0, 20.0, 30.0]

"""
    segregation_case() -> (case, arrays)

A closed vertical column with the heavy phase on top. It MUST fall.

The pressure is initialised at the discrete total-pressure balance of the MIXTURE density, so
the only potential difference left across the face is the one between the phases: water is
heavier than the mixture and oil is lighter, and nothing else is pushing. The case is not
expected to stay still — it is expected to move, in one direction — which is why it is a
separate fixture from the hydrostatic one and not a tolerance on it.
"""
function segregation_case()
    fluids = educational_fluids()
    centers = uniform_cell_centers(COLUMN_DIMS, COLUMN_EXTENT)
    n_cells = prod(COLUMN_DIMS)
    rho_sc = Float64.(fluids["density_sc_kg_m3"])
    comp = Float64.(fluids["compressibility_pa_inv"])
    p_sc = Float64(fluids["p_sc_pa"])
    mixture_density(p, sw) =
        sw * rho_sc[1] * exp(comp[1] * (p - p_sc)) + (1 - sw) * rho_sc[2] * exp(comp[2] * (p - p_sc))

    pressure = fill(COLUMN_TOP_PRESSURE_PA, n_cells)
    for c in 2:n_cells
        dz = centers[c, 3] - centers[c - 1, 3]
        # Two fixed-point sweeps: rho moves by parts in 1e5 over 50 m, so this is converged
        # far below the pressure the check below is compared against.
        for _ in 1:3
            rho_face = 0.5 * (
                mixture_density(pressure[c - 1], SEGREGATION_SW[c - 1]) +
                mixture_density(pressure[c], SEGREGATION_SW[c])
            )
            pressure[c] = pressure[c - 1] + STANDARD_GRAVITY_M_S2 * dz * rho_face
        end
    end

    case = analytic_skeleton(
        case_id = "analytic-segregation",
        dims = COLUMN_DIMS,
        extent = COLUMN_EXTENT,
        fluids = fluids,
        wells = Any[analytic_well("MON1", [0]; reference_depth = DATUM_M)],
        controls = shut_monitor_controls("MON1", SEGREGATION_EDGES_DAYS, 1),
        report_edges_days = copy(SEGREGATION_EDGES_DAYS),
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => centers,
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(SEGREGATION_PERMEABILITY_M2, 3, n_cells),
        "pressure_pa" => pressure,
        "sw" => copy(SEGREGATION_SW),
    )
    return (case, arrays)
end

# ======================================================================================
# the registry and the runner
# ======================================================================================

"""
    analytic_case(name::Symbol, options::AbstractDict) -> (case, arrays)

One registered analytic fixture, fully materialized, in the exchange's own JSON shape.

A name that is not registered is an explicit error rather than a plausible-looking stub: a
silently wrong reference case is worse than a missing one, because every later comparison
inherits it.
"""
function analytic_case(name::Symbol, options::AbstractDict = Dict{String,Any}())
    if name === :closed_cell_pvt
        return closed_pvt_case(1)
    elseif name === :closed_box_pvt
        return closed_pvt_case(8)
    elseif name === :broken_pvt
        return closed_pvt_case(1; broken_density = true)
    elseif name === :bl
        return bl_case(Int(get(options, "nx", 128)))
    elseif name === :hydrostatic
        return hydrostatic_case()
    elseif name === :segregation
        return segregation_case()
    end
    error(
        "analytic_case: fixture $(repr(name)) is not registered; this build has " *
        ":closed_cell_pvt, :closed_box_pvt, :broken_pvt, :bl, :hydrostatic and :segregation",
    )
end

"""
    run_analytic(name::Symbol, options::AbstractDict) -> Dict

Build and run one analytic fixture through the adapter, and return everything it produced.

The physics goes through `build_ow`, `build_forces` and `run_forward` — the same three the
worker's own `run_job` reaches — so what is verified here is the operator the production path
uses and not a copy of it. `options` carries the fixture's own parameters (`nx` for the BL
grids) and any solver keyword `run_forward` forwards to `simulate_reservoir`.

A fixture whose PHYSICS cannot be assembled comes back as a status rather than an exception:
the classification is `SOReconAdapter.failure_status`, the one the worker uses, so a fixture
that reports `PHYSICALLY_INVALID` is reporting what a real job would have recorded.
"""
function run_analytic(name::Symbol, options::AbstractDict = Dict{String,Any}())
    case, arrays = analytic_case(name, options)
    solver = Dict{Symbol,Any}(
        Symbol(k) => v for (k, v) in options if k != "nx"
    )
    out = Dict{String,Any}(
        "name" => String(name),
        "case" => case,
        "arrays" => serialisable_arrays(arrays),
    )
    # Construction and simulation are separate phases on purpose: a fixture that fails to
    # BUILD has to be distinguishable from one that failed to converge, and 9.5 requires a
    # broken PVT to be the former.
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
    out["native"] = native_diagnostics(physical, case, arrays)
    out["extraction"] = ADAPTER.run_forward(case, arrays; solver...)
    out["status"] = String(out["extraction"]["status"])
    return out
end

"""
    native_diagnostics(physical, case, arrays) -> Dict

What only Julia can see about the model that was just built, BEFORE a single timestep.

The two-point gravity difference is read as the PARAMETER JutulDarcy will actually use, the
densities come from the model's own `PhaseMassDensities` variable, and the face residual is
assembled the way `darcy_permeability_potential_differences` assembles it:

    potential = -T * ( (p[r] - p[l]) + gdz * rho_face ),  gdz = -g * (z[r] - z[l])

with `rho_face` the ARITHMETIC average of the two cells (`Jutul.face_average`). Python
recomputes the same number from the PUBLISHED states and compares; agreement between a value
read out of the parameter set and one computed from the published `bw` is evidence, where
either alone would only be a restatement.
"""
function native_diagnostics(physical, case::AbstractDict, arrays::AbstractDict)
    rmodel = physical.model.models[:Reservoir]
    neighbors = rmodel.data_domain[:neighbors]
    gdz = collect(Float64.(physical.parameters[:Reservoir][:TwoPointGravityDifference]))
    evaluated = ADAPTER.evaluated_state(physical.model, physical.parameters, physical.state0)
    reservoir = evaluated[:Reservoir]
    pressure = collect(Float64.(reservoir[:Pressure]))
    densities = reservoir[:PhaseMassDensities]
    saturations = reservoir[:Saturations]
    centroids = rmodel.data_domain[:cell_centroids]

    faces = Any[]
    for f in axes(neighbors, 2)
        l, r = Int(neighbors[1, f]), Int(neighbors[2, f])
        rho_face = [0.5 * (densities[ph, l] + densities[ph, r]) for ph in axes(densities, 1)]
        push!(
            faces,
            Dict{String,Any}(
                "face" => f - 1,
                "left_cell" => l - 1,
                "right_cell" => r - 1,
                "dz_m" => centroids[3, r] - centroids[3, l],
                "gdz" => gdz[f],
                "pressure_difference_pa" => pressure[r] - pressure[l],
                "face_density_kg_m3" => rho_face,
                # The quantity the phase flux is driven by, per phase, up to -T.
                "potential_residual_pa" =>
                    [(pressure[r] - pressure[l]) + gdz[f] * rho for rho in rho_face],
            ),
        )
    end
    return Dict{String,Any}(
        "cell_center_depth_m" => collect(Float64.(centroids[3, :])),
        "pressure_pa" => pressure,
        "phase_densities_kg_m3" => rows_of(densities),
        "saturations" => rows_of(saturations),
        "pore_volume_m3" => collect(Float64.(JutulDarcy.pore_volume(physical.model, physical.parameters))),
        "faces" => faces,
    )
end

# ======================================================================================
# the diagnostic (plan Task 9.8). Only runs as a program.
# ======================================================================================

"""Largest absolute per-step balance residual of one extraction, over both statements."""
function worst_balance_residual(extraction::AbstractDict)
    surface = balance_residual(
        extraction["inventory_m3_sc"], extraction["net_surface_source_m3_sc"],
    )
    connection = balance_residual(
        extraction["reservoir_inventory_m3_sc"], extraction["net_connection_source_m3_sc"],
    )
    return maximum(
        maximum(maximum(abs.(r)) for r in residuals) for residuals in (surface, connection)
    )
end

function selftest_analytic()
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "gravity_constant" => Jutul.gravity_constant,
    )
    fixtures = Dict{String,Any}()

    @testset "E01 analytic oil-water verification fixtures" begin
        # --- 9.5 the closed / PVT fixtures ---------------------------------------------
        for (key, name, n_cells) in
            (("closed_cell_pvt", :closed_cell_pvt, 1), ("closed_box_pvt", :closed_box_pvt, 8))
            run = run_analytic(name, Dict{String,Any}("timesteps" => :none))
            @test run["status"] == "COMPLETE"
            extraction = run["extraction"]
            @test length(extraction["chunk"]["dt_s"]) == length(CLOSED_EDGES_DAYS) - 1
            @test sum(extraction["chunk"]["dt_s"]) ≈ CLOSED_EDGES_DAYS[end] * SECONDS_PER_DAY
            # A single layer: no face carries any gravity head at all, which is what makes a
            # uniform state an exact steady state rather than an approximate one.
            # One layer along x: n-1 faces, every one of them horizontal, so the native
            # gravity head is zero across all of them. On the single cell there are none at
            # all, and the count is asserted so that "all of them" is not vacuously true.
            @test length(run["native"]["faces"]) == n_cells - 1
            @test all(f["gdz"] == 0.0 for f in run["native"]["faces"])
            @test length(run["native"]["cell_center_depth_m"]) == n_cells
            # Source-free: every connection of the shut monitoring well is closed and the
            # native cross term through it is an exact zero, not a small number.
            @test !isempty(extraction["connections"])
            @test all(!c["connection_open"] for c in extraction["connections"])
            @test all(c["total_mass_kg_s"] == 0.0 for c in extraction["connections"])
            @test worst_balance_residual(extraction) < 1.0e-6
            fixtures[key] = run
        end

        # --- 9.5 a PVT that cannot describe a fluid, refused BEFORE the solver -----------
        broken = run_analytic(:broken_pvt)
        @test broken["phase"] == "build"
        @test broken["status"] == "PHYSICALLY_INVALID"
        # Not INVALID_INPUT: every field is present, of the right type and of the right
        # length, so the case is well formed and only its physics is impossible.
        @test broken["is_invalid_case_input"] == false
        @test occursin("density_sc_kg_m3", broken["reason"])
        @test !haskey(broken, "extraction")
        # And the same case with a positive density does build, so the refusal is the density.
        @test run_analytic(:closed_cell_pvt, Dict{String,Any}("timesteps" => :none))["status"] ==
              "COMPLETE"
        fixtures["broken_pvt"] = broken

        # --- 9.6 the Buckley-Leverett refinement ----------------------------------------
        for nx in (32, 64, 128)
            options = bl_solver_options()
            options["nx"] = nx
            run = run_analytic(:bl, options)
            @test run["status"] == "COMPLETE"
            extraction = run["extraction"]
            dt = extraction["chunk"]["dt_s"]
            report_dt = BL_HORIZON_DAYS * SECONDS_PER_DAY / BL_REPORT_STEPS
            # Every accepted substep is the report step or its half, and nothing else.
            @test all(d -> isapprox(d, report_dt) || isapprox(d, report_dt / 2), dt)
            @test sum(dt) ≈ BL_HORIZON_DAYS * SECONDS_PER_DAY
            @test extraction["control_infeasible_reason"] === nothing
            # The injector really held the rate the reference is written for.
            injected = sum(
                -w["surface_water_m3_s"][k] * dt[k] for (name, w) in extraction["wells"]
                if name == "INJ1" for k in eachindex(dt)
            )
            @test injected ≈ -BL_INJECTED_PV * BL_PORE_VOLUME_M3 rtol = 1.0e-6
            # The front has not reached the outlet: the reference is a pre-breakthrough one.
            @test extraction["states"]["sw"][end][end] < 1.0e-6
            @test worst_balance_residual(extraction) < 1.0e-4
            fixtures["bl_$(nx)"] = run
        end

        # --- 9.7 the hydrostatic column -------------------------------------------------
        column = run_analytic(:hydrostatic, Dict{String,Any}("timesteps" => :none))
        @test column["status"] == "COMPLETE"
        native = column["native"]
        @test native["cell_center_depth_m"] == [DATUM_M + 25.0, DATUM_M + 75.0]
        # z is depth, positive down: the deeper cell is at the higher pressure.
        @test native["pressure_pa"][2] > native["pressure_pa"][1]
        @test length(native["faces"]) == 1
        face = native["faces"][1]
        @test face["dz_m"] ≈ 50.0
        @test face["gdz"] ≈ -STANDARD_GRAVITY_M_S2 * 50.0
        # The discrete face residual BEFORE the first timestep: a real number, measured from
        # the native gravity parameter and the model's own densities, not asserted to be zero.
        head = abs(face["gdz"] * face["face_density_kg_m3"][1])
        @test abs(face["potential_residual_pa"][1]) / head < 1.0e-5
        # The EXTENT of the single-phase defect `hydrostatic_case` documents, pinned rather
        # than described. `run_forward` runs at `info_level = -1`, so the cut messages are
        # suppressed and the only trace is an unconditional `@warn` on a subprocess's stderr
        # that the launcher discards — a prose note about "two cuts" would let two become
        # twenty, or the degeneracy change character on a dependency bump, entirely in
        # silence. The solver's own counters are in the payload and are asserted exactly:
        # this is the fixture that defines verified physics for the rest of the plan, so its
        # known fragility is the thing that has to break first.
        solver = column["extraction"]["solver"]
        @test solver["cut_steps"] == SINGLE_PHASE_CUT_STEPS
        @test solver["accepted_steps"] == SINGLE_PHASE_ACCEPTED_STEPS
        @test length(column["extraction"]["chunk"]["dt_s"]) == SINGLE_PHASE_ACCEPTED_STEPS
        # Whatever the cuts, the accepted substeps still tile the schedule exactly and every
        # report edge is still one of their boundaries (`requested_states` refuses otherwise).
        @test sum(column["extraction"]["chunk"]["dt_s"]) ≈ CLOSED_EDGES_DAYS[end] * SECONDS_PER_DAY
        measured["single_phase_degeneracy"] = Dict{String,Any}(
            "fixture" => "hydrostatic",
            "report_steps" => length(CLOSED_EDGES_DAYS) - 1,
            "cut_steps" => solver["cut_steps"],
            "accepted_steps" => solver["accepted_steps"],
            "nonlinear_iterations" => solver["nonlinear_iterations"],
            "expected_cut_steps" => SINGLE_PHASE_CUT_STEPS,
            "expected_accepted_steps" => SINGLE_PHASE_ACCEPTED_STEPS,
        )
        # And no artificial drift afterwards.
        states = column["extraction"]["states"]
        @test maximum(
            maximum(abs.(states["sw"][k] .- states["sw"][1])) for k in eachindex(states["sw"])
        ) < 1.0e-6
        fixtures["hydrostatic"] = column

        # --- 9.7 gravity segregation ----------------------------------------------------
        segregation = run_analytic(:segregation, Dict{String,Any}("timesteps" => :none))
        @test segregation["status"] == "COMPLETE"
        seg_states = segregation["extraction"]["states"]
        depths = segregation["native"]["cell_center_depth_m"]
        pore = segregation["native"]["pore_volume_m3"]
        water_depth(sw) = sum(sw .* pore .* depths) / sum(sw .* pore)
        start_depth = water_depth(seg_states["sw"][1])
        end_depth = water_depth(seg_states["sw"][end])
        # DEMONSTRATED, not asserted by construction: the heavy phase started above the light
        # one and its centre of mass moved DOWN.
        @test seg_states["sw"][1] == SEGREGATION_SW
        @test end_depth > start_depth
        fixtures["segregation"] = segregation
    end

    measured["fixtures"] = fixtures
    measured["status"] = "ok"
    return measured
end

"""Parse the diagnostic to run and the optional `--out` the project launcher appends."""
function parse_analytic_args(args::Vector{String})
    out = nothing
    requested = false
    i = 1
    while i <= length(args)
        if args[i] == "--test-analytic"
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

function analytic_main(args::Vector{String})
    requested, out = parse_analytic_args(args)
    requested || error(
        "nothing to do: pass --test-analytic to run the analytic oil-water verification " *
        "fixtures of plan Task 9",
    )
    payload = try
        selftest_analytic()
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
    analytic_main(collect(ARGS))
    println("so-recon verification: analytic diagnostic passed")
end
