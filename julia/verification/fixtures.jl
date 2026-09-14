# E01.0 — verification fixtures for the native oil-water model, and the standalone
# diagnostic that checks the constructor against them.
#
# Two uses, one file. `verification_case` is a plain definition: including this file from a
# verification script, from a later gate or from the worker gives you the fixtures and
# nothing else — no tests run, nothing is printed, no adapter is loaded. Running the file
# AS A PROGRAM with `--test-model` (the constructor, plan Task 5.2) or `--test-controls`
# (the calendar, the controls and the completion events, plan Task 6.6) additionally loads
# the adapter and executes the checks below. These are addressed diagnostics, not a serial
# gate: the gate drives the persistent worker instead of paying a cold Julia start per
# assertion, and each diagnostic pays exactly one cold start for all of its claims.
#
# A fixture here is FULLY MATERIALIZED. `verification_case` returns the case mapping in the
# exchange's own JSON shape plus the arrays that a real case would reference through
# `ArrayRef`s, already in the semantic layout `build_ow` expects. Nothing is read from disk
# and no external dataset is downloaded: a verification fixture whose meaning depended on
# somebody's copy of an SPE deck would not be a verification of anything.
#
# Numbers come from the plan's §3.1 educational PVT/Corey block. They are duplicated here
# because Julia cannot import a Pydantic default; the Python physics test compares this
# file's `educational_fluids()` against `FluidSpec()` field by field, so the duplication is
# checked rather than trusted.

using Test
using Jutul
using JutulDarcy
using JSON

#: The exchange's conversion, spelled the same way on both sides (plan 3.1).
const MILLIDARCY_M2 = 9.869233e-16
const SECONDS_PER_DAY = 86400.0
const STANDARD_GRAVITY_M_S2 = 9.80665

#: Every fixture cell is a 10 m cube, so a pore volume is a number one can check by hand.
const CELL_EDGE_M = 10.0

#: The depth of the top face of every fixture box. z is depth, positive down, and ABSOLUTE:
#: a fixture origined at zero would let a grid-origin bug through unseen, because "the model
#: ignored the declared centres" and "the declared centres were at the origin anyway" would
#: produce the same numbers. A thousand metres is an ordinary reservoir datum and makes the
#: two cases distinguishable.
const DATUM_M = 1000.0

"""
    cartesian_cell_centers(nx, ny, nz) -> Matrix (n_cells, 3)

The `(cell, dim)` centre array the exchange declares, for a box of 10 m cubes whose top face
is at `DATUM_M`. Cells are laid out `cell_id = i + nx*(j + ny*k)`, zero-based, exactly as
`case_io.cartesian_neighbors` does it on the Python side.
"""
function cartesian_cell_centers(nx::Int, ny::Int, nz::Int)
    centers = Matrix{Float64}(undef, nx * ny * nz, 3)
    for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        cell = i + nx * (j + ny * k) + 1
        centers[cell, 1] = CELL_EDGE_M * i + 0.5 * CELL_EDGE_M
        centers[cell, 2] = CELL_EDGE_M * j + 0.5 * CELL_EDGE_M
        centers[cell, 3] = DATUM_M + CELL_EDGE_M * k + 0.5 * CELL_EDGE_M
    end
    return centers
end

#: The grid size the family of fixtures defaults to. `:closed_cell` is a single cell and
#: refuses any other size rather than quietly ignoring the request.
const DEFAULT_NX = 16
const DEFAULT_NZ = 1

"""
    educational_fluids()

The plan's §3.1 two-phase oil-water model, in the JSON field names the exchange uses.
Tuples are ordered (water, oil) — the same order as the phases of the system the adapter
builds, which is what makes `viscosity_pa_s[1]` the water viscosity and not a coincidence.
"""
function educational_fluids()
    return Dict{String,Any}(
        "kind" => "OW",
        "density_sc_kg_m3" => [1000.0, 800.0],
        "viscosity_pa_s" => [0.001, 0.003],
        "compressibility_pa_inv" => [4.0e-10, 1.0e-9],
        "p_sc_pa" => 101325.0,
        "t_sc_k" => 288.15,
        "corey_exponents" => [2.0, 2.0],
        "residual_saturations" => [0.2, 0.2],
        "kr_endpoints" => [1.0, 1.0],
        "pc_model" => "zero",
        "educational" => true,
        "analytical_limit" => false,
    )
end

"""
    verification_case(name::Symbol; nx = 16, nz = 1) -> (case, arrays)

One small fixture, fully materialized: `(case, arrays)` is exactly what `build_ow` takes.

Implemented names:

* `:closed_cell` — a single 10 m cube with its top face at `DATUM_M`, porosity 0.2, 100 mD
  isotropic, 1.5e7 Pa, Sw 0.3, no wells, one report interval of a day. Its pore volume is
  1000 m³ × 0.2 = 200 m³, which is the number the constructor check below asserts, and its
  centre is at 1005 m depth, which is where the built model has to put it. The fixture is
  one cell BY DEFINITION, so
  asking it for a different `nx` or `nz` is an error rather than a request it ignores;
  those arguments shape the grid fixtures (`:bl`, `:hydrostatic`, `:two_layer`,
  `:boundary`, `:five_spot`) that plan Tasks 9–10 add.
* `:two_interval_controls` — the calendar fixture of plan Task 6: a closed 4×1×2 box with
  one injector and one producer, driven over two REAL months (January and the 29-day
  February of 2020) by four control intervals. It carries a rate control, a completion
  event inside a month, a field shutdown and a bhp control, which is what makes it the
  case the control mapping is checked against. It is 4×1×2 by definition and refuses a
  resize for the same reason `:closed_cell` does.

Any other name is an explicit error. A fixture whose physics has not been decided is not
stubbed with plausible-looking numbers here: a silently wrong reference case is worse than
a missing one, because every later comparison would inherit it.
"""
function verification_case(name::Symbol; nx::Int = DEFAULT_NX, nz::Int = DEFAULT_NZ)
    if name === :closed_cell
        return closed_cell_case(nx, nz)
    elseif name === :two_interval_controls
        return two_interval_controls_case(nx, nz)
    end
    error(
        "verification_case: fixture $(repr(name)) is not implemented in this build. " *
        "Only :closed_cell and :two_interval_controls exist so far; :bl, :hydrostatic, " *
        ":two_layer, :boundary and :five_spot are named by the plan but their physics is " *
        "decided in Tasks 9-10, and a stub would be a wrong reference case rather than a " *
        "missing one.",
    )
end

function closed_cell_case(nx::Int, nz::Int)
    (nx == DEFAULT_NX && nz == DEFAULT_NZ) || error(
        "verification_case: :closed_cell is a single cell by definition and cannot be " *
        "resized (asked for nx=$(nx), nz=$(nz)); nx and nz shape the grid fixtures Tasks " *
        "9-10 add, and silently returning one cell would make every later comparison wrong.",
    )
    shape = [1, 1, 1]
    n_cells = 1
    case = Dict{String,Any}(
        "schema_version" => "case-1",
        "case_id" => "verification-closed-cell",
        "grid" => Dict{String,Any}(
            "shape" => shape,
            "extent_m" => [CELL_EDGE_M, CELL_EDGE_M, CELL_EDGE_M],
            "z_positive" => "down",
        ),
        # Constant pore volume, declared rather than assumed (plan 3.1).
        "rock" => Dict{String,Any}("rock_compressibility_pa_inv" => 0.0),
        "fluids" => educational_fluids(),
        "wells" => Any[],
        "controls" => Any[],
        "boundary" => Dict{String,Any}("kind" => "closed", "cells" => Any[]),
        "report_edges_s" => [0.0, SECONDS_PER_DAY],
        "gravity_m_s2" => STANDARD_GRAVITY_M_S2,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => cartesian_cell_centers(1, 1, 1),
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, n_cells),
        "pressure_pa" => fill(1.5e7, n_cells),
        "sw" => fill(0.3, n_cells),
    )
    return (case, arrays)
end

# --------------------------------------------------------------------------------------
# :two_interval_controls — the calendar fixture of plan Task 6
# --------------------------------------------------------------------------------------

#: The fixture's box: four cells along x, two layers, all of them 10 m cubes.
const CONTROLS_NX = 4
const CONTROLS_NZ = 2

#: Real months, not an average one. January 2020 is 31 days and February 2020 is 29,
#: because 2020 is a leap year; the Python side derives the same two numbers from
#: `so_recon.simulator.schedule.month_edges` and the integration test compares them.
const CONTROLS_MONTH_DAYS = (31.0, 29.0)

#: Where the four control intervals begin and end, in days from the start date. Day 45 is
#: the completion event and day 52 the field shutdown: both are INSIDE February, and a
#: schedule that rounded them to a month boundary would lose them.
const CONTROLS_EVENT_DAYS = (0.0, 31.0, 45.0, 52.0, 60.0)

#: The rate both wells run on while they run at all, m3_sc/day. Small enough that the
#: producer reaches it against a bottom-hole limit 150 bar below the initial pressure —
#: the identity `Vo + Vw = q_liquid * uptime` is only meaningful where the rate is reached.
const CONTROLS_RATE_M3_DAY = 2.0

#: The producer's recorded bottom-hole floor and the injector's ceiling, Pa. These are the
#: ONLY limits the fixture has: `build_forces` disables JutulDarcy's own defaults.
const CONTROLS_PRODUCER_BHP_FLOOR_PA = 5.0e6
const CONTROLS_INJECTOR_BHP_CEILING_PA = 4.0e7

#: The bhp both wells are put on for the last interval: the producer below the initial
#: pressure so that it really produces under it, and the injector above it so that it
#: really injects. The injector's bhp interval is what builds
#: `InjectorControl(BottomHolePressureTarget(...), mixture; density)`, a constructor no
#: rate control reaches — a rate control that later switches onto its bhp limit goes
#: through `replace_target`, which INHERITS the mixture and the density instead.
const CONTROLS_PRODUCER_BHP_TARGET_PA = 1.9e7
const CONTROLS_INJECTOR_BHP_TARGET_PA = 2.2e7

const CONTROLS_INITIAL_PRESSURE_PA = 2.0e7

"""
    control_segment(; kwargs...) -> Dict

One `ControlSegment` in the exchange's own JSON shape (plan 3.2). Times are seconds from
the case start and `value` is the HUMAN quantity: m3_sc/day for a rate, Pa for a bhp.
"""
function control_segment(;
    start_day::Real,
    end_day::Real,
    well_id::AbstractString,
    role::AbstractString,
    target::AbstractString,
    value::Real,
    bhp_limit_pa::Union{Nothing,Real},
    connection_open::Vector{Bool},
)
    return Dict{String,Any}(
        "start_s" => Float64(start_day) * SECONDS_PER_DAY,
        "end_s" => Float64(end_day) * SECONDS_PER_DAY,
        "well_id" => String(well_id),
        "role" => String(role),
        "target" => String(target),
        "value" => Float64(value),
        "bhp_limit_pa" => bhp_limit_pa === nothing ? nothing : Float64(bhp_limit_pa),
        "connection_open" => Any[connection_open...],
    )
end

"""
    two_interval_controls_case(nx, nz) -> (case, arrays)

A closed 4×1×2 box with an injector on the i=0 column and a producer on the i=3 column,
driven over January and February 2020 by four control intervals:

| interval | days | INJ1 | PRO1 |
|---|---|---|---|
| 0 | 0–31 | water rate, both open | liquid rate, both open |
| 1 | 31–45 | water rate, both open | liquid rate, LOWER completion shut |
| 2 | 45–52 | shut, isolated | shut, isolated |
| 3 | 52–60 | bhp, both open | bhp, both open |

Interval 0 is the rate branch, interval 3 the bhp branch for BOTH roles, interval 1 a
completion event inside a month and interval 2 a month with part of its uptime removed.
Every interval restates the role and the mask of both wells: nothing is inherited.
"""
function two_interval_controls_case(nx::Int, nz::Int)
    (nx == DEFAULT_NX && nz == DEFAULT_NZ) || error(
        "verification_case: :two_interval_controls is a 4x1x2 box by definition and cannot " *
        "be resized (asked for nx=$(nx), nz=$(nz)).",
    )
    n_cells = CONTROLS_NX * CONTROLS_NZ
    # cell_id = i + nx*(j + ny*k), zero-based: 0 and 4 are the i=0 column, 3 and 7 the i=3
    # column, each of them one cell in the upper layer and one in the lower.
    injector_cells = [0, CONTROLS_NX]
    producer_cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1]
    both = Bool[true, true]
    upper_only = Bool[true, false]
    isolated = Bool[false, false]
    d0, d1, d2, d3, d4 = CONTROLS_EVENT_DAYS

    injecting(a, b, mask) = control_segment(
        start_day = a,
        end_day = b,
        well_id = "INJ1",
        role = "injector",
        target = "water_rate",
        value = CONTROLS_RATE_M3_DAY,
        bhp_limit_pa = CONTROLS_INJECTOR_BHP_CEILING_PA,
        connection_open = mask,
    )
    producing(a, b, mask) = control_segment(
        start_day = a,
        end_day = b,
        well_id = "PRO1",
        role = "producer",
        target = "liquid_rate",
        value = CONTROLS_RATE_M3_DAY,
        bhp_limit_pa = CONTROLS_PRODUCER_BHP_FLOOR_PA,
        connection_open = mask,
    )
    shut(a, b, well) = control_segment(
        start_day = a,
        end_day = b,
        well_id = well,
        role = "shut",
        target = "disabled",
        value = 0.0,
        bhp_limit_pa = nothing,
        connection_open = isolated,
    )

    case = Dict{String,Any}(
        "schema_version" => "case-1",
        "case_id" => "verification-two-interval-controls",
        "start_date" => "2020-01-01",
        "grid" => Dict{String,Any}(
            "shape" => [CONTROLS_NX, 1, CONTROLS_NZ],
            "extent_m" =>
                [CELL_EDGE_M * CONTROLS_NX, CELL_EDGE_M, CELL_EDGE_M * CONTROLS_NZ],
            "z_positive" => "down",
        ),
        "rock" => Dict{String,Any}("rock_compressibility_pa_inv" => 0.0),
        "fluids" => educational_fluids(),
        "wells" => Any[
            well_spec("INJ1", injector_cells),
            well_spec("PRO1", producer_cells),
        ],
        "controls" => Any[
            injecting(d0, d1, both),
            injecting(d1, d2, both),
            shut(d2, d3, "INJ1"),
            # The injector's own bhp branch, which no rate control can reach.
            control_segment(
                start_day = d3,
                end_day = d4,
                well_id = "INJ1",
                role = "injector",
                target = "bhp",
                value = CONTROLS_INJECTOR_BHP_TARGET_PA,
                bhp_limit_pa = nothing,
                connection_open = both,
            ),
            producing(d0, d1, both),
            # The completion event: the lower perforation is closed from day 45, which is
            # inside February and must stay inside February.
            producing(d1, d2, upper_only),
            shut(d2, d3, "PRO1"),
            control_segment(
                start_day = d3,
                end_day = d4,
                well_id = "PRO1",
                role = "producer",
                target = "bhp",
                value = CONTROLS_PRODUCER_BHP_TARGET_PA,
                bhp_limit_pa = nothing,
                connection_open = both,
            ),
        ],
        "boundary" => Dict{String,Any}("kind" => "closed", "cells" => Any[]),
        # Two REAL months: 31 days and the leap February's 29, never two months of 30.
        "report_edges_s" => [
            0.0,
            CONTROLS_MONTH_DAYS[1] * SECONDS_PER_DAY,
            sum(CONTROLS_MONTH_DAYS) * SECONDS_PER_DAY,
        ],
        # The control-rate unit travels with the case (plan 3.1); Julia divides by 86400.
        "units" => Dict{String,Any}("control_rate" => "m3_sc/day"),
        "gravity_m_s2" => STANDARD_GRAVITY_M_S2,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => cartesian_cell_centers(CONTROLS_NX, 1, CONTROLS_NZ),
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, n_cells),
        "pressure_pa" => fill(CONTROLS_INITIAL_PRESSURE_PA, n_cells),
        "sw" => fill(0.3, n_cells),
    )
    return (case, arrays)
end

"""One `WellSpec` in the exchange's JSON shape, with the crossflow the backend can express."""
function well_spec(well_id::AbstractString, cells::Vector{Int}; model::AbstractString = "multisegment")
    return Dict{String,Any}(
        "well_id" => String(well_id),
        "cells" => cells,
        "radius_m" => 0.1,
        "reference_depth_m" => DATUM_M,
        "model" => String(model),
        # The native wellbore always couples its connections; `check_crossflow` refuses the
        # other declaration over more than one connection rather than pretending otherwise.
        "allow_crossflow" => true,
    )
end

# ======================================================================================
# Standalone diagnostic (plan 5.2/5.7). Nothing below runs on a plain `include`.
# ======================================================================================

"""
    selftest_wellbore_case()

A construction probe, not a registered fixture: a closed 4×1×2 box of the same educational
fluid, with its top face at `DATUM_M` and one simple and one multisegment well, used only to
exercise the plumbing that a single cell cannot reach — the per-submodel viscosity
parameter, the face gravity of a vertical connection and a well datum that is not the
perforation depth. It carries no controls and makes no physical claim, so it is deliberately
NOT reachable through `verification_case`, whose names later tasks pin to reference physics.
"""
function selftest_wellbore_case()
    nx, nz = 4, 2
    n_cells = nx * nz
    case = Dict{String,Any}(
        "schema_version" => "case-1",
        "case_id" => "selftest-wellbore-probe",
        "grid" => Dict{String,Any}(
            "shape" => [nx, 1, nz],
            "extent_m" => [CELL_EDGE_M * nx, CELL_EDGE_M, CELL_EDGE_M * nz],
            "z_positive" => "down",
        ),
        "rock" => Dict{String,Any}("rock_compressibility_pa_inv" => 0.0),
        "fluids" => educational_fluids(),
        "wells" => Any[
            # cell_id = i + nx*(j + ny*k), zero-based in the exchange: 0 and 4 are the two
            # cells of the column at i=0, which is a vertical two-node well.
            # The datum is the top face of the reservoir box, 5 m above the shallowest
            # perforation centre — a wellhead reference, not a number that has drifted away
            # from the cells it belongs to.
            Dict{String,Any}(
                "well_id" => "INJ1",
                "cells" => [0, 4],
                "radius_m" => 0.1,
                "reference_depth_m" => DATUM_M,
                "model" => "multisegment",
                # True because the native wellbore couples its connections and cannot be
                # told not to; `check_crossflow` refuses the other declaration over more
                # than one connection rather than simulating the opposite semantics.
                "allow_crossflow" => true,
            ),
            Dict{String,Any}(
                "well_id" => "PRO1",
                "cells" => [3],
                "radius_m" => 0.1,
                "reference_depth_m" => DATUM_M,
                "model" => "simple",
                "allow_crossflow" => false,
            ),
        ],
        "controls" => Any[],
        "boundary" => Dict{String,Any}("kind" => "closed", "cells" => Any[]),
        "report_edges_s" => [0.0, SECONDS_PER_DAY],
        "gravity_m_s2" => STANDARD_GRAVITY_M_S2,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => cartesian_cell_centers(nx, 1, nz),
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, n_cells),
        "pressure_pa" => fill(1.5e7, n_cells),
        "sw" => fill(0.3, n_cells),
    )
    return (case, arrays)
end

"""Read a phase-indexed parameter column as a plain vector, for reporting."""
phase_column(x::AbstractMatrix) = Float64[x[i, 1] for i in axes(x, 1)]

"""Evaluate the model's OWN density variable, through the dispatch the simulator uses."""
function native_densities(model, pressures::Vector{Float64})
    rmodel = model.models[:Reservoir]
    rho_def = Jutul.get_secondary_variables(rmodel)[:PhaseMassDensities]
    rho = zeros(Float64, 2, length(pressures))
    Jutul.update_secondary_variable!(rho, rho_def, rmodel, (Pressure = pressures,), eachindex(pressures))
    return rho
end

function selftest(adapter::Module)
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "gravity_constant" => Jutul.gravity_constant,
        "fluids" => educational_fluids(),
    )

    @testset "E01 native oil-water constructor" begin
        # --- 5.1 the constructor check the plan spells out -----------------------------
        case, arrays = verification_case(:closed_cell)
        physical = adapter.build_ow(case, arrays)
        @test all(physical.state0[:Reservoir][:Pressure] .> 0.0)
        @test sum(physical.state0[:Reservoir][:Saturations][:, 1]) ≈ 1.0
        @test sum(pore_volume(physical.model, physical.parameters)) ≈ 200.0

        # --- 5.3 geometry and phase model ----------------------------------------------
        rmodel = physical.model.models[:Reservoir]
        domain = JutulDarcy.reservoir_domain(physical.model)
        sys = rmodel.system
        phases = collect(string.(typeof.(JutulDarcy.get_phases(sys))))
        @test phases == ["AqueousPhase", "LiquidPhase"]
        @test number_of_cells(rmodel.domain) == 1
        @test domain[:porosity] == arrays["porosity"]
        @test domain[:permeability] == arrays["permeability_m2"]
        @test physical.state0[:Reservoir][:Pressure] == arrays["pressure_pa"]
        # The model sits at the datum the case declared, not at the mesh's own origin. A
        # grid origined at zero would put this cell's centre at 5 m instead of 1005 m.
        centroids = domain[:cell_centroids]
        @test permutedims(centroids, (2, 1)) ≈ arrays["cell_centers_m"]
        @test centroids[3, 1] == DATUM_M + 0.5 * CELL_EDGE_M
        # Water first, oil second: a swapped phase order would still sum to one.
        @test physical.state0[:Reservoir][:Saturations][1, 1] == 0.3
        @test physical.state0[:Reservoir][:Saturations][2, 1] == 0.7

        kr = Jutul.get_secondary_variables(rmodel)[:RelativePermeabilities]
        @test kr isa BrooksCoreyRelativePermeabilities
        @test collect(kr.exponents) == [2.0, 2.0]
        @test collect(kr.residuals) == [0.2, 0.2]
        @test collect(kr.endpoints) == [1.0, 1.0]

        rho_def = Jutul.get_secondary_variables(rmodel)[:PhaseMassDensities]
        @test rho_def isa ConstantCompressibilityDensities
        @test collect(rho_def.reference_densities) == [1000.0, 800.0]
        @test collect(rho_def.reference_pressure) == [101325.0, 101325.0]
        @test collect(rho_def.compressibility) == [4.0e-10, 1.0e-9]

        measured["closed_cell"] = Dict{String,Any}(
            "n_cells" => number_of_cells(rmodel.domain),
            "phases" => phases,
            "pore_volume_m3" => sum(pore_volume(physical.model, physical.parameters)),
            "pressure_pa" => collect(physical.state0[:Reservoir][:Pressure]),
            "saturations" => collect(eachrow(physical.state0[:Reservoir][:Saturations])),
            "permeability_m2" => collect(domain[:permeability][:, 1]),
            "declared_cell_centers_m" => collect(eachrow(arrays["cell_centers_m"])),
            "mesh_cell_centers_m" => collect(eachcol(centroids)),
        )

        # --- the datum is validated, not assumed ---------------------------------------
        # A declared centre the shape and extent cannot produce is refused by name, with the
        # offending cell, axis and distance in the message. Ignoring `cell_centers_m` would
        # make this pass silently, which is the failure mode the check exists for.
        #
        # It has to be a MULTI-cell case: the origin is derived from cell 0, so a grid of one
        # cell has nothing left to contradict and the check is vacuous there by construction.
        bad_case, bad_arrays = selftest_wellbore_case()
        bad_arrays["cell_centers_m"][6, 1] += 2.5  # zero-based cell 5, x axis
        refusal = try
            adapter.build_ow(bad_case, bad_arrays)
            nothing
        catch err
            err
        end
        @test refusal isa adapter.InvalidCaseInput
        refusal_message = refusal === nothing ? "" : sprint(showerror, refusal)
        @test occursin("cell_centers_m", refusal_message)
        @test occursin("zero-based cell 5", refusal_message)
        @test occursin("x axis", refusal_message)
        @test occursin("2.5", refusal_message)
        measured["rejected_geometry_message"] = refusal_message

        # --- 5.5 SI and PVT, against numbers a person can check ------------------------
        p_sc = 101325.0
        pressures = [p_sc, 1.5e7]
        rho = native_densities(physical.model, pressures)
        rho_sc = [1000.0, 800.0]
        compressibility = [4.0e-10, 1.0e-9]
        expected = [rho_sc[ph] * exp(compressibility[ph] * (pressures[i] - p_sc)) for ph in 1:2, i in 1:2]
        @test rho ≈ expected
        @test all(rho .> 0.0)
        # B = rho_sc/rho: exactly one at the reference pressure, and below one above it.
        @test rho[1, 1] == 1000.0 && rho[2, 1] == 800.0
        @test rho_sc[1] / rho[1, 1] == 1.0 && rho_sc[2] / rho[2, 1] == 1.0
        @test rho[1, 2] > rho[1, 1] && rho[2, 2] > rho[2, 1]
        @test rho_sc[1] / rho[1, 2] < 1.0 && rho_sc[2] / rho[2, 2] < 1.0
        # Oil is the lighter phase and the more viscous one; neither pair is swapped.
        @test rho[1, 1] > rho[2, 1]
        mu = phase_column(physical.parameters[:Reservoir][:PhaseViscosities])
        @test mu == [0.001, 0.003]
        @test mu[2] / mu[1] == 3.0

        measured["pvt"] = Dict{String,Any}(
            "p_pa" => pressures,
            "rho_w_kg_m3" => collect(rho[1, :]),
            "rho_o_kg_m3" => collect(rho[2, :]),
            "b_w" => [rho_sc[1] / rho[1, i] for i in 1:2],
            "b_o" => [rho_sc[2] / rho[2, i] for i in 1:2],
            "viscosity_pa_s" => mu,
            "viscosity_ratio" => mu[2] / mu[1],
        )

        # --- 5.4 viscosity on every submodel, gravity, constant pore volume -------------
        wcase, warrays = selftest_wellbore_case()
        probe = adapter.build_ow(wcase, warrays)
        viscosities = Dict{String,Any}()
        non_flow = String[]
        for (name, submodel) in pairs(probe.model.models)
            if name == :Reservoir || JutulDarcy.model_or_domain_is_well(submodel)
                viscosities[string(name)] =
                    phase_column(probe.parameters[name][:PhaseViscosities])
            else
                push!(non_flow, string(name))
                @test !haskey(probe.parameters[name], :PhaseViscosities)
            end
        end
        @test sort(collect(keys(viscosities))) == ["INJ1", "PRO1", "Reservoir"]
        @test all(v -> v == [0.001, 0.003], values(viscosities))
        # The filter is a filter, not a tautology: the facility carries no phase viscosity.
        @test non_flow == ["Facility"]

        @test Jutul.gravity_constant == STANDARD_GRAVITY_M_S2
        @test wcase["gravity_m_s2"] == Jutul.gravity_constant
        wdomain = JutulDarcy.reservoir_domain(probe.model)
        neighbors = wdomain[:neighbors]
        z = vec(wdomain[:cell_centroids][3, :])
        # Absolute depth again, this time over two layers: 1005 m and 1015 m, not 5 and 15.
        @test permutedims(wdomain[:cell_centroids], (2, 1)) ≈ warrays["cell_centers_m"]
        @test sort(unique(z)) == [DATUM_M + 0.5 * CELL_EDGE_M, DATUM_M + 1.5 * CELL_EDGE_M]
        native_gdz = compute_face_gdz(neighbors, z)
        gdz = probe.parameters[:Reservoir][:TwoPointGravityDifference]
        @test gdz == native_gdz
        # z is depth, positive down, and gdz = -g*(z_r - z_l): a face between a shallow
        # cell and the cell below it is negative and exactly one cell edge of head, while
        # a face inside a layer has no z difference and therefore none of it.
        vertical = [i for i in eachindex(gdz) if z[neighbors[2, i]] != z[neighbors[1, i]]]
        horizontal = [i for i in eachindex(gdz) if z[neighbors[2, i]] == z[neighbors[1, i]]]
        @test !isempty(vertical) && !isempty(horizontal)
        @test all(gdz[i] == -STANDARD_GRAVITY_M_S2 * CELL_EDGE_M for i in vertical)
        @test all(gdz[i] == 0.0 for i in horizontal)

        # Pore volume is a PARAMETER and stays one: no pressure-dependent replacement, so
        # the same numbers come back after the pressure moves.
        wres = probe.model.models[:Reservoir]
        @test haskey(Jutul.get_parameters(wres), :FluidVolume)
        @test !haskey(Jutul.get_secondary_variables(wres), :FluidVolume)
        @test !haskey(Jutul.get_secondary_variables(wres), :StaticFluidVolume)
        @test wcase["rock"]["rock_compressibility_pa_inv"] == 0.0
        pv_before = copy(pore_volume(probe.model, probe.parameters))
        probe.state0[:Reservoir][:Pressure] .= 3.0e7
        pv_after = pore_volume(probe.model, probe.parameters)
        @test pv_after == pv_before
        @test sum(pv_after) ≈ 8 * 200.0

        measured["wellbore_probe"] = Dict{String,Any}(
            "cell_center_depth_m" => sort(unique(z)),
            "well_reference_depth_m" => wcase["wells"][1]["reference_depth_m"],
            "submodel_viscosities_pa_s" => viscosities,
            "submodels_without_viscosity" => non_flow,
            "face_gdz" => collect(gdz),
            "native_face_gdz" => collect(native_gdz),
            "vertical_face_gdz" => collect(gdz[vertical]),
            "horizontal_face_gdz" => collect(gdz[horizontal]),
            "pore_volume_m3" => sum(pv_before),
            "pore_volume_after_pressure_change_m3" => sum(pv_after),
        )

        # --- 5.6 no singleton model: A -> B -> A rebuilds from scratch ------------------
        acase, aarrays = verification_case(:closed_cell)
        bcase, barrays = verification_case(:closed_cell)
        barrays["porosity"] = fill(0.4, 1)
        a1 = adapter.build_ow(acase, aarrays)
        b = adapter.build_ow(bcase, barrays)
        a2 = adapter.build_ow(acase, aarrays)
        pv_a1 = sum(pore_volume(a1.model, a1.parameters))
        pv_b = sum(pore_volume(b.model, b.parameters))
        pv_a2 = sum(pore_volume(a2.model, a2.parameters))
        @test pv_a1 == pv_a2 == 200.0
        @test pv_b == 400.0
        @test a1.state0 !== a2.state0
        @test a1.model !== a2.model

        measured["rebuild"] = Dict{String,Any}(
            "pore_volume_a1_m3" => pv_a1,
            "pore_volume_b_m3" => pv_b,
            "pore_volume_a2_m3" => pv_a2,
        )
    end

    measured["status"] = "ok"
    return measured
end

# ======================================================================================
# The controls diagnostic (plan Task 6.6). Also only runs as a program.
# ======================================================================================

"""
    compile_intervals(case) -> (edges, month_index, controls)

Group a case's flat control segments into the per-interval lists `build_forces` takes:
the sorted union of the report edges and every segment boundary, and for each of those
intervals exactly one control per well.

This is the VERIFICATION side of `so_recon.simulator.schedule.compile_schedule`. The
production path compiles the schedule in Python and ships it; a fixture cannot import a
Python function, so the grouping is spelled again here and the integration test compares
the two — the same way `educational_fluids()` is compared against `FluidSpec()` rather than
trusted.
"""
function compile_intervals(case::AbstractDict)
    report = Float64.(case["report_edges_s"])
    segments = collect(case["controls"])
    boundaries = Set{Float64}(report)
    for s in segments
        push!(boundaries, Float64(s["start_s"]))
        push!(boundaries, Float64(s["end_s"]))
    end
    edges = sort(collect(boundaries))
    wells = sort(unique(String(s["well_id"]) for s in segments))
    month_index = Int[]
    controls = Vector{Vector{Any}}()
    month = 1
    for i in 1:(length(edges) - 1)
        a, b = edges[i], edges[i + 1]
        while report[month + 1] <= a
            month += 1
        end
        push!(month_index, month - 1)  # zero-based, like the Python schedule
        group = Any[]
        for w in wells
            covering = [
                s for s in segments if String(s["well_id"]) == w &&
                Float64(s["start_s"]) <= a && Float64(s["end_s"]) >= b
            ]
            length(covering) == 1 || error(
                "compile_intervals: well $(w) has $(length(covering)) controls on interval " *
                "$(i - 1) [$(a), $(b))",
            )
            push!(group, covering[1])
        end
        push!(controls, group)
    end
    return (edges = edges, month_index = month_index, controls = controls)
end

"""Uptime, in seconds, of one well on each compiled interval: open surface AND open holes."""
function interval_uptime_s(schedule, well_id::AbstractString)
    out = Float64[]
    for i in eachindex(schedule.controls)
        c = only(x for x in schedule.controls[i] if String(x["well_id"]) == well_id)
        flowing = String(c["role"]) != "shut" && any(Bool.(c["connection_open"]))
        push!(out, flowing ? schedule.edges[i + 1] - schedule.edges[i] : 0.0)
    end
    return out
end

"""
    selftest_single_well_case(; ...) -> (case, arrays)

A one-well construction probe for the control diagnostics below: a closed box of the same
educational fluid over one interval of `days`, carrying exactly one control segment. Like
`selftest_wellbore_case` it is deliberately NOT reachable through `verification_case`,
because it makes no reference claim of its own — it isolates one native behaviour.
"""
function selftest_single_well_case(;
    nz::Int,
    cells::Vector{Int},
    pressure::Vector{Float64},
    role::AbstractString,
    target::AbstractString,
    value::Real,
    bhp_limit_pa::Union{Nothing,Real},
    connection_open::Vector{Bool},
    days::Real = 1.0,
    well_model::AbstractString = "multisegment",
)
    nx = CONTROLS_NX
    n_cells = nx * nz
    length(pressure) == n_cells || error("selftest_single_well_case: pressure/grid mismatch")
    case = Dict{String,Any}(
        "schema_version" => "case-1",
        "case_id" => "selftest-control-probe",
        "grid" => Dict{String,Any}(
            "shape" => [nx, 1, nz],
            "extent_m" => [CELL_EDGE_M * nx, CELL_EDGE_M, CELL_EDGE_M * nz],
            "z_positive" => "down",
        ),
        "rock" => Dict{String,Any}("rock_compressibility_pa_inv" => 0.0),
        "fluids" => educational_fluids(),
        "wells" => Any[well_spec("PRO1", cells; model = well_model)],
        "controls" => Any[control_segment(
            start_day = 0.0,
            end_day = days,
            well_id = "PRO1",
            role = role,
            target = target,
            value = value,
            bhp_limit_pa = bhp_limit_pa,
            connection_open = connection_open,
        )],
        "boundary" => Dict{String,Any}("kind" => "closed", "cells" => Any[]),
        "report_edges_s" => [0.0, Float64(days) * SECONDS_PER_DAY],
        "units" => Dict{String,Any}("control_rate" => "m3_sc/day"),
        "gravity_m_s2" => STANDARD_GRAVITY_M_S2,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => cartesian_cell_centers(nx, 1, nz),
        "porosity" => fill(0.2, n_cells),
        "permeability_m2" => fill(100.0 * MILLIDARCY_M2, 3, n_cells),
        "pressure_pa" => copy(pressure),
        "sw" => fill(0.3, n_cells),
    )
    return (case, arrays)
end

"""Build, drive and simulate one fixture. Returns everything the checks below read."""
function drive_fixture(adapter::Module, case::AbstractDict, arrays::AbstractDict)
    schedule = compile_intervals(case)
    physical = adapter.build_ow(case, arrays)
    forces = [
        adapter.build_forces(physical.model, group, case["boundary"]) for group in schedule.controls
    ]
    dt = diff(schedule.edges)
    state0 = deepcopy(physical.state0)
    result = simulate_reservoir(
        physical.state0,
        physical.model,
        dt;
        parameters = physical.parameters,
        forces = forces,
        info_level = -1,
    )
    states = result.result.states
    length(states) == length(dt) || error(
        "drive_fixture: the solver reported $(length(states)) states for $(length(dt)) intervals; " *
        "the fixture did not run to the end of its schedule",
    )
    evidence = adapter.control_evidence(physical.model, states, forces, schedule.controls)
    return (; physical, schedule, forces, dt, states, evidence, state0)
end

"""Produced or injected volume of one phase over one interval, m3_sc, production positive."""
interval_volume_m3(evidence_step, key, dt_s) = evidence_step[key] * dt_s / SECONDS_PER_DAY

function selftest_controls(adapter::Module)
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
    )

    @testset "E01 calendar controls and completion events" begin
        # --- 6.6 the two-interval rate/bhp fixture -------------------------------------
        case, arrays = verification_case(:two_interval_controls)
        fixture_run = drive_fixture(adapter, case, arrays)
        schedule, forces, evidence = fixture_run.schedule, fixture_run.forces, fixture_run.evidence

        # The calendar reached the model as two REAL months plus the two events inside
        # February. Twelve 30-day months would put these edges somewhere else entirely.
        @test schedule.edges ≈ collect(CONTROLS_EVENT_DAYS) .* SECONDS_PER_DAY
        @test schedule.month_index == [0, 1, 1, 1]
        @test case["report_edges_s"] ≈ [0.0, 31.0, 60.0] .* SECONDS_PER_DAY

        # --- every well, every interval: role, target, limit and mask restated ---------
        control_by_interval = Dict{String,Vector{String}}()
        mask_by_interval = Dict{String,Vector{Vector{Float64}}}()
        limits_by_interval = Dict{String,Vector{Any}}()
        for well in ("INJ1", "PRO1")
            name = Symbol(well)
            control_by_interval[well] = String[]
            mask_by_interval[well] = Vector{Float64}[]
            limits_by_interval[well] = Any[]
            for f in forces
                push!(control_by_interval[well], string(nameof(typeof(f[:Facility].control[name]))))
                push!(mask_by_interval[well], collect(f[name].mask.values))
                limit = f[:Facility].limits[name]
                push!(limits_by_interval[well], limit === nothing ? nothing : limit.bhp)
            end
        end
        @test control_by_interval["PRO1"] ==
              ["ProducerControl", "ProducerControl", "DisabledControl", "ProducerControl"]
        @test control_by_interval["INJ1"] ==
              ["InjectorControl", "InjectorControl", "DisabledControl", "InjectorControl"]
        # The producer's completion closes on interval 1, is fully isolated on interval 2
        # and is OPEN AGAIN on interval 3: a mask is restated, never inherited.
        @test mask_by_interval["PRO1"] == [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
        @test mask_by_interval["INJ1"] == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

        # --- limits: only what the case recorded ---------------------------------------
        @test limits_by_interval["PRO1"] ==
              [CONTROLS_PRODUCER_BHP_FLOOR_PA, CONTROLS_PRODUCER_BHP_FLOOR_PA, nothing, nothing]
        # The injector's ceiling is recorded while it runs on a rate; a well already ON bhp
        # carries no limit, for either role.
        @test limits_by_interval["INJ1"] == [
            CONTROLS_INJECTOR_BHP_CEILING_PA,
            CONTROLS_INJECTOR_BHP_CEILING_PA,
            nothing,
            nothing,
        ]
        # And the native defaults really were disabled: JutulDarcy WOULD have put a rate
        # floor under the bhp producer of interval 3, and the forces carry no limit at all.
        native_default = JutulDarcy.default_limits(forces[4][:Facility].control[:PRO1])
        @test haskey(native_default, :rate_lower)
        @test forces[4][:Facility].limits[:PRO1] === nothing

        # --- the injected stream, as BUILT on every branch that constructs one ----------
        # `InjectorControl` carries the surface mixture and the surface density of what is
        # being injected, and a wrong value there is silent wrong physics rather than an
        # error. The bhp branch builds its own `InjectorControl` from scratch, so it is
        # checked here beside the rate branch rather than inherited from it: a rate control
        # that later hits its bhp limit goes through `replace_target`, which copies the
        # mixture and density across and could not reveal a mistake in this constructor.
        injector_targets = String[]
        injector_mixtures = Any[]
        injector_densities = Any[]
        for f in forces
            c = f[:Facility].control[:INJ1]
            push!(injector_targets, string(nameof(typeof(c.target))))
            if c isa InjectorControl
                push!(injector_mixtures, collect(c.injection_mixture))
                push!(injector_densities, c.mixture_density)
            else
                push!(injector_mixtures, nothing)
                push!(injector_densities, nothing)
            end
        end
        @test injector_targets == [
            "SurfaceWaterRateTarget",
            "SurfaceWaterRateTarget",
            "DisabledTarget",
            "BottomHolePressureTarget",
        ]
        # Pure water by mass, water first: the phase order `OW_PHASES` fixes, on every
        # branch including the bhp one.
        @test injector_mixtures == [[1.0, 0.0], [1.0, 0.0], nothing, [1.0, 0.0]]
        # And the surface density is the SYSTEM's own water density, not a default of 1.0.
        rho_w_sc_declared = educational_fluids()["density_sc_kg_m3"][1]
        @test injector_densities ==
              [rho_w_sc_declared, rho_w_sc_declared, nothing, rho_w_sc_declared]
        @test rho_w_sc_declared != 1.0
        @test forces[4][:Facility].control[:INJ1].target.value ≈
              CONTROLS_INJECTOR_BHP_TARGET_PA

        # --- the native sign convention, on the record ---------------------------------
        pro = evidence["PRO1"]
        inj = evidence["INJ1"]
        @test pro[1]["operating_target"] == "lrat"
        # Production is NEGATIVE natively and POSITIVE in the public number beside it.
        @test pro[1]["native_lrat_m3_s"] ≈ -CONTROLS_RATE_M3_DAY / SECONDS_PER_DAY
        @test pro[1]["liquid_rate_m3_day"] ≈ CONTROLS_RATE_M3_DAY
        @test pro[1]["oil_rate_m3_day"] > 0.0
        @test pro[1]["water_rate_m3_day"] > 0.0
        # An injector's native surface rate has the other sign, so the public water rate of
        # an injector comes out negative: the sign is a direction, not an absolute value.
        @test inj[1]["operating_target"] == "wrat"
        @test inj[1]["water_rate_m3_day"] ≈ -CONTROLS_RATE_M3_DAY
        # The injector's bhp interval runs, and runs on the bhp it was built with: the
        # constructor is exercised by the solver, not merely constructed and discarded.
        @test inj[4]["operating_target"] == "bhp"
        @test inj[4]["bhp_pa"] ≈ CONTROLS_INJECTOR_BHP_TARGET_PA
        # Still injecting, which is negative under the production-positive convention.
        @test inj[4]["water_rate_m3_day"] < 0.0

        # --- Vo + Vw = q_liquid * uptime, where the rate is actually reached ------------
        uptime = interval_uptime_s(schedule, "PRO1")
        @test uptime ≈ [31.0, 14.0, 0.0, 8.0] .* SECONDS_PER_DAY
        liquid_volumes = Float64[]
        for i in 1:4
            vo = interval_volume_m3(pro[i], "oil_rate_m3_day", fixture_run.dt[i])
            vw = interval_volume_m3(pro[i], "water_rate_m3_day", fixture_run.dt[i])
            push!(liquid_volumes, vo + vw)
        end
        # The first two intervals are on the rate control, so the identity is exact: the
        # control pins the surface liquid rate at every step, and oil plus water IS that rate.
        for i in (1, 2)
            @test liquid_volumes[i] ≈
                  CONTROLS_RATE_M3_DAY * uptime[i] / SECONDS_PER_DAY rtol = 1e-10
            @test pro[i]["honoured"]
        end
        # The third interval has no uptime at all: no volume, and no phase split to report.
        @test uptime[3] == 0.0
        @test liquid_volumes[3] == 0.0
        @test pro[3]["oil_rate_m3_day"] == 0.0
        @test pro[3]["water_rate_m3_day"] == 0.0
        @test pro[3]["operating_target"] == "disabled"
        # The fourth interval is the bhp branch: it still produces, and under its target.
        @test pro[4]["operating_target"] == "bhp"
        @test pro[4]["bhp_pa"] ≈ CONTROLS_PRODUCER_BHP_TARGET_PA
        @test liquid_volumes[4] > 0.0

        # Nothing in this fixture was infeasible.
        @test adapter.infeasible_controls(evidence) === nothing

        measured["two_interval_controls"] = Dict{String,Any}(
            "edges_s" => schedule.edges,
            "month_index" => schedule.month_index,
            "report_edges_s" => Float64.(case["report_edges_s"]),
            "start_date" => case["start_date"],
            "control_rate_unit" => case["units"]["control_rate"],
            # Both shapes: the case's own flat segments, which the Python test parses back
            # into `ControlSegment`s and compiles with `compile_schedule`, and the grouping
            # this side made of them. Agreement between the two is what keeps the two
            # compilers from drifting apart.
            "controls" => case["controls"],
            "controls_by_interval" => schedule.controls,
            "control_types" => control_by_interval,
            # As BUILT, per interval: the injector's target type and the surface stream it
            # would inject. The bhp interval is the one no rate control can reach.
            "injector_targets" => injector_targets,
            "injector_mixtures" => injector_mixtures,
            "injector_densities_kg_m3" => injector_densities,
            "masks" => mask_by_interval,
            "recorded_bhp_limits_pa" => limits_by_interval,
            "uptime_s" => uptime,
            "liquid_volume_m3" => liquid_volumes,
            "evidence" => evidence,
            "native_default_limit_for_bhp_producer" => String.(collect(keys(native_default))),
        )

        # --- the refusals, which no valid case can reach ---------------------------------
        # These are the guards that stand between a malformed control and silent wrong
        # physics, and a guard nobody exercised is a guard nobody knows works. They need no
        # simulation: the model built above is enough, so the whole block is free.
        model = fixture_run.physical.model
        boundary = case["boundary"]
        interval0 = schedule.controls[1]
        inj_control = only(c for c in interval0 if String(c["well_id"]) == "INJ1")
        pro_control = only(c for c in interval0 if String(c["well_id"]) == "PRO1")
        rho_w_sc = educational_fluids()["density_sc_kg_m3"][1]
        refusals = Dict{String,String}()

        function expect_refusal!(label::String, thunk)
            err = try
                thunk()
                nothing
            catch caught
                caught
            end
            @test err isa adapter.InvalidCaseInput
            message = err === nothing ? "" : sprint(showerror, err)
            refusals[label] = message
            return message
        end

        """A copy of one control segment with some of its fields replaced."""
        altered(c, changes) = merge(Dict{String,Any}(c), Dict{String,Any}(changes))

        # A hole in the schedule is a refusal, not a well that kept going. `setup_forces`
        # would otherwise fill the missing well in with DisabledControl().
        hole_message = expect_refusal!(
            "missing_well",
            () -> adapter.build_forces(
                model,
                [c for c in interval0 if String(c["well_id"]) != "INJ1"],
                boundary,
            ),
        )
        @test occursin("INJ1", hole_message)
        @test occursin("never inherited", hole_message)

        # Two controls for one well on one interval: which one would have won is not a
        # question this adapter answers.
        duplicate_message = expect_refusal!(
            "duplicate_well",
            () -> adapter.build_forces(model, [inj_control, pro_control, pro_control], boundary),
        )
        @test occursin("PRO1", duplicate_message)
        @test occursin("two controls on one interval", duplicate_message)

        # A control for a well the model does not have — a renamed or misspelled well would
        # otherwise leave its real counterpart silently disabled.
        ghost_message = expect_refusal!(
            "unknown_well",
            () -> adapter.build_forces(
                model,
                [inj_control, pro_control, altered(pro_control, ("well_id" => "GHOST",))],
                boundary,
            ),
        )
        @test occursin("GHOST", ghost_message)
        @test occursin("the model does not have", ghost_message)

        # A mask SHORTER than the well's connections is the dangerous one:
        # `apply_perforation_mask!` iterates `eachindex(mask)`, so Jutul would apply the
        # entries it was given and leave every remaining perforation fully open — a
        # completion event that half-happened, with no complaint from the backend. Python
        # cannot catch it either, because `ControlSegment` never sees the model's
        # connection count. This guard is the only place it can be caught.
        short_message = expect_refusal!(
            "mask_too_short",
            () -> adapter.build_forces(
                model,
                [inj_control, altered(pro_control, ("connection_open" => Any[true],))],
                boundary,
            ),
        )
        @test occursin("PRO1", short_message)
        @test occursin("1 entries for 2 connections", short_message)
        # And longer, which Jutul would simply ignore the tail of.
        long_message = expect_refusal!(
            "mask_too_long",
            () -> adapter.build_forces(
                model,
                [
                    inj_control,
                    altered(pro_control, ("connection_open" => Any[true, true, false],)),
                ],
                boundary,
            ),
        )
        @test occursin("3 entries for 2 connections", long_message)
        # A partial-completion multiplier is a real `PerforationMask` feature and NOT part
        # of the E01 contract: scaling a connection to half strength would be a physical
        # change nobody declared.
        fraction_message = expect_refusal!(
            "mask_not_boolean",
            () -> adapter.build_forces(
                model,
                [inj_control, altered(pro_control, ("connection_open" => Any[true, 0.5],))],
                boundary,
            ),
        )
        @test occursin("connection_open[1] is 0.5", fraction_message)
        @test occursin("not partially open", fraction_message)

        # SPEC 9.1 at the native boundary, not only in the Python contract: a producer is
        # never given a phase rate, and a rate of zero is a shut well.
        phase_rate_message = expect_refusal!(
            "producer_phase_rate",
            () -> adapter.native_control(
                altered(pro_control, ("target" => "water_rate",)),
                rho_w_sc,
            ),
        )
        @test occursin("SPEC 9.1", phase_rate_message)
        zero_rate_message = expect_refusal!(
            "zero_rate",
            () -> adapter.native_control(altered(pro_control, ("value" => 0.0,)), rho_w_sc),
        )
        @test occursin("role='shut'", zero_rate_message)
        # A recorded limit that is not a pressure.
        limit_message = expect_refusal!(
            "non_positive_bhp_limit",
            () -> adapter.control_limits(altered(pro_control, ("bhp_limit_pa" => -1.0,))),
        )
        @test occursin("must be positive when given", limit_message)

        # A boundary this build does not implement is named, not stubbed; and a closed one
        # that names cells is a contradiction rather than a closed boundary.
        boundary_message = expect_refusal!(
            "unimplemented_boundary",
            () -> adapter.build_forces(
                model,
                interval0,
                Dict{String,Any}("kind" => "pressure_water", "cells" => Any[0]),
            ),
        )
        @test occursin("pressure_water", boundary_message)
        @test occursin("Tasks 9-10", boundary_message)
        closed_message = expect_refusal!(
            "closed_boundary_with_cells",
            () -> adapter.build_forces(
                model,
                interval0,
                Dict{String,Any}("kind" => "closed", "cells" => Any[0]),
            ),
        )
        @test occursin("closed boundary names no cells", closed_message)

        measured["refusals"] = refusals
        measured["missing_control_message"] = hole_message

        # --- a crossflow declaration the backend cannot honour --------------------------
        no_crossflow_case, no_crossflow_arrays = verification_case(:two_interval_controls)
        no_crossflow_case["wells"][2]["allow_crossflow"] = false
        refusal = try
            adapter.build_ow(no_crossflow_case, no_crossflow_arrays)
            nothing
        catch err
            err
        end
        @test refusal isa adapter.InvalidCaseInput
        crossflow_message = refusal === nothing ? "" : sprint(showerror, refusal)
        @test occursin("allow_crossflow=false", crossflow_message)
        measured["crossflow_refusal_message"] = crossflow_message

        # --- 6.6 a rate nobody could deliver --------------------------------------------
        # The well is asked for 100 000 m3_sc/day against a bottom-hole floor 150 bar below
        # the reservoir. JutulDarcy does not fail: it switches the well onto the floor and
        # keeps simulating, which is exactly why the requested and operating targets have
        # to be compared instead of the result being read as a fulfilled rate.
        bad_case, bad_arrays = selftest_single_well_case(
            nz = 2,
            cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "producer",
            target = "liquid_rate",
            value = 1.0e5,
            bhp_limit_pa = CONTROLS_PRODUCER_BHP_FLOOR_PA,
            connection_open = Bool[true, true],
        )
        bad = drive_fixture(adapter, bad_case, bad_arrays)
        step = bad.evidence["PRO1"][1]
        @test !step["honoured"]
        @test step["requested_target"] == "lrat"
        @test step["requested_value"] == 1.0e5
        @test step["operating_target"] == "bhp"
        @test step["bhp_pa"] ≈ CONTROLS_PRODUCER_BHP_FLOOR_PA
        @test 0.0 < step["liquid_rate_m3_day"] < 1.0e5
        reason = adapter.infeasible_controls(bad.evidence)
        @test reason !== nothing
        @test occursin("requested lrat=100000.0", reason)
        @test occursin("operated on bhp", reason)
        measured["infeasible"] = Dict{String,Any}("evidence" => step, "reason" => reason)

        # The same limit in the other direction. `control_limits` writes one `:bhp` key for
        # both roles and lets JutulDarcy read it as a FLOOR under a producer and a CEILING
        # over an injector; an injector that cannot take its water proves the second half of
        # that, which the producer above cannot.
        over_case, over_arrays = selftest_single_well_case(
            nz = 2,
            cells = [0, CONTROLS_NX],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "injector",
            target = "water_rate",
            value = 1.0e5,
            bhp_limit_pa = CONTROLS_INJECTOR_BHP_CEILING_PA,
            connection_open = Bool[true, true],
        )
        over = drive_fixture(adapter, over_case, over_arrays)
        over_step = over.evidence["PRO1"][1]
        @test !over_step["honoured"]
        @test over_step["requested_target"] == "wrat"
        @test over_step["operating_target"] == "bhp"
        @test over_step["bhp_pa"] ≈ CONTROLS_INJECTOR_BHP_CEILING_PA
        # Injection is negative under the public production-positive convention, and what
        # the well actually took is far below the 100 000 m3_sc/day it was asked for.
        @test -1.0e5 < over_step["water_rate_m3_day"] < 0.0
        over_reason = adapter.infeasible_controls(over.evidence)
        @test over_reason !== nothing
        @test occursin("requested wrat=100000.0", over_reason)
        measured["infeasible_injector"] =
            Dict{String,Any}("evidence" => over_step, "reason" => over_reason)

        # --- 6.5 full isolation: every mask false means no connection flux at all --------
        # One layer, uniform pressure, closed box: with the connections shut NOTHING can
        # move a cell, so the reservoir must come back bit for bit unchanged. The contrast
        # run is the same case with the connections open.
        isolation = Dict{String,Any}()
        for (label, mask) in (("isolated", Bool[false, false]), ("open", Bool[true, true]))
            iso_case, iso_arrays = selftest_single_well_case(
                nz = 1,
                cells = [0, CONTROLS_NX - 1],
                pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX),
                role = "producer",
                target = "bhp",
                value = CONTROLS_PRODUCER_BHP_FLOOR_PA,
                bhp_limit_pa = nothing,
                connection_open = mask,
            )
            iso = drive_fixture(adapter, iso_case, iso_arrays)
            dp = maximum(abs.(iso.states[end][:Reservoir][:Pressure] .- iso_arrays["pressure_pa"]))
            ds = maximum(abs.(
                iso.states[end][:Reservoir][:Saturations] .-
                iso.state0[:Reservoir][:Saturations],
            ))
            isolation[label] = Dict{String,Any}(
                "max_abs_dp_pa" => dp,
                "max_abs_dsw" => ds,
                "liquid_rate_m3_day" => iso.evidence["PRO1"][1]["liquid_rate_m3_day"],
                "well_mass_kg" => sum(iso.states[end][:PRO1][:TotalMasses]),
            )
        end
        # Zero connection mass flux, exactly: not a tolerance, an untouched reservoir.
        @test isolation["isolated"]["max_abs_dp_pa"] == 0.0
        @test isolation["isolated"]["max_abs_dsw"] == 0.0
        # The contrast proves the check is not vacuous.
        @test isolation["open"]["max_abs_dp_pa"] > 1.0e6
        @test isolation["open"]["liquid_rate_m3_day"] > 1.0
        # What still leaves the surface of the isolated well is the WELLBORE's own fluid
        # decompressing from 200 bar to 50 bar, not reservoir production: one day of it,
        # weighed at the heavier phase's surface density, is a small fraction of the mass
        # standing in the bore (about 1.2%, which is c_t times that 150 bar), and it is two
        # orders below what the same well produces with its connections open.
        rho_w_sc = educational_fluids()["density_sc_kg_m3"][1]
        surface_mass_kg = abs(isolation["isolated"]["liquid_rate_m3_day"]) * rho_w_sc
        @test surface_mass_kg < 0.05 * isolation["isolated"]["well_mass_kg"]
        @test isolation["isolated"]["liquid_rate_m3_day"] <
              0.01 * isolation["open"]["liquid_rate_m3_day"]
        isolation["isolated"]["surface_mass_over_one_day_kg"] = surface_mass_kg
        measured["isolation"] = isolation

        # --- 6.5 a shut SURFACE is not a shut well: DisabledControl and crossflow --------
        # Two layers at different pressures and one wellbore across both. Under
        # `DisabledControl` the net surface rate is exactly zero, and the connections stay
        # coupled: fluid enters the deep perforation and leaves the shallow one. That is
        # the native zero-net-surface-rate formulation, demonstrated rather than asserted.
        crossflow = Dict{String,Any}()
        layered = vcat(fill(1.8e7, CONTROLS_NX), fill(2.2e7, CONTROLS_NX))
        for (label, mask) in (("coupled", Bool[true, true]), ("isolated", Bool[false, false]))
            xf_case, xf_arrays = selftest_single_well_case(
                nz = 2,
                cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
                pressure = layered,
                role = "shut",
                target = "disabled",
                value = 0.0,
                bhp_limit_pa = nothing,
                connection_open = mask,
            )
            xf = drive_fixture(adapter, xf_case, xf_arrays)
            p = xf.states[end][:Reservoir][:Pressure]
            crossflow[label] = Dict{String,Any}(
                "operating_target" => xf.evidence["PRO1"][1]["operating_target"],
                "liquid_rate_m3_day" => xf.evidence["PRO1"][1]["liquid_rate_m3_day"],
                "surface_mass_rate_kg_s" =>
                    only(xf.states[end][:Facility][:TotalSurfaceMassRate]),
                # Zero-based cells 3 and 7: the shallow and the deep perforation.
                "upper_perforated_pressure_pa" => p[CONTROLS_NX],
                "lower_perforated_pressure_pa" => p[2 * CONTROLS_NX],
                "well_segment_mass_flux_kg_s" => collect(xf.states[end][:PRO1][:TotalMassFlux]),
            )
        end
        for label in ("coupled", "isolated")
            @test crossflow[label]["operating_target"] == "disabled"
            # A disabled well is a well with no SURFACE rate. Exactly zero, both ways.
            @test crossflow[label]["surface_mass_rate_kg_s"] == 0.0
            @test crossflow[label]["liquid_rate_m3_day"] == 0.0
        end
        # And yet the coupled well moved fluid: relative to the isolated run the shallow
        # perforated cell ends up at a HIGHER pressure and the deep one at a LOWER one.
        @test crossflow["coupled"]["upper_perforated_pressure_pa"] >
              crossflow["isolated"]["upper_perforated_pressure_pa"]
        @test crossflow["coupled"]["lower_perforated_pressure_pa"] <
              crossflow["isolated"]["lower_perforated_pressure_pa"]
        # The wellbore segment between the two perforations carries that flow, and closing
        # the completions takes it away: DisabledControl alone does NOT isolate a well.
        coupled_flux = abs(crossflow["coupled"]["well_segment_mass_flux_kg_s"][2])
        isolated_flux = abs(crossflow["isolated"]["well_segment_mass_flux_kg_s"][2])
        @test coupled_flux > 50.0 * isolated_flux
        measured["crossflow"] = crossflow
    end

    measured["status"] = "ok"
    return measured
end

# ======================================================================================
# The outputs diagnostic (plan Task 7.8). Also only runs as a program.
# ======================================================================================

"""Serialise a matrix as an explicit list of ROWS, so no layout has to be guessed."""
rows_of(m::AbstractMatrix) = [collect(Float64.(r)) for r in eachrow(m)]

"""The fixture arrays in the row-explicit shape the Python side rebuilds a case from."""
function serialisable_arrays(arrays::AbstractDict)
    return Dict{String,Any}(
        "cell_centers_m" => rows_of(arrays["cell_centers_m"]),
        "permeability_m2" => rows_of(arrays["permeability_m2"]),
        "porosity" => collect(Float64.(arrays["porosity"])),
        "pressure_pa" => collect(Float64.(arrays["pressure_pa"])),
        "sw" => collect(Float64.(arrays["sw"])),
    )
end

"""Per-step balance residual of one extraction, as `ΔI - source`, per component."""
function balance_residual(inventory, source)
    return [
        [inventory[k + 1][c] - inventory[k][c] - source[k][c] for c in eachindex(source[k])]
        for k in eachindex(source)
    ]
end

function selftest_outputs(adapter::Module)
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
    )

    @testset "E01 accepted-substep integrals and outputs" begin
        # --- 7.3/7.5/7.6 the calendar fixture, integrated over its accepted substeps -----
        case, arrays = verification_case(:two_interval_controls)
        payload = adapter.run_forward(case, arrays; max_timestep = 5 * SECONDS_PER_DAY)
        @test payload["status"] == "COMPLETE"
        chunk = payload["chunk"]
        dt = chunk["dt_s"]
        @test all(>(0.0), dt)
        @test sum(dt) ≈ sum(CONTROLS_MONTH_DAYS) * SECONDS_PER_DAY
        # Every accepted substep, and nothing else: the solver's own accepted count.
        @test length(dt) == payload["solver"]["accepted_steps"]
        @test payload["solver"]["cut_steps"] == 0
        # More substeps than report intervals, so the integral really is over substeps.
        @test length(dt) > length(chunk["edges_s"]) - 1

        # Conservation, both ways. The reservoir against the native connection cross term,
        # and the reservoir PLUS the wellbores against the surface flux: the two differ by
        # the well storage, which is why they are kept apart.
        surface_residual = balance_residual(
            payload["inventory_m3_sc"], payload["net_surface_source_m3_sc"],
        )
        connection_residual = balance_residual(
            payload["reservoir_inventory_m3_sc"], payload["net_connection_source_m3_sc"],
        )
        @test maximum(maximum(abs.(r)) for r in surface_residual) < 1.0e-2
        @test maximum(maximum(abs.(r)) for r in connection_residual) < 1.0e-2

        # --- 7.3 `result.result`, not `result.states` -----------------------------------
        # The high-level states are `map(x -> x[:Reservoir], states)`: no facility and no
        # wellbore, so no well rate can be read from them. This is the choice the plan
        # names, pinned by a test rather than by a comment.
        iso_case, iso_arrays = selftest_single_well_case(
            nz = 2,
            cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "producer",
            target = "bhp",
            value = CONTROLS_PRODUCER_BHP_FLOOR_PA,
            bhp_limit_pa = nothing,
            connection_open = Bool[false, false],
            days = 1.0,
        )
        iso_schedule = adapter.compile_intervals(iso_case)
        iso_physical = adapter.build_ow(iso_case, iso_arrays)
        iso_forces = [
            adapter.build_forces(iso_physical.model, g, iso_case["boundary"])
            for g in iso_schedule.controls
        ]
        iso_state0 = deepcopy(iso_physical.state0)
        iso_result = simulate_reservoir(
            iso_physical.state0,
            iso_physical.model,
            diff(iso_schedule.edges_s);
            parameters = iso_physical.parameters,
            forces = iso_forces,
            info_level = -1,
            output_substates = true,
        )
        high_level = sort(string.(collect(keys(iso_result.states[1]))))
        raw = sort(string.(collect(keys(iso_result.result.states[1]))))
        @test !("Facility" in high_level) && !("PRO1" in high_level)
        @test "Facility" in raw && "PRO1" in raw
        measured["state_keys"] = Dict{String,Any}("high_level" => high_level, "raw" => raw)

        # --- 7.4 a fully masked well moves exactly nothing through its connections -------
        iso_payload = adapter.extract_interval(
            iso_result,
            iso_physical.model,
            iso_physical.parameters,
            iso_forces,
            iso_state0;
            controls_by_interval = iso_schedule.controls,
            edges_s = iso_schedule.edges_s,
            state_times_s = Float64.(iso_case["report_edges_s"]),
        )
        masked = iso_payload["connections"]
        @test !isempty(masked)
        @test all(!c["connection_open"] for c in masked)
        # Not a tolerance: a masked connection multiplies the native cross term by zero.
        @test all(c["total_mass_kg_s"] == 0.0 for c in masked)
        @test all(c["water_mass_kg_s"] == 0.0 && c["oil_mass_kg_s"] == 0.0 for c in masked)
        measured["masked_connections"] = masked

        # The contrast, so the zero above is the mask and not an empty extraction.
        open_case, open_arrays = selftest_single_well_case(
            nz = 2,
            cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "producer",
            target = "bhp",
            value = CONTROLS_PRODUCER_BHP_FLOOR_PA,
            bhp_limit_pa = nothing,
            connection_open = Bool[true, true],
            days = 1.0,
        )
        open_payload = adapter.run_forward(open_case, open_arrays)
        @test all(c["connection_open"] for c in open_payload["connections"])
        # The same well, the same day, the same bottom-hole target, with the completions
        # open: every connection carries a real flux (about 0.09 kg/s here) rather than the
        # exact zero above. Without this the masked assertion would also pass on an
        # extraction that computed nothing at all.
        @test all(abs(c["total_mass_kg_s"]) > 1.0e-3 for c in open_payload["connections"])
        measured["open_connections"] = open_payload["connections"]

        # --- 7.4 a SimpleWell's connection flux, and the field it depends on -------------
        # `setup_well(...; simple_well = true)` builds a `SimpleWell` with `explicit_dp =
        # true`, whose connection pressure drop JutulDarcy 0.3.11 keeps as an EXTRA STATE
        # FIELD rather than a variable; `select_minimum_output_variables!` does not list it,
        # so by default it is never stored, and `perforation_phase_potential_difference`
        # reads it in preference to the density head. Measured with it missing: 6.8 m³_sc of
        # reservoir mass unaccounted for over a single day. `request_extra_outputs!` asks the
        # well submodel to record it — a change to what is stored, not to what is solved —
        # and `evaluated_state` restores the stored value over the zeros `setup_state!`
        # re-initialises. Both halves are checked here: that it conserves WITH the field, and
        # that the extraction still refuses WITHOUT it.
        simple_case, simple_arrays = selftest_single_well_case(
            nz = 2,
            cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "producer",
            target = "bhp",
            value = CONTROLS_PRODUCER_BHP_FLOOR_PA,
            bhp_limit_pa = nothing,
            connection_open = Bool[true, true],
            well_model = "simple",
            days = 5.0,
        )
        simple_schedule = adapter.compile_intervals(simple_case)
        simple_physical = adapter.build_ow(simple_case, simple_arrays)
        # What the model would have recorded on its own, and what asking adds.
        simple_wm = simple_physical.model.models[:PRO1]
        @test !(:ConnectionPressureDrop in simple_wm.output_variables)
        @test adapter.request_extra_outputs!(simple_physical.model) ==
              Dict("PRO1" => ["ConnectionPressureDrop"])
        @test :ConnectionPressureDrop in simple_wm.output_variables
        # Asking for it a second time is a no-op, and Jutul agrees the field is askable.
        @test adapter.request_extra_outputs!(simple_physical.model) == Dict{String,Any}()
        Jutul.check_output_variables(simple_physical.model)

        simple_forces = [
            adapter.build_forces(simple_physical.model, g, simple_case["boundary"])
            for g in simple_schedule.controls
        ]
        simple_state0 = deepcopy(simple_physical.state0)
        simple_result = simulate_reservoir(
            simple_physical.state0,
            simple_physical.model,
            diff(simple_schedule.edges_s);
            parameters = simple_physical.parameters,
            forces = simple_forces,
            info_level = -1,
            output_substates = true,
            max_timestep = SECONDS_PER_DAY,
        )
        simple_states, _, _ = Jutul.expand_to_ministeps(simple_result.result)
        # It really is stored now, and it is not a vector of zeros.
        @test haskey(simple_states[1][:PRO1], :ConnectionPressureDrop)
        @test all(!iszero, simple_states[1][:PRO1][:ConnectionPressureDrop])
        simple = adapter.extract_interval(
            simple_result,
            simple_physical.model,
            simple_physical.parameters,
            simple_forces,
            simple_state0;
            controls_by_interval = simple_schedule.controls,
            edges_s = simple_schedule.edges_s,
            state_times_s = Float64.(simple_case["report_edges_s"]),
        )
        @test simple["stored_extra_state_fields"] == Dict("PRO1" => ["ConnectionPressureDrop"])
        @test any(abs(c["total_mass_kg_s"]) > 1.0e-3 for c in simple["connections"])
        simple_residual = balance_residual(
            simple["reservoir_inventory_m3_sc"], simple["net_connection_source_m3_sc"],
        )
        # 6.8 m³_sc without the field; measured 9.3e-5 m³_sc with it.
        @test maximum(maximum(abs.(r)) for r in simple_residual) < 1.0e-2
        measured["simple_well_extraction"] = simple
        measured["simple_well_residual_m3_sc"] =
            maximum(maximum(abs.(r)) for r in simple_residual)
        measured["simple_well_connection_pressure_drop_pa"] =
            collect(simple_states[1][:PRO1][:ConnectionPressureDrop])

        # And the guard is still a guard: strip the stored field out of one substate and the
        # extraction refuses rather than reading the re-initialised zeros. This needs no
        # second simulation — it is the same states with one key taken away.
        stripped = deepcopy(simple_states[1])
        delete!(stripped[:PRO1], :ConnectionPressureDrop)
        simple_refusal = try
            adapter.evaluated_state(
                simple_physical.model, simple_physical.parameters, stripped,
            )
            nothing
        catch err
            err
        end
        @test simple_refusal isa adapter.InvalidCaseInput
        simple_message = simple_refusal === nothing ? "" : sprint(showerror, simple_refusal)
        @test occursin("ConnectionPressureDrop", simple_message)
        @test occursin("PRO1", simple_message)
        @test occursin("request_extra_outputs!", simple_message)
        measured["simple_well_refusal"] = simple_message

        # --- 7.3 the same integral under real timestep cuts -----------------------------
        # A deliberately starved nonlinear budget makes the solver fail and cut. The cut
        # trial states are NOT in the accepted list, the accepted ones still tile the
        # chunk, and the rate-controlled month still integrates to the volume its control
        # pins — which is the property a cut must not be able to break.
        cut = adapter.run_forward(case, arrays; max_nonlinear_iterations = 3)
        @test cut["status"] == "COMPLETE"
        @test cut["solver"]["cut_steps"] > 0
        @test length(cut["chunk"]["dt_s"]) == cut["solver"]["accepted_steps"]
        @test all(>(0.0), cut["chunk"]["dt_s"])
        @test sum(cut["chunk"]["dt_s"]) ≈ sum(CONTROLS_MONTH_DAYS) * SECONDS_PER_DAY

        # --- 7.8 an unreachable rate still extracts, and says it was unreachable --------
        bad_case, bad_arrays = selftest_single_well_case(
            nz = 2,
            cells = [CONTROLS_NX - 1, 2 * CONTROLS_NX - 1],
            pressure = fill(CONTROLS_INITIAL_PRESSURE_PA, CONTROLS_NX * 2),
            role = "producer",
            target = "liquid_rate",
            value = 1.0e5,
            bhp_limit_pa = CONTROLS_PRODUCER_BHP_FLOOR_PA,
            connection_open = Bool[true, true],
        )
        infeasible = adapter.run_forward(bad_case, bad_arrays)
        @test infeasible["status"] == "COMPLETE"
        @test infeasible["control_infeasible_reason"] !== nothing
        @test occursin("requested lrat=100000.0", infeasible["control_infeasible_reason"])
        measured["infeasible_reason"] = infeasible["control_infeasible_reason"]
        measured["infeasible_evidence"] = infeasible["control_evidence"]

        # --- what the Python side reads back --------------------------------------------
        measured["case"] = case
        measured["arrays"] = serialisable_arrays(arrays)
        measured["extraction"] = payload
        measured["extraction_with_cuts"] = cut
        measured["isolated_extraction"] = iso_payload
    end

    measured["status"] = "ok"
    return measured
end

"""Parse the diagnostic to run and the optional `--out` the project launcher appends."""
function parse_selftest_args(args::Vector{String})
    out = nothing
    mode = nothing
    modes = Dict(
        "--test-model" => :model,
        "--test-controls" => :controls,
        "--test-outputs" => :outputs,
    )
    i = 1
    while i <= length(args)
        if haskey(modes, args[i])
            mode === nothing ||
                error("pick one diagnostic: $(repr(mode)) was already requested")
            mode = modes[args[i]]
            i += 1
        elseif args[i] == "--out"
            i + 1 <= length(args) || error("missing value for --out")
            out = args[i + 1]
            i += 2
        else
            error("unexpected argument $(args[i])")
        end
    end
    return (mode, out)
end

function selftest_main(args::Vector{String}, adapter::Module)
    mode, out = parse_selftest_args(args)
    mode === nothing && error(
        "nothing to do: pass --test-model for the constructor diagnostic, --test-controls " *
        "for the calendar and control diagnostic, or --test-outputs for the accepted-substep " *
        "integrals and their balance",
    )
    diagnostics = Dict(
        :model => selftest,
        :controls => selftest_controls,
        :outputs => selftest_outputs,
    )
    payload = try
        diagnostics[mode](adapter)
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
    # Loading the adapter belongs to the diagnostic, not to the fixtures: `include` returns
    # the module the file defines, so the fixtures never depend on it at definition time and
    # this file can be included from the adapter's own side of the tree without a cycle.
    adapter_module = include(joinpath(@__DIR__, "..", "adapter", "SOReconAdapter.jl"))
    requested_mode, _ = parse_selftest_args(collect(ARGS))
    selftest_main(collect(ARGS), adapter_module)
    println("so-recon verification: $(requested_mode) diagnostic passed")
end
