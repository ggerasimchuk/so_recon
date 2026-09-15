# E01.13 — the black-oil capability fixtures, and the diagnostic that runs them.
#
# Two fixtures and one refusal, all driven through the SAME `build_physical`, `build_forces`
# and chunked `run_forward_native` the worker reaches. There is no second solver here and no
# reimplemented PVT: SPEC 8.1 leaves JutulDarcy the only operational backend.
#
# * `bo_closed` — one cell, one shut monitor well, no source at all. A closed black-oil
#   system at rest. Nothing may move, and the component gas inventory — free plus dissolved —
#   must be the number it started as. This is the capability's equivalent of 9.5's closed
#   cell, and it is here for the same reason: whatever moves is the solver's own arithmetic.
#
# * `bo_depletion` — a (4,1,1) box initialised well ABOVE its own bubble point with a single
#   producer whose bottom-hole pressure steps down over six report intervals, from above that
#   bubble point to below it. Free gas must appear, the dissolved ratio must fall, and the
#   component gas balance must close including the dissolved term. It is run through
#   `run_forward_native` with `chunk_months = 1`, which is EXACTLY the driver and exactly the
#   chunking the worker uses, so the continuous trajectory this publishes and the restart pair
#   the Python side runs through the production worker are numerically comparable rather than
#   two different discretisations of the same case.
#
# * `unknown_physics_class` — a case declaring a class this build does not have. It must come
#   back `INVALID_INPUT` from the dispatcher, never a default to the oil-water model.
#
# Scoring is NOT here. Julia publishes the extraction and Python scores it against
# `configs/e01_blackoil_tolerances.yml`; a fixture that graded itself could not fail.
#
# The capability's standing, restated where it is easy to find: this is an EDUCATIONAL
# capability on an academic benchmark PVT. It does not choose the physics of the field case,
# it is not the paired oil-water/black-oil sensitivity study with matched oil inventory —
# that is E06 — and its verdict is a separate gate from the oil-water one (plan 13.5).

using Test
using Jutul
using JutulDarcy
using JSON
using SHA

# `analytic.jl` carries Task 9's shared fixture vocabulary — `uniform_cell_centers`,
# `analytic_well`, `analytic_skeleton`, `shut_monitor_controls` — and the `ADAPTER` handle,
# and its `PROGRAM_FILE` guard makes including it inert. The black-oil fixtures are built out
# of that same vocabulary on purpose: a second copy of the grid, datum and control helpers
# would be a second set of conventions to keep in step.
include(joinpath(@__DIR__, "analytic.jl"))

# ======================================================================================
# the capability's fluid, as the exchange writes it
# ======================================================================================

#: Initial reservoir pressure of the depletion fixture: 200 bar, far above the bubble point
#: of the dissolved ratio below, and inside the exported saturation table's own range.
const BO_INITIAL_PRESSURE_PA = 200.0e5

#: The dissolved gas-oil ratio the cases start at. The exported PVTO saturation table gives
#: 53.67 bar as the bubble point of this ratio, which is what the producer's bottom-hole
#: pressure steps across.
const BO_INITIAL_RS = 50.0

#: Initial water saturation. Above `Swc = 0.1`, so the water is mobile and the oil-gas pair
#: is not sharing the cell with an immobile film alone.
const BO_INITIAL_SW = 0.2

"""
    blackoil_fluids() -> Dict

The capability's fluid block, in the JSON field names the exchange uses.

The surface densities are the pinned benchmark's own, permuted into PHASE order (water, oil,
gas) — see `adapter/blackoil.jl` for why the permutation is explicit. The PVT hashes are
filled in by the diagnostic, which is the only place that has the exported tables.
"""
function blackoil_fluids(; initial_rs::Float64 = BO_INITIAL_RS)
    pvt = ADAPTER.blackoil_pvt()
    fluid = Dict{String,Any}(
        "kind" => "BO",
        "pvt_source" => ADAPTER.BO_PVT_SOURCE,
        "pvt_table_hashes" => Dict{String,Any}(),
        "density_sc_kg_m3" => copy(pvt.rhoS),
        "initial_rs_m3_m3" => initial_rs,
        "rv" => 0.0,
        "p_sc_pa" => 101325.0,
        "t_sc_k" => 288.15,
        "corey_exponents" => [2.0, 2.0, 2.0],
        "residual_saturations" => [0.1, 0.1, 0.0],
        "kr_endpoints" => [1.0, 1.0, 1.0],
        "relperm_definition" =>
            "BrooksCoreyRelativePermeabilities(3, [2,2,2], [0.1,0.1,0.0], [1,1,1])",
        "hysteresis" => "none",
        "pc_model" => "zero",
        "educational" => true,
        "academic_benchmark" => true,
        "analytical_limit" => false,
    )
    # The tables are hashed into the fluid HERE, so the case record that reaches Python
    # already names the numbers the model was built from; a fluid that travelled with an
    # empty hash block would be a PVT identified by the name of a function.
    return with_pvt_hashes(
        fluid, ADAPTER.blackoil_pvt_export(pvt), ADAPTER.blackoil_relperm_export(fluid)
    )
end

"""
    with_pvt_hashes(fluid, pvt_export, relperm_export) -> Dict

Stamp the fluid with the SHA-256 of every table the model was built from.

The digest is of the exported NUMBERS, canonically serialised, together with the JutulDarcy
tree and version they came out of — which is inside the export itself. A PVT identified only
by the name of the function that returned it would be identified by a name.
"""
function with_pvt_hashes(fluid::AbstractDict, pvt_export::AbstractDict, relperm_export::AbstractDict)
    out = Dict{String,Any}(fluid)
    out["pvt_table_hashes"] = ADAPTER.blackoil_table_hashes(pvt_export, relperm_export)
    return out
end

# ======================================================================================
# the fixtures
# ======================================================================================

#: What `Sw + So + Sg` really sums to in this build, and why it is not exactly one.
#:
#: `JutulDarcy/src/blackoil/variables/varswitch.jl:update_saturations!` reconstructs the oil
#: and gas saturations from the black-oil unknown as
#: `rem = 1 - Sw + MINIMUM_COMPOSITIONAL_SATURATION`, and that constant is `1e-10`
#: (`src/multicomponent/multicomponent.jl:2`). The three saturations of every black-oil cell
#: therefore sum to `1 + 1e-10` BY CONSTRUCTION, in the pinned engine, on every step.
#:
#: It is recorded rather than renormalised away. The measured drift is compared against this
#: native constant with a small margin for float64 round-off, so a real saturation defect —
#: anything an order of magnitude above the engine's own epsilon — still fails.
const BO_NATIVE_SATURATION_EPSILON = 1.0e-10
const BO_SATURATION_SUM_TOLERANCE = 2.0e-10

#: The closed fixture's horizon: three report steps of a day, like 9.5's closed cell.
const BO_CLOSED_EDGES_DAYS = [0.0, 1.0, 2.0, 3.0]

#: The depletion fixture's six report intervals, in days from 2020-01-01. They are the real
#: calendar months January to June 2020, because `chunk_months = 1` chunks on report steps
#: and a restart is taken at one of these boundaries.
const BO_DEPLETION_EDGES_DAYS = [0.0, 31.0, 60.0, 91.0, 121.0, 152.0, 182.0]

#: The producer's bottom-hole pressure, one value per report interval, in Pa. It starts at
#: twice the 53.67 bar bubble point of `BO_INITIAL_RS` and ends well below it, so the phase
#: transition happens inside the fixture rather than at its edge. Calibrated with the tank
#: constant below: the measured free gas saturation is 0 through the first three intervals,
#: 0.015 at the end of the fourth, and 0.092 at the end of the sixth.
const BO_DEPLETION_BHP_PA = [110.0e5, 80.0e5, 60.0e5, 45.0e5, 36.0e5, 30.0e5]

#: Which report step the restart pair is split after — the step gas APPEARS in, so that the
#: checkpoint carries a state that already holds free gas.
#:
#: That is the point of the split and not an arbitrary place to cut it. A black-oil primary
#: unknown is a `BlackOilX`: a value together with the PHASE STATE it belongs to (`OilOnly`
#: or `OilAndGas`). A continuation that restored saturations alone would resume an
#: undersaturated cell as undersaturated whatever its pressure said, so splitting after the
#: transition is what makes the native checkpoint's phase state a thing the comparison can
#: fail on.
const BO_RESTART_AFTER_STEP = 4

const BO_POROSITY = 0.2
const BO_PERMEABILITY_M2 = 100.0 * MILLIDARCY_M2

#: The depletion fixture's own geometry and permeability, CALIBRATED rather than inherited.
#:
#: A closed tank produced at a fixed bottom-hole pressure approaches that pressure with a time
#: constant of roughly `PV * c_t / (WI * lambda)`. On the closed fixture's 10 m cubes at 100 mD
#: that constant is a couple of DAYS: the box equalised with the well inside the first report
#: interval, the rate fell to 8e-4 m3/day, and JutulDarcy (which shuts a well it cannot flow)
#: reported the operating control as `disabled` — a `CONTROL_INFEASIBLE` result and not a
#: depletion at all. Measured, not guessed: see the calibration table in the task report.
#:
#: These values put that constant at a few weeks, so the reservoir TRACKS the falling
#: bottom-hole pressure with a lag, the producer keeps flowing on the control it was given for
#: every interval, and the last two intervals take the cells below the bubble point.
const BO_DEPLETION_CELL_EDGE_M = 60.0
const BO_DEPLETION_PERMEABILITY_M2 = 8.0 * MILLIDARCY_M2

"""
    blackoil_case(name) -> (case, arrays)

One registered black-oil fixture, fully materialised in the exchange's own JSON shape.
"""
function blackoil_case(name::Symbol)
    if name === :bo_closed
        return blackoil_closed_case()
    elseif name === :bo_depletion
        return blackoil_depletion_case()
    elseif name === :unknown_physics_class
        case, arrays = blackoil_closed_case()
        broken = Dict{String,Any}(case)
        fluid = Dict{String,Any}(broken["fluids"])
        fluid["kind"] = "XYZ"
        broken["fluids"] = fluid
        return (broken, arrays)
    end
    error(
        "blackoil_case: fixture $(repr(name)) is not registered; this build has :bo_closed, " *
        ":bo_depletion and :unknown_physics_class",
    )
end

"""One cell, one shut monitor well, no source. A closed black-oil system that must not move."""
function blackoil_closed_case()
    dims = (1, 1, 1)
    extent = (CELL_EDGE_M, CELL_EDGE_M, CELL_EDGE_M)
    case = analytic_skeleton(
        case_id = "blackoil-closed-cell",
        dims = dims,
        extent = extent,
        fluids = blackoil_fluids(),
        wells = Any[analytic_well("MONBO", [0]; reference_depth = DATUM_M)],
        controls = shut_monitor_controls("MONBO", BO_CLOSED_EDGES_DAYS, 1),
        report_edges_days = copy(BO_CLOSED_EDGES_DAYS),
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => uniform_cell_centers(dims, extent),
        "porosity" => fill(BO_POROSITY, 1),
        "permeability_m2" => fill(BO_PERMEABILITY_M2, 3, 1),
        "pressure_pa" => fill(BO_INITIAL_PRESSURE_PA, 1),
        "sw" => fill(BO_INITIAL_SW, 1),
    )
    return (case, arrays)
end

"""A (4,1,1) box and one producer whose bottom-hole pressure steps across the bubble point.

The keywords exist so the fixture's own tank constant can be MEASURED rather than guessed:
the pore volume and the permeability together decide how fast the box equalises with the
well, and a box that equalises inside one report interval is a box whose producer stops
flowing and is then shut by the engine — a `CONTROL_INFEASIBLE` result rather than a
depletion. The defaults are the calibrated ones and the diagnostic never passes anything else.
"""
function blackoil_depletion_case(;
    cell_edge_m::Float64 = BO_DEPLETION_CELL_EDGE_M,
    permeability_m2::Float64 = BO_DEPLETION_PERMEABILITY_M2,
    bhp_pa::Vector{Float64} = collect(BO_DEPLETION_BHP_PA),
)
    dims = (4, 1, 1)
    n_cells = prod(dims)
    extent = (cell_edge_m * dims[1], cell_edge_m, cell_edge_m)
    edges = copy(BO_DEPLETION_EDGES_DAYS)
    length(bhp_pa) == length(edges) - 1 || error(
        "blackoil_depletion_case: one bottom-hole pressure per report interval is required",
    )
    controls = Any[
        control_segment(
            start_day = edges[i],
            end_day = edges[i + 1],
            well_id = "PBO",
            role = "producer",
            target = "bhp",
            value = bhp_pa[i],
            bhp_limit_pa = nothing,
            connection_open = [true],
        ) for i in 1:length(bhp_pa)
    ]
    case = analytic_skeleton(
        case_id = "blackoil-depletion-4-cell",
        dims = dims,
        extent = extent,
        fluids = blackoil_fluids(),
        # The producer sits at the far end of the box, so the depletion has to travel the
        # length of it: a well in the first cell would drop the whole model's pressure at
        # once and the phase transition would be a single cell's.
        wells = Any[analytic_well("PBO", [n_cells - 1]; reference_depth = DATUM_M)],
        controls = controls,
        report_edges_days = edges,
    )
    arrays = Dict{String,Any}(
        "cell_centers_m" => uniform_cell_centers(dims, extent),
        "porosity" => fill(BO_POROSITY, n_cells),
        "permeability_m2" => fill(permeability_m2, 3, n_cells),
        "pressure_pa" => fill(BO_INITIAL_PRESSURE_PA, n_cells),
        "sw" => fill(BO_INITIAL_SW, n_cells),
    )
    return (case, arrays)
end

# ======================================================================================
# driving a fixture
# ======================================================================================

#: The solver settings the E01 worker drives every forward with (`suites.E01_SOLVER`), in the
#: keywords the native driver takes. They are stated here so that the continuous trajectory
#: this diagnostic publishes and the restart pair the Python side runs through the worker are
#: the SAME discretisation of the same case, and a difference between them is a restart
#: defect rather than a solver setting nobody matched.
function blackoil_solver_options()
    return Dict{String,Any}(
        "max_timestep" => 5.0 * SECONDS_PER_DAY,
        "max_nonlinear_iterations" => 15,
    )
end

"""
    run_blackoil(name, native_dir; chunk_months) -> Dict

Build and run one black-oil fixture through the production drivers, and return everything.

Construction and simulation are separate phases, exactly as in `analytic.jl`: a fixture that
fails to BUILD has to be distinguishable from one that failed to converge, and the unknown
physics class is the former.
"""
function run_blackoil(name::Symbol, native_dir::AbstractString; chunk_months::Int = 1)
    case, arrays = blackoil_case(name)
    out = Dict{String,Any}(
        "name" => String(name),
        "case" => case,
        "arrays" => serialisable_arrays(arrays),
        "solver_options" => blackoil_solver_options(),
        "chunk_months" => chunk_months,
    )
    physical = try
        ADAPTER.build_physical(case, arrays)
    catch err
        out["status"] = ADAPTER.failure_status(err)
        out["phase"] = "build"
        out["reason"] = sprint(showerror, err)
        out["is_invalid_case_input"] = err isa ADAPTER.InvalidCaseInput
        return out
    end
    out["phase"] = "solve"
    out["native"] = blackoil_native_diagnostics(physical)
    solver = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in blackoil_solver_options())
    mkpath(native_dir)
    out["extraction"] = ADAPTER.run_forward_native(
        case, arrays, native_dir; chunk_months = chunk_months, solver...,
    )
    out["status"] = String(out["extraction"]["status"])
    return out
end

"""
    blackoil_native_diagnostics(physical) -> Dict

What only Julia can see about the black-oil model that was just built, BEFORE a timestep.

The phases, the reference densities and the relative permeability are read off the OBJECTS
the solver will use — not off the case that asked for them — so a case whose declaration and
whose model disagreed would be visible here rather than in a result that looks plausible.
"""
function blackoil_native_diagnostics(physical)
    rmodel = physical.model.models[:Reservoir]
    system = rmodel.system
    kr = Jutul.get_secondary_variables(rmodel)[:RelativePermeabilities]
    evaluated = ADAPTER.evaluated_state(physical.model, physical.parameters, physical.state0)
    reservoir = evaluated[:Reservoir]
    saturations = reservoir[:Saturations]
    return Dict{String,Any}(
        "phases" => [string(typeof(p)) for p in JutulDarcy.get_phases(system)],
        "n_phases" => length(JutulDarcy.get_phases(system)),
        "reference_densities_kg_m3" =>
            Float64[d for d in JutulDarcy.reference_densities(system)],
        "has_disgas" => JutulDarcy.has_disgas(system),
        "has_vapoil" => JutulDarcy.has_vapoil(system),
        "relative_permeability" => string(typeof(kr)),
        "kr_exponents" => Float64[x for x in kr.exponents],
        "kr_residuals" => Float64[x for x in kr.residuals],
        "kr_endpoints" => Float64[x for x in kr.endpoints],
        "pore_volume_m3" =>
            sum(Float64.(JutulDarcy.pore_volume(physical.model, physical.parameters))),
        "initial_saturations" =>
            [Float64[saturations[ph, c] for c in axes(saturations, 2)] for ph in axes(saturations, 1)],
        "initial_rs" => collect(Float64.(reservoir[:Rs])),
        "gravity_m_s2" => Jutul.gravity_constant,
    )
end

# ======================================================================================
# 13.3 — the phase dispatch regression the oil-water results have to survive
# ======================================================================================

"""
    phase_dispatch_evidence() -> Dict

Task 13.3 changed `native_control`'s signature to take a phase count. This is the evidence
that at `n_phases = 2` it produces what it produced before.

The injected mixture is compared against `WATER_INJECTION_MIXTURE` — the literal the oil-water
results were produced with — and the controls themselves are compared object for object
between the two-argument call and the explicit three-argument one. The three-phase mixture is
shown beside them so that the change is visible as well as proved harmless.
"""
function phase_dispatch_evidence()
    rho_w = 1000.0
    segments = Dict{String,Any}(
        "injector_bhp" => control_segment(
            start_day = 0.0, end_day = 1.0, well_id = "INJ", role = "injector",
            target = "bhp", value = 2.2e7, bhp_limit_pa = nothing, connection_open = [true],
        ),
        "injector_rate" => control_segment(
            start_day = 0.0, end_day = 1.0, well_id = "INJ", role = "injector",
            target = "water_rate", value = 2.0, bhp_limit_pa = nothing,
            connection_open = [true],
        ),
        "producer_rate" => control_segment(
            start_day = 0.0, end_day = 1.0, well_id = "PRO", role = "producer",
            target = "liquid_rate", value = 2.0, bhp_limit_pa = nothing,
            connection_open = [true],
        ),
    )
    controls = Dict{String,Any}()
    for (label, segment) in segments
        default = ADAPTER.native_control(segment, rho_w)
        explicit = ADAPTER.native_control(segment, rho_w, 2)
        three = ADAPTER.native_control(segment, rho_w, 3)
        # `InjectorControl` holds its mixture as an ARRAY, and Julia's structural equality
        # compares array fields by identity, so two controls built from two separately
        # allocated `[1.0, 0.0]` are never `==`. The comparison that means something is
        # therefore made field by field: the same native type, the same target, and the same
        # mixture BY VALUE.
        mixture(c) = hasproperty(c, :injection_mixture) ? Float64[x for x in c.injection_mixture] :
                     Float64[]
        target(c) = hasproperty(c, :target) ? string(c.target) : string(typeof(c))
        controls[label] = Dict{String,Any}(
            "type" => string(typeof(default)),
            "default_equals_explicit_two_phase" =>
                typeof(default) == typeof(explicit) && target(default) == target(explicit) &&
                mixture(default) == mixture(explicit),
            "target" => target(default),
            "target_unchanged_at_three_phases" => target(three) == target(default),
            "two_phase_mixture" => mixture(default),
            "three_phase_mixture" => mixture(three),
        )
    end
    return Dict{String,Any}(
        "water_injection_mixture_two_phase" => Float64[x for x in ADAPTER.WATER_INJECTION_MIXTURE],
        "water_injection_mixture_from_count" =>
            Dict{String,Any}(string(n) => ADAPTER.water_injection_mixture(n) for n in (1, 2, 3)),
        "controls" => controls,
    )
end

# ======================================================================================
# the diagnostic (plan 13.2, 13.4). Only runs as a program.
# ======================================================================================

"""Free and dissolved gas inventory of one published state, in m3_sc, and their sum."""
function gas_inventory(states::AbstractDict, index::Int)
    pv = Float64.(states["pore_volume_m3"][index])
    sg = Float64.(states["sg"][index])
    so = Float64.(states["so"][index])
    bo = Float64.(states["bo"][index])
    bg = Float64.(states["bg"][index])
    rs = Float64.(states["rs"][index])
    free = sum(pv .* sg ./ bg)
    dissolved = sum(rs .* pv .* so ./ bo)
    return (free = free, dissolved = dissolved, total = free + dissolved)
end

function selftest_blackoil(native_root::AbstractString)
    pvt = ADAPTER.blackoil_pvt()
    pvt_export = ADAPTER.blackoil_pvt_export(pvt)
    relperm_export = ADAPTER.blackoil_relperm_export(blackoil_fluids())
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "gravity_constant" => Jutul.gravity_constant,
        "pvt_export" => pvt_export,
        "relperm_export" => relperm_export,
        "phase_dispatch" => phase_dispatch_evidence(),
        "saturated_rs_at_initial_pressure" => ADAPTER.saturated_rs(pvt, BO_INITIAL_PRESSURE_PA),
        "bubble_point_pa" => bubble_point_pressure(pvt, BO_INITIAL_RS),
        "initial_rs_m3_m3" => BO_INITIAL_RS,
        "restart_after_report_step" => BO_RESTART_AFTER_STEP,
        "restart_after_time_s" =>
            BO_DEPLETION_EDGES_DAYS[BO_RESTART_AFTER_STEP + 1] * SECONDS_PER_DAY,
        "native_saturation_epsilon" => BO_NATIVE_SATURATION_EPSILON,
        "saturation_sum_tolerance" => BO_SATURATION_SUM_TOLERANCE,
    )
    fixtures = Dict{String,Any}()

    @testset "E01 educational black-oil capability" begin
        # --- 13.3 the phase dispatch regression -----------------------------------------
        dispatch = measured["phase_dispatch"]
        @test dispatch["water_injection_mixture_two_phase"] == [1.0, 0.0]
        @test dispatch["water_injection_mixture_from_count"]["2"] == [1.0, 0.0]
        @test dispatch["water_injection_mixture_from_count"]["3"] == [1.0, 0.0, 0.0]
        for (_label, control) in dispatch["controls"]
            @test control["default_equals_explicit_two_phase"]
            @test control["target_unchanged_at_three_phases"]
            mixture = control["two_phase_mixture"]
            if !isempty(mixture)
                @test mixture == [1.0, 0.0]
                @test control["three_phase_mixture"] == [1.0, 0.0, 0.0]
            end
        end

        # --- the exported PVT is the pinned benchmark's own numbers ----------------------
        @test pvt_export["academic_benchmark"]
        @test pvt_export["reference_densities_kg_m3"] == [1037.84, 786.507, 0.969758]
        @test pvt_export["reference_densities_deck_order_kg_m3"] == [786.507, 1037.84, 0.969758]
        @test length(pvt_export["pvto"]["rs"]) == length(pvt_export["pvto"]["sat_pressure_pa"])

        # --- 13.1 an unknown physics class is INVALID_INPUT ------------------------------
        unknown = run_blackoil(:unknown_physics_class, joinpath(native_root, "unknown"))
        @test unknown["status"] == "INVALID_INPUT"
        @test unknown["phase"] == "build"
        @test unknown["is_invalid_case_input"]
        @test !haskey(unknown, "extraction")
        fixtures["unknown_physics_class"] = unknown

        # --- bo_closed: a closed black-oil system at rest --------------------------------
        closed = run_blackoil(:bo_closed, joinpath(native_root, "bo_closed"))
        @test closed["status"] == "COMPLETE"
        fixtures["bo_closed"] = closed
        closed_states = closed["extraction"]["states"]
        @test haskey(closed_states, "sg") && haskey(closed_states, "rs") &&
              haskey(closed_states, "bg")
        @test closed["native"]["n_phases"] == 3
        @test closed["native"]["has_disgas"]
        @test !closed["native"]["has_vapoil"]
        @test closed["extraction"]["components"] == ["water", "oil", "gas"]
        first_gas = gas_inventory(closed_states, 1)
        last_gas = gas_inventory(closed_states, length(closed_states["times_s"]))
        @test first_gas.free <= 1.0e-10
        @test first_gas.dissolved > 0.0
        @test abs(last_gas.total - first_gas.total) <= 1.0e-8 * first_gas.total
        @test all(c["total_mass_kg_s"] == 0.0 for c in closed["extraction"]["connections"])
        measured["bo_closed"] = Dict{String,Any}(
            "initial_free_gas_m3_sc" => first_gas.free,
            "initial_dissolved_gas_m3_sc" => first_gas.dissolved,
            "final_free_gas_m3_sc" => last_gas.free,
            "final_dissolved_gas_m3_sc" => last_gas.dissolved,
        )

        # --- bo_depletion: gas comes out of solution below the bubble point --------------
        depletion = run_blackoil(:bo_depletion, joinpath(native_root, "bo_depletion"))
        @test depletion["status"] == "COMPLETE"
        fixtures["bo_depletion"] = depletion
        states = depletion["extraction"]["states"]
        n = length(states["times_s"])
        @test n == length(BO_DEPLETION_EDGES_DAYS)
        @test maximum(states["sg"][1]) <= 1.0e-10
        @test maximum(states["sg"][n]) > 1.0e-3
        # The split of 13.4's restart pair is taken after the step gas appears in, so the
        # checkpoint has to be a state that already holds free gas. Asserted here rather
        # than assumed: a fixture recalibrated into a different schedule would otherwise
        # move the transition past the split without anything noticing.
        @test maximum(states["sg"][BO_RESTART_AFTER_STEP + 1]) > 1.0e-3
        @test maximum(states["sg"][BO_RESTART_AFTER_STEP]) <= 1.0e-10
        @test minimum(states["rs"][n]) < BO_INITIAL_RS
        @test minimum(states["pressure_pa"][n]) < bubble_point_pressure(pvt, BO_INITIAL_RS)
        for k in 1:n
            total = states["sw"][k] .+ states["so"][k] .+ states["sg"][k]
            @test maximum(abs.(total .- 1.0)) <= BO_SATURATION_SUM_TOLERANCE
            @test minimum(states["sg"][k]) >= -1.0e-8
            @test minimum(states["rs"][k]) >= 0.0
            @test minimum(states["bo"][k]) > 0.0
            @test minimum(states["bg"][k]) > 0.0
        end
        first_gas = gas_inventory(states, 1)
        last_gas = gas_inventory(states, n)
        saturation_drift = maximum(
            maximum(abs.(states["sw"][k] .+ states["so"][k] .+ states["sg"][k] .- 1.0))
            for k in 1:n
        )
        measured["bo_depletion"] = Dict{String,Any}(
            "saturation_sum_drift" => saturation_drift,
            "max_sg_per_state" => [maximum(states["sg"][k]) for k in 1:n],
            "min_pressure_pa_per_state" => [minimum(states["pressure_pa"][k]) for k in 1:n],
            "max_sg_at_restart_boundary" => maximum(states["sg"][BO_RESTART_AFTER_STEP + 1]),
            "cell_edge_m" => BO_DEPLETION_CELL_EDGE_M,
            "permeability_m2" => BO_DEPLETION_PERMEABILITY_M2,
            "bhp_schedule_pa" => collect(BO_DEPLETION_BHP_PA),
            "bubble_point_pa" => bubble_point_pressure(pvt, BO_INITIAL_RS),
            "initial_free_gas_m3_sc" => first_gas.free,
            "initial_dissolved_gas_m3_sc" => first_gas.dissolved,
            "final_free_gas_m3_sc" => last_gas.free,
            "final_dissolved_gas_m3_sc" => last_gas.dissolved,
            "final_max_sg" => maximum(states["sg"][n]),
            "final_min_pressure_pa" => minimum(states["pressure_pa"][n]),
            "final_min_rs" => minimum(states["rs"][n]),
        )
        @test last_gas.free > first_gas.free
    end

    measured["fixtures"] = fixtures
    measured["status"] = "ok"
    return measured
end

"""The pressure at which the exported saturation table dissolves exactly `rs`."""
function bubble_point_pressure(pvt, rs::Float64)
    lo, hi = extrema(pvt.pvto.tab[1].sat_pressure)
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        if ADAPTER.saturated_rs(pvt, mid) < rs
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function parse_blackoil_args(args::Vector{String})
    out = Dict{String,Any}("mode" => nothing, "out" => nothing, "native" => nothing)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--test-blackoil"
            out["mode"] = "test"
        elseif a == "--out"
            i += 1
            out["out"] = args[i]
        elseif a == "--native-dir"
            i += 1
            out["native"] = args[i]
        else
            error("blackoil.jl: unknown argument $(repr(a))")
        end
        i += 1
    end
    return out
end

function blackoil_main(args::Vector{String})
    options = parse_blackoil_args(args)
    options["mode"] == "test" ||
        error("blackoil.jl: this diagnostic runs with --test-blackoil")
    native_root = options["native"] === nothing ? mktempdir(; cleanup = false) :
                  String(options["native"])
    mkpath(native_root)
    payload = try
        selftest_blackoil(native_root)
    catch err
        Base.showerror(stderr, err, catch_backtrace())
        Dict{String,Any}("status" => "error", "message" => sprint(showerror, err))
    end
    if options["out"] !== nothing
        path = String(options["out"])
        mkpath(dirname(path))
        open(path, "w") do io
            JSON.print(io, payload)
            print(io, "\n")
        end
    else
        JSON.print(stdout, payload)
        println()
    end
    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    payload = blackoil_main(ARGS)
    exit(payload["status"] == "ok" ? 0 : 1)
end
