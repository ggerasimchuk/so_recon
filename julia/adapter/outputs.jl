# E01.7 — what a finished simulation is asked for, and nothing it is not.
#
# `model.jl` builds the model, `controls.jl` builds the forces it is driven with, and this
# file reads the RESULT. The split is the same one as before: nothing here constructs a
# mesh, a control or a variable, and nothing here computes a flux of its own.
#
# THE ACCEPTED SUBSTEP IS THE UNIT. `Jutul.expand_to_ministeps(result.result)` is the
# supported way to reach the solver's own accepted mini-steps, and it is given
# `result.result` — the raw `SimResult` — rather than the high-level `result.states`,
# because the latter is `map(x -> x[:Reservoir], states)` and carries no facility and no
# wellbore at all. `report_timesteps(..., ministeps = true)` keeps only `m[:success]`, and
# `solve_timestep!` pushes a substate only after a mini-step converged, so a failed or cut
# trial step is not in this list: what comes back tiles the chunk exactly, with every `dt`
# positive and `sum(dt)` equal to the chunk duration. Those three properties are asserted
# here rather than assumed, because they are what makes `Σ rate·dt` the same backward-Euler
# quadrature the solver used instead of a re-interpolation of it (SPEC 9.3).
#
# THE FLOW IS TAKEN FROM THE NATIVE CROSS TERM, NOT DISTRIBUTED BY kh. A connection's
# component mass flux is `multisegment_well_perforation_flux!` (or `simple_well_perforation_flux!`
# for a `SimpleWell`, which is the same dispatch `update_cross_term_in_entity!` makes),
# evaluated on the substep's OWN upwind state and multiplied by the interval's real
# `PerforationMask`. Splitting a surface rate over connections by well index would put flow
# through a completion the case had shut, and would agree with the surface total while being
# wrong everywhere it matters.
#
# SECONDARY FIELDS ARE RESTORED BY THE NATIVE UPDATE. A substate carries primary variables
# and whatever `extra_outputs` kept — for this model that is `Pressure`, `Saturations` and
# `TotalMasses` on the reservoir — so densities and mobilities have to be recomputed before
# any flux can be evaluated. That is done with `Jutul.evaluate_all_secondary_variables`,
# which runs the MODEL's own variable definitions over the substate merged with the model's
# parameters. Reconstructing a phase composition from surface fractions instead would be a
# second, unvalidated PVT living beside the one the solver used.
#
# THE INVENTORY INCLUDES THE WELLBORE. `TotalMasses` is a component mass in kg, in the
# reservoir and in every well node; divided by the phase's reference density it is the
# standard-volume inventory the balance is evaluated on. The surface flux does NOT have to
# equal the sum of the connection fluxes at every instant — the difference is the wellbore
# filling or emptying — so the two balances are kept apart: the reservoir against the
# connections, and the reservoir plus the wells against the surface.

#: The shape of the payload this file writes. The Python side checks it: a reader that
#: guesses which version it is looking at is a reader that will one day guess wrong.
const EXTRACT_SCHEMA_VERSION = "forward-extract-1"

#: Water first, oil second — the order `OW_PHASES` fixes and the whole exchange uses.
const EXTRACT_COMPONENTS = ("water", "oil")

#: How far a substep boundary may sit from a requested state time and still be it. The
#: schedule's times are day counts times 86400, so this is far above float64 round-off at
#: reservoir time scales and far below any event anybody schedules.
const TIME_MATCH_TOLERANCE_S = 1e-6

#: Fields the native perforation flux reads that are EXTRA STATE FIELDS rather than
#: variables, and are therefore not stored unless a model is asked to store them.
#:
#: There is one in JutulDarcy 0.3.11. A `SimpleWell` built with `explicit_dp = true` — the
#: constructor's default — keeps its connection pressure drop in
#: `initialize_extra_state_fields!` (`src/facility/wells/stdwells.jl:71`), and
#: `select_minimum_output_variables!` does not list it, so `get_output_state` never copies
#: it out. `perforation_phase_potential_difference` (`src/facility/cross_terms.jl:74`) reads
#: it in preference to the density head, so without it every connection flux of such a well
#: is computed from zeros: measured 6.8 m³_sc of reservoir mass unaccounted for over a single
#: day. `request_extra_outputs!` asks for it and `evaluated_state` restores it.
const PERFORATION_EXTRA_STATE_FIELDS = (:ConnectionPressureDrop,)

# --------------------------------------------------------------------------------------
# the schedule, as the production path compiles it
# --------------------------------------------------------------------------------------

"""
    compile_intervals(case) -> (edges_s, month_index, controls)

Group a case's flat control segments into the per-interval lists `build_forces` takes: the
sorted union of the report edges and every segment boundary, and for each of those intervals
exactly one control per well.

This is the Julia side of `so_recon.simulator.schedule.compile_schedule`, which is the gate
the production path passes through FIRST — a gap, an overlap or a segment reaching past the
last report edge is refused there, in Python, before a Julia process is asked for anything.
The same refusals are restated here because the two compilers must not be able to drift
apart silently: a well with no control on an interval would otherwise be filled in by
`setup_forces` with `DisabledControl()`, and a schedule with a hole in it would become a
well that quietly stopped producing. The integration test compares the two edge lists.
"""
function compile_intervals(case::AbstractDict)
    report = Float64.(case["report_edges_s"])
    length(report) >= 2 ||
        invalid("compile_intervals: report_edges_s must describe at least one report interval")
    report[1] == 0.0 ||
        invalid("compile_intervals: report_edges_s must start at 0 s, got $(report[1])")
    all(report[i + 1] > report[i] for i in 1:(length(report) - 1)) ||
        invalid("compile_intervals: report_edges_s must be strictly increasing, got $(report)")

    segments = collect(get(case, "controls", Any[]))
    isempty(segments) && invalid("compile_intervals: a schedule needs at least one control segment")
    horizon = report[end]
    boundaries = Set{Float64}(report)
    for s in segments
        Float64(s["end_s"]) <= horizon || invalid(
            "compile_intervals: well $(well_name(s)) has a control segment ending at " *
            "$(s["end_s"]) s, past the last report edge at $(horizon) s",
        )
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
            length(covering) == 1 || invalid(
                "compile_intervals: well $(w) has $(length(covering)) controls on interval " *
                "$(i - 1) [$(a), $(b)) s; every well carries exactly one control on every " *
                "interval and nothing is inherited",
            )
            push!(group, covering[1])
        end
        push!(controls, group)
    end
    return (edges_s = edges, month_index = month_index, controls = controls)
end

# --------------------------------------------------------------------------------------
# native state reconstruction
# --------------------------------------------------------------------------------------

"""
    evaluated_state(model, parameters, state) -> Dict{Symbol, Any}

Every submodel's secondary variables, recomputed from the substate by the MODEL's own
variable definitions.

`Jutul.evaluate_all_secondary_variables` merges the submodel's parameters into the state and
runs `update_secondary_variables_state!` over it, so what comes back carries the primary
variables the solver stored, the parameters the model was built with (well indices,
perforation gravity differences, viscosities) and the densities, mobilities and component
masses the model's own variables define. The facility is skipped: it holds a control
configuration rather than a field, and has no secondary variables to restore.
"""
function evaluated_state(model, parameters, state::AbstractDict)
    out = Dict{Symbol,Any}()
    for name in vcat([:Reservoir], well_names(model))
        haskey(state, name) ||
            invalid("extract_interval: the stored state carries no submodel $(name)")
        stored = state[name]
        evaluated = Jutul.evaluate_all_secondary_variables(
            model.models[name],
            stored,
            parameters[name],
        )
        # An EXTRA STATE FIELD is not a secondary variable and is not recomputed from
        # anything: `setup_state!` ends by calling `initialize_extra_state_fields!`, which
        # assigns the field its initial value unconditionally — zeros, for a `SimpleWell`'s
        # explicit connection pressure drop. `perforation_phase_potential_difference` prefers
        # that field over the density head whenever it is present, so evaluating a substate
        # and using the result directly would compute every perforation flux from those
        # zeros. The value the solver actually used is restored from the stored substate
        # instead, and a substate that does not carry it is refused rather than guessed.
        for field in PERFORATION_EXTRA_STATE_FIELDS
            haskey(evaluated, field) || continue
            haskey(stored, field) || invalid(
                "extract_interval: well $(name) carries an explicit $(field), which " *
                "JutulDarcy $(pkgversion(JutulDarcy)) keeps as an extra state field rather " *
                "than a variable, and this run did not store it. The native perforation flux " *
                "reads that field in preference to the density head, so a flux rebuilt here " *
                "would come from a re-initialised zero and would conserve nothing. Call " *
                "request_extra_outputs!(model) before simulating",
            )
            evaluated[field] = copy(stored[field])
        end
        out[name] = evaluated
    end
    return out
end

"""Component inventory of one evaluated submodel, in standard m³ (`mass / rho_sc`)."""
function component_inventory(evaluated, name::Symbol, rhoS::Vector{Float64})
    masses = evaluated[name][:TotalMasses]
    return Float64[sum(view(masses, c, :)) / rhoS[c] for c in eachindex(rhoS)]
end

"""
    request_extra_outputs!(model) -> Dict{String, Vector{String}}

Ask every well submodel to STORE the extra state fields its perforation flux depends on.

A `SimulationModel` carries its own `output_variables`, and `get_output_state`
(`Jutul/src/models.jl:1048`) copies exactly those; Jutul's own `check_output_variables`
enumerates `initialize_extra_state_fields!` alongside the variables when it decides what a
model may legally be asked for, so an extra state field is a permitted output and is simply
not one of the defaults. Pushing it on is therefore a change to what is RECORDED and not to
what is solved: no equation, variable or parameter moves, and the model hash covers none of
it.

This has to be called on the model BEFORE it is simulated. `run_forward` does it; a driver
that assembles its own simulation must do it too, and `evaluated_state` refuses rather than
silently reading a re-initialised zero if it was not.

Returns what was added, per well, so a diagnostic can show the mechanism fired.
"""
function request_extra_outputs!(model)
    added = Dict{String,Vector{String}}()
    for name in well_names(model)
        submodel = model.models[name]
        probe = Jutul.JutulStorage()
        Jutul.initialize_extra_state_fields!(probe, submodel)
        gained = String[]
        for field in PERFORATION_EXTRA_STATE_FIELDS
            (haskey(probe, field) && !(field in submodel.output_variables)) || continue
            push!(submodel.output_variables, field)
            push!(gained, String(field))
        end
        unique!(submodel.output_variables)
        isempty(gained) || (added[String(name)] = gained)
    end
    return added
end

"""
    interval_mask(force, well) -> Vector

The completion multipliers one interval's forces really carry for one well.

`build_forces` always sets a `PerforationMask`, and this extraction depends on it: without
one there is no way to tell a connection the case shut from one it left open, and every
connection flux below would be reported as if the whole completion were producing. A force
without a mask is therefore named rather than defaulted to all-open.
"""
function interval_mask(force, well::Symbol)
    well_force = get(force, well, nothing)
    (well_force !== nothing && haskey(well_force, :mask) && well_force.mask !== nothing) || invalid(
        "extract_interval: the forces of this interval carry no PerforationMask for well " *
        "$(well); a connection flux read without one cannot tell a shut completion from an " *
        "open one",
    )
    return well_force.mask.values
end

"""The reservoir-well cross terms of this model, keyed by the well they belong to."""
function reservoir_well_cross_terms(model)
    out = Dict{Symbol,Any}()
    for ctp in model.cross_terms
        if ctp.cross_term isa JutulDarcy.ReservoirFromWellFlowCT && ctp.target == :Reservoir
            out[ctp.source] = ctp.cross_term
        end
    end
    return out
end

"""
    connection_component_flux(system, reservoir, well, cross_term, mask) -> Matrix (nph, nconn)

Component mass flux through each of one well's connections, kg/s, POSITIVE out of the
reservoir — the same sign `update_cross_term_in_entity!` gives the reservoir equation.

The connection tuple is built by JutulDarcy's own `cross_term_perforation_get_conn`, so the
well index, the perforation gravity difference and the pressure difference are the ones the
solver used. The interval's `PerforationMask` multiplies the result exactly as
`apply_force_to_cross_term!` does, so a shut completion moves nothing rather than moving its
share of a surface total.

`update_cross_term_in_entity!` picks between two kernels on one test — whether the well
state carries `MassFractions` — and that test is restated here. Every well this adapter
builds is a `SimulationModel` over an `ImmiscibleSystem` with `Saturations` as the well's
primary variable, so the test is false and `multisegment_well_perforation_flux!` is what
runs, for a `SimpleWell` domain as much as for a `MultiSegmentWell` one. A model that did
carry `MassFractions` would need `simple_well_perforation_flux!` instead; rather than carry
an untested branch for a model E01 does not build, that case is refused by name.
"""
function connection_component_flux(system, reservoir, well, cross_term, mask::Vector{Float64})
    state_res = Jutul.convert_to_immutable_storage(reservoir)
    state_well = Jutul.convert_to_immutable_storage(well)
    haskey(state_well, :MassFractions) && invalid(
        "extract_interval: this well carries MassFractions, so its native perforation flux " *
        "is simple_well_perforation_flux! rather than the multisegment kernel. No E01 model " *
        "builds such a well, and this extraction will not guess which kernel the solver used",
    )
    rhoS = collect(JutulDarcy.reference_densities(system))
    n_phases = length(rhoS)
    n_conn = length(cross_term.reservoir_cells)
    length(mask) == n_conn || invalid(
        "extract_interval: the interval's perforation mask has $(length(mask)) entries for " *
        "$(n_conn) connections; `apply_perforation_mask!` iterates `eachindex(mask)`, so a " *
        "short mask would leave the remaining perforations fully open",
    )
    flux = zeros(Float64, n_phases, n_conn)
    out = zeros(Float64, n_phases)
    for i in 1:n_conn
        conn = JutulDarcy.cross_term_perforation_get_conn(cross_term, i, state_well, state_res)
        JutulDarcy.multisegment_well_perforation_flux!(
            out, system, state_res, state_well, rhoS, conn,
        )
        for ph in 1:n_phases
            flux[ph, i] = Jutul.value(out[ph]) * mask[i]
        end
    end
    return flux
end

# --------------------------------------------------------------------------------------
# the extraction
# --------------------------------------------------------------------------------------

"""
    extract_interval(result, model, parameters, forces, state0; kwarg...) -> Dict

Everything a forward result is built from, read out of one finished simulation.

The plan's interface is `extract_interval(result, model, forces)`; the installed
`Jutul 0.4.31` / `JutulDarcy 0.3.11` need two more inputs and they are taken here rather
than reconstructed. `parameters` is required because the well index and the perforation
gravity difference a connection flux is built from are model PARAMETERS and are not in any
substate, and because the model's own secondary-variable update needs them; `state0` is
required because the inventory of the first substep is the state BEFORE it, which the
solver never stores as a result.

`controls_by_interval` and `edges_s` come from `compile_intervals`, `state_times_s` names
the times the states are wanted at, and `chunk_start_s` places this chunk on the case's own
clock so that a later chunked driver can offset it without this function knowing about
chunking at all.
"""
function extract_interval(
    result,
    model,
    parameters,
    forces,
    state0::AbstractDict;
    controls_by_interval,
    edges_s::Vector{Float64},
    state_times_s::Vector{Float64},
    chunk_start_s::Float64 = 0.0,
)
    sim_result = result.result
    states, dt, report_index = Jutul.expand_to_ministeps(sim_result)
    n = length(states)
    n > 0 || invalid("extract_interval: the simulation produced no accepted substep at all")
    (length(dt) == n && length(report_index) == n) || invalid(
        "extract_interval: expand_to_ministeps returned $(n) states, $(length(dt)) durations " *
        "and $(length(report_index)) report indices",
    )
    all(>(0.0), dt) || invalid(
        "extract_interval: an accepted substep has a non-positive duration $(minimum(dt)) s; " *
        "a failed or cut trial step is not an accepted substep",
    )
    horizon = edges_s[end] - edges_s[1]
    abs(sum(dt) - horizon) <= TIME_MATCH_TOLERANCE_S || invalid(
        "extract_interval: the accepted substeps total $(sum(dt)) s over a chunk of " *
        "$(horizon) s; the simulation did not run to the end of its schedule",
    )

    # Boundaries, absolute on the case's clock. `edges_s` are the interval boundaries and
    # `report_index` says which interval each substep belongs to, so a substep that crossed
    # one would be a solver that ignored its own report step: it is refused, not split.
    ends = chunk_start_s .+ cumsum(dt)
    starts = vcat(chunk_start_s, ends[1:(end - 1)])
    for k in 1:n
        lo = chunk_start_s + (edges_s[report_index[k]] - edges_s[1])
        hi = chunk_start_s + (edges_s[report_index[k] + 1] - edges_s[1])
        (starts[k] >= lo - TIME_MATCH_TOLERANCE_S && ends[k] <= hi + TIME_MATCH_TOLERANCE_S) ||
            invalid(
                "extract_interval: accepted substep $(k - 1) spans [$(starts[k]), $(ends[k])) s " *
                "but belongs to interval $(report_index[k] - 1) [$(lo), $(hi)) s; one substep " *
                "never crosses an event or month boundary",
            )
    end

    # One force object per accepted substep, whichever way the caller supplied them: the
    # mask below is read positionally, and `well_output` takes either shape.
    step_forces = forces isa AbstractVector ? forces[report_index] : fill(forces, n)
    controls_by_step = [controls_by_interval[i] for i in report_index]
    evidence = control_evidence(model, states, step_forces, controls_by_step)

    rmodel = model.models[:Reservoir]
    system = rmodel.system
    rhoS = Float64[d for d in JutulDarcy.reference_densities(system)]
    n_phases = length(rhoS)
    n_phases == length(EXTRACT_COMPONENTS) || invalid(
        "extract_interval: this extraction describes $(length(EXTRACT_COMPONENTS)) components, " *
        "the system has $(n_phases)",
    )
    cross_terms = reservoir_well_cross_terms(model)
    names = well_names(model)

    # Surface quantities, per well. `well_output` with an Int target is the component MASS
    # rate `q_t * mix` — the same product the facility cross term uses — and with a phase
    # target it is the surface volumetric rate. Both come back with the native sign
    # (negative for production); the public split happens on the Python side, once.
    wells_out = Dict{String,Any}()
    surface_mass = Dict{Symbol,Vector{Vector{Float64}}}()
    for name in names
        water = well_output(model, states, name, step_forces, SurfaceWaterRateTarget)
        oil = well_output(model, states, name, step_forces, SurfaceOilRateTarget)
        bhp = well_output(model, states, name, step_forces, BottomHolePressureTarget)
        operating = well_output(model, states, name, step_forces, :control)
        surface_mass[name] = [
            Float64[well_output(model, states, name, step_forces, c)[k] for k in 1:n]
            for c in 1:n_phases
        ]
        wells_out[String(name)] = Dict{String,Any}(
            "cells" => Int[c - 1 for c in cross_terms[name].reservoir_cells],
            "surface_water_m3_s" => collect(Float64.(water)),
            "surface_oil_m3_s" => collect(Float64.(oil)),
            "surface_component_mass_kg_s" => surface_mass[name],
            "bhp_pa" => collect(Float64.(bhp)),
            "operating_target" => String[String(t) for t in operating],
        )
    end

    # The substeps, one at a time: the inventory at the end of each, the connection flux
    # through each perforation, and the source that crossed the surface.
    inventory = Vector{Vector{Float64}}()
    reservoir_inventory = Vector{Vector{Float64}}()
    connection_rows = Any[]
    connection_source = Vector{Vector{Float64}}()
    surface_source = Vector{Vector{Float64}}()

    # The inventory of the FIRST substep is the state before it, which the solver never
    # stores as a result: it is evaluated from `state0` here.
    initial = evaluated_state(model, parameters, state0)
    push!(reservoir_inventory, component_inventory(initial, :Reservoir, rhoS))
    push!(
        inventory,
        reservoir_inventory[end] .+
        sum([component_inventory(initial, w, rhoS) for w in names]; init = zeros(n_phases)),
    )

    for k in 1:n
        evaluated = evaluated_state(model, parameters, states[k])
        into_reservoir = zeros(Float64, n_phases)
        for name in names
            mask = Float64.(interval_mask(step_forces[k], name))
            flux = connection_component_flux(
                system,
                evaluated[:Reservoir],
                evaluated[name],
                cross_terms[name],
                mask,
            )
            # `flux` is positive OUT of the reservoir, so as a source INTO it, it is negative.
            into_reservoir .-= vec(sum(flux, dims = 2))
            target = String(wells_out[String(name)]["operating_target"][k])
            bhp = wells_out[String(name)]["bhp_pa"][k]
            for i in axes(flux, 2)
                push!(
                    connection_rows,
                    Dict{String,Any}(
                        "well_id" => String(name),
                        "connection_id" => i - 1,
                        "cell_id" => cross_terms[name].reservoir_cells[i] - 1,
                        "step" => k - 1,
                        "water_mass_kg_s" => flux[1, i],
                        "oil_mass_kg_s" => flux[2, i],
                        "total_mass_kg_s" => sum(view(flux, :, i)),
                        "connection_open" => mask[i] != 0.0,
                        "actual_target" => target,
                        "bhp_pa" => bhp,
                    ),
                )
            end
        end
        # Into standard volume, like the inventory it is balanced against: the cross term
        # is a component MASS flux in kg/s, and the inventory is `TotalMasses / rho_sc`.
        push!(connection_source, (into_reservoir ./ rhoS) .* dt[k])
        push!(
            surface_source,
            Float64[sum(surface_mass[w][c][k] for w in names) / rhoS[c] * dt[k] for c in 1:n_phases],
        )
        push!(reservoir_inventory, component_inventory(evaluated, :Reservoir, rhoS))
        push!(
            inventory,
            reservoir_inventory[end] .+
            sum([component_inventory(evaluated, w, rhoS) for w in names]; init = zeros(n_phases)),
        )
    end

    accepted, cut, iterations = solver_counters(sim_result)
    # Which extra state fields the perforation flux depended on were really in the stored
    # substates. Empty for a multisegment well, which has none; non-empty for a `SimpleWell`,
    # where it is the difference between a conserving flux and one computed from zeros.
    stored_extra = Dict{String,Any}(
        String(name) => String[
            String(f) for f in PERFORATION_EXTRA_STATE_FIELDS if haskey(states[1][name], f)
        ] for name in names
    )
    return Dict{String,Any}(
        "schema_version" => EXTRACT_SCHEMA_VERSION,
        "components" => collect(EXTRACT_COMPONENTS),
        "reference_densities_kg_m3" => rhoS,
        "stored_extra_state_fields" => stored_extra,
        "chunk" => Dict{String,Any}(
            "horizon_start_s" => chunk_start_s,
            "horizon_end_s" => chunk_start_s + horizon,
            # On the case's own clock, like the substep boundaries beside them: a chunk
            # whose edges were left local would put every interval of a later chunk back at
            # the start of the case.
            "edges_s" => chunk_start_s .+ (edges_s .- edges_s[1]),
            "start_s" => collect(starts),
            "end_s" => collect(ends),
            "dt_s" => collect(dt),
            "interval_index" => Int[i - 1 for i in report_index],
        ),
        "wells" => wells_out,
        "connections" => connection_rows,
        "inventory_m3_sc" => inventory,
        "reservoir_inventory_m3_sc" => reservoir_inventory,
        "net_surface_source_m3_sc" => surface_source,
        "net_connection_source_m3_sc" => connection_source,
        "states" => requested_states(
            model, parameters, states, state0, starts, ends, rhoS, state_times_s,
        ),
        "solver" => Dict{String,Any}(
            "accepted_steps" => accepted,
            "cut_steps" => cut,
            "nonlinear_iterations" => iterations,
        ),
        "control_evidence" => evidence,
        "control_infeasible_reason" => infeasible_controls(evidence),
    )
end

"""
Accepted mini-steps, cut ones and nonlinear iterations, from the solver's own reports.

A report reaches this function in one of TWO shapes, and reading only the first was worth a
whole count. A report still in memory carries `:steps`, one entry per nonlinear iteration.
A report read back from an `output_path` does not: `get_output_report` at the default
`report_level = 0` (`Jutul 0.4.31 src/simulator/io.jl:35`) deletes `:steps` and puts
`stats_ministep(...)` in its place, whose `linearizations` field is the count those entries
were. A chunked or restarted run reads every report back from disk, so taking `:steps`
alone would publish `nonlinear_iterations = 0` for every one of them — a measurement
silently replaced by a zero, which is exactly what a cost record must never carry.
"""
function solver_counters(sim_result)
    accepted, cut, iterations = 0, 0, 0
    for report in sim_result.reports[eachindex(sim_result.states)]
        for ministep in report[:ministeps]
            ministep[:success] ? (accepted += 1) : (cut += 1)
            iterations += if haskey(ministep, :steps)
                length(ministep[:steps])
            elseif haskey(ministep, :stats)
                ministep[:stats].linearizations
            else
                invalid(
                    "extract_interval: a mini-step report carries neither :steps nor :stats, " *
                    "so its nonlinear iterations cannot be counted; a cost record does not " *
                    "record an unmeasured zero",
                )
            end
        end
    end
    return (accepted, cut, iterations)
end

"""
    requested_states(...) -> Dict

The cell fields at the times that were asked for, and only at those times.

A requested time has to BE an accepted substep boundary (or the chunk start): interpolating
between two substeps would publish a state the solver never computed, under a date somebody
will read as a measurement. `B_alpha = rho_alpha_sc / rho_alpha(p)` is formed from the
model's own `PhaseMassDensities` variable rather than from the analytic expression, so the
published formation volume factor is the one the flux used.

The model's geometry and rock are NOT among these fields: they are inputs the case already
publishes and hashes, and copying them once per timestep would make a result heavier than
the model it came from. The pore volume IS a per-time field of the contract — E01 fixes
`rock_compressibility` at 0, so the rows are equal, and that equality is a fact about this
case rather than a shape the file is allowed to assume.
"""
function requested_states(
    model,
    parameters,
    states,
    state0::AbstractDict,
    starts::AbstractVector,
    ends::AbstractVector,
    rhoS::Vector{Float64},
    state_times_s::Vector{Float64},
)
    isempty(state_times_s) &&
        invalid("extract_interval: state_times_s must request at least one state")
    available = vcat(Float64(starts[1]), Float64.(ends))
    pv = collect(Float64.(pore_volume(model, parameters)))
    n_cells = length(pv)

    times = Float64[]
    pressure, sw, so, pore, bw, bo = (Vector{Vector{Float64}}() for _ in 1:6)
    for requested in state_times_s
        index = findfirst(t -> abs(t - requested) <= TIME_MATCH_TOLERANCE_S, available)
        index === nothing && invalid(
            "extract_interval: a state was requested at $(requested) s, which is not an " *
            "accepted substep boundary of this chunk (boundaries run from $(available[1]) s " *
            "to $(available[end]) s); a state is never interpolated between two substeps",
        )
        state = index == 1 ? state0 : states[index - 1]
        evaluated = Jutul.evaluate_all_secondary_variables(
            model.models[:Reservoir], state[:Reservoir], parameters[:Reservoir],
        )
        saturations = evaluated[:Saturations]
        rho = evaluated[:PhaseMassDensities]
        size(saturations, 2) == n_cells || invalid(
            "extract_interval: the stored state describes $(size(saturations, 2)) cells, the " *
            "model has $(n_cells)",
        )
        push!(times, available[index])
        push!(pressure, collect(Float64.(evaluated[:Pressure])))
        push!(sw, Float64[saturations[1, c] for c in 1:n_cells])
        push!(so, Float64[saturations[2, c] for c in 1:n_cells])
        push!(pore, copy(pv))
        push!(bw, Float64[rhoS[1] / rho[1, c] for c in 1:n_cells])
        push!(bo, Float64[rhoS[2] / rho[2, c] for c in 1:n_cells])
    end
    return Dict{String,Any}(
        "times_s" => times,
        "pressure_pa" => pressure,
        "sw" => sw,
        "so" => so,
        "pore_volume_m3" => pore,
        "bw" => bw,
        "bo" => bo,
    )
end

# --------------------------------------------------------------------------------------
# one whole forward
# --------------------------------------------------------------------------------------

"""
    run_forward(case, arrays; state_times_s = nothing, solver...) -> Dict

Build this case, drive it over its own compiled schedule and extract the result.

The status this returns describes the SIMULATION: `COMPLETE` when the solver reached the end
of the schedule and the extraction holds together, and a named refusal otherwise. It is not
the status of a forward result — that is decided where the outputs are published, because
whether a `CONTROL_INFEASIBLE` evidence table becomes a `CONTROL_INFEASIBLE` result is a
question about the record, not about the solver (plan 3.3).

`solver` keywords are forwarded to `simulate_reservoir`; `output_substates = true` is not
one of them, because without it there are no accepted substeps to integrate and this whole
file would silently report report-step averages instead. Nor is `request_extra_outputs!`,
for the same reason: without it a `SimpleWell`'s connection flux would be read off a
re-initialised zero.
"""
function run_forward(case::AbstractDict, arrays::AbstractDict; state_times_s = nothing, solver...)
    schedule = compile_intervals(case)
    physical = build_ow(case, arrays)
    # Before anything is simulated: what is RECORDED has to include the extra state fields
    # the perforation flux reads, or they are gone by the time the result is read.
    request_extra_outputs!(physical.model)
    forces = [
        build_forces(physical.model, group, case["boundary"]) for group in schedule.controls
    ]
    dt = diff(schedule.edges_s)
    state0 = deepcopy(physical.state0)
    requested = state_times_s === nothing ? Float64.(case["report_edges_s"]) :
                Float64.(collect(state_times_s))

    result = simulate_reservoir(
        physical.state0,
        physical.model,
        dt;
        parameters = physical.parameters,
        forces = forces,
        info_level = -1,
        output_substates = true,
        solver...,
    )
    length(result.result.states) == length(dt) || return Dict{String,Any}(
        "schema_version" => EXTRACT_SCHEMA_VERSION,
        "status" => "NUMERICAL_FAILURE",
        "reason" => string(
            "the solver reported ", length(result.result.states), " report steps for ",
            length(dt), " scheduled intervals; the case did not run to the end of its ",
            "schedule and no partial integral is published for it",
        ),
    )
    payload = extract_interval(
        result,
        physical.model,
        physical.parameters,
        forces,
        state0;
        controls_by_interval = schedule.controls,
        edges_s = schedule.edges_s,
        state_times_s = requested,
    )
    payload["status"] = "COMPLETE"
    payload["reason"] = nothing
    return payload
end
