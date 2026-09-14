# E01.6 — the driving forces of one report interval: controls, limits and completions.
#
# `model.jl` builds the model; this file builds the forces the model is driven with, and
# the split is deliberate. A model is what the case IS; forces are what happens to it, and
# they change on every interval of the schedule. Nothing here constructs a mesh, a system
# or a variable, and nothing in `model.jl` constructs a control.
#
# THE INTERVAL IS THE UNIT, AND NOTHING IS INHERITED. `build_forces` is given the controls
# of exactly one interval and restates, for every well of the model, its role, its target,
# its recorded limit and its completion mask. A well the interval does not name is refused
# rather than defaulted: JutulDarcy's own `setup_forces` silently fills a missing well in
# with `DisabledControl()`, which would turn a schedule with a hole in it into a well that
# quietly stopped producing — a plausible result rather than an error.
#
# Conventions converted here, exactly once:
#
# * Human control rates are m3_sc/day and native ones m3_sc/s (plan 3.1), so every rate is
#   divided by `SECONDS_PER_DAY` on the way in. The case must declare the unit
#   (`units.control_rate`), and the Python contract validates that declaration.
# * Production is POSITIVE in the case and in every public table, and NEGATIVE natively:
#   `ProducerControl` refuses a target that is not negative. The sign flip happens on the
#   line below that divides by 86400, and nowhere else.
# * A producer is given a TOTAL standard liquid rate and an injector a standard water rate
#   (SPEC 9.1). Two separately prescribed phase rates on a producer would prescribe the
#   very thing the forward model exists to predict, so they are refused here as well as in
#   the Python contract.
#
# LIMITS ARE ONLY WHAT THE CASE RECORDED. `setup_reservoir_forces` is called with
# `set_default_limits = false`, so JutulDarcy's convenience defaults — a 1 atm floor under
# every rate producer, a minimum rate under every bhp injector — are NOT applied. A well
# switching to a limit nobody wrote down would make the reported control a guess. The only
# limit this file ever sets is `bhp_limit_pa`, when the segment carries one.

#: 1 day = 86400 s (plan 3.1). The exchange carries day rates; Jutul wants rates per second.
const SECONDS_PER_DAY = 86400.0

#: The injected stream of the educational oil-water case: pure water, by mass, in the
#: water-first phase order `OW_PHASES` fixes. The whole stream is water because SPEC 9.1
#: gives an injector a standard WATER rate; there is no case in E01 that injects oil.
#:
#: This is the one line black oil will change. When a third component arrives, this becomes
#: `vcat(1.0, zeros(n_phases - 1))` and `native_control` takes the phase count as a third
#: argument; nothing else in this file depends on the length of the mixture.
const WATER_INJECTION_MIXTURE = [1.0, 0.0]

#: What the case's `target` is called once it is a native operating target. These are
#: JutulDarcy's own short names (`translate_target_to_symbol`), which is what
#: `full_well_outputs` reports a well's operating control as — so a requested target and an
#: achieved one are comparable without a second naming convention.
const NATIVE_TARGET_SYMBOL = Dict(
    "liquid_rate" => :lrat,
    "water_rate" => :wrat,
    "bhp" => :bhp,
    "disabled" => :disabled,
)

"""
    native_control(c, rho_water_sc) -> WellControlForce

The JutulDarcy control one `ControlSegment` asks for.

`c` is the segment as it crosses the exchange: `role`, `target`, `value`, and (for a rate
control) `bhp_limit_pa`. `rho_water_sc` is the system's own surface water density, which is
what an injector's surface stream is measured against.

Refuses, as `InvalidCaseInput`, anything the E01 control contract does not express: a shut
well with a target, a rate that is not positive, a non-positive pressure, and above all a
producer asked for a per-phase rate (SPEC 9.1).
"""
function native_control(c::AbstractDict, rho_water_sc::Float64)
    role = String(c["role"])
    target = String(c["target"])
    value = Float64(c["value"])

    if role == "shut"
        target == "disabled" || invalid(
            "control for well $(well_name(c)): a shut well carries target 'disabled', got " *
            "$(repr(target))",
        )
        value == 0.0 || invalid(
            "control for well $(well_name(c)): a disabled control carries value 0.0, got $(value)",
        )
        return DisabledControl()
    end
    role in ("producer", "injector") ||
        invalid("control for well $(well_name(c)): unknown role $(repr(role))")
    target == "disabled" && invalid(
        "control for well $(well_name(c)): target 'disabled' belongs to role 'shut', not " *
        "$(repr(role))",
    )

    if target == "bhp"
        value > 0.0 || invalid(
            "control for well $(well_name(c)): a bhp target is an absolute pressure in Pa " *
            "and must be positive, got $(value)",
        )
        bhp = BottomHolePressureTarget(value)
        return role == "producer" ? ProducerControl(bhp) :
               InjectorControl(bhp, WATER_INJECTION_MIXTURE; density = rho_water_sc)
    end
    # A rate of exactly zero is not a rate control: it is a shut well. Jutul would refuse a
    # producer target of -0.0 anyway ("Producer target rate must be negative"), and saying
    # so here names the fix instead of the symptom.
    value > 0.0 || invalid(
        "control for well $(well_name(c)): rate $(value) is not positive. Rates are stated " *
        "positive in m3_sc/day (production is positive in public tables), and a well that " *
        "does not flow is role='shut' with target='disabled', never a rate of zero",
    )
    if role == "producer"
        target == "liquid_rate" || invalid(
            "control for well $(well_name(c)): a producer is controlled on a TOTAL standard " *
            "liquid rate or on bhp, not on the phase rate $(repr(target)) (SPEC 9.1); two " *
            "separately prescribed phase rates would prescribe what the forward model predicts",
        )
        return ProducerControl(SurfaceLiquidRateTarget(-value / SECONDS_PER_DAY))
    end
    target == "water_rate" || invalid(
        "control for well $(well_name(c)): an injector is controlled on a standard water " *
        "rate or on bhp, not on $(repr(target)) (SPEC 9.1)",
    )
    return InjectorControl(
        SurfaceWaterRateTarget(value / SECONDS_PER_DAY),
        WATER_INJECTION_MIXTURE;
        density = rho_water_sc,
    )
end

"""
    control_limits(c) -> NamedTuple or nothing

The operating limits this segment RECORDED, and nothing else.

Only `bhp_limit_pa` on a rate control produces a limit. Its direction is not written here
because JutulDarcy decides it from the control: `check_well_limit` reads `:bhp` as a lower
limit for a `ProducerControl` and an upper limit for an `InjectorControl`. A control that is
already on bhp, and a shut well, carry no limit at all.

`nothing` means no limit, which is what `set_default_limits = false` then leaves in place.
"""
function control_limits(c::AbstractDict)
    target = String(c["target"])
    (target == "bhp" || target == "disabled") && return nothing
    limit = get(c, "bhp_limit_pa", nothing)
    limit === nothing && return nothing
    value = Float64(limit)
    value > 0.0 || invalid(
        "control for well $(well_name(c)): bhp_limit_pa is an absolute pressure in Pa and " *
        "must be positive when given, got $(value)",
    )
    return (bhp = value,)
end

"""
    perforation_mask(c, n_perforations) -> PerforationMask

The completion state of one well over one interval, as the native multiplier on its well
indices. `connection_open` is a tuple of BOOLEANS, one per connection, in the order the
well's `cells` were declared: a partial-completion multiplier is not part of the E01
contract, so a number here is refused rather than scaled.

This is a different thing from a surface shutdown. A masked connection cannot exchange mass
with the reservoir at all; a shut SURFACE (`DisabledControl`) still lets the wellbore talk
to every open connection.
"""
function perforation_mask(c::AbstractDict, n_perforations::Int)
    flags = collect(get(c, "connection_open", Any[]))
    length(flags) == n_perforations || invalid(
        "control for well $(well_name(c)): connection_open has $(length(flags)) entries for " *
        "$(n_perforations) connections",
    )
    multipliers = Float64[]
    for (i, flag) in enumerate(flags)
        flag isa Bool || invalid(
            "control for well $(well_name(c)): connection_open[$(i - 1)] is $(repr(flag)); " *
            "E01 completions are open or shut, not partially open",
        )
        push!(multipliers, flag ? 1.0 : 0.0)
    end
    return PerforationMask(multipliers)
end

"""
    build_forces(model, controls, boundary) -> forces

The driving forces of ONE interval: a control, its recorded limits and a completion mask
for every well of the model, plus the boundary.

`controls` is the list of `ControlSegment` mappings that hold on this interval — exactly
one per well. A well that is missing is an `InvalidCaseInput`, and so is a well named
twice or a well the model does not have: a schedule with a hole in it must not become a
well that silently stopped.

The surface density an injector's stream is measured against is read from the MODEL's own
system rather than from the case, so an injection mixture cannot be scaled by a density
that disagrees with the fluid the model was built from.
"""
function build_forces(model, controls, boundary)
    bc = boundary_conditions(model, boundary)
    rho_water_sc = Float64(JutulDarcy.reference_densities(model.models[:Reservoir].system)[1])
    names = well_names(model)

    by_well = Dict{Symbol,Any}()
    for c in controls
        name = Symbol(well_name(c))
        haskey(by_well, name) && invalid(
            "build_forces: well $(name) carries two controls on one interval; each well has " *
            "exactly one control on each interval",
        )
        name in names ||
            invalid("build_forces: control names well $(name), which the model does not have")
        by_well[name] = c
    end
    for name in names
        haskey(by_well, name) || invalid(
            "build_forces: well $(name) has no control on this interval. A role, a target and " *
            "a completion mask are restated on every interval and never inherited; a missing " *
            "control is a hole in the schedule, not a well that kept doing what it was doing",
        )
    end

    control = Dict{Symbol,Any}()
    limits = Dict{Symbol,Any}()
    for (name, c) in by_well
        control[name] = native_control(c, rho_water_sc)
        limits[name] = control_limits(c)
    end
    # set_default_limits = false: only the limits the case recorded, never JutulDarcy's
    # convenience floors. A well that switched to a limit nobody wrote down would report an
    # operating control that is not in the case.
    forces = setup_reservoir_forces(
        model;
        control = control,
        limits = limits,
        set_default_limits = false,
        bc = bc,
    )
    for (name, c) in by_well
        forces[name] =
            setup_forces(model.models[name]; mask = perforation_mask(c, n_perforations(model, name)))
    end
    return forces
end

"""Every well submodel of this model, in the order the model holds them."""
function well_names(model)
    return Symbol[k for (k, m) in pairs(model.models) if JutulDarcy.model_or_domain_is_well(m)]
end

"""How many reservoir connections one well has — the length its mask must have."""
function n_perforations(model, name::Symbol)
    well = Jutul.physical_representation(model.models[name].domain)
    return length(well.perforations.self)
end

"""The well this control mapping is about, for an error message that names it."""
well_name(c::AbstractDict) = String(get(c, "well_id", "<unnamed>"))

"""
Boundary forces for the reservoir. `closed` is the absence of them.

`pressure_water` is the E01 boundary of plan Task 10.6, and it is an INFINITE one: a
`FlowBoundaryCondition` holds `pressure` fixed for the whole run whatever crosses it, so it
is an unbounded source and sink of water and never a finite aquifer. A case that needs a
FINITE store attaches a real buffer cell with a real pore volume instead — that cell is in
the grid, in the inventory and in the balance, and its pressure falls as it gives water up.
Calling the fixed-pressure model a finite aquifer would be claiming a storage it does not
have; the two are different cases and are kept apart by name.

The native flux is `JutulDarcy.compute_bc_mass_fluxes`: `q_tot = trans_flow * (p_cell - p_bc)`
is positive OUT of the domain and is then split by the reservoir cell's own mobilities on the
way out, or by the boundary's declared `fractional_flow` on the way in. Nothing about that
kernel is written here, and `outputs.jl` reads the same one when it accounts for what crossed.

The injected stream is declared, not inferred: `fractional_flow = (1, 0)` makes the inflow
pure water and `density` is the system's own STANDARD water density, so the standard volume
the balance records for an influx is exactly its native mass divided by `rho_w_sc`. Taking
the water density at the boundary pressure instead would be a different boundary — about 0.6%
denser at 15 MPa under the §3.1 water compressibility — and that is a modelling choice rather
than a correction, so the one the case gets is the one written down here.
"""
function boundary_conditions(model, boundary)
    kind = String(get(boundary, "kind", ""))
    if kind == "closed"
        cells = collect(get(boundary, "cells", Any[]))
        isempty(cells) ||
            invalid("build_forces: a closed boundary names no cells, got $(cells)")
        return nothing
    end
    kind == "pressure_water" || invalid(
        "build_forces: boundary kind $(repr(kind)) is not implemented in this build; E01 " *
        "builds 'closed' and 'pressure_water'",
    )

    cells = collect(get(boundary, "cells", Any[]))
    isempty(cells) &&
        invalid("build_forces: a pressure_water boundary names at least one cell")
    pressure = get(boundary, "pressure_pa", nothing)
    pressure === nothing && invalid(
        "build_forces: a pressure_water boundary carries pressure_pa, which this case leaves " *
        "null; a boundary pressure is never defaulted",
    )
    p = Float64(pressure)
    p > 0.0 || invalid(
        "build_forces: boundary.pressure_pa is an absolute pressure in Pa and must be " *
        "positive, got $(p)",
    )
    trans = get(boundary, "trans_flow", nothing)
    trans === nothing && invalid(
        "build_forces: a pressure_water boundary carries trans_flow, which this case leaves " *
        "null; the conductance of a boundary decides how much it supports and is never guessed",
    )
    t_flow = Float64(trans)
    t_flow > 0.0 ||
        invalid("build_forces: boundary.trans_flow must be positive, got $(t_flow)")

    fractional = Float64.(collect(get(boundary, "fractional_flow", [1.0, 0.0])))
    length(fractional) == PHASE_COUNT || invalid(
        "build_forces: boundary.fractional_flow gives one fraction per phase " *
        "($(PHASE_COUNT)), got $(fractional)",
    )
    all(>=(0.0), fractional) && sum(fractional) == 1.0 || invalid(
        "build_forces: boundary.fractional_flow must be non-negative and sum to exactly 1, " *
        "got $(fractional)",
    )
    fractional == [1.0, 0.0] || invalid(
        "build_forces: a pressure_water boundary supplies pure water; " *
        "fractional_flow must be [1, 0], got $(fractional)",
    )

    n_cells = Jutul.number_of_cells(model.models[:Reservoir].domain)
    rho_water_sc = Float64(JutulDarcy.reference_densities(model.models[:Reservoir].system)[1])
    seen = Set{Int}()
    return [
        begin
            zero_based = Int(c)
            (zero_based >= 0 && zero_based < n_cells) || invalid(
                "build_forces: boundary cell $(zero_based) is outside the grid's " *
                "$(n_cells) cells; boundary cell ids are zero-based",
            )
            zero_based in seen &&
                invalid("build_forces: boundary cell $(zero_based) is named twice")
            push!(seen, zero_based)
            FlowBoundaryCondition(
                zero_based + 1,
                p;
                fractional_flow = fractional,
                density = rho_water_sc,
                trans_flow = t_flow,
            )
        end for c in cells
    ]
end

# --------------------------------------------------------------------------------------
# what the wells were ACTUALLY operating on
# --------------------------------------------------------------------------------------

"""
    control_evidence(model, states, forces, controls_by_step) -> Dict

For every well and every report step: the target the case REQUESTED, the target the well
was actually operating on, and the rates and bottom-hole pressure it reached.

This is the evidence a `CONTROL_INFEASIBLE` result has to carry. A rate a well cannot reach
makes JutulDarcy switch it onto its bhp limit and keep simulating; the numbers still look
like a forward, and without this comparison the achieved bhp control would be reported as
the requested rate control — a demanded rate silently replaced by whatever was possible.

Rates come back in the case's own human units (m3_sc/day) and with the case's own sign
convention — PRODUCTION POSITIVE, which makes an injector's reported water rate negative —
alongside the raw native rate, so that the negative native producer sign stays on the record
rather than being converted away.
"""
function control_evidence(model, states, forces, controls_by_step)
    length(controls_by_step) == length(states) || invalid(
        "control_evidence: $(length(controls_by_step)) intervals of controls for " *
        "$(length(states)) reported states",
    )
    out = Dict{String,Any}()
    for name in well_names(model)
        operating = well_output(model, states, name, forces, :control)
        bhp = well_output(model, states, name, forces, BottomHolePressureTarget)
        lrat = well_output(model, states, name, forces, SurfaceLiquidRateTarget)
        orat = well_output(model, states, name, forces, SurfaceOilRateTarget)
        wrat = well_output(model, states, name, forces, SurfaceWaterRateTarget)
        steps = Any[]
        for step in eachindex(states)
            c = interval_control(controls_by_step[step], String(name))
            requested = NATIVE_TARGET_SYMBOL[String(c["target"])]
            push!(
                steps,
                Dict{String,Any}(
                    "requested_target" => String(requested),
                    "requested_value" => Float64(c["value"]),
                    "operating_target" => String(operating[step]),
                    "honoured" => operating[step] === requested,
                    # Native first, so the sign convention is on the record, then the
                    # public one: production positive, m3_sc/day (plan 3.1).
                    "native_lrat_m3_s" => lrat[step],
                    "liquid_rate_m3_day" => -lrat[step] * SECONDS_PER_DAY,
                    "oil_rate_m3_day" => -orat[step] * SECONDS_PER_DAY,
                    "water_rate_m3_day" => -wrat[step] * SECONDS_PER_DAY,
                    "bhp_pa" => bhp[step],
                ),
            )
        end
        out[String(name)] = steps
    end
    return out
end

"""The control of one well inside one interval's control list."""
function interval_control(controls, well_id::AbstractString)
    for c in controls
        well_name(c) == well_id && return c
    end
    invalid("control_evidence: no control for well $(well_id) on this interval")
end

"""
    infeasible_controls(evidence) -> nothing or String

`nothing` when every well ran on the target its case asked for, and otherwise the reason a
`CONTROL_INFEASIBLE` result carries: which well, which step, what was demanded and what the
well was actually operating on when it could not be reached.
"""
function infeasible_controls(evidence::AbstractDict)
    problems = String[]
    for well in sort(collect(keys(evidence))), (step, s) in enumerate(evidence[well])
        s["honoured"] && continue
        push!(
            problems,
            "well $(well) step $(step): requested $(s["requested_target"])=" *
            "$(s["requested_value"]) but operated on $(s["operating_target"]) at " *
            "$(round(s["liquid_rate_m3_day"], sigdigits = 6)) m3_sc/day and " *
            "$(round(s["bhp_pa"], sigdigits = 6)) Pa",
        )
    end
    isempty(problems) && return nothing
    return string(
        "the demanded control could not be reached, so the well operated on a limit ",
        "instead: ",
        join(problems, "; "),
    )
end
