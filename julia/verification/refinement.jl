# E01.10.7-10.8 — the P1 fixtures: the five-spot, and the grid/time refinement study.
#
# Everything here is a DISCRETIZATION SENSITIVITY measurement and nothing here is a posterior.
# Running the same continuous problem on a finer grid and a shorter timestep says how much of
# an answer is the physics and how much is the mesh; it does not say the fine answer is right,
# and no line in this file or in the checks that read it claims otherwise.
#
# TWO REFINEMENTS, FOR TWO DIFFERENT REASONS.
#
# * Buckley-Leverett at 64 cells with the report step, and at 128 cells with HALF of it. This
#   pair has an analytic answer outside the simulator — `so_recon.validation.physics.bl_saturation`
#   is the exact weak solution of the limit the fixture declares — so what is measured is the
#   error against a formula, on both members, and whether refining reduced it. Refining space
#   and time together is the only honest way to refine a first-order upwind scheme: halving
#   the cells alone raises the Courant number of every step.
#
# * The five-spot at 16x16 and at 32x32. This pair has no analytic answer, so what is measured
#   is AGREEMENT: the two are compared on a fixed 8x8 grid of physical zones that both meshes
#   tile exactly, plus the standard-volume inventory and every monthly phase integral.
#
# WHAT IS KEPT FIXED ACROSS A REFINEMENT, AND WHY IT MATTERS.
#
# The continuous field is the same one: the five-spot rock is homogeneous, so the fine grid
# carries the identical porosity and permeability rather than a second draw of a random field,
# and the zone pore volumes agree to round-off — which the check verifies rather than assumes.
# The wells are at the same PHYSICAL locations: a well that perforated one coarse cell
# perforates the whole block of fine cells that replaced it (`refine_cells`), so the completion
# covers the same rock. Its well index is NOT carried over: `setup_well` recomputes it natively
# from the same radius in the refined geometry, which is what a refinement means. Tuning a fine
# WI until it reproduced the coarse rate would be fitting the answer.
#
# The five-spot is 256 cells with 5 wells over 36 months and its refinement is 1024 cells, so
# both belong to P1_LOOP and neither may be run under P0_VERIFY, whose ceiling is 3 wells and
# 12 report intervals.

using Test
using Dates
using Jutul
using JutulDarcy
using JSON

# `operations.jl` carries the five-spot and, through `analytic.jl`, the Buckley-Leverett core
# and the adapter handle. Both `PROGRAM_FILE` guards make including them inert.
include(joinpath(@__DIR__, "operations.jl"))

#: The refinement pair of the five-spot: the 256-cell case plan 10.7 names, and one whole
#: doubling of it. 32/16 = 2, so every coarse cell is exactly four fine ones.
const FIVE_SPOT_GRIDS = (16, 32)

#: The fixed physical support both five-spot grids are compared on: 8x8 zones over the same
#: 400 x 400 m extent, so each zone is 50 x 50 m. 16 and 32 are both whole multiples of 8, so
#: a zone is 2x2 coarse cells and 4x4 fine ones and every cell lies entirely inside one zone.
#: That exactness is why no geometry library is needed and why the zone pore volumes have to
#: agree — the check measures that agreement instead of assuming it.
const FIVE_SPOT_SUPPORT_SIDE = 8

#: The Buckley-Leverett refinement pair: cells and the divisor applied to the report step.
const BL_REFINEMENT = ((64, 1), (128, 2))

#: `five_spot_symmetry_abs_max` of `configs/e01_tolerances.yml`, restated on this side because
#: this process has no YAML reader and the diagnostic should fail where the number is produced.
#:
#: A hand-copied threshold is a threshold that can go stale, and this one carries real weight:
#: `so_recon.validation.physics` pins its `five_spot` fixture to the 16x16 grid, so the 32x32
#: refinement's symmetry is gated HERE and nowhere else. It is therefore EXPORTED in the
#: payload below, and `tests/integration/test_e01_physics.py` asserts it equals the frozen YAML
#: value and re-scores both grids' measured reflections against that value. Editing the YAML
#: without editing this line now turns that test red instead of leaving the fine grid on a
#: stale gate.
const FIVE_SPOT_SYMMETRY_ABS_MAX = 1.0e-4

"""
    bl_refinement_options(nx, dt_divisor) -> Dict

`bl_solver_options()` with its timestep divided. `timesteps = :none` takes the report step
whole, and `initial_dt`/`max_timestep` then decide the substep; dividing both by `dt_divisor`
is the time half of this refinement. `max_timestep_cuts = 1` is inherited, so the accepted
substeps are that dt and its half and nothing else.
"""
function bl_refinement_options(nx::Int, dt_divisor::Int)
    options = bl_solver_options()
    options["initial_dt"] = options["initial_dt"] / dt_divisor
    options["max_timestep"] = options["max_timestep"] / dt_divisor
    options["nx"] = nx
    return options
end

"""
    support_zone_ids(nx, side) -> Vector{Int}

The zero-based support zone each cell of an `nx x nx` grid belongs to, in the exchange's own
`i + nx*j` cell order. `side` divides `nx` exactly, so the map is a partition of whole cells
and the intersection pore volume of a cell with its zone is the cell's own pore volume.
"""
function support_zone_ids(nx::Int, side::Int)
    factor, remainder = divrem(nx, side)
    remainder == 0 ||
        error("support_zone_ids: a $(side)-zone support does not tile an $(nx)-cell grid")
    zones = Vector{Int}(undef, nx * nx)
    for j in 0:(nx - 1), i in 0:(nx - 1)
        zones[i + nx * j + 1] = (i ÷ factor) + side * (j ÷ factor)
    end
    return zones
end

"""
    mirror_symmetry_abs(field, nx) -> (about_x, about_y)

The largest absolute difference between a square field and its own reflection in each axis.

The five-spot's wells are placed symmetrically about the grid's mirror `i -> nx-1-i`, so a
converged answer has to be symmetric about BOTH axes. Measuring the reflection directly is
what makes that a fact about the published field rather than about how it was built; nothing
in the fixture forces it, and an asymmetric well placement or a directional solver artefact
would show up here as a number.
"""
function mirror_symmetry_abs(field::AbstractVector{Float64}, nx::Int)
    length(field) == nx * nx ||
        error("mirror_symmetry_abs: $(length(field)) values for an $(nx)x$(nx) grid")
    at(i, j) = field[i + nx * j + 1]
    about_x = 0.0
    about_y = 0.0
    for j in 0:(nx - 1), i in 0:(nx - 1)
        about_x = max(about_x, abs(at(i, j) - at(nx - 1 - i, j)))
        about_y = max(about_y, abs(at(i, j) - at(i, nx - 1 - j)))
    end
    return (about_x, about_y)
end

"""The well indices of one run, summed per well: what `setup_well` computed for that geometry."""
function well_index_totals(run::AbstractDict)
    indices = run["native"]["well_indices"]
    return Dict{String,Any}(name => sum(Float64.(values)) for (name, values) in indices)
end

function selftest_refinement()
    measured = Dict{String,Any}(
        "julia_version" => string(VERSION),
        "jutul_version" => string(pkgversion(Jutul)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "gravity_constant" => Jutul.gravity_constant,
        "support_side" => FIVE_SPOT_SUPPORT_SIDE,
        "five_spot_grids" => collect(FIVE_SPOT_GRIDS),
        # Exported so the Python side can prove this copy is still the frozen one.
        "five_spot_symmetry_abs_max" => FIVE_SPOT_SYMMETRY_ABS_MAX,
    )
    fixtures = Dict{String,Any}()
    timings = Dict{String,Any}()

    @testset "E01 five-spot and the grid/time refinement study" begin
        # --- 10.8 Buckley-Leverett: 64 cells at dt, 128 cells at dt/2 --------------------
        for (nx, divisor) in BL_REFINEMENT
            options = bl_refinement_options(nx, divisor)
            elapsed = @elapsed run = run_analytic(:bl, options)
            @test run["status"] == "COMPLETE"
            extraction = run["extraction"]
            @test extraction["control_infeasible_reason"] === nothing
            # Every accepted substep is the refined report step or its half, and nothing else.
            dt = extraction["chunk"]["dt_s"]
            target = options["max_timestep"]
            @test all(d -> isapprox(d, target) || isapprox(d, target / 2), dt)
            @test sum(dt) ≈ BL_HORIZON_DAYS * SECONDS_PER_DAY
            # Pre-breakthrough on BOTH grids: the analytic reference is only the answer while
            # the shock is still inside the core.
            @test extraction["states"]["sw"][end][end] < 1.0e-6
            fixtures["bl_$(nx)"] = run
            timings["bl_$(nx)"] = elapsed
        end
        # Refining time really did refine time: the fine member's substep is half the coarse
        # member's, so this is a joint grid/time refinement and not a grid one wearing its name.
        @test length(fixtures["bl_128"]["extraction"]["chunk"]["dt_s"]) ==
              2 * length(fixtures["bl_64"]["extraction"]["chunk"]["dt_s"])

        # --- 10.7 / 10.8 the five-spot at both resolutions -------------------------------
        for nx in FIVE_SPOT_GRIDS
            elapsed = @elapsed run = run_operational(:five_spot, Dict{String,Any}("nx" => nx))
            @test run["status"] == "COMPLETE"
            extraction = run["extraction"]
            @test extraction["control_infeasible_reason"] === nothing
            # 36 real calendar months, and a published state at every month edge.
            @test length(extraction["states"]["times_s"]) == FIVE_SPOT_MONTHS + 1
            # GRAVITY IS PRESENT and the case does not switch it off: it is one layer, so every
            # horizontal face carries a gravity head of exactly zero and there is no other kind
            # of face. The native parameter is what is read, not the fixture's intention.
            gdz = run["native"]["two_point_gravity_difference"]
            @test !isempty(gdz)
            @test all(==(0.0), gdz)
            @test all(==(run["native"]["cell_center_depth_m"][1]),
                      run["native"]["cell_center_depth_m"])
            fixtures["five_spot_$(nx)"] = run
            timings["five_spot_$(nx)"] = elapsed
        end

        # --- 10.7 symmetry about both axes ------------------------------------------------
        symmetry = Dict{String,Any}()
        for nx in FIVE_SPOT_GRIDS
            states = fixtures["five_spot_$(nx)"]["extraction"]["states"]
            final_so = Float64.(states["so"][end])
            about_x, about_y = mirror_symmetry_abs(final_so, nx)
            symmetry["five_spot_$(nx)"] = Dict{String,Any}(
                "so_mirror_x_abs" => about_x,
                "so_mirror_y_abs" => about_y,
                # Not vacuous: the field really varies across the pattern, so a symmetric
                # answer is a statement and not the absence of one.
                "so_range" => maximum(final_so) - minimum(final_so),
            )
            # The gate is the FIXED one: `five_spot_symmetry_abs_max` in
            # `configs/e01_tolerances.yml`, restated here so the Julia side fails first and
            # loudly. Python scores the published field against the same number.
            @test about_x <= FIVE_SPOT_SYMMETRY_ABS_MAX
            @test about_y <= FIVE_SPOT_SYMMETRY_ABS_MAX
            @test maximum(final_so) - minimum(final_so) > 0.05
        end
        measured["five_spot_symmetry"] = symmetry

        # --- 10.8 the well index is RECOMPUTED, not carried over --------------------------
        coarse_wi = well_index_totals(fixtures["five_spot_16"])
        fine_wi = well_index_totals(fixtures["five_spot_32"])
        @test sort(collect(keys(coarse_wi))) == sort(collect(keys(fine_wi)))
        for name in keys(coarse_wi)
            # Native Peaceman on a smaller cell gives a different number for the same radius.
            # It is reported, not gated: what a refinement must not do is TUNE it back.
            @test fine_wi[name] != coarse_wi[name]
        end
        measured["five_spot_well_index_total"] =
            Dict{String,Any}("16" => coarse_wi, "32" => fine_wi)

        # --- 10.8 the support both grids are compared on ----------------------------------
        support = Dict{String,Any}()
        for nx in FIVE_SPOT_GRIDS
            zones = support_zone_ids(nx, FIVE_SPOT_SUPPORT_SIDE)
            @test length(zones) == nx * nx
            @test sort(unique(zones)) == collect(0:(FIVE_SPOT_SUPPORT_SIDE^2 - 1))
            support["five_spot_$(nx)"] = zones
        end
        measured["support_zone_ids"] = support
    end

    measured["fixtures"] = fixtures
    measured["wall_clock_s"] = timings
    measured["status"] = "ok"
    return measured
end

"""Parse the diagnostic to run and the optional `--out` the project launcher appends."""
function parse_refinement_args(args::Vector{String})
    out = nothing
    requested = false
    i = 1
    while i <= length(args)
        if args[i] == "--test-refinement"
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

function refinement_main(args::Vector{String})
    requested, out = parse_refinement_args(args)
    requested || error(
        "nothing to do: pass --test-refinement to run the five-spot and the grid/time " *
        "refinement study of plan Task 10.7-10.8",
    )
    payload = try
        selftest_refinement()
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
    refinement_main(collect(ARGS))
    println("so-recon verification: refinement diagnostic passed")
end
