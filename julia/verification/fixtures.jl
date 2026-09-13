# E01.0 — verification fixtures for the native oil-water model, and the standalone
# diagnostic that checks the constructor against them.
#
# Two uses, one file. `verification_case` is a plain definition: including this file from a
# verification script, from a later gate or from the worker gives you the fixtures and
# nothing else — no tests run, nothing is printed, no adapter is loaded. Running the file
# AS A PROGRAM with `--test-model` additionally loads the adapter and executes the checks
# below, which is the addressed diagnostic of plan Task 5.2, not a serial gate: the gate
# drives the persistent worker instead of paying a cold Julia start per assertion.
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

Any other name is an explicit error. A fixture whose physics has not been decided is not
stubbed with plausible-looking numbers here: a silently wrong reference case is worse than
a missing one, because every later comparison would inherit it.
"""
function verification_case(name::Symbol; nx::Int = DEFAULT_NX, nz::Int = DEFAULT_NZ)
    if name !== :closed_cell
        error(
            "verification_case: fixture $(repr(name)) is not implemented in this build. " *
            "Only :closed_cell exists so far; :bl, :hydrostatic, :two_layer, :boundary and " *
            ":five_spot are named by the plan but their physics is decided in Tasks 9-10, " *
            "and a stub would be a wrong reference case rather than a missing one.",
        )
    end
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
                "allow_crossflow" => false,
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

"""Parse `--test-model` and the optional `--out` the project launcher appends."""
function parse_selftest_args(args::Vector{String})
    out = nothing
    requested = false
    i = 1
    while i <= length(args)
        if args[i] == "--test-model"
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

function selftest_main(args::Vector{String}, adapter::Module)
    requested, out = parse_selftest_args(args)
    requested || error("nothing to do: pass --test-model to run the constructor diagnostic")
    payload = try
        selftest(adapter)
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
    selftest_main(collect(ARGS), adapter_module)
    println("so-recon verification: constructor diagnostic passed")
end
