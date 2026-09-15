# E01.13 — the small EDUCATIONAL black-oil capability, beside the oil-water model.
#
# What this file is for, and what it is deliberately not for.
#
# It is a CAPABILITY: evidence that this adapter can build, drive, balance and restart a
# three-phase black-oil system with dissolved gas on an academic benchmark PVT. It is not a
# choice of physics for the field case, it is not a paired oil-water/black-oil sensitivity
# study with matched oil inventory — that is E06 — and nothing in it feeds the oil-water
# deliverables of this stage (plan 13.5).
#
# Every number of the fluid comes from the PINNED JutulDarcy tree. `blackoil_bench_pvt(:spe1)`
# ships inside the package (`src/blackoil/data.jl`); no artifact is downloaded, no SPE9 deck
# is read, and no plotting package is loaded. `blackoil_pvt_export` re-emits the tables that
# were really constructed — not the literals someone copied out of a source file — so the
# fixture artifact this capability publishes is the numbers the model was built from.
#
# THE ONE CORRECTION THIS FILE MAKES TO THE PINNED EXAMPLE. `blackoil_bench_pvt(:spe1)`
# returns `rhoS = [786.507 1037.84 0.969758]`, which is the ECLIPSE `DENSITY` keyword order
# (oil, water, gas). `setup_reservoir_model_from_blackoil_tables` wants `reference_densities`
# in PHASE order, aqueous first, because it pairs `reference_densities[i]` with `pvt[i]` and
# `pvt` is `[pvtw, pvto, pvdg]`. The pinned tree's own
# `src/test_utils/setup_multimodel.jl:setup_mini_wellcase(::Val{:bo_spe1})` passes the vector
# through unpermuted, which gives that educational mini-case a 786.5 kg/m3 "water" and a
# 1037.8 kg/m3 "oil". This file permutes explicitly — `[water, oil, gas]` — and states the
# permutation in the exported fixture, because a surface density that is attached to the
# wrong phase turns every standard volume this capability publishes into a different number.
#
# NO STATE OUTLIVES A CALL, exactly as in `model.jl`: `build_blackoil` returns the model, the
# parameters and the state it built and keeps nothing.

#: Aqueous, liquid, vapour. The order `StandardBlackOilSystem` is built with, the order
#: `reference_densities` is read in, and the order every (water, oil, gas) tuple of a
#: `BlackOilFluidSpec` is written in.
const BO_PHASES = (AqueousPhase(), LiquidPhase(), VaporPhase())
const BO_PHASE_COUNT = length(BO_PHASES)

#: The physics class a case declares to reach this constructor.
const BO_FLUID_KIND = "BO"

#: How the pinned benchmark's `rhoS` is permuted into phase order. `rhoS` is (oil, water,
#: gas); the system wants (water, oil, gas).
const BO_DENSITY_PERMUTATION = (2, 1, 3)

#: The native call the tables come out of, spelled as it is called. It travels into the case
#: record, so a reader can re-derive the fixture rather than take its word.
const BO_PVT_SOURCE = "JutulDarcy.blackoil_bench_pvt(:spe1)"

"""
    blackoil_pvt() -> NamedTuple

The pinned academic benchmark PVT, with its surface densities already in PHASE order.

`rhoS_deck` is kept beside `rhoS` so the permutation above is visible in the record rather
than only in this comment.
"""
function blackoil_pvt()
    setup = JutulDarcy.blackoil_bench_pvt(:spe1)
    pvt = setup[:pvt]
    deck = Float64.(vec(collect(setup[:rhoS])))
    length(deck) == BO_PHASE_COUNT || error(
        "blackoil_pvt: the pinned benchmark returned $(length(deck)) reference densities for " *
        "$(BO_PHASE_COUNT) phases",
    )
    return (
        pvtw = pvt[1],
        pvto = pvt[2],
        pvdg = pvt[3],
        rhoS = Float64[deck[i] for i in BO_DENSITY_PERMUTATION],
        rhoS_deck = deck,
        name = String(setup[:name]),
    )
end

"""
    blackoil_pvt_export(pvt) -> Dict

Every number of the constructed tables, as plain JSON-able arrays.

This is what gets hashed and published as the versioned PVT fixture artifact. It is read off
the OBJECTS the model is built from — `ConstMuBTable`, `PVTOTable`, `MuBTable` — rather than
copied from the literals in the pinned source, so a table that changed in the package would
change this export and therefore the hash the case records.
"""
function blackoil_pvt_export(pvt = blackoil_pvt())
    w = pvt.pvtw.tab[1]
    o = pvt.pvto.tab[1]
    g = pvt.pvdg.tab[1]
    return Dict{String,Any}(
        "schema_version" => "e01-blackoil-pvt-1",
        "source" => BO_PVT_SOURCE,
        "benchmark" => pvt.name,
        "academic_benchmark" => true,
        "note" => string(
            "An academic benchmark PVT that ships inside the pinned JutulDarcy. It is NOT a ",
            "Romashka PVT and this capability does not choose the physics of a field case.",
        ),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "jutul_version" => string(pkgversion(Jutul)),
        "phase_order" => ["water", "oil", "gas"],
        "reference_densities_kg_m3" => pvt.rhoS,
        "reference_densities_deck_order_kg_m3" => pvt.rhoS_deck,
        "density_permutation_deck_to_phase" => collect(BO_DENSITY_PERMUTATION),
        "pvtw" => Dict{String,Any}(
            "p_ref_pa" => Float64(w.p_ref),
            "b_ref" => Float64(w.b_ref),
            "b_c" => Float64(w.b_c),
            "mu_ref_pa_s" => Float64(w.mu_ref),
            "mu_c" => Float64(w.mu_c),
        ),
        "pvto" => Dict{String,Any}(
            "pos" => Int[i for i in o.pos],
            "rs" => Float64[x for x in o.rs],
            "pressure_pa" => Float64[x for x in o.pressure],
            "sat_pressure_pa" => Float64[x for x in o.sat_pressure],
            "shrinkage" => Float64[x for x in o.shrinkage],
            "viscosity_pa_s" => Float64[x for x in o.viscosity],
        ),
        "pvdg" => Dict{String,Any}(
            "pressure_pa" => Float64[x for x in g.pressure],
            "shrinkage" => Float64[x for x in g.shrinkage],
            "viscosity_pa_s" => Float64[x for x in g.viscosity],
        ),
    )
end

"""
    blackoil_relperm_export(fluid) -> Dict

The three-phase relative permeability definition this capability registers, as data.

It is an EXPLICIT approximation — three independent Brooks-Corey curves, no hysteresis, no
three-phase oil model — and it is hashed beside the PVT so that the record says which
approximation produced the numbers. A field case would choose a Stone or LET model instead,
and that choice is not this task's to make (plan 13.3, 13.5).
"""
function blackoil_relperm_export(fluid::AbstractDict)
    return Dict{String,Any}(
        "schema_version" => "e01-blackoil-relperm-1",
        "definition" => "BrooksCoreyRelativePermeabilities",
        "phases" => ["water", "oil", "gas"],
        "n_phases" => BO_PHASE_COUNT,
        "exponents" => Float64.(fluid["corey_exponents"]),
        "residual_saturations" => Float64.(fluid["residual_saturations"]),
        "endpoints" => Float64.(fluid["kr_endpoints"]),
        "hysteresis" => "none",
        "three_phase_oil_model" => "none: independent per-phase Corey curves",
        "note" => string(
            "An explicit capability approximation, not a chosen field relative permeability ",
            "model.",
        ),
    )
end

"""
    assert_physical_blackoil_fluid(fluid)

Refuse a black-oil fluid whose physics does not exist, BEFORE a model is assembled from it.

The same rule `assert_physical_pvt` applies to the oil-water fluid, for the fields a
black-oil case carries: three positive surface densities, a positive reference pressure, a
non-negative initial dissolved GOR, no vaporised oil, and residual saturations that leave the
three phases a state they can share. Plain `error`, not `InvalidCaseInput`, so the job ends
as `PHYSICALLY_INVALID` at construction and never as a non-converging solve.
"""
function assert_physical_blackoil_fluid(fluid::AbstractDict)
    rho = Float64.(fluid["density_sc_kg_m3"])
    (length(rho) == BO_PHASE_COUNT && all(isfinite, rho) && all(>(0.0), rho)) || error(
        "build_blackoil: fluids.density_sc_kg_m3 = $(rho) is not physical; a black-oil case " *
        "gives one positive, finite surface density per phase (water, oil, gas)",
    )
    p_sc = Float64(fluid["p_sc_pa"])
    (isfinite(p_sc) && p_sc > 0.0) || error(
        "build_blackoil: fluids.p_sc_pa = $(p_sc) is not a physical reference pressure",
    )
    rs0 = Float64(fluid["initial_rs_m3_m3"])
    (isfinite(rs0) && rs0 >= 0.0) || error(
        "build_blackoil: fluids.initial_rs_m3_m3 = $(rs0) is not a physical dissolved gas-oil " *
        "ratio; it is a non-negative m3_sc of gas per m3_sc of oil",
    )
    rv = Float64(get(fluid, "rv", 0.0))
    rv == 0.0 || error(
        "build_blackoil: this capability is disgas-only — the constructor is given PVTO and " *
        "PVDG and no PVTG — so fluids.rv must be exactly 0.0, got $(rv)",
    )
    residual = Float64.(fluid["residual_saturations"])
    exponents = Float64.(fluid["corey_exponents"])
    endpoints = Float64.(fluid["kr_endpoints"])
    for (label, values) in
        (("residual_saturations", residual), ("corey_exponents", exponents), ("kr_endpoints", endpoints))
        length(values) == BO_PHASE_COUNT || error(
            "build_blackoil: fluids.$(label) gives one value per phase ($(BO_PHASE_COUNT)), " *
            "got $(values)",
        )
    end
    all(x -> 0.0 <= x < 1.0, residual) || error(
        "build_blackoil: fluids.residual_saturations = $(residual) must lie in [0,1)",
    )
    sum(residual) < 1.0 || error(
        "build_blackoil: fluids.residual_saturations = $(residual): Swc + Sorw + Sgc must be " *
        "below 1, or the three phases have no state they can share",
    )
    (all(>(0.0), endpoints) && all(<=(1.0), endpoints)) || error(
        "build_blackoil: fluids.kr_endpoints = $(endpoints) must be positive and at most 1",
    )
    all(>(0.0), exponents) || error(
        "build_blackoil: fluids.corey_exponents = $(exponents) must be positive",
    )
    return nothing
end

"""
    saturated_rs(pvt, pressure) -> Float64

The dissolved gas-oil ratio at the bubble point, read off the EXPORTED saturation table.

`setup_reservoir_model_from_blackoil_tables` builds the system's `rs_max` from exactly this
pair — `pvto.tab[k].sat_pressure` against `pvto.tab[k].rs` — so a check made here is a check
against the table the model will really use rather than against a second copy of it.
"""
function saturated_rs(pvt, pressure::Float64)
    tab = pvt.pvto.tab[1]
    interpolator = Jutul.get_1d_interpolator(tab.sat_pressure, tab.rs)
    return Float64(interpolator(pressure))
end

"""
    assert_initial_state_is_on_the_table(pvt, pressures, rs0)

Before the run: the declared initial pressure and Rs have to be a state this PVT describes.

Two separate things are checked and both matter. The pressures must lie inside the range the
saturation table covers, because outside it the interpolator returns an extrapolated ratio
nobody tabulated; and the declared Rs must not exceed the saturated Rs at the cell's own
pressure, because a cell initialised above its own bubble point is a cell holding more
dissolved gas than the fluid can dissolve — the capability's whole subject is gas coming OUT
of solution, and starting on the wrong side of that line would make the first timestep the
phase transition rather than the depletion.
"""
function assert_initial_state_is_on_the_table(pvt, pressures::AbstractVector{Float64}, rs0::Float64)
    tab = pvt.pvto.tab[1]
    lo, hi = extrema(tab.sat_pressure)
    p_lo, p_hi = extrema(pressures)
    (p_lo >= lo && p_hi <= hi) || invalid(
        "build_blackoil: the initial pressure range [$(p_lo), $(p_hi)] Pa is outside the " *
        "exported saturation table [$(lo), $(hi)] Pa; an initial state off the table would be " *
        "an extrapolated Rs nobody tabulated",
    )
    worst = minimum(saturated_rs(pvt, p) for p in pressures)
    rs0 <= worst + 1e-9 || invalid(
        "build_blackoil: the case declares initial_rs_m3_m3 = $(rs0), and the exported " *
        "saturation table gives at most $(worst) m3/m3 at the initial pressure; a cell cannot " *
        "start with more dissolved gas than its own bubble point allows",
    )
    return nothing
end

"""
    build_blackoil(case, arrays) -> (; model, parameters, state0)

The whole physical constructor of the educational black-oil capability.

The dispatcher in `model.jl` reaches it when the case declares `fluids.kind == "BO"`. Its
shape is deliberately the same as `build_ow`'s: a Cartesian mesh at the case's own datum, a
reservoir domain, the wells the case names, and then the property objects that make it THIS
model rather than JutulDarcy's defaults. Everything geometric is shared with `build_ow`
through `cartesian_origin`, `check_cell_centers`, `check_crossflow` and `well_cells`, so the
two constructors cannot drift apart on the conventions they both have to honour.

What is black-oil specific:

* the system is a `StandardBlackOilSystem` built by the pinned
  `setup_reservoir_model_from_blackoil_tables` from PVTW, PVTO and PVDG — disgas, no vapoil;
* `Rs` and `Saturations` are asked for as outputs, because a result that could not report
  the dissolved ratio would not be evidence of dissolved gas;
* the relative permeability is REPLACED by three independent Brooks-Corey curves before
  `setup_parameters`, so the parameter set is built for the variables that will be used;
* the initial state is the case's own pressure and water saturation with a `BlackOilX` per
  cell, which is the phase state and not merely a saturation — the thing a native restart has
  to carry.
"""
function build_blackoil(case::AbstractDict, arrays::AbstractDict)
    fluid = case["fluids"]
    String(get(fluid, "kind", "")) == BO_FLUID_KIND || invalid(
        "build_blackoil: this constructor builds the educational black-oil system; the case " *
        "declares fluids.kind $(repr(get(fluid, "kind", nothing)))",
    )
    assert_physical_blackoil_fluid(fluid)

    @assert Jutul.gravity_constant == STANDARD_GRAVITY_M_S2
    declared_gravity = Float64(case["gravity_m_s2"])
    declared_gravity == Jutul.gravity_constant || invalid(
        "build_blackoil: the case declares gravity_m_s2=$(declared_gravity); the black-oil " *
        "dispatcher accepts only the native $(Jutul.gravity_constant) m/s^2",
    )

    rock = get(case, "rock", Dict{String,Any}())
    rock_compressibility = Float64(get(rock, "rock_compressibility_pa_inv", 0.0))
    rock_compressibility == 0.0 || invalid(
        "build_blackoil: pore volume is constant in E01; rock_compressibility_pa_inv must be " *
        "exactly 0, got $(rock_compressibility)",
    )

    dims = Tuple(Int.(case["grid"]["shape"]))
    extent = Tuple(Float64.(case["grid"]["extent_m"]))
    (length(dims) == 3 && length(extent) == 3) || invalid(
        "build_blackoil: the grid must be three-dimensional, got shape $(dims) and extent " *
        "$(extent)",
    )
    (all(>(0), dims) && all(>(0.0), extent)) || invalid(
        "build_blackoil: the grid must have a positive size in every direction, got shape " *
        "$(dims) and extent_m $(extent)",
    )
    n_cells = prod(dims)
    check_array_shapes(arrays, n_cells)

    centers = arrays["cell_centers_m"]
    mesh = CartesianMesh(dims, extent; origin = cartesian_origin(dims, extent, centers))
    domain = reservoir_domain(
        mesh;
        permeability = arrays["permeability_m2"],
        porosity = arrays["porosity"],
    )
    check_cell_centers(domain, centers)

    declared_wells = get(case, "wells", Any[])
    foreach(check_crossflow, declared_wells)
    wells = [
        setup_well(
            domain,
            well_cells(w, n_cells);
            name = Symbol(w["well_id"]),
            radius = Float64(w["radius_m"]),
            simple_well = w["model"] == "simple",
            reference_depth = Float64(w["reference_depth_m"]),
        ) for w in declared_wells
    ]

    pvt = blackoil_pvt()
    declared_rho = Float64.(fluid["density_sc_kg_m3"])
    # The case's own densities have to BE the benchmark's, in phase order. A case that
    # declared something else would be declaring a PVT this constructor does not build: the
    # shrinkage, viscosity and density tables are the benchmark's and the surface densities
    # they are paired with are not separately adjustable.
    maximum(abs.(declared_rho .- pvt.rhoS)) <= 1e-6 * maximum(pvt.rhoS) || invalid(
        "build_blackoil: the case declares density_sc_kg_m3 = $(declared_rho) and the pinned " *
        "benchmark's tables are paired with $(pvt.rhoS) in (water, oil, gas) order; the " *
        "surface densities of a deck PVT are not separately adjustable",
    )

    model = JutulDarcy.setup_reservoir_model_from_blackoil_tables(
        domain;
        pvtw = pvt.pvtw,
        pvto = pvt.pvto,
        pvdg = pvt.pvdg,
        reference_densities = pvt.rhoS,
        wells = wells,
        # Rs and the saturations are what a black-oil result IS. `extra_outputs = false` —
        # the oil-water constructor's setting — would leave a result that cannot report
        # either, and `Saturations` is a secondary variable here rather than a primary one.
        extra_outputs = [:Rs, :Saturations],
    )

    rmodel = model.models[:Reservoir]
    system = rmodel.system
    JutulDarcy.has_disgas(system) || error(
        "build_blackoil: the constructed system has no dissolved gas; the capability's " *
        "subject is gas coming out of solution",
    )
    JutulDarcy.has_vapoil(system) && error(
        "build_blackoil: the constructed system has vaporised oil, which this capability " *
        "declares rv = 0 for",
    )

    # The educational three-phase relative permeability, BEFORE `setup_parameters`, so the
    # parameter set is built for the variables that will be used.
    kr = BrooksCoreyRelativePermeabilities(
        BO_PHASE_COUNT,
        Float64.(fluid["corey_exponents"]),
        Float64.(fluid["residual_saturations"]),
        Float64.(fluid["kr_endpoints"]),
    )
    replace_variables!(model; RelativePermeabilities = kr)

    parameters = setup_parameters(model)
    assert_constant_pore_volume(model)

    pressure = Float64.(arrays["pressure_pa"])
    sw = Float64.(arrays["sw"])
    rs0 = Float64(fluid["initial_rs_m3_m3"])
    assert_initial_state_is_on_the_table(pvt, pressure, rs0)
    all(s -> 0.0 <= s <= 1.0, sw) ||
        invalid("build_blackoil: the initial water saturation is outside [0,1]")

    # One `BlackOilX` per cell: the PHASE STATE, not a saturation. `BlackOilX(system, p; ...)`
    # is the pinned high-level initializer; with `sg = 0` it fills `so = 1 - sw` and decides
    # `OilOnly` or `OilAndGas` from the saturation table itself, which is exactly the decision
    # this capability must not make by hand.
    bo = [BlackOilX(system, pressure[c]; sw = sw[c], rs = rs0) for c in 1:n_cells]
    state0 = setup_reservoir_state(
        model;
        Pressure = pressure,
        ImmiscibleSaturation = sw,
        BlackOilUnknown = bo,
    )
    return (; model, parameters, state0)
end
