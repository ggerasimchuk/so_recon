# E01.0 — the explicit educational oil-water model, built on native JutulDarcy.
#
# `build_ow` is the whole physical constructor: a Cartesian mesh, a reservoir domain, the
# wells the case names, a two-phase immiscible system, and the three property objects that
# make it the plan's §3.1 educational model rather than JutulDarcy's defaults — constant
# compressibility densities, Brooks-Corey relative permeabilities, and one viscosity per
# phase on every submodel that carries fluid. Nothing here implements physics of its own:
# every object below is a JutulDarcy constructor, because JutulDarcy is the only
# operational backend (SPEC 8.1).
#
# NO STATE OUTLIVES A CALL. Everything is a local, and `build_ow` returns it; there is no
# module-level model, parameter set or state anywhere in this file. Jobs A, B and A again
# therefore get three independently built models from the same warm process, and the second
# A cannot inherit anything B allocated.
#
# Conventions this file is responsible for converting, exactly once, at construction:
#
# * `cell_id = i + nx*(j + ny*k)` is ZERO-BASED in the exchange and one-based in Julia. The
#   `+ 1` happens here, on the well connection lists, and nowhere else.
# * z is depth, positive down. Gravity is the native constant and only the native constant;
#   the face term is JutulDarcy's own `TwoPointGravityDifference`, derived from the domain's
#   z centroids, so this file never writes a gravity formula of its own.
# * Pore volume is constant: `rock_compressibility_pa_inv` must be 0 and the pore volume
#   stays the default `FluidVolume` PARAMETER, never replaced by a pressure-dependent
#   secondary variable.

#: Water first, oil second. Every (water, oil) tuple in the case — densities, viscosities,
#: compressibilities, Corey exponents — is read against this order.
const OW_PHASES = (AqueousPhase(), LiquidPhase())
const PHASE_COUNT = length(OW_PHASES)

const STANDARD_GRAVITY_M_S2 = 9.80665

#: What a job can be answered with while the output axis is not integrated. It is a refusal,
#: not a degraded success: a forward that produced no states is not a forward.
const OUTPUTS_UNAVAILABLE =
    "outputs unavailable: the model, parameters and initial state were constructed, but " *
    "this build integrates no time axis and therefore has no states to report"

#: The arrays `build_ow` reads, where the case manifest keeps each one's `ArrayRef`, and the
#: axis order the exchange declares for it (plan 3.2). The layout is read from the file, not
#: assumed: an `(n_times, n_cells)` dataspace arrives in Julia with its dimensions reversed.
const REQUIRED_ARRAYS = (
    ("porosity", ("rock", "porosity"), ("cell",)),
    ("permeability_m2", ("rock", "permeability_m2"), ("dim", "cell")),
    ("pressure_pa", ("initial", "pressure_pa"), ("cell",)),
    ("sw", ("initial", "sw"), ("cell",)),
)

# --------------------------------------------------------------------------------------
# arrays
# --------------------------------------------------------------------------------------

"""
    load_arrays(case, root) -> Dict{String, Any}

Read every array `build_ow` needs, and read nothing else.

Each one is located through the `ArrayRef` the case carries, re-hashed against the digest
that reference declares, and converted into the semantic layout by the `axis_order` recorded
on the dataset — never by assuming how HDF5 dimensions come back. A missing file, a digest
that does not match, an axis order that is not the declared one, a wrong shape or a
nonfinite value is an error here: the adapter does not repair an input, and it does not
build a model out of bytes nobody vouched for.
"""
function load_arrays(case::AbstractDict, root::AbstractString)
    arrays = Dict{String,Any}()
    for (name, location, axes) in REQUIRED_ARRAYS
        arrays[name] = read_verified_array(name, array_ref(case, name, location), root, axes)
    end
    return arrays
end

"""Follow a case's field path to the `ArrayRef` mapping it should hold."""
function array_ref(case::AbstractDict, name::AbstractString, location::Tuple)
    node = case
    for key in location
        (node isa AbstractDict && haskey(node, key)) ||
            error("$(name): the case has no $(join(location, ".")); it cannot build a model")
        node = node[key]
    end
    node isa AbstractDict ||
        error("$(name): $(join(location, ".")) is not an array reference mapping")
    for key in ("path", "dataset", "sha256", "shape", "unit", "axis_order")
        haskey(node, key) || error("$(name): the array reference is missing $(key)")
    end
    return node
end

function read_verified_array(
    name::AbstractString,
    ref::AbstractDict,
    root::AbstractString,
    expected_axes::Tuple,
)
    relative = String(ref["path"])
    path = joinpath(root, relative)
    isfile(path) || error("$(name): the declared array $(relative) does not exist")
    digest = bytes2hex(open(sha256, path))
    declared_digest = get(ref, "sha256", nothing)
    if !(declared_digest isa AbstractString) || digest != declared_digest
        error(
            "$(name): $(relative) hashes to $(digest), which is not the declared " *
            "$(repr(declared_digest))",
        )
    end
    declared_axes = Tuple(String.(ref["axis_order"]))
    declared_axes == expected_axes ||
        error("$(name): the case declares axis_order $(declared_axes), expected $(expected_axes)")
    dataset = String(ref["dataset"])
    raw, axis_order, unit = h5open(path, "r") do file
        haskey(file, dataset) ||
            error("$(name): dataset $(repr(dataset)) is missing from $(relative)")
        stored = file[dataset]
        (read(stored), read_attribute(stored, "axis_order"), read_attribute(stored, "unit"))
    end
    Tuple(String.(axis_order)) == expected_axes || error(
        "$(name): $(relative) records axis_order $(Tuple(String.(axis_order))), " *
        "expected $(expected_axes)",
    )
    String(unit) == String(ref["unit"]) ||
        error("$(name): $(relative) records unit $(repr(String(unit))), expected $(repr(String(ref["unit"])))")
    # The file holds the dataspace in the declared axis order; Julia is column-major and
    # hands the dimensions back reversed, so the conversion is explicit rather than implied.
    values = ndims(raw) > 1 ? permutedims(raw, ndims(raw):-1:1) : raw
    declared_shape = Tuple(Int.(ref["shape"]))
    size(values) == declared_shape ||
        error("$(name): $(relative) holds $(size(values)), the case declares $(declared_shape)")
    converted = Float64.(values)
    all(isfinite, converted) || error("$(name): $(relative) holds nonfinite values")
    return converted
end

# --------------------------------------------------------------------------------------
# the physical constructor
# --------------------------------------------------------------------------------------

"""
    build_ow(case, arrays) -> NamedTuple{(:model, :parameters, :state0)}

Build the educational oil-water model of plan §3.1 from a verified case and its arrays.

`arrays` holds the normalised fields `porosity`, `permeability_m2`, `pressure_pa` and `sw`
— either from `load_arrays`, or materialized directly by a verification fixture. The
returned model, parameters and state are freshly allocated and owned by the caller.
"""
function build_ow(case::AbstractDict, arrays::AbstractDict)
    fluid = case["fluids"]
    String(get(fluid, "kind", "")) == "OW" || error(
        "build_ow: this adapter builds the educational oil-water system; the case declares " *
        "fluids.kind $(repr(get(fluid, "kind", nothing)))",
    )

    # Gravity is native, and only native. The adapter promises no gravity keyword: a zero-g
    # analytical limit is a registered analytic fixture that zeroes the reservoir's own
    # TwoPointGravityDifference parameter, never a different global constant and never here.
    @assert Jutul.gravity_constant == STANDARD_GRAVITY_M_S2
    declared_gravity = Float64(case["gravity_m_s2"])
    declared_gravity == Jutul.gravity_constant || error(
        "build_ow: the case declares gravity_m_s2=$(declared_gravity); the oil-water " *
        "dispatcher accepts only the native $(Jutul.gravity_constant) m/s^2",
    )

    rock = get(case, "rock", Dict{String,Any}())
    rock_compressibility = Float64(get(rock, "rock_compressibility_pa_inv", 0.0))
    rock_compressibility == 0.0 || error(
        "build_ow: pore volume is constant in E01; rock_compressibility_pa_inv must be " *
        "exactly 0, got $(rock_compressibility)",
    )

    dims = Tuple(Int.(case["grid"]["shape"]))
    extent = Tuple(Float64.(case["grid"]["extent_m"]))
    # Three dimensions, always: a two-dimensional mesh has no z, so JutulDarcy would hand
    # back a face gravity of zeros and the model would silently lose its buoyancy.
    (length(dims) == 3 && length(extent) == 3) ||
        error("build_ow: the grid must be three-dimensional, got shape $(dims) and extent $(extent)")
    n_cells = prod(dims)
    check_array_shapes(arrays, n_cells)

    mesh = CartesianMesh(dims, extent)
    domain = reservoir_domain(
        mesh;
        permeability = arrays["permeability_m2"],
        porosity = arrays["porosity"],
    )
    wells = [
        setup_well(
            domain,
            well_cells(w, n_cells);
            name = Symbol(w["well_id"]),
            radius = Float64(w["radius_m"]),
            simple_well = w["model"] == "simple",
            reference_depth = Float64(w["reference_depth_m"]),
        ) for w in get(case, "wells", Any[])
    ]

    rhoS = Float64.(fluid["density_sc_kg_m3"])
    sys = ImmiscibleSystem(OW_PHASES; reference_densities = rhoS)
    model = setup_reservoir_model(domain, sys; wells = wells, extra_outputs = false)

    # B_alpha = rho_alpha_sc / rho_alpha(p) with rho = rho_sc * exp(c * (p - p_sc)): one
    # reference pressure, one reference density and one compressibility per phase, so B is
    # exactly 1 at the standard pressure and density grows with pressure (plan 3.1).
    rho = ConstantCompressibilityDensities(
        p_ref = Float64(fluid["p_sc_pa"]),
        density_ref = rhoS,
        compressibility = Float64.(fluid["compressibility_pa_inv"]),
    )
    kr = BrooksCoreyRelativePermeabilities(
        sys,
        Float64.(fluid["corey_exponents"]),
        Float64.(fluid["residual_saturations"]),
        Float64.(fluid["kr_endpoints"]),
    )
    replace_variables!(model; PhaseMassDensities = rho, RelativePermeabilities = kr)

    parameters = setup_parameters(model)

    # One viscosity per phase, the same in the reservoir and in every well: a well left on
    # the 1e-3 default would move oil at the water's mobility, which no case asked for.
    mu = Float64.(fluid["viscosity_pa_s"])
    length(mu) == PHASE_COUNT ||
        error("build_ow: viscosity_pa_s must give one value per phase, got $(mu)")
    for (name, submodel) in pairs(model.models)
        if name == :Reservoir || JutulDarcy.model_or_domain_is_well(submodel)
            parameters[name][:PhaseViscosities] .= reshape(mu, PHASE_COUNT, 1)
        end
    end

    assert_constant_pore_volume(model)

    sw = arrays["sw"]
    state0 = setup_reservoir_state(
        model;
        Pressure = arrays["pressure_pa"],
        Saturations = vcat(sw', (1 .- sw)'),
    )
    return (; model, parameters, state0)
end

function check_array_shapes(arrays::AbstractDict, n_cells::Int)
    for name in ("porosity", "pressure_pa", "sw")
        haskey(arrays, name) || error("build_ow: the arrays do not carry $(name)")
        length(arrays[name]) == n_cells ||
            error("build_ow: $(name) has $(length(arrays[name])) values for $(n_cells) cells")
    end
    haskey(arrays, "permeability_m2") || error("build_ow: the arrays do not carry permeability_m2")
    size(arrays["permeability_m2"]) == (3, n_cells) || error(
        "build_ow: permeability_m2 has shape $(size(arrays["permeability_m2"])), " *
        "expected (3, $(n_cells))",
    )
    return nothing
end

"""Convert one well's zero-based connection list into Julia's one-based cell indices."""
function well_cells(well::AbstractDict, n_cells::Int)
    cells = Int.(well["cells"])
    isempty(cells) && error("build_ow: well $(well["well_id"]) has no connection cells")
    for c in cells
        (0 <= c < n_cells) || error(
            "build_ow: well $(well["well_id"]) names cell $(c), outside the zero-based " *
            "range [0, $(n_cells))",
        )
    end
    return cells .+ 1
end

"""
Refuse a model whose pore volume could move with pressure.

`pore_volume` reads `FluidVolume`, which JutulDarcy sets up as a PARAMETER. Replacing it
with a pressure-dependent secondary variable is exactly how a rock compressibility would
enter, and plan §3.1 fixes it at zero — so the invariant is asserted where the model is
built rather than discovered later in a mass balance.
"""
function assert_constant_pore_volume(model)
    rmodel = model.models[:Reservoir]
    haskey(Jutul.get_parameters(rmodel), :FluidVolume) || error(
        "build_ow: the reservoir has no FluidVolume parameter; pore volume would not be " *
        "the constant plan 3.1 requires",
    )
    for name in (:FluidVolume, :StaticFluidVolume)
        haskey(Jutul.get_secondary_variables(rmodel), name) && error(
            "build_ow: $(name) is a secondary variable, so pore volume would follow " *
            "pressure; E01 fixes rock_compressibility at 0",
        )
    end
    return nothing
end

# --------------------------------------------------------------------------------------
# the worker entry point
# --------------------------------------------------------------------------------------

"""
    run_job(job, root) -> NamedTuple{(:status, :reason, :model)}

The worker's entry point into physics: build this job's model from the case its descriptor
names, and describe what was built.

The descriptor's identity checks have already been made by the worker; this side still
re-hashes every array it reads, because the adapter does not take a caller's word for which
bytes it is modelling. The model, its parameters and its initial state are locals of this
call and are released when it returns.

The answer is deliberately NOT `COMPLETE`. A forward result is complete when it carries the
whole requested time axis and its verified outputs; this build integrates no time axis, so
it reports the refusal and the model it was able to construct.
"""
function run_job(job::AbstractDict, root::AbstractString)
    case_path = joinpath(root, String(job["case_path"]))
    case = JSON.parse(read(case_path, String))
    physical = build_ow(case, load_arrays(case, root))
    return (
        status = "INVALID_INPUT",
        reason = OUTPUTS_UNAVAILABLE,
        model = describe_model(physical),
    )
end

"""What was actually constructed, in a form a JSON result record can carry."""
function describe_model(physical)
    model = physical.model
    rmodel = model.models[:Reservoir]
    wells = String[]
    viscosities = Dict{String,Any}()
    for (name, submodel) in pairs(model.models)
        is_well = JutulDarcy.model_or_domain_is_well(submodel)
        is_well && push!(wells, string(name))
        if name == :Reservoir || is_well
            viscosity = physical.parameters[name][:PhaseViscosities]
            viscosities[string(name)] = Float64[viscosity[i, 1] for i in 1:PHASE_COUNT]
        end
    end
    return Dict{String,Any}(
        "n_cells" => number_of_cells(rmodel.domain),
        "phases" => [string(typeof(p)) for p in JutulDarcy.get_phases(rmodel.system)],
        "reference_densities_kg_m3" =>
            Float64[d for d in JutulDarcy.reference_densities(rmodel.system)],
        "pore_volume_m3" => sum(pore_volume(model, physical.parameters)),
        "viscosities_pa_s" => viscosities,
        "wells" => sort(wells),
        "gravity_m_s2" => Jutul.gravity_constant,
    )
end
