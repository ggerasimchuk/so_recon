# E01.0 — the explicit educational oil-water model, built on native JutulDarcy.
#
# `build_ow` is the whole physical constructor: a Cartesian mesh at the case's own datum, a
# reservoir domain, the wells the case names, a two-phase immiscible system, and the three
# property objects that make it the plan's §3.1 educational model rather than JutulDarcy's
# defaults — constant compressibility densities, Brooks-Corey relative permeabilities, and
# one viscosity per phase on every submodel that carries fluid. Nothing here implements
# physics of its own: every object below is a JutulDarcy constructor, because JutulDarcy is
# the only operational backend (SPEC 8.1).
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
# * z is depth, positive down, and ABSOLUTE. `CartesianMesh(dims, extent)` would origin the
#   grid at zero and put a reservoir declared at 1000 m depth at 5 m, leaving
#   `grid.cell_centers_m` — an input the model hash covers — unread. The mesh is therefore
#   origined at the datum that array declares, and every centroid the mesh produces is
#   compared back against it, so a declared input of F is never silently ignored.
# * Gravity is the native constant and only the native constant; the face term is
#   JutulDarcy's own `TwoPointGravityDifference`, derived from the domain's z centroids, so
#   this file never writes a gravity formula of its own.
# * Pore volume is constant: `rock_compressibility_pa_inv` must be 0 and the pore volume
#   stays the default `FluidVolume` PARAMETER, never replaced by a pressure-dependent
#   secondary variable.

#: Water first, oil second. Every (water, oil) tuple in the case — densities, viscosities,
#: compressibilities, Corey exponents — is read against this order.
const OW_PHASES = (AqueousPhase(), LiquidPhase())
const PHASE_COUNT = length(OW_PHASES)

const STANDARD_GRAVITY_M_S2 = 9.80665

"""
    InvalidCaseInput(message)

The case itself is the problem: an array that is not the bytes it claims, a geometry that
does not describe the grid it declares, a fluid this adapter does not build, or a gravity it
does not promise.

The worker reports this as `INVALID_INPUT`. Anything else that fails inside a JutulDarcy
constructor is reported as `PHYSICALLY_INVALID` instead, because there the input was well
formed and the physics still could not be assembled from it.
"""
struct InvalidCaseInput <: Exception
    message::String
end

Base.showerror(io::IO, err::InvalidCaseInput) = print(io, err.message)

invalid(message::AbstractString) = throw(InvalidCaseInput(message))

"""
    failure_status(err) -> String

Which forward status an exception raised inside the adapter becomes.

There is exactly one rule and it lives here, so the worker and the verification fixtures
classify the same failure the same way. `InvalidCaseInput` is the case being the problem —
bytes that are not what they claim, a geometry that does not describe its own grid, a fluid
this adapter does not build — and it is `INVALID_INPUT`. Anything else is `PHYSICALLY_INVALID`:
the input was well formed and the physics still could not be assembled from it, which is
exactly what a PVT with a non-positive density is.
"""
failure_status(err) = err isa InvalidCaseInput ? "INVALID_INPUT" : "PHYSICALLY_INVALID"

"""
    assert_physical_pvt(fluid)

Refuse a PVT that cannot describe a fluid, BEFORE a model is assembled from it.

A negative or zero standard density, a non-positive viscosity, a negative compressibility or
a non-positive reference pressure is not a malformed case: every field is present, of the
right type and of the right length. It is a case whose physics does not exist, and the
difference matters because a forward must not discover it as a non-converging Newton loop
three timesteps in and then be reported as a numerical failure. Refusing here, with a plain
error rather than an `InvalidCaseInput`, is what makes the job end as `PHYSICALLY_INVALID`
before the solver is ever entered (plan Task 9.5).

Python's `FluidSpec` refuses the same values when a case is built through the contract; this
is the other end, for every case that reaches the adapter as JSON.
"""
function assert_physical_pvt(fluid::AbstractDict)
    for (field, rule, predicate) in (
        ("density_sc_kg_m3", "positive", >(0.0)),
        ("viscosity_pa_s", "positive", >(0.0)),
        ("compressibility_pa_inv", "non-negative", >=(0.0)),
    )
        values = Float64.(fluid[field])
        (all(isfinite, values) && all(predicate, values)) && continue
        error(
            "build_ow: fluids.$(field) = $(values) is not physical; every entry must be " *
            "$(rule) and finite. The case is well formed and its physics is not: this is " *
            "refused at construction, not discovered as a non-converging solve",
        )
    end
    p_sc = Float64(fluid["p_sc_pa"])
    (isfinite(p_sc) && p_sc > 0.0) || error(
        "build_ow: fluids.p_sc_pa = $(p_sc) is not a physical reference pressure; it must " *
        "be positive and finite",
    )
    return nothing
end

#: How far a declared cell centre may sit from the centroid the mesh actually produces. A
#: micrometre is far below anything with geometric meaning and far above float64 round-off,
#: which at reservoir depths is a few picometres.
const CELL_CENTER_TOLERANCE_M = 1e-6

#: The arrays `build_ow` reads, where the case manifest keeps each one's `ArrayRef`, and the
#: axis order the exchange declares for it (plan 3.2). The layout is read from the file, not
#: assumed: an `(n_times, n_cells)` dataspace arrives in Julia with its dimensions reversed.
const REQUIRED_ARRAYS = (
    ("cell_centers_m", ("grid", "cell_centers_m"), ("cell", "dim")),
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
nonfinite value is an `InvalidCaseInput`: the adapter does not repair an input, and it does
not build a model out of bytes nobody vouched for.
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
            invalid("$(name): the case has no $(join(location, ".")); it cannot build a model")
        node = node[key]
    end
    node isa AbstractDict ||
        invalid("$(name): $(join(location, ".")) is not an array reference mapping")
    for key in ("path", "dataset", "sha256", "shape", "unit", "axis_order")
        haskey(node, key) || invalid("$(name): the array reference is missing $(key)")
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
    isfile(path) || invalid("$(name): the declared array $(relative) does not exist")
    digest = bytes2hex(open(sha256, path))
    declared_digest = get(ref, "sha256", nothing)
    if !(declared_digest isa AbstractString) || digest != declared_digest
        invalid(
            "$(name): $(relative) hashes to $(digest), which is not the declared " *
            "$(repr(declared_digest))",
        )
    end
    declared_axes = Tuple(String.(ref["axis_order"]))
    declared_axes == expected_axes ||
        invalid("$(name): the case declares axis_order $(declared_axes), expected $(expected_axes)")
    dataset = String(ref["dataset"])
    raw, axis_order, unit = h5open(path, "r") do file
        haskey(file, dataset) || invalid("$(name): dataset $(repr(dataset)) is missing from $(relative)")
        stored = file[dataset]
        (read(stored), read_attribute(stored, "axis_order"), read_attribute(stored, "unit"))
    end
    Tuple(String.(axis_order)) == expected_axes || invalid(
        "$(name): $(relative) records axis_order $(Tuple(String.(axis_order))), " *
        "expected $(expected_axes)",
    )
    String(unit) == String(ref["unit"]) || invalid(
        "$(name): $(relative) records unit $(repr(String(unit))), expected " *
        "$(repr(String(ref["unit"])))",
    )
    # The file holds the dataspace in the declared axis order; Julia is column-major and
    # hands the dimensions back reversed, so the conversion is explicit rather than implied.
    values = ndims(raw) > 1 ? permutedims(raw, ndims(raw):-1:1) : raw
    declared_shape = Tuple(Int.(ref["shape"]))
    size(values) == declared_shape ||
        invalid("$(name): $(relative) holds $(size(values)), the case declares $(declared_shape)")
    converted = Float64.(values)
    all(isfinite, converted) || invalid("$(name): $(relative) holds nonfinite values")
    return converted
end

# --------------------------------------------------------------------------------------
# the physical constructor
# --------------------------------------------------------------------------------------

"""
    build_ow(case, arrays) -> NamedTuple{(:model, :parameters, :state0)}

Build the educational oil-water model of plan §3.1 from a verified case and its arrays.

`arrays` holds the normalised fields `cell_centers_m`, `porosity`, `permeability_m2`,
`pressure_pa` and `sw` — either from `load_arrays`, or materialized directly by a
verification fixture. The returned model, parameters and state are freshly allocated and
owned by the caller.

A case whose geometry, fluid, gravity or rock compressibility this adapter cannot honour is
refused with `InvalidCaseInput`, never quietly adapted.
"""
function build_ow(case::AbstractDict, arrays::AbstractDict)
    fluid = case["fluids"]
    String(get(fluid, "kind", "")) == "OW" || invalid(
        "build_ow: this adapter builds the educational oil-water system; the case declares " *
        "fluids.kind $(repr(get(fluid, "kind", nothing)))",
    )

    # A PVT that cannot describe a fluid is refused before anything is built from it, so
    # the job ends as PHYSICALLY_INVALID at construction and never as a non-converging solve.
    assert_physical_pvt(fluid)

    # Gravity is native, and only native. The adapter promises no gravity keyword: a zero-g
    # analytical limit is a registered analytic fixture that zeroes the reservoir's own
    # TwoPointGravityDifference parameter, never a different global constant and never here.
    @assert Jutul.gravity_constant == STANDARD_GRAVITY_M_S2
    declared_gravity = Float64(case["gravity_m_s2"])
    declared_gravity == Jutul.gravity_constant || invalid(
        "build_ow: the case declares gravity_m_s2=$(declared_gravity); the oil-water " *
        "dispatcher accepts only the native $(Jutul.gravity_constant) m/s^2",
    )

    rock = get(case, "rock", Dict{String,Any}())
    rock_compressibility = Float64(get(rock, "rock_compressibility_pa_inv", 0.0))
    rock_compressibility == 0.0 || invalid(
        "build_ow: pore volume is constant in E01; rock_compressibility_pa_inv must be " *
        "exactly 0, got $(rock_compressibility)",
    )

    dims = Tuple(Int.(case["grid"]["shape"]))
    extent = Tuple(Float64.(case["grid"]["extent_m"]))
    # Three dimensions, always: a two-dimensional mesh has no z, so JutulDarcy would hand
    # back a face gravity of zeros and the model would silently lose its buoyancy.
    (length(dims) == 3 && length(extent) == 3) ||
        invalid("build_ow: the grid must be three-dimensional, got shape $(dims) and extent $(extent)")
    (all(>(0), dims) && all(>(0.0), extent)) || invalid(
        "build_ow: the grid must have a positive size in every direction, got shape $(dims) " *
        "and extent_m $(extent)",
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
    # Before any of them is built: a crossflow declaration this backend cannot express is
    # refused, never quietly replaced by the semantics it is able to offer.
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
        invalid("build_ow: viscosity_pa_s must give one value per phase, got $(mu)")
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
        haskey(arrays, name) || invalid("build_ow: the arrays do not carry $(name)")
        length(arrays[name]) == n_cells ||
            invalid("build_ow: $(name) has $(length(arrays[name])) values for $(n_cells) cells")
    end
    for (name, shape) in (("permeability_m2", (3, n_cells)), ("cell_centers_m", (n_cells, 3)))
        haskey(arrays, name) || invalid("build_ow: the arrays do not carry $(name)")
        size(arrays[name]) == shape ||
            invalid("build_ow: $(name) has shape $(size(arrays[name])), expected $(shape)")
    end
    return nothing
end

"""
    cartesian_origin(dims, extent, centers) -> NTuple{3, Float64}

Where the grid's first corner has to be for cell 0 to sit where the case says it does.

`cell_id = i + nx*(j + ny*k)` makes zero-based cell 0 the row 1 of `cell_centers_m`, and a
uniform Cartesian cell puts its centre half a cell from that corner in each direction.
"""
function cartesian_origin(dims::Tuple, extent::Tuple, centers::AbstractMatrix)
    return Tuple(Float64[centers[1, d] - 0.5 * extent[d] / dims[d] for d in 1:3])
end

"""
Prove the mesh reproduces every centre the case declared, and refuse it if it does not.

This is deliberately a comparison against `domain[:cell_centroids]` — what JutulDarcy will
actually use for face gravity, well placement and equilibration — rather than the same
formula evaluated twice. It therefore also checks the CELL ORDERING: a case written with
`cell_id = i + nx*(j + ny*k)` against a mesh that numbered its cells some other way
disagrees here, loudly and at construction, instead of three tasks later in a saturation
field nobody can explain.
"""
function check_cell_centers(domain, declared::AbstractMatrix)
    centroids = domain[:cell_centroids]
    n_cells = size(centroids, 2)
    size(declared, 1) == n_cells || invalid(
        "build_ow: grid.cell_centers_m describes $(size(declared, 1)) cells, the mesh has " *
        "$(n_cells)",
    )
    worst, cell, axis = 0.0, 0, 0
    for c in 1:n_cells, d in 1:3
        delta = abs(centroids[d, c] - declared[c, d])
        if delta > worst
            worst, cell, axis = delta, c, d
        end
    end
    worst <= CELL_CENTER_TOLERANCE_M || invalid(
        "build_ow: grid.cell_centers_m is not the uniform Cartesian grid that shape and " *
        "extent_m imply. The worst disagreement is at zero-based cell $(cell - 1) on the " *
        "$(("x", "y", "z")[axis]) axis: the case declares $(declared[cell, axis]) m, the mesh " *
        "places that centroid at $(centroids[axis, cell]) m, a difference of $(worst) m " *
        "against a tolerance of $(CELL_CENTER_TOLERANCE_M) m",
    )
    return nothing
end

"""
    check_crossflow(well)

Refuse a well whose declared crossflow semantics this backend cannot express.

`allow_crossflow` is a physical switch the case must state (plan 3.2), and JutulDarcy
0.3.11 implements only one side of it: the string "crossflow" does not appear anywhere in
either pinned package. A well is a wellbore, and a wellbore with more than one connection
can always take fluid in at one and put it out at another — under a rate control, under a
bhp control, and under `DisabledControl`, whose whole formulation is a net surface rate of
zero with the connections still coupled. There is no keyword that turns that off.

So a multi-connection well that declares `allow_crossflow = false` is refused with the
reason, rather than simulated with the semantics it asked not to have. A well with a single
connection has nowhere to cross to, so both declarations are honoured there.
"""
function check_crossflow(well::AbstractDict)
    allow = get(well, "allow_crossflow", nothing)
    allow isa Bool || invalid(
        "build_ow: well $(get(well, "well_id", "<unnamed>")) must declare allow_crossflow as " *
        "a boolean, got $(repr(allow)); a physical switch is stated, never inherited",
    )
    (allow || length(well["cells"]) < 2) && return nothing
    invalid(
        "build_ow: well $(well["well_id"]) declares allow_crossflow=false over " *
        "$(length(well["cells"])) connections, which JutulDarcy $(pkgversion(JutulDarcy)) " *
        "cannot express: a multi-connection wellbore always couples its connections, and " *
        "no native control or force switches that off. Refusing rather than simulating the " *
        "opposite semantics under the case's own label",
    )
end

"""Convert one well's zero-based connection list into Julia's one-based cell indices."""
function well_cells(well::AbstractDict, n_cells::Int)
    cells = Int.(well["cells"])
    isempty(cells) && invalid("build_ow: well $(well["well_id"]) has no connection cells")
    for c in cells
        (0 <= c < n_cells) || invalid(
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
built rather than discovered later in a mass balance. This one is NOT an `InvalidCaseInput`:
no case can ask for it, so reaching it would mean this adapter built the wrong model.
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
    run_job(job, root; kwarg...) -> NamedTuple

The worker's entry point into physics: run this job's case to a native extraction.

The descriptor's identity checks have already been made by the worker; this side still
re-hashes every array it reads and the solver configuration it is driven by, because the
adapter does not take a caller's word for which bytes it is modelling. The model, its
parameters and its initial state are locals of `run_forward_native` and are released when it
returns, which is what keeps job B's result independent of job A's.

A job that names a `resume_from` checkpoint has every piece of that checkpoint's metadata
verified and its bytes re-hashed BEFORE a single native state is read, and the verified files
are copied into this job's own native working directory: the parent's restart bytes are never
written to. A checkpoint from another model, another environment or another schedule prefix,
or one whose files moved, is `INVALID_INPUT` — never a numerical retry.

The status describes the SIMULATION. `COMPLETE` means the solver reached the end of the
schedule and every chunk was extracted; whether that becomes a `COMPLETE` forward RESULT is
decided where the outputs are published, because a control the well could not hold is a
question about the record rather than about the solver (plan 3.3).
"""
function run_job(
    job::AbstractDict,
    root::AbstractString;
    native_dir::AbstractString,
    environment_lock_hash::AbstractString,
    static_hash::AbstractString,
    schedule_prefix_hashes = Any[],
    stop_requested = _ -> false,
)
    case_path = joinpath(root, String(job["case_path"]))
    case = JSON.parse(read(case_path, String))
    arrays = load_arrays(case, root)
    solver = read_solver_config(joinpath(root, String(job["solver_config_path"])))

    request = get(job, "output_request", Dict{String,Any}())
    requested_times = haskey(request, "state_times_s") ?
                      Float64.(collect(request["state_times_s"])) : nothing
    chunk_months = Int(get(request, "chunk_months", 1))
    keep_native_restart = Bool(get(request, "keep_native_restart", false))

    resume = get(job, "resume_from", nothing)
    resume_from_step = 0
    if resume !== nothing
        manifest_path = joinpath(root, String(resume["manifest_path"]))
        manifest = verify_restart(
            resume, manifest_path, static_hash, schedule_prefix_hashes, environment_lock_hash,
        )
        resume_from_step = stage_restart(manifest, manifest_path, native_dir)
    end

    payload = run_forward_native(
        case, arrays, native_dir;
        state_times_s = requested_times,
        chunk_months = chunk_months,
        resume_from_step = resume_from_step,
        stop_requested = stop_requested,
        max_timestep = solver.max_timestep_days * SECONDS_PER_DAY,
        max_nonlinear_iterations = solver.max_nonlinear_iterations,
    )

    completed_step = Int(get(payload, "completed_report_step", 0))
    completed_time = Float64(get(payload, "completed_time_s", 0.0))
    checkpoint = nothing
    if keep_native_restart && completed_step >= 1
        checkpoint = (
            completed_report_step = completed_step,
            completed_time_s = completed_time,
            model_hash = get(case, "model_hash", nothing),
            static_hash = static_hash,
            case_sha256 = get(job, "case_sha256", nothing),
            schedule_prefix_hash =
                prefix_hash_for(schedule_prefix_hashes, completed_time, native_dir),
            environment_lock_hash = environment_lock_hash,
        )
    end
    return (
        status = String(payload["status"]),
        reason = payload["reason"],
        model = pop!(payload, "model", nothing),
        payload = payload,
        native_dir = native_dir,
        checkpoint = checkpoint,
    )
end

"""
    prefix_hash_for(hashes, completed_time_s, where) -> String

The digest of the schedule PREFIX this checkpoint stops at.

The Python side canonicalises the case, so it computes one digest per report edge and sends
them all; this side picks the one for the time it actually reached. A checkpoint whose time
has no digest is refused rather than published with a placeholder — a restart that could not
prove its prefix is a restart nobody can safely continue.
"""
function prefix_hash_for(hashes, completed_time_s::Float64, where::AbstractString)
    for entry in hashes
        abs(Float64(entry["completed_time_s"]) - completed_time_s) <= TIME_MATCH_TOLERANCE_S &&
            return String(entry["schedule_prefix_hash"])
    end
    invalid(
        "run_job: the job carries no schedule prefix digest for the checkpoint at " *
        "$(completed_time_s) s reached in $(where); a checkpoint without one cannot be " *
        "continued from safely",
    )
end

"""
    read_solver_config(path) -> NamedTuple

The two numbers a job may say about how the solver is driven, and nothing else.

`max_timestep_days` and `max_nonlinear_iterations` bound EFFORT. A configuration carrying
anything else is refused rather than partially honoured: SPEC 3.3 lets a numerical retry
halve the step and raise the iteration limit «неизменными convergence tolerances и
физическими входами», and a file this side would silently ignore half of is a file that
could carry a relaxed tolerance nobody noticed.
"""
function read_solver_config(path::AbstractString)
    isfile(path) ||
        invalid("read_solver_config: the solver configuration $(path) does not exist")
    payload = JSON.parse(String(copy(read(path))))
    payload isa AbstractDict ||
        invalid("read_solver_config: $(path) is not a JSON object")
    allowed = Set(["max_timestep_days", "max_nonlinear_iterations"])
    extra = sort(collect(setdiff(Set(keys(payload)), allowed)))
    isempty(extra) || invalid(
        "read_solver_config: $(path) carries $(extra), which this adapter does not apply. A " *
        "solver configuration states the maximum timestep and the nonlinear-iteration limit " *
        "and nothing else",
    )
    for name in sort(collect(allowed))
        haskey(payload, name) ||
            invalid("read_solver_config: $(path) does not state $(name)")
    end
    days = Float64(payload["max_timestep_days"])
    iterations = Int(payload["max_nonlinear_iterations"])
    days > 0.0 ||
        invalid("read_solver_config: max_timestep_days must be positive, got $(days)")
    iterations >= 1 || invalid(
        "read_solver_config: max_nonlinear_iterations must be at least one, got $(iterations)",
    )
    return (max_timestep_days = days, max_nonlinear_iterations = iterations)
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
    depths = vec(JutulDarcy.reservoir_domain(model)[:cell_centroids][3, :])
    return Dict{String,Any}(
        "n_cells" => number_of_cells(rmodel.domain),
        # Absolute, from the case's own datum: the depths the model was really built at.
        "cell_center_depth_m" => [minimum(depths), maximum(depths)],
        "phases" => [string(typeof(p)) for p in JutulDarcy.get_phases(rmodel.system)],
        "reference_densities_kg_m3" =>
            Float64[d for d in JutulDarcy.reference_densities(rmodel.system)],
        "pore_volume_m3" => sum(pore_volume(model, physical.parameters)),
        "viscosities_pa_s" => viscosities,
        "wells" => sort(wells),
        "gravity_m_s2" => Jutul.gravity_constant,
    )
end
