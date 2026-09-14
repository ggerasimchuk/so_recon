# E01.8 — native continuation, one calendar month at a time, and the checkpoint that
# proves a continuation is a continuation of THIS run.
#
# `model.jl` builds the model, `controls.jl` the forces, `outputs.jl` reads a finished
# simulation. This file owns the two things none of them can: how a forward is cut into
# chunks that are simulated and extracted one at a time, and how the native state at a
# report-step boundary becomes an immutable artefact somebody else's process can resume
# from.
#
# WHY NOT THE HIGH-LEVEL WRAPPER. `simulate_reservoir(...; output_path, restart = k+1)`
# does run the right steps, and for a single unchunked run it is the right call. It cannot
# be the chunk driver, because of what happens after the loop: `simulate!` ends in
# `retrieve_output!`, which — with the default `output_states = true` — reads `range = 1:n`
# back from the output directory, where `n` is the WHOLE schedule and not the part this
# call simulated (`Jutul 0.4.31 src/simulator/io.jl:11`). A twelfth month restarted at step
# 12 therefore returns twelve months of states in memory, and `extract_interval` on it
# would integrate the first eleven a second time. So the driver here sets
# `output_states = false` and `output_reports = false`, which makes `retrieve_output!`
# return immediately, and reads back exactly `first:last` of the chunk in hand with
# `Jutul.read_results(range = ...)`. `extra_outputs = false` proves nothing on its own;
# what bounds memory is that only one chunk is ever materialised, and the payload records
# `gc_live_bytes` after a collection at every chunk boundary so that the claim is measured
# rather than asserted.
#
# WHAT THE NATIVE RESTART ACTUALLY DOES (read, not assumed, in
# `Jutul 0.4.31 src/simulator/simulator.jl:680`): `restart = k` begins at step k and reads
# the state written for step `k-1`; `restart = true` resolves to `maximum(valid indices)+1`;
# `restart = false`, `nothing`, `0` or `1` all mean "start from the beginning". The state it
# reads is `get_output_state`, so it is exactly the model's `output_variables` — which is
# why `request_extra_outputs!` matters here twice over: without it a `SimpleWell`'s
# connection pressure drop is neither extracted NOR restored. `deserialize_restart` also
# reads back the previous step's report and takes its LAST mini-step `dt` as the initial
# step size, and reads `in_memory_reports` reports back so the iteration-based timestep
# selector sees the same history a continuous run would. That is what makes a continuation
# numerically the same run rather than a new one starting from the same numbers.
#
# THE CHECKPOINT IS NOT THE WORKING DIRECTORY. A job simulates into a mutable native
# directory of its own. What it PUBLISHES is a copy, staged and renamed into place, with a
# manifest naming every file and its SHA-256, the model hash, the schedule-prefix hash, the
# environment lock hash and the step and time actually reached. A resume copies that
# snapshot into its own new working directory and continues there, so the parent's bytes are
# never written to — and a manifest from a different model, a different environment or a
# different schedule prefix, or a file whose bytes moved, is INVALID_INPUT and never a
# numerical retry.

const RESTART_MANIFEST_FILENAME = "restart_manifest.json"
const RESTART_MANIFEST_SCHEMA = "restart-manifest-1"
const NATIVE_RESTART_FORMAT = "Jutul-native"
const NATIVE_STEP_PREFIX = "jutul_"
const NATIVE_STEP_SUFFIX = ".jld2"

"""Digest of one file's bytes, in the hex the whole exchange speaks."""
restart_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

"""
    chunk_ranges(month_index, chunk_months) -> Vector{UnitRange{Int}}

Group compiled intervals into chunks of whole calendar months.

`month_index[i]` is the zero-based report month interval `i` lies inside, so a completion
event inside a month does NOT start a chunk: the chunk boundary is a month boundary, which
is the boundary a restart is taken at and the boundary monthly volumes are reported over.
"""
function chunk_ranges(month_index::AbstractVector{Int}, chunk_months::Int)
    chunk_months >= 1 ||
        invalid("chunk_ranges: chunk_months must be at least one month, got $(chunk_months)")
    n = length(month_index)
    n >= 1 || invalid("chunk_ranges: a schedule has at least one interval")
    ranges = UnitRange{Int}[]
    i = 1
    while i <= n
        group = div(month_index[i], chunk_months)
        j = i
        while j < n && div(month_index[j + 1], chunk_months) == group
            j += 1
        end
        push!(ranges, i:j)
        i = j + 1
    end
    return ranges
end

"""The native step files present in a working directory, as `(index, filename)` pairs."""
function native_step_files(dir::AbstractString)
    isdir(dir) || return Tuple{Int,String}[]
    out = Tuple{Int,String}[]
    for name in readdir(dir)
        (startswith(name, NATIVE_STEP_PREFIX) && endswith(name, NATIVE_STEP_SUFFIX)) || continue
        body = name[(length(NATIVE_STEP_PREFIX) + 1):(end - length(NATIVE_STEP_SUFFIX))]
        index = tryparse(Int, body)
        index === nothing && continue
        push!(out, (index, name))
    end
    sort!(out; by = first)
    return out
end

"""
    write_checkpoint(native_dir, destination; kwargs...) -> Dict

Publish the native state of a finished chunk as an IMMUTABLE snapshot.

Every `jutul_i.jld2` up to `completed_report_step` is copied (never moved, never linked:
the working directory stays writable and the snapshot stays fixed) into a staging directory
which is then renamed into place, so a process killed half way leaves unfinished staging
rather than a checkpoint that claims to be complete. The manifest is written inside the
staging directory, before the rename, and lists each file's relative path and SHA-256
alongside the identity a continuation has to match.
"""
function write_checkpoint(
    native_dir::AbstractString,
    destination::AbstractString;
    completed_report_step::Int,
    completed_time_s::Float64,
    model_hash,
    static_hash,
    case_sha256,
    schedule_prefix_hash,
    environment_lock_hash,
)
    completed_report_step >= 1 || invalid(
        "write_checkpoint: a checkpoint is taken at a completed report step, got " *
        "$(completed_report_step)",
    )
    present = native_step_files(native_dir)
    indices = [i for (i, _) in present]
    wanted = collect(1:completed_report_step)
    missing_steps = setdiff(wanted, indices)
    isempty(missing_steps) || invalid(
        "write_checkpoint: the native working directory is missing steps " *
        "$(missing_steps); a checkpoint names every step a continuation reads back",
    )

    ispath(destination) && invalid(
        "write_checkpoint: $(destination) already exists; a checkpoint is published once",
    )
    mkpath(dirname(destination))
    staging = string(destination, ".staging-", getpid(), "-", time_ns())
    mkpath(staging)
    files = Any[]
    try
        for step in wanted
            name = string(NATIVE_STEP_PREFIX, step, NATIVE_STEP_SUFFIX)
            cp(joinpath(native_dir, name), joinpath(staging, name))
            push!(
                files,
                Dict{String,Any}(
                    "path" => name,
                    "step" => step,
                    "sha256" => restart_sha256(joinpath(staging, name)),
                    "bytes" => filesize(joinpath(staging, name)),
                ),
            )
        end
        manifest = Dict{String,Any}(
            "schema_version" => RESTART_MANIFEST_SCHEMA,
            "native_format" => NATIVE_RESTART_FORMAT,
            "completed_report_step" => completed_report_step,
            "completed_time_s" => completed_time_s,
            "model_hash" => model_hash,
            # Everything that determines F except the schedule. A continuation may change
            # what happens after this point and nothing before it, so THIS is what must be
            # identical and the model hash is not.
            "static_hash" => static_hash,
            "case_sha256" => case_sha256,
            "schedule_prefix_hash" => schedule_prefix_hash,
            "environment_lock_hash" => environment_lock_hash,
            "julia" => string(VERSION),
            "Jutul" => string(pkgversion(Jutul)),
            "JutulDarcy" => string(pkgversion(JutulDarcy)),
            "files" => files,
        )
        open(joinpath(staging, RESTART_MANIFEST_FILENAME), "w") do io
            JSON.print(io, manifest)
            print(io, "\n")
        end
        mv(staging, destination)
        return manifest
    catch
        rm(staging; force = true, recursive = true)
        rethrow()
    end
end

"""
    verify_restart(ref, manifest_path, case, lock_hash) -> Dict

Check EVERY piece of a checkpoint's metadata before a single native byte is read.

`ref` is the `RestartRef` the descriptor carried; `manifest_path` is where it says the
manifest lives. The manifest's own bytes are hashed against the reference, then the identity
is compared field by field — the static half of the model, the schedule prefix, the
environment lock, the native format and the step and time reached — and only then is each
native file re-hashed against the digest the manifest declares.

The MODEL hash is compared against the descriptor's reference and NOT against the case being
resumed: a continuation is allowed to change the policy after its checkpoint, which changes
the model hash by construction. What may not move is the static half and the prefix, and both
are checked here.

Every failure here is an `InvalidCaseInput`, which the worker reports as `INVALID_INPUT`.
None of them is a numerical retry: a checkpoint from a different version, a changed prefix
or a corrupted file is not a run that nearly converged.
"""
function verify_restart(
    ref::AbstractDict,
    manifest_path::AbstractString,
    static_hash,
    schedule_prefix_hashes,
    lock_hash,
)
    isfile(manifest_path) || invalid(
        "verify_restart: the checkpoint manifest $(get(ref, "manifest_path", manifest_path)) " *
        "does not exist; an unfinished staging directory is not a checkpoint",
    )
    actual = restart_sha256(manifest_path)
    declared = get(ref, "sha256", nothing)
    actual == declared || invalid(
        "verify_restart: the checkpoint manifest hashes to $(actual), but the descriptor " *
        "declares $(repr(declared))",
    )
    manifest = JSON.parse(String(copy(read(manifest_path))))
    directory = dirname(manifest_path)

    get(manifest, "schema_version", nothing) == RESTART_MANIFEST_SCHEMA || invalid(
        "verify_restart: the checkpoint declares schema_version " *
        "$(repr(get(manifest, "schema_version", nothing))); this build reads " *
        "$(RESTART_MANIFEST_SCHEMA)",
    )
    for (field, expected) in (
        ("native_format", NATIVE_RESTART_FORMAT),
        # The half of the model a continuation may NOT change: grid, rock, fluids, wells,
        # initial state, boundary. The model hash itself is deliberately not compared —
        # changing the policy after the checkpoint changes it, and that is allowed.
        ("static_hash", static_hash),
        ("environment_lock_hash", lock_hash),
    )
        get(manifest, field, nothing) == expected || invalid(
            "verify_restart: the checkpoint's $(field) is " *
            "$(repr(get(manifest, field, nothing))), not $(repr(expected)); a continuation " *
            "of a different model or environment is INVALID_INPUT, never a numerical retry",
        )
    end
    for field in ("model_hash", "schedule_prefix_hash", "environment_lock_hash",
                  "completed_report_step", "native_format")
        manifest[field] == get(ref, field, nothing) || invalid(
            "verify_restart: the checkpoint's $(field) is $(repr(manifest[field])), but the " *
            "descriptor's restart reference declares $(repr(get(ref, field, nothing)))",
        )
    end
    abs(Float64(manifest["completed_time_s"]) - Float64(get(ref, "completed_time_s", NaN))) <=
    TIME_MATCH_TOLERANCE_S || invalid(
        "verify_restart: the checkpoint stops at $(manifest["completed_time_s"]) s, but the " *
        "descriptor's restart reference declares $(get(ref, "completed_time_s", nothing)) s",
    )

    # And the half it may not change EITHER side of: what was already simulated. The
    # digest is the Python side's canonical one, computed from the case being resumed, so
    # agreement means this case's history up to the checkpoint is the history the checkpoint
    # was taken after — a rewritten past is INVALID_INPUT.
    expected_prefix =
        prefix_hash_for(schedule_prefix_hashes, Float64(manifest["completed_time_s"]), manifest_path)
    manifest["schedule_prefix_hash"] == expected_prefix || invalid(
        "verify_restart: the checkpoint was taken after a schedule prefix hashing to " *
        "$(manifest["schedule_prefix_hash"]), and this case's prefix up to " *
        "$(manifest["completed_time_s"]) s hashes to $(expected_prefix); a continuation of a " *
        "rewritten history is INVALID_INPUT, never a numerical retry",
    )

    files = manifest["files"]
    for entry in files
        path = joinpath(directory, String(entry["path"]))
        isfile(path) || invalid(
            "verify_restart: the checkpoint names $(entry["path"]), which does not exist",
        )
        digest = restart_sha256(path)
        digest == entry["sha256"] || invalid(
            "verify_restart: $(entry["path"]) hashes to $(digest), but the checkpoint " *
            "manifest declares $(entry["sha256"]); a corrupted restart file is INVALID_INPUT",
        )
    end
    steps = sort(Int[entry["step"] for entry in files])
    steps == collect(1:Int(manifest["completed_report_step"])) || invalid(
        "verify_restart: the checkpoint holds steps $(steps) for a completed report step " *
        "$(manifest["completed_report_step"]); a gap would make Jutul read a state nobody wrote",
    )
    return manifest
end

"""
    stage_restart(manifest, manifest_path, native_dir)

Copy a verified checkpoint into a NEW native working directory and leave the parent alone.

The continuation writes `jutul_{k+1}.jld2` onwards into `native_dir`, so it must not be the
directory the checkpoint lives in: writing there would modify an immutable artefact that
another job may still resume from. Copying is what makes the parent's bytes unreachable
from this job.
"""
function stage_restart(manifest::AbstractDict, manifest_path::AbstractString, native_dir::AbstractString)
    source = dirname(manifest_path)
    abspath(source) == abspath(native_dir) && invalid(
        "stage_restart: a continuation must not write into the checkpoint it resumes from " *
        "($(native_dir)); the parent's restart bytes are immutable",
    )
    mkpath(native_dir)
    for entry in manifest["files"]
        name = String(entry["path"])
        cp(joinpath(source, name), joinpath(native_dir, name); force = true)
    end
    return Int(manifest["completed_report_step"])
end

# --------------------------------------------------------------------------------------
# merging the chunks back into one extraction
# --------------------------------------------------------------------------------------

"""
    merge_extractions(payloads, interval_offsets, requested_times) -> Dict

One whole-horizon extraction from a sequence of per-chunk ones.

Three things are NOT concatenations and are the reason this exists:

* the INVENTORY of a chunk starts with the state before its first substep, which is the
  previous chunk's last state. Every chunk after the first therefore contributes
  `inventory[2:end]`, or the boundary state would be counted twice and the balance would
  close over a horizon nobody simulated;
* `interval_index` is local to the chunk it was extracted from, and is shifted here onto
  the schedule's own numbering;
* a `connections` row names the substep it belongs to by index, which is likewise local.

`requested_times` is the axis the CALLER asked for. Each chunk is extracted at the
requested times that fall inside it, and at its own end when none do — `requested_states`
refuses an empty request, and a chunk still has to be extracted to be integrated. Those
filler states are dropped here, so the published axis is exactly the requested one.
"""
function merge_extractions(
    payloads::AbstractVector,
    interval_offsets::AbstractVector{Int},
    requested_times::AbstractVector{Float64},
)
    isempty(payloads) && invalid("merge_extractions: there is nothing to merge")
    length(payloads) == length(interval_offsets) ||
        invalid("merge_extractions: one interval offset per chunk is required")
    head = payloads[1]

    edges = Float64[]
    starts, ends, dts = Float64[], Float64[], Float64[]
    intervals = Int[]
    connections = Any[]
    boundary_rows = Any[]
    inventory, reservoir_inventory = Any[], Any[]
    surface_source, connection_source, boundary_source = Any[], Any[], Any[]
    evidence = Dict{String,Any}()
    wells = Dict{String,Any}()
    accepted, cut, iterations = 0, 0, 0
    state_times = Float64[]
    state_fields = Dict{String,Any}(
        name => Any[] for name in ("pressure_pa", "sw", "so", "pore_volume_m3", "bw", "bo")
    )
    substep_offset = 0

    for (index, payload) in enumerate(payloads)
        chunk = payload["chunk"]
        chunk_edges = Float64.(chunk["edges_s"])
        if isempty(edges)
            append!(edges, chunk_edges)
        else
            abs(chunk_edges[1] - edges[end]) <= TIME_MATCH_TOLERANCE_S || invalid(
                "merge_extractions: chunk $(index - 1) starts at $(chunk_edges[1]) s but the " *
                "previous one ended at $(edges[end]) s; the chunks must tile the horizon",
            )
            append!(edges, chunk_edges[2:end])
        end
        append!(starts, Float64.(chunk["start_s"]))
        append!(ends, Float64.(chunk["end_s"]))
        append!(dts, Float64.(chunk["dt_s"]))
        append!(intervals, Int.(chunk["interval_index"]) .+ interval_offsets[index])

        for row in payload["connections"]
            merged = Dict{String,Any}(row)
            merged["step"] = Int(row["step"]) + substep_offset
            push!(connections, merged)
        end

        for row in payload["boundary"]
            merged = Dict{String,Any}(row)
            merged["step"] = Int(row["step"]) + substep_offset
            push!(boundary_rows, merged)
        end

        if index == 1
            append!(inventory, payload["inventory_m3_sc"])
            append!(reservoir_inventory, payload["reservoir_inventory_m3_sc"])
        else
            # The chunk's own first inventory row IS the previous chunk's last: one state,
            # not two. Dropping it here is what keeps the prefix out of the suffix's sum.
            append!(inventory, payload["inventory_m3_sc"][2:end])
            append!(reservoir_inventory, payload["reservoir_inventory_m3_sc"][2:end])
        end
        append!(surface_source, payload["net_surface_source_m3_sc"])
        append!(connection_source, payload["net_connection_source_m3_sc"])
        append!(boundary_source, payload["net_boundary_source_m3_sc"])

        for (name, well) in payload["wells"]
            if !haskey(wells, name)
                wells[name] = Dict{String,Any}(
                    "cells" => well["cells"],
                    "surface_water_m3_s" => Float64[],
                    "surface_oil_m3_s" => Float64[],
                    "surface_component_mass_kg_s" =>
                        [Float64[] for _ in well["surface_component_mass_kg_s"]],
                    "bhp_pa" => Float64[],
                    "operating_target" => String[],
                )
            end
            target = wells[name]
            target["cells"] == well["cells"] || invalid(
                "merge_extractions: well $(name) perforates $(well["cells"]) in chunk " *
                "$(index - 1) and $(target["cells"]) earlier; the model changed between chunks",
            )
            append!(target["surface_water_m3_s"], Float64.(well["surface_water_m3_s"]))
            append!(target["surface_oil_m3_s"], Float64.(well["surface_oil_m3_s"]))
            append!(target["bhp_pa"], Float64.(well["bhp_pa"]))
            append!(target["operating_target"], String.(well["operating_target"]))
            for (component, values) in enumerate(well["surface_component_mass_kg_s"])
                append!(target["surface_component_mass_kg_s"][component], Float64.(values))
            end
        end

        for (name, steps) in payload["control_evidence"]
            append!(get!(evidence, name, Any[]), steps)
        end

        solver = payload["solver"]
        accepted += Int(solver["accepted_steps"])
        cut += Int(solver["cut_steps"])
        iterations += Int(solver["nonlinear_iterations"])

        states = payload["states"]
        for (position, time) in enumerate(Float64.(states["times_s"]))
            any(abs(time - r) <= TIME_MATCH_TOLERANCE_S for r in requested_times) || continue
            any(abs(time - t) <= TIME_MATCH_TOLERANCE_S for t in state_times) && invalid(
                "merge_extractions: the state at $(time) s appears in more than one chunk; a " *
                "boundary snapshot belongs to exactly one of them",
            )
            push!(state_times, time)
            for (name, rows) in state_fields
                push!(rows, states[name][position])
            end
        end

        substep_offset += length(chunk["dt_s"])
    end

    length(inventory) == length(surface_source) + 1 || invalid(
        "merge_extractions: $(length(inventory)) inventory rows for " *
        "$(length(surface_source)) sources; the inventory includes the state each step " *
        "started from and nothing else",
    )

    merged_states = Dict{String,Any}("times_s" => state_times)
    for (name, rows) in state_fields
        merged_states[name] = rows
    end

    return Dict{String,Any}(
        "schema_version" => head["schema_version"],
        "components" => head["components"],
        "reference_densities_kg_m3" => head["reference_densities_kg_m3"],
        "stored_extra_state_fields" => head["stored_extra_state_fields"],
        "chunk" => Dict{String,Any}(
            "horizon_start_s" => starts[1],
            "horizon_end_s" => ends[end],
            "edges_s" => edges,
            "start_s" => starts,
            "end_s" => ends,
            "dt_s" => dts,
            "interval_index" => intervals,
        ),
        "wells" => wells,
        "connections" => connections,
        "boundary" => boundary_rows,
        "inventory_m3_sc" => inventory,
        "reservoir_inventory_m3_sc" => reservoir_inventory,
        "net_surface_source_m3_sc" => surface_source,
        "net_connection_source_m3_sc" => connection_source,
        "net_boundary_source_m3_sc" => boundary_source,
        "states" => merged_states,
        "solver" => Dict{String,Any}(
            "accepted_steps" => accepted,
            "cut_steps" => cut,
            "nonlinear_iterations" => iterations,
        ),
        "control_evidence" => evidence,
        "control_infeasible_reason" => infeasible_controls(evidence),
    )
end

# --------------------------------------------------------------------------------------
# the chunk driver
# --------------------------------------------------------------------------------------

#: What the well group's own step counter is put back to before each chunk is simulated.
#: Jutul's progress recorder starts every `simulate!` call at step 1, so 0 is a value the
#: comparison below can never see from the recorder.
const CONTROL_STEP_INDEX_RESET = 0

"""
    neutralise_control_step_index(state, report) -> state

Clear the well group's own step counter on its way into an output state.

THIS IS NOT A TIDY-UP; without it a chunked or restarted run silently runs one mini-step of
every chunk on the PREVIOUS chunk's controls. `update_before_step_multimodel!` for a
`WellGroupModel` (`JutulDarcy 0.3.11 src/facility/controls.jl:8`) decides whether to apply
the control the forces ask for with

    is_new_step = cfg.step_index != recorder.recorder.step
    changed = (is_new_step && newctrl != oldctrl) || well_was_disabled

and those two numbers live in different frames the moment a run is chunked.
`cfg.step_index` is a field of `WellGroupConfiguration`, which IS carried across a native
restart: it is an output variable of the facility, `Jutul.update_values!` for it copies
`old.step_index = new.step_index` (`JutulDarcy 0.3.11 src/facility/types.jl:862`), and
`initial_setup!` applies exactly that through `reset_variables!`. The recorder is not
carried: `recorder_reset!` puts it back to `step = 1` at the top of EVERY `simulate!` call
(`Jutul 0.4.31 src/simulator/recorder.jl:60`).

So a chunk of one report step stores `step_index == 1`, and the next chunk's first mini-step
also sees `current_step == 1`: `is_new_step` is false and a control change landing exactly on
the chunk boundary is skipped. Measured on the six-month fixture, whose wells swap roles at
the start of month 4: that mini-step kept the previous month's control while
`valid_surface_rate_for_control` had already clamped the total surface rate to the NEW
control's sign, `update_primary_variable!` read the result as a rate "approaching zero" and
disabled both wells for the whole mini-step. The run published `CONTROL_INFEASIBLE` — the
honest verdict on a control that was not honoured, and a control that was not honoured only
because the run was chunked.

`config[:output_function]` is the supported place to do this: it runs on the state
`store_output!` is about to keep, before it is written or returned. Storing a counter the
next call cannot mistake for its own restores exactly the comparison a continuous run makes
at a report-step boundary — in a continuous run `is_new_step` is true there too, because the
recorder has moved on. No equation, variable or parameter is touched: this is bookkeeping
about which step the controls were last applied on, and nothing else.
"""
function neutralise_control_step_index(state, report)
    for (_, substate) in pairs(state)
        substate isa AbstractDict || continue
        haskey(substate, :WellGroupConfiguration) || continue
        substate[:WellGroupConfiguration].step_index = CONTROL_STEP_INDEX_RESET
    end
    return state
end

"""Live heap after a collection, which is what "retained between chunks" means."""
function retained_bytes()
    GC.gc()
    return isdefined(Base, :gc_live_bytes) ? Int(Base.gc_live_bytes()) : -1
end

"""
    run_forward_native(case, arrays, native_dir; kwarg...) -> Dict

Drive one forward as a sequence of calendar-month chunks and extract every one of them.

`resume_from_step` is how many report steps the native working directory ALREADY holds —
zero for a fresh job, and the checkpoint's completed step for a continuation that has staged
one. Those steps are not simulated again; they are extracted from the bytes the parent wrote,
which is what makes a resumed run one run rather than two half-runs stapled together.

`stop_requested` is given the time the run has reached and is consulted only at chunk
boundaries. It never touches a job input: a stop leaves the chunks that finished and reports
`INCOMPLETE_BUDGET` for the rest, with a checkpoint the next command continues from.
"""
function run_forward_native(
    case::AbstractDict,
    arrays::AbstractDict,
    native_dir::AbstractString;
    state_times_s = nothing,
    chunk_months::Int = 1,
    resume_from_step::Int = 0,
    stop_requested = _ -> false,
    solver...,
)
    schedule = compile_intervals(case)
    physical = build_ow(case, arrays)
    # Before anything is simulated: a `SimpleWell`'s connection pressure drop is an extra
    # STATE field, so without this it is neither extracted nor written into the restart.
    request_extra_outputs!(physical.model)
    forces = [build_forces(physical.model, group, case["boundary"]) for group in schedule.controls]
    dt = diff(schedule.edges_s)
    n_steps = length(dt)
    ranges = chunk_ranges(schedule.month_index, chunk_months)
    requested = state_times_s === nothing ? Float64.(case["report_edges_s"]) :
                Float64.(collect(state_times_s))
    initial_state = deepcopy(physical.state0)

    (0 <= resume_from_step <= n_steps) || invalid(
        "run_forward_native: a continuation was asked to resume from report step " *
        "$(resume_from_step) of a schedule with $(n_steps) of them",
    )
    resume_from_step == 0 || any(last(r) == resume_from_step for r in ranges) || invalid(
        "run_forward_native: report step $(resume_from_step) is not the end of a chunk of " *
        "$(chunk_months) month(s); a restart is taken at a month boundary",
    )

    mkpath(native_dir)
    jutul_case = Jutul.JutulCase(
        physical.model, dt, forces;
        state0 = physical.state0, parameters = physical.parameters,
    )
    simulator, config = setup_reservoir_simulator(
        jutul_case;
        output_path = native_dir,
        output_substates = true,
        # `retrieve_output!` would otherwise read `1:n` of the WHOLE schedule back into
        # memory at the end of every chunk (Jutul 0.4.31 src/simulator/io.jl:11). The chunk
        # in hand is read explicitly below instead.
        output_states = false,
        output_reports = false,
        # The controls have to be restated on the first mini-step of every chunk; see
        # `neutralise_control_step_index` for the two counters that otherwise agree by
        # accident across a native restart.
        output_function = neutralise_control_step_index,
        info_level = -1,
        solver...,
    )

    completed = resume_from_step
    stopped = nothing
    failure = nothing
    for range in ranges
        last(range) <= completed && continue
        if !isnothing(stopped) || !isnothing(failure)
            break
        end
        first(range) == completed + 1 || invalid(
            "run_forward_native: chunk $(first(range)):$(last(range)) does not continue from " *
            "the $(completed) report steps already on disk",
        )
        restart = completed == 0 ? false : completed + 1
        simulate!(
            simulator, dt[1:last(range)];
            forces = forces[1:last(range)], config = config, restart = restart,
        )
        present = native_step_files(native_dir)
        reached = isempty(present) ? 0 : maximum(i for (i, _) in present)
        if reached < last(range)
            failure = string(
                "the solver reached report step ", reached, " of ", last(range),
                " in the chunk starting at ", schedule.edges_s[first(range)], " s; the case did ",
                "not run to the end of its schedule and no partial integral is published for it",
            )
            break
        end
        completed = last(range)
        if completed < n_steps && stop_requested(schedule.edges_s[completed + 1])
            stopped = string(
                "a stop was requested after the chunk ending at ",
                schedule.edges_s[completed + 1], " s; ", n_steps - completed,
                " report steps of this case were not simulated and are not published as a ",
                "month in which nothing flowed",
            )
        end
    end

    completed >= 1 || return Dict{String,Any}(
        "schema_version" => EXTRACT_SCHEMA_VERSION,
        "status" => failure === nothing ? "INCOMPLETE_BUDGET" : "NUMERICAL_FAILURE",
        "reason" => failure === nothing ?
                    "no chunk of this case was completed, so there is nothing to publish" :
                    failure,
        "completed_report_step" => 0,
        "completed_time_s" => schedule.edges_s[1],
        "report_steps" => n_steps,
        "model" => describe_model(physical),
    )

    payloads = Any[]
    offsets = Int[]
    diagnostics = Any[]
    # The counters of the chunks this call did NOT simulate. They are re-extracted from the
    # parent's own `.jld2` files so that the published integrals cover the whole horizon,
    # and the merged totals below therefore include them — but nobody spent them here. The
    # ledger sums what was spent, so this is what the Python side subtracts.
    resumed_accepted, resumed_cut, resumed_iterations = 0, 0, 0
    for range in ranges
        last(range) <= completed || break
        edges = schedule.edges_s[first(range):(last(range) + 1)]
        window = [
            t for t in requested if
            (first(range) == 1 ? t >= edges[1] - TIME_MATCH_TOLERANCE_S :
             t > edges[1] + TIME_MATCH_TOLERANCE_S) && t <= edges[end] + TIME_MATCH_TOLERANCE_S
        ]
        isempty(window) && push!(window, edges[end])
        states = Vector{Jutul.JUTUL_OUTPUT_TYPE}()
        reports = Any[]
        Jutul.read_results(
            native_dir;
            read_states = true, read_reports = true,
            states = states, reports = reports,
            range = first(range):last(range), verbose = false,
        )
        chunk_state0 = first(range) == 1 ? initial_state :
                       Jutul.read_restart(native_dir, first(range) - 1; read_report = false)[1]
        payload = extract_interval(
            (result = Jutul.SimResult(states, reports, Dates.now()),),
            physical.model,
            physical.parameters,
            forces[range],
            chunk_state0;
            controls_by_interval = schedule.controls[range],
            edges_s = collect(Float64, edges),
            state_times_s = collect(Float64, window),
            chunk_start_s = Float64(edges[1]),
        )
        push!(payloads, payload)
        push!(offsets, first(range) - 1)
        if last(range) <= resume_from_step
            resumed_accepted += Int(payload["solver"]["accepted_steps"])
            resumed_cut += Int(payload["solver"]["cut_steps"])
            resumed_iterations += Int(payload["solver"]["nonlinear_iterations"])
        end
        # The chunk's full substates and reports go out of scope HERE, and the number below
        # is what a later test reads to show that nothing of them was retained.
        states = nothing
        reports = nothing
        chunk_state0 = nothing
        push!(
            diagnostics,
            Dict{String,Any}(
                "first_report_step" => first(range) - 1,
                "last_report_step" => last(range) - 1,
                "retained_bytes" => retained_bytes(),
                "maxrss_bytes" => Int(Sys.maxrss()),
                "accepted_steps" => payload["solver"]["accepted_steps"],
            ),
        )
    end

    merged = merge_extractions(payloads, offsets, requested)
    # What was actually constructed, for the diagnostic the worker record carries. It is a
    # description of the model, never a substitute for the outputs beside it.
    merged["model"] = describe_model(physical)
    merged["chunk_diagnostics"] = diagnostics
    # Always present, zero for a fresh job: a reader that had to guess whether a missing
    # field meant "nothing was resumed" or "this worker does not report it" would guess
    # wrong for exactly the job whose counters are double-counted without it.
    merged["resumed_solver"] = Dict{String,Any}(
        "accepted_steps" => resumed_accepted,
        "cut_steps" => resumed_cut,
        "nonlinear_iterations" => resumed_iterations,
    )
    merged["completed_report_step"] = completed
    merged["completed_time_s"] = schedule.edges_s[completed + 1]
    merged["report_steps"] = n_steps
    if failure !== nothing
        merged["status"] = "NUMERICAL_FAILURE"
        merged["reason"] = failure
    elseif stopped !== nothing
        merged["status"] = "INCOMPLETE_BUDGET"
        merged["reason"] = stopped
    else
        merged["status"] = "COMPLETE"
        merged["reason"] = nothing
    end
    return merged
end
