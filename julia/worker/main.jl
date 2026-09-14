# E01.0 — the persistent Julia worker.
#
# One process, many jobs. Jutul, JutulDarcy, JSON and HDF5 are loaded once here and then
# reused, because paying that cost per forward would spend more time compiling than
# simulating. Every job is nevertheless isolated: its inputs come from an immutable
# descriptor, its state is built inside a function and released when that function
# returns, and it writes into a directory of its own. There is deliberately NO model
# global in this file — a model that outlived a job would make the next job's result
# depend on which jobs preceded it, which is exactly the reproducibility E01 is about.
#
# Protocol. One JSON object per line on stdin:
#
#     {"op":"ping",     "job_id":…, "root":…, "inputs":{name: relative path}}
#     {"op":"run",      "job_id":…, "root":…, "job_path":…, "stdout_path":…, "stderr_path":…,
#                       "native_dir":…, "environment_lock_hash":…, "static_hash":…,
#                       "schedule_prefix_hashes":[…], "stop_path":…}
#     {"op":"shutdown"}
#
# and one JSON object per line on stdout: a READY frame at start (pid, versions, protocol
# version), then one reply per request carrying job_id, status, result_path and
# result_sha256. STDOUT CARRIES PROTOCOL FRAMES ONLY. Native logging — anything Jutul,
# JutulDarcy or a linear solver prints — is redirected into the job's own stdout/stderr
# files for the duration of the job, and the reply is written after that redirect has been
# undone, so a solver's chatter can never be parsed as a response.
#
# Identity. A job runs on the bytes its descriptor names, or it does not run. Before
# anything else, this side re-hashes the case manifest, every array the case references
# and the solver configuration, and compares each digest with the one the descriptor
# declares; the case's own model_hash must match the descriptor's too. The Python side
# stamps what only it knows (the environment lock hash) onto the result.
#
# Once those checks pass, `run` hands the job to the JutulDarcy adapter, which builds the
# model, drives it one calendar month at a time and extracts every chunk, all inside one
# call whose locals die with it. The extraction travels in the published record; the
# reply carries only the status, the path and the digest, so a success is a success
# somebody can read back and re-hash rather than a claim on the wire.
#
# A COMPLETE here is a statement about the SIMULATION: the solver reached the end of its
# schedule and every chunk was extracted. Whether that becomes a COMPLETE forward RESULT is
# decided on the Python side, which holds the request the job was made from and is the only
# side that can say whether the axis delivered is the axis that was asked for.

using Jutul, JutulDarcy, JSON, HDF5, SHA
using LinearAlgebra: BLAS

# The adapter is loaded once, with the packages. It holds no model: `run_job` builds one per
# job and returns it, which is what keeps job B's result independent of job A.
include(joinpath(@__DIR__, "..", "adapter", "SOReconAdapter.jl"))
using .SOReconAdapter

const PROTOCOL_VERSION = "worker-2"
const JOB_SCHEMA_VERSION = "job-1"
# worker-result-2: the record gained the two fields the Python side now reads out of it,
# `extraction` (the whole native extraction a success is published from) and `restart` (the
# checkpoint the job left behind). Both are load-bearing, so the shape has a new name.
const RESULT_SCHEMA_VERSION = "worker-result-2"
const RESULT_FILENAME = "result.json"

#: Where a published result keeps its immutable native checkpoint. It is created inside the
#: staging directory and renamed into place with the record, so it is never half a snapshot.
const CHECKPOINT_DIRNAME = "checkpoint"

"""Write one protocol frame. Never called while stdout is redirected to a job log."""
function emit(payload::AbstractDict)
    JSON.print(stdout, payload)
    print(stdout, "\n")
    flush(stdout)
    return nothing
end

version_of(m::Module) = (v = pkgversion(m); v === nothing ? "unknown" : string(v))

sha256_of(path::AbstractString) = bytes2hex(open(sha256, path))

read_json(path::AbstractString) = JSON.parse(String(copy(read(path))))

"""
Re-hash one declared input and record both the digest and any mismatch.

Returns true when the bytes on disk are the bytes the descriptor named. A missing file and
a wrong digest are both refusals: neither is repaired, and neither is passed on to physics.
"""
function verify_input!(
    problems::Vector{String},
    checked::Dict{String,Any},
    name::AbstractString,
    path::AbstractString,
    declared,
)
    if !isfile(path)
        push!(problems, "$(name): declared input $(path) does not exist")
        checked[name] = nothing
        return false
    end
    actual = sha256_of(path)
    checked[name] = actual
    if !(declared isa AbstractString) || actual != declared
        push!(problems, "$(name): sha256 $(actual) does not match the declared $(repr(declared))")
        return false
    end
    return true
end

"""
Collect every ArrayRef reachable from the case manifest, keyed by where it was found.

An ArrayRef is recognised structurally — a mapping carrying `path`, `dataset` and
`sha256` — so a case that gains a new array in a later schema is covered here without this
file being taught its name.
"""
function array_refs(node, prefix::String = "case", out::Dict{String,Any} = Dict{String,Any}())
    if node isa AbstractDict
        if haskey(node, "path") && haskey(node, "dataset") && haskey(node, "sha256")
            out[prefix] = node
            return out
        end
        for (key, value) in node
            array_refs(value, string(prefix, ".", key), out)
        end
    elseif node isa AbstractVector
        for (index, value) in enumerate(node)
            array_refs(value, string(prefix, "[", index - 1, "]"), out)
        end
    end
    return out
end

"""
Publish a job's result into its own directory, atomically.

The record is built inside a staging directory and the whole directory is renamed into
place, so a reader never sees a half-written result and a crash leaves unfinished staging
rather than a result that looks finished. A destination that already exists is refused:
`result_dir` is unique per job (plan 3.2), and silently writing into someone else's
directory would attribute one job's output to another.
"""
function publish_result(
    root::AbstractString,
    result_dir::AbstractString,
    payload::AbstractDict;
    populate = nothing,
)
    destination = joinpath(root, result_dir)
    if ispath(destination)
        error("result directory $(result_dir) already exists; every job writes its own")
    end
    mkpath(dirname(destination))
    staging = string(destination, ".staging-", getpid(), "-", time_ns())
    mkpath(staging)
    try
        # Anything else the result owns — a native checkpoint, say — is built INSIDE the
        # staging directory, so the whole result appears at once or not at all. A process
        # killed here leaves staging nobody accepts, never a checkpoint half copied.
        populate === nothing || populate(staging)
        open(joinpath(staging, RESULT_FILENAME), "w") do io
            JSON.print(io, payload)
            print(io, "\n")
        end
        mv(staging, destination)
    catch
        rm(staging; force = true, recursive = true)
        rethrow()
    end
    return (
        joinpath(result_dir, RESULT_FILENAME),
        sha256_of(joinpath(destination, RESULT_FILENAME)),
    )
end

"""
Whether an operator's stop request applies at this chunk boundary.

The request is a FILE the control plane wrote and this side only ever reads: an operator
asking for a stop must not be able to change what a job simulates. A request with no
`after_completed_time_s` is the unconditional one — finish the chunk in hand and start
nothing else; one that names a time asks for a particular completed month ("finish March"),
and is not due until the run has reached it. A file that cannot be parsed is treated as the
unconditional request: a stop nobody can read is still a stop somebody asked for.
"""
function stop_is_due(path::AbstractString, completed_time_s::Float64)
    isfile(path) || return false
    request = try
        JSON.parse(String(copy(read(path))))
    catch
        return true
    end
    request isa AbstractDict || return true
    after = get(request, "after_completed_time_s", nothing)
    after === nothing && return true
    return completed_time_s >= Float64(after) - 1e-6
end

"""
Run one job. Everything it needs is a local of this function and dies with it.

The model, its parameters and its initial state are built by the adapter inside this same
scope, for the same reason: what the next job sees must not depend on what this one
allocated. Jobs A, B and A again get three independently built models from one process.
"""
function execute_job(request::AbstractDict)
    job_id = String(request["job_id"])
    root = String(request["root"])
    job = read_json(String(request["job_path"]))

    # This is native logging, and it lands in the job's own stdout file: `handle_run` has
    # already redirected stdout for the duration of this call. Nothing printed from here
    # can reach the protocol stream, which is the whole reason the redirect is scoped the
    # way it is — the reply is emitted by the caller, after the redirect is undone.
    println("so-recon worker: job ", job_id, " attempt ", get(job, "attempt", "?"),
            " protocol ", PROTOCOL_VERSION)

    problems = String[]
    checked = Dict{String,Any}()
    physics_class = "unknown"

    if get(job, "job_id", nothing) != job_id
        push!(
            problems,
            "descriptor job_id $(repr(get(job, "job_id", nothing))) is not the requested $(job_id)",
        )
    end
    if get(job, "schema_version", nothing) != JOB_SCHEMA_VERSION
        push!(
            problems,
            "descriptor schema_version $(repr(get(job, "schema_version", nothing))) is not " *
            "$(JOB_SCHEMA_VERSION)",
        )
    end

    case_path = joinpath(root, String(job["case_path"]))
    case_ok = verify_input!(problems, checked, "case", case_path, get(job, "case_sha256", nothing))
    verify_input!(
        problems,
        checked,
        "solver_config",
        joinpath(root, String(job["solver_config_path"])),
        get(job, "solver_config_sha256", nothing),
    )

    if case_ok
        case = read_json(case_path)
        if get(case, "model_hash", nothing) != get(job, "model_hash", nothing)
            push!(
                problems,
                "case model_hash $(repr(get(case, "model_hash", nothing))) is not the " *
                "descriptor's $(repr(get(job, "model_hash", nothing)))",
            )
        end
        fluids = get(case, "fluids", Dict{String,Any}())
        physics_class = String(get(fluids, "kind", "unknown"))
        for (name, ref) in array_refs(case)
            verify_input!(
                problems,
                checked,
                name,
                joinpath(root, String(ref["path"])),
                get(ref, "sha256", nothing),
            )
        end
    end

    # Bytes first, physics second. A job whose inputs do not hash to what the descriptor
    # declared never reaches the adapter at all; one whose inputs check out is built into a
    # model here, inside this scope, and the model is released when this function returns.
    status = "INVALID_INPUT"
    built = nothing
    extraction = nothing
    checkpoint = nothing
    native_dir = String(request["native_dir"])
    lock_hash = String(request["environment_lock_hash"])
    stop_path = get(request, "stop_path", nothing)
    stop_requested = stop_path === nothing ? (_ -> false) :
                     (completed_time_s -> stop_is_due(String(stop_path), completed_time_s))
    reason = if !isempty(problems)
        string("declared inputs do not match their bytes: ", join(problems, "; "))
    else
        try
            outcome = SOReconAdapter.run_job(
                job, root;
                native_dir = native_dir,
                environment_lock_hash = lock_hash,
                static_hash = String(request["static_hash"]),
                schedule_prefix_hashes = get(request, "schedule_prefix_hashes", Any[]),
                stop_requested = stop_requested,
            )
            status = outcome.status
            built = outcome.model
            extraction = outcome.payload
            checkpoint = outcome.checkpoint
            outcome.reason
        catch err
            # The adapter's own checks on the case say INVALID_INPUT — a geometry that does
            # not describe the grid it declares, a fluid this adapter does not build, an
            # array that is not the bytes it claims. A failure anywhere else means the input
            # was well formed and the physics still could not be assembled from it.
            if err isa SOReconAdapter.InvalidCaseInput
                status = "INVALID_INPUT"
                string("the case was refused by the model constructor: ",
                       sprint(showerror, err))
            else
                status = "PHYSICALLY_INVALID"
                string("the case could not be built into a JutulDarcy model: ",
                       sprint(showerror, err))
            end
        end
    end

    restart_record = checkpoint === nothing ? nothing : Dict{String,Any}(
        "manifest_path" =>
            joinpath(String(job["result_dir"]), CHECKPOINT_DIRNAME, RESTART_MANIFEST_FILENAME),
        "completed_report_step" => checkpoint.completed_report_step,
        "completed_time_s" => checkpoint.completed_time_s,
        "model_hash" => checkpoint.model_hash,
        "schedule_prefix_hash" => checkpoint.schedule_prefix_hash,
        "environment_lock_hash" => checkpoint.environment_lock_hash,
        "native_format" => SOReconAdapter.NATIVE_RESTART_FORMAT,
    )
    record = Dict(
        "schema_version" => RESULT_SCHEMA_VERSION,
        "job_id" => job_id,
        "attempt" => get(job, "attempt", nothing),
        "status" => status,
        "reason" => reason,
        "physics_class" => physics_class,
        "model" => built,
        # The whole native extraction. It travels in the RECORD rather than in the reply so
        # that the bytes the Python side publishes from are bytes it re-hashed first.
        "extraction" => extraction,
        "restart" => restart_record,
        "case_sha256" => get(job, "case_sha256", nothing),
        "model_hash" => get(job, "model_hash", nothing),
        "solver_config_sha256" => get(job, "solver_config_sha256", nothing),
        "checked_inputs" => checked,
        "worker" => Dict(
            "protocol" => PROTOCOL_VERSION,
            "pid" => getpid(),
            "julia" => string(VERSION),
            "Jutul" => version_of(Jutul),
            "JutulDarcy" => version_of(JutulDarcy),
        ),
    )

    reply = Dict{String,Any}(
        "job_id" => job_id,
        "status" => status,
        "reason" => reason,
        "physics_class" => physics_class,
        "result_path" => nothing,
        "result_sha256" => nothing,
    )
    # The solver's counters are deliberately NOT on the wire. They are physics, and physics
    # travels in the published record whose bytes the other side re-hashes: a number the
    # transport read off an unverified frame is a number nobody vouched for. They are in
    # `record["extraction"]["solver"]`, beside the `resumed_solver` counters that say how
    # much of that total this job did not spend.
    try
        result_path, result_sha = publish_result(
            root, String(job["result_dir"]), record;
            populate = checkpoint === nothing ? nothing : staging -> SOReconAdapter.write_checkpoint(
                native_dir,
                joinpath(staging, CHECKPOINT_DIRNAME);
                completed_report_step = checkpoint.completed_report_step,
                completed_time_s = checkpoint.completed_time_s,
                model_hash = checkpoint.model_hash,
                static_hash = checkpoint.static_hash,
                case_sha256 = checkpoint.case_sha256,
                schedule_prefix_hash = checkpoint.schedule_prefix_hash,
                environment_lock_hash = checkpoint.environment_lock_hash,
            ),
        )
        reply["result_path"] = result_path
        reply["result_sha256"] = result_sha
    catch err
        # A result nobody can read back is not a result. A refusal keeps its own status —
        # it is still the truth about the physics — but a success that could not be
        # published becomes a protocol failure rather than an unverifiable claim.
        reply["reason"] = string(reason, "; the result record could not be published: ",
                                 sprint(showerror, err))
        status == "COMPLETE" && (reply["status"] = "PROTOCOL_FAILURE")
    end
    return reply
end

"""Run a job with native logging sent to the job's own files, never to the protocol."""
function handle_run(request::AbstractDict)
    stdout_path = String(request["stdout_path"])
    stderr_path = String(request["stderr_path"])
    mkpath(dirname(stdout_path))
    mkpath(dirname(stderr_path))
    # The redirect is scoped to the job. The reply is emitted by the caller, after this
    # returns and stdout is the protocol stream again.
    return open(stdout_path, "a") do out_io
        open(stderr_path, "a") do err_io
            redirect_stdout(out_io) do
                redirect_stderr(err_io) do
                    execute_job(request)
                end
            end
        end
    end
end

function handle_ping(request::AbstractDict)
    root = String(request["root"])
    inputs = Dict{String,Any}()
    for (name, relative) in get(request, "inputs", Dict{String,Any}())
        path = joinpath(root, String(relative))
        inputs[String(name)] = isfile(path) ? sha256_of(path) : nothing
    end
    return Dict{String,Any}(
        "job_id" => String(request["job_id"]),
        "status" => "COMPLETE",
        "protocol" => PROTOCOL_VERSION,
        "pid" => getpid(),
        "inputs" => inputs,
    )
end

"""Handle one line. Returns :shutdown when the session is over, :continue otherwise."""
function handle_line(line::AbstractString)
    request = nothing
    try
        request = JSON.parse(line)
    catch err
        emit(Dict{String,Any}(
            "job_id" => nothing,
            "status" => "PROTOCOL_FAILURE",
            "reason" => string("request is not JSON: ", sprint(showerror, err)),
        ))
        return :continue
    end
    if !(request isa AbstractDict)
        emit(Dict{String,Any}(
            "job_id" => nothing,
            "status" => "PROTOCOL_FAILURE",
            "reason" => "a request must be a JSON object",
        ))
        return :continue
    end
    op = get(request, "op", nothing)
    op == "shutdown" && return :shutdown
    job_id = get(request, "job_id", nothing)
    try
        if op == "ping"
            emit(handle_ping(request))
        elseif op == "run"
            emit(handle_run(request))
        else
            emit(Dict{String,Any}(
                "job_id" => job_id,
                "status" => "PROTOCOL_FAILURE",
                "reason" => string("unknown op ", repr(op)),
            ))
        end
    catch err
        # A job that threw still owes an answer: a silent worker is indistinguishable
        # from a hung one, and the caller would have to wait out its whole timeout.
        emit(Dict{String,Any}(
            "job_id" => job_id,
            "status" => "PROTOCOL_FAILURE",
            "reason" => string("worker raised while handling ", repr(op), ": ",
                               sprint(showerror, err)),
        ))
    end
    return :continue
end

function serve()
    while !eof(stdin)
        line = readline(stdin)
        isempty(strip(line)) && continue
        try
            handle_line(line) === :shutdown && break
        finally
            # The job's locals are out of scope by now, so this is where releasing them
            # actually returns memory rather than merely marking it collectable.
            GC.gc()
        end
    end
    return nothing
end

function main()
    BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1")))
    emit(Dict{String,Any}(
        "event" => "ready",
        "protocol" => PROTOCOL_VERSION,
        "pid" => getpid(),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "versions" => Dict(
            "julia" => string(VERSION),
            "Jutul" => version_of(Jutul),
            "JutulDarcy" => version_of(JutulDarcy),
            "JSON" => version_of(JSON),
            "HDF5" => version_of(HDF5),
        ),
    ))
    try
        serve()
    finally
        flush(stdout)
    end
    return nothing
end

main()
