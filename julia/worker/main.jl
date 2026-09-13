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
#     {"op":"run",      "job_id":…, "root":…, "job_path":…, "stdout_path":…, "stderr_path":…}
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
# model, its parameters and its initial state inside one call and lets them go when that
# call returns. The adapter cannot answer COMPLETE either: integrating the requested time
# axis and publishing its outputs is a later stage, so a job is answered with the model it
# was able to build and an explicit refusal — a transport must not be able to manufacture a
# successful physics result out of a constructor.

using Jutul, JutulDarcy, JSON, HDF5, SHA
using LinearAlgebra: BLAS

# The adapter is loaded once, with the packages. It holds no model: `run_job` builds one per
# job and returns it, which is what keeps job B's result independent of job A.
include(joinpath(@__DIR__, "..", "adapter", "SOReconAdapter.jl"))
using .SOReconAdapter

const PROTOCOL_VERSION = "worker-1"
const JOB_SCHEMA_VERSION = "job-1"
const RESULT_SCHEMA_VERSION = "worker-result-1"
const RESULT_FILENAME = "result.json"

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
function publish_result(root::AbstractString, result_dir::AbstractString, payload::AbstractDict)
    destination = joinpath(root, result_dir)
    if ispath(destination)
        error("result directory $(result_dir) already exists; every job writes its own")
    end
    mkpath(dirname(destination))
    staging = string(destination, ".staging-", getpid(), "-", time_ns())
    mkpath(staging)
    open(joinpath(staging, RESULT_FILENAME), "w") do io
        JSON.print(io, payload)
        print(io, "\n")
    end
    mv(staging, destination)
    return (
        joinpath(result_dir, RESULT_FILENAME),
        sha256_of(joinpath(destination, RESULT_FILENAME)),
    )
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
    reason = if !isempty(problems)
        string("declared inputs do not match their bytes: ", join(problems, "; "))
    else
        try
            outcome = SOReconAdapter.run_job(job, root)
            status = outcome.status
            built = outcome.model
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

    record = Dict(
        "schema_version" => RESULT_SCHEMA_VERSION,
        "job_id" => job_id,
        "attempt" => get(job, "attempt", nothing),
        "status" => status,
        "reason" => reason,
        "physics_class" => physics_class,
        "model" => built,
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
        "cost" =>
            Dict("accepted_steps" => 0, "cut_steps" => 0, "nonlinear_iterations" => 0),
    )
    try
        result_path, result_sha = publish_result(root, String(job["result_dir"]), record)
        reply["result_path"] = result_path
        reply["result_sha256"] = result_sha
    catch err
        reply["reason"] = string(reason, "; the failure record could not be published: ",
                                 sprint(showerror, err))
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
