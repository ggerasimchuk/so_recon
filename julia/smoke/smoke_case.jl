# E00 smoke: tiny 1D oil-water JutulDarcy case. Not a verification test (that is E05);
# it proves the locked Julia environment runs end to end and produces deterministic
# scalar summaries. The single input is case.json, written by Python; the script returns
# the SHA-256 of the bytes it actually read, so the Python side can prove the round trip.
using JutulDarcy, Jutul, JSON, SHA

function parse_cli(args::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || error("unexpected argument $(key)")
        i + 1 <= length(args) || error("missing value for $(key)")
        opts[key[3:end]] = args[i + 1]
        i += 2
    end
    haskey(opts, "case") || error("--case <case.json> is required")
    haskey(opts, "out") || error("--out <file.json> is required")
    return opts
end

function read_case(path::AbstractString)
    raw = read(path)                       # exact bytes on disk
    input_sha = bytes2hex(sha256(raw))
    case = JSON.parse(String(copy(raw)))   # copy: String(::Vector{UInt8}) takes ownership
    return case, input_sha
end

# AbstractDict, not Dict: JSON.jl 1.x parses objects into JSON.Object{String,Any}, which is
# an AbstractDict but NOT a Dict, so a ::Dict signature never matches and dispatch fails at
# run time. AbstractDict accepts both that and JSON.jl 0.21's plain Dict.
function run_smoke(case::AbstractDict)
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)

    grid = case["grid"]
    rock = case["rock"]
    fluids = case["fluids"]
    initial = case["initial"]
    schedule = case["schedule"]
    controls = case["controls"]

    nx = Int(grid["nx"])
    nsteps = Int(schedule["n_steps"])

    g = CartesianMesh(
        (nx, 1, 1),
        (nx * Float64(grid["dx_m"]), Float64(grid["dy_m"]), Float64(grid["dz_m"])) .* meter,
    )
    domain = reservoir_domain(
        g,
        permeability = Float64(rock["permeability_darcy"]) * Darcy,
        porosity = Float64(rock["porosity"]),
    )
    injector = setup_vertical_well(domain, 1, 1, name = :Injector)
    producer = setup_vertical_well(domain, nx, 1, name = :Producer)

    rhoWS = Float64(fluids["water_density_kg_m3"])kg / meter^3
    rhoOS = Float64(fluids["oil_density_kg_m3"])kg / meter^3
    sys = ImmiscibleSystem((AqueousPhase(), LiquidPhase()), reference_densities = [rhoWS, rhoOS])
    model = setup_reservoir_model(domain, sys, wells = [injector, producer])
    parameters = setup_parameters(model)
    state0 = setup_reservoir_state(
        model,
        Pressure = Float64(initial["pressure_bar"])bar,
        Saturations = [Float64(initial["water_saturation"]), Float64(initial["oil_saturation"])],
    )

    dt = fill(Float64(schedule["dt_days"])day, nsteps)
    pv = pore_volume(model, parameters)
    inj_rate = Float64(controls["injected_pore_volume_fraction"]) * sum(pv) / sum(dt)
    i_ctrl = InjectorControl(TotalRateTarget(inj_rate), [1.0, 0.0], density = rhoWS)
    p_ctrl = ProducerControl(BottomHolePressureTarget(Float64(controls["producer_bhp_bar"])bar))
    forces = setup_reservoir_forces(model, control = Dict(:Injector => i_ctrl, :Producer => p_ctrl))

    wd, states, t = simulate_reservoir(
        state0, model, dt, parameters = parameters, forces = forces, info_level = -1
    )

    orat = wd[:Producer, :orat]      # surface oil rate, negative for production
    wrat_inj = wd[:Injector, :wrat]  # surface water rate, positive for injection
    cum_oil = -sum(orat .* dt)
    cum_winj = sum(wrat_inj .* dt)
    so_final = states[end][:Saturations][2, :]
    mean_so = sum(so_final .* pv) / sum(pv)

    return Dict(
        "status" => "ok",
        "case_schema_version" => string(case["schema_version"]),
        "julia_version" => string(VERSION),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "jutul_version" => string(pkgversion(Jutul)),
        "nx" => nx,
        "n_steps" => nsteps,
        "cumulative_oil_m3" => cum_oil,
        "cumulative_water_injected_m3" => cum_winj,
        "mean_so_final" => mean_so,
    )
end

function main(args::Vector{String})
    t0 = time()
    result = try
        opts = parse_cli(args)
        case, input_sha = read_case(opts["case"])
        r = run_smoke(case)
        r["input_sha256"] = input_sha
        r["wall_time_s"] = time() - t0
        out = opts["out"]
        mkpath(dirname(abspath(out)))
        open(out, "w") do io
            write(io, JSON.json(r))   # JSON.json exists in both JSON.jl 0.21 and 1.x
        end
        r
    catch err
        r = Dict(
            "status" => "error",
            "message" => sprint(showerror, err),
            "wall_time_s" => time() - t0,
        )
        # Best effort: still report through --out when it was parsed successfully.
        try
            opts = parse_cli(args)
            mkpath(dirname(abspath(opts["out"])))
            open(opts["out"], "w") do io
                write(io, JSON.json(r))
            end
        catch
            println(stderr, r["message"])
        end
        r
    end
    return result["status"] == "ok" ? 0 : 1
end

exit(main(ARGS))
