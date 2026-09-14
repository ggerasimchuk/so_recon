"""E01.0: an array written by Python survives a real Julia round trip with its semantics.

The transpose hazard is the whole point. HDF5 stores a `(n_times, n_cells)` dataspace;
Julia is column-major and its HDF5 bindings hand that back as an `(n_cells, n_times)`
array, so a side that merely checks the shape can be wrong about every value while looking
right. The fixture is asymmetric — `[[1,2,4],[8,16,32]]` — and Julia is made to send back
explicit `(time_index, cell_index, value)` triples, which this test compares one by one.

Julia also decodes the case JSON, so the shared manifest shape is checked by both sides.

There is no Julia production file yet — the persistent worker is Task 4 and the JutulDarcy
adapter Task 5 — so the probe script is written here and run from a temporary directory.
Nothing under `julia/` is added by this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunContext
from so_recon.simulator.case_io import CASE_MANIFEST_FILENAME, read_array, write_array, write_case
from so_recon.simulator.contracts import (
    CONTROL_RATE_UNIT,
    SECONDS_PER_DAY,
    TIME_CELL_AXES,
    ArrayRef,
)
from so_recon.simulator.julia_bridge import (
    JuliaNotFoundError,
    SubprocessJuliaLauncher,
    find_julia,
)
from tests.forward_case import ASYMMETRIC, build_case, write_case_arrays

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 600

# What Julia must send back: the value it read, stamped with the zero-based indices it read
# it at. Both terms are integer-valued, so the comparison is exact in float64.
EXPECTED = ASYMMETRIC + 1000.0 * np.arange(2, dtype=np.float64)[:, None] + np.arange(3)[None, :]

PROBE_JL = r"""
# E01.0 exchange probe, written by tests/integration/test_e01_exchange.py.
# Reads a Python-written (n_times, n_cells) array and a case manifest, then writes back
# both the stamped array and the explicit semantic (time, cell, value) triples.
using HDF5, JSON, SHA

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
    for required in ("in", "case", "h5out", "out")
        haskey(opts, required) || error("--$(required) is required")
    end
    return opts
end

function check_case(path::AbstractString)
    raw = read(path)                       # exact bytes on disk
    case = JSON.parse(String(copy(raw)))   # copy: String(::Vector{UInt8}) takes ownership
    for key in ("schema_version", "spec_version", "case_id", "model_hash",
                "grid", "controls", "units")
        haskey(case, key) || error("case manifest is missing $(key)")
    end
    case["schema_version"] == "case-1" ||
        error("unexpected case schema_version $(case["schema_version"])")
    case["spec_version"] == "4.0" || error("unexpected spec_version $(case["spec_version"])")
    length(case["grid"]["shape"]) == 3 || error("grid shape is not three-dimensional")
    # Control rates arrive as human day rates; this side divides by 86400 when it builds
    # the model (Task 6). Assert the convention rather than assume it: a case carrying
    # native m3_sc/s would otherwise run every rate control 86400 times off.
    haskey(case["units"], "control_rate") || error("case manifest is missing units.control_rate")
    case["units"]["control_rate"] == "m3_sc/day" ||
        error("unexpected control rate unit $(case["units"]["control_rate"])")
    for c in case["controls"]
        if c["target"] in ("liquid_rate", "water_rate")
            native = Float64(c["value"]) / 86400.0
            native > 0.0 || error("non-positive native rate for well $(c["well_id"])")
        end
    end
    return case, bytes2hex(sha256(raw))
end

function main(args::Vector{String})
    opts = parse_cli(args)
    try
        case, case_sha = check_case(opts["case"])

        native, axis_order, unit = h5open(opts["in"], "r") do f
            d = f["states"]
            (read(d), read_attribute(d, "axis_order"), read_attribute(d, "unit"))
        end
        axis_order == ["time", "cell"] || error("unexpected axis_order $(axis_order)")

        # The HDF5 dataspace is (n_times, n_cells); Julia is column-major, so it arrives
        # here with its dimensions reversed. Convert EXPLICITLY, by the recorded axis
        # order, never by assuming a layout.
        n_cells, n_times = size(native)
        semantic = permutedims(native, (2, 1))          # (time, cell)

        stamped = similar(semantic)
        triples = Array{Float64}(undef, 3, n_times * n_cells)
        row = 1
        for t in 1:n_times, c in 1:n_cells
            # Zero-based indices, as in Python and in the exchange (cell_id = i + nx*(j + ny*k)).
            ti, ci = t - 1, c - 1
            stamped[t, c] = semantic[t, c] + 1000.0 * ti + ci
            triples[1, row] = ti
            triples[2, row] = ci
            triples[3, row] = semantic[t, c]
            row += 1
        end

        h5open(opts["h5out"], "w") do f
            # Writing a Julia (n_cells, n_times) array produces the (n_times, n_cells)
            # dataspace the Python side declares, so the conversion is explicit here too.
            f["roundtrip"] = permutedims(stamped, (2, 1))
            attributes(f["roundtrip"])["axis_order"] = ["time", "cell"]
            attributes(f["roundtrip"])["unit"] = unit
            f["triples"] = triples
            attributes(f["triples"])["axis_order"] = ["triple", "field"]
            attributes(f["triples"])["unit"] = "1"
        end

        open(opts["out"], "w") do io
            JSON.print(io, Dict(
                "status" => "ok",
                "case_sha256" => case_sha,
                "case_id" => case["case_id"],
                "model_hash" => case["model_hash"],
                "n_times" => n_times,
                "n_cells" => n_cells,
                "axis_order" => axis_order,
                "unit" => unit,
                "control_rate_unit" => case["units"]["control_rate"],
                "native_rates_m3_sc_per_s" => [
                    Float64(c["value"]) / 86400.0 for c in case["controls"]
                    if c["target"] in ("liquid_rate", "water_rate")
                ],
                "julia_version" => string(VERSION),
            ))
        end
    catch err
        open(opts["out"], "w") do io
            JSON.print(io, Dict("status" => "error", "message" => sprint(showerror, err)))
        end
        exit(1)
    end
end

main(ARGS)
"""


@pytest.mark.julia
def test_python_arrays_survive_a_real_julia_round_trip(tmp_project: Path) -> None:
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        julia_exe = find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")
    launcher = SubprocessJuliaLauncher(
        julia_exe=julia_exe, project=ROOT / "julia", timeout_s=TIMEOUT_S
    )

    paths = ProjectPaths.default(tmp_project)
    states = write_array(
        paths.artifacts / "exchange" / "states.h5",
        "states",
        ASYMMETRIC,
        unit="1",
        axis_order=TIME_CELL_AXES,
        paths=paths,
    )
    case = build_case(write_case_arrays(paths))
    ctx = RunContext.start(command="exchange", argv=[], cfg=None, paths=paths)
    case_ref = write_case(case, paths, ctx)
    case_path = ctx.run_dir / CASE_MANIFEST_FILENAME

    script = tmp_project / "exchange_probe.jl"
    script.write_text(PROBE_JL, encoding="utf-8")
    h5out = paths.artifacts / "exchange" / "roundtrip.h5"
    summary_path = paths.artifacts / "exchange" / "summary.json"
    launcher.launch(
        script,
        [
            "--in",
            str(paths.root / states.path),
            "--case",
            str(case_path),
            "--h5out",
            str(h5out),
        ],
        summary_path,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok", summary
    # Julia read the manifest Python published, and read exactly those bytes.
    assert summary["case_sha256"] == case_ref.sha256
    assert summary["model_hash"] == case.model_hash
    assert summary["case_id"] == case.case_id
    assert summary["axis_order"] == ["time", "cell"]
    # Julia read the declared control-rate unit and the day rates behind it.
    assert summary["control_rate_unit"] == CONTROL_RATE_UNIT == "m3_sc/day"
    assert summary["native_rates_m3_sc_per_s"] == pytest.approx(
        [c.value / SECONDS_PER_DAY for c in case.controls if c.target.endswith("rate")]
    )
    assert summary["unit"] == "1"
    # The array kept its orientation across the language boundary, not merely its size.
    assert (summary["n_times"], summary["n_cells"]) == (2, 3)

    # The file Julia wrote is read back through the ordinary contract reader.
    roundtrip = read_array(
        ArrayRef(
            path=paths.relative(h5out),
            dataset="roundtrip",
            sha256=sha256_file(h5out),
            shape=(2, 3),
            dtype="float64",
            unit="1",
            axis_order=TIME_CELL_AXES,
        ),
        paths,
    )
    assert roundtrip.tolist() == EXPECTED.tolist()

    # Pair by pair, not shape by shape: the value Julia saw at (time t, cell c) must be the
    # value Python wrote at (t, c).
    with h5py.File(h5out, "r") as handle:
        triples = np.asarray(handle["triples"][()])
    assert triples.shape == (6, 3)
    seen = {(int(t), int(c)): float(v) for t, c, v in triples}
    assert seen == {(t, c): float(ASYMMETRIC[t, c]) for t in range(2) for c in range(3)}


@pytest.mark.julia
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"schema_version": "case-1"}, "missing"),
        (
            {
                "schema_version": "case-1",
                "spec_version": "4.0",
                "case_id": "case-x",
                "model_hash": "a" * 64,
                "grid": {"shape": [2, 2, 2]},
                "controls": [],
                # Native rates, contradicting the conversion this side performs. Python
                # refuses such a case at construction, so the only way to prove the Julia
                # check is live is to hand-build the manifest and feed it in.
                "units": {"control_rate": "m3_sc/s"},
            },
            "unexpected control rate unit",
        ),
    ],
    ids=["missing-keys", "native-control-rate"],
)
def test_julia_refuses_a_case_manifest_of_the_wrong_shape(
    tmp_project: Path, payload: dict[str, object], reason: str
) -> None:
    """The shared JSON shape is checked by the Julia decoder too, not only by Python."""
    if not (ROOT / "julia" / "Manifest.toml").is_file():
        pytest.skip("julia/Manifest.toml missing; run make setup-julia")
    try:
        julia_exe = find_julia()
    except JuliaNotFoundError:
        pytest.skip("julia executable not found")
    launcher = SubprocessJuliaLauncher(
        julia_exe=julia_exe, project=ROOT / "julia", timeout_s=TIMEOUT_S
    )

    paths = ProjectPaths.default(tmp_project)
    states = write_array(
        paths.artifacts / "exchange" / "states.h5",
        "states",
        ASYMMETRIC,
        unit="1",
        axis_order=TIME_CELL_AXES,
        paths=paths,
    )
    manifest = paths.artifacts / "exchange" / "handbuilt.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    script = tmp_project / "exchange_probe.jl"
    script.write_text(PROBE_JL, encoding="utf-8")
    with pytest.raises(RuntimeError, match=reason):
        launcher.launch(
            script,
            [
                "--in",
                str(paths.root / states.path),
                "--case",
                str(manifest),
                "--h5out",
                str(paths.artifacts / "exchange" / "roundtrip.h5"),
            ],
            paths.artifacts / "exchange" / "summary.json",
        )
