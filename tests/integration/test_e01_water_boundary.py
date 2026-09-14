"""The native water boundary refuses oil before it can become a mass source."""

from __future__ import annotations

import json
from pathlib import Path

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.simulator.julia_bridge import SubprocessJuliaLauncher, find_julia

ROOT = Path(__file__).resolve().parents[2]


def test_native_pressure_water_boundary_rejects_oil_and_preserves_water_flux(
    tmp_path: Path,
) -> None:
    # Constructor/kernel probe only: no forward simulation or budgeted solver attempt.
    script = tmp_path / "water_boundary.jl"
    script.write_text(
        """
using Test
include(ARGS[1])
include(ARGS[2])
case, arrays = verification_case(:closed_cell)
physical = SOReconAdapter.build_ow(case, arrays)
boundary = Dict{String,Any}(
    "kind" => "pressure_water", "cells" => [0], "pressure_pa" => 2e7,
    "trans_flow" => 1e-12, "fractional_flow" => [1.0, 0.0],
)
for fractions in ([0.0, 1.0], [0.5, 0.5], [1.0 - 1e-12, 1e-12])
    bad = merge(boundary, Dict("fractional_flow" => fractions))
    err = try
        SOReconAdapter.boundary_conditions(physical.model, bad)
        nothing
    catch e
        e
    end
    @test err isa SOReconAdapter.InvalidCaseInput
    @test occursin("pure water", sprint(showerror, err))
end
bc = SOReconAdapter.boundary_conditions(physical.model, boundary)
state = SOReconAdapter.evaluated_state(physical.model, physical.parameters, physical.state0)
flux = SOReconAdapter.boundary_component_flux(
    physical.model, physical.model.models[:Reservoir].system, state[:Reservoir], bc,
)
@test flux[1, 1] < 0.0
@test flux[2, 1] == 0.0
@test bc[1].density == 1000.0
open(ARGS[end], "w") do io
    JSON.print(io, Dict("status" => "ok", "water_mass_kg_s" => flux[1, 1]))
end
""",
        encoding="utf-8",
    )
    result = tmp_path / "boundary.json"
    SubprocessJuliaLauncher(find_julia(), ROOT / "julia", P0_VERIFY_PROFILE.job_timeout_s).launch(
        script,
        [
            str(ROOT / "julia/adapter/SOReconAdapter.jl"),
            str(ROOT / "julia/verification/fixtures.jl"),
        ],
        result,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["water_mass_kg_s"] < 0.0
