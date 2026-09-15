"""Native BO construction must bind declared provenance to the tables it uses."""

import json
from pathlib import Path

from so_recon.simulator.julia_bridge import SubprocessJuliaLauncher, find_julia

ROOT = Path(__file__).resolve().parents[2]


def test_native_blackoil_rejects_forged_pvt_provenance(tmp_path: Path) -> None:
    script = tmp_path / "provenance.jl"
    script.write_text(
        """
include(ARGS[1])
case, arrays = blackoil_case(:bo_closed)
@testset "BO provenance" begin
    physical = ADAPTER.build_physical(case, arrays)
    @test physical !== nothing
    native = ADAPTER.evaluated_state(physical.model, physical.parameters, physical.state0)
    @test ADAPTER.assert_blackoil_phase_properties(native[:Reservoir]) === nothing
    for field in (:PhaseMassDensities, :PhaseViscosities), value in (0.0, -1.0, NaN, Inf)
        other = field == :PhaseMassDensities ? :PhaseViscosities : :PhaseMassDensities
        broken = Dict(field => fill(value, 3, 1), other => ones(3, 1))
        @test_throws ErrorException ADAPTER.assert_blackoil_phase_properties(broken)
    end
    for field in ("pvtw", "pvto", "pvdg", "relperm", "tables", "jutuldarcy_version")
        bad = deepcopy(case)
        bad["fluids"]["pvt_table_hashes"][field] = repeat("0", 64)
        @test_throws ADAPTER.InvalidCaseInput ADAPTER.build_physical(bad, arrays)
    end
    bad = deepcopy(case)
    bad["fluids"]["pvt_source"] = "unsupported-table-provider"
    @test_throws ADAPTER.InvalidCaseInput ADAPTER.build_physical(bad, arrays)
    bad = deepcopy(case)
    bad["fluids"]["corey_exponents"][1] = 3.0
    @test_throws ADAPTER.InvalidCaseInput ADAPTER.build_physical(bad, arrays)
    bad = deepcopy(case)
    delete!(bad["fluids"]["pvt_table_hashes"], "pvto")
    @test_throws ADAPTER.InvalidCaseInput ADAPTER.build_physical(bad, arrays)
end
open(ARGS[end], "w") do io
    JSON.print(io, Dict("status" => "ok"))
end
""",
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    SubprocessJuliaLauncher(find_julia(), ROOT / "julia", 180.0).launch(
        script,
        [str(ROOT / "julia/verification/blackoil.jl")],
        result,
    )
    assert json.loads(result.read_text())["status"] == "ok"
