# E01.0 — the SO-RECON JutulDarcy adapter.
#
# This file is a seam and nothing else: imports, includes and exports. There is no logic
# here and, deliberately, no module-level model, parameter set or state. A singleton model
# living in the adapter would make the result of job B depend on whether job A ran first,
# which is precisely the reproducibility E01 exists to establish; `model.jl` therefore
# builds everything inside `build_ow` and hands it back, so a job's model dies with the
# call that made it while the expensive, immutable part — the loaded and specialised
# packages — stays warm in the worker process.
#
# The numerical engine is JutulDarcy. Nothing in this tree implements a solver, a flux or a
# relative permeability of its own (SPEC 8.1).

module SOReconAdapter

using Jutul
using JutulDarcy
using HDF5
using JSON
using SHA

include("model.jl")
# Forces after the model: `controls.jl` refuses a case through `invalid`, which `model.jl`
# defines together with the exception the worker maps to INVALID_INPUT.
include("controls.jl")
# Outputs last: reading a finished result needs both the model it was built from and the
# forces it was driven with, and refuses through the same `invalid`.
include("outputs.jl")

export build_ow, load_arrays, run_job, InvalidCaseInput
export build_forces, native_control, control_limits, perforation_mask
export control_evidence, infeasible_controls
export compile_intervals, extract_interval, run_forward
export request_extra_outputs!, evaluated_state

end # module
