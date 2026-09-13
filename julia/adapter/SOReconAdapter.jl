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

export build_ow, load_arrays, run_job

end # module
