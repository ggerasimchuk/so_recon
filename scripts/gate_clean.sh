#!/usr/bin/env bash
# Same gate, but with a throwaway Julia depot: proves julia/Manifest.toml instantiates
# from scratch. Slow (full JutulDarcy download + precompile), so it is a separate target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

TMP_DEPOT="$(mktemp -d "${TMPDIR:-/tmp}/so-recon-depot.XXXXXX")"
cleanup() { rm -rf "$TMP_DEPOT"; }
trap cleanup EXIT

export JULIA_DEPOT_PATH="$TMP_DEPOT"
echo "using throwaway JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

# No exec: the EXIT trap must still run to remove the temporary depot.
"$ROOT/scripts/gate.sh"
