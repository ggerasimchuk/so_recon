#!/usr/bin/env bash
# E00 gate: clean Python environment, locked Julia environment, manifests, smoke, tests.
# pipefail lives here because GNU Make 3.81 silently ignores .SHELLFLAGS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

UV="${UV:-uv}"
JULIA_BIN="${JULIA:-julia}"

LOG_DIR="artifacts/gate"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gate-$(date -u +%Y%m%dT%H%M%SZ).log"

# Files that must be byte-identical before and after a gate run (invariant I5).
DETERMINISTIC_PATHS=(
  uv.lock
  julia/Manifest.toml
  julia/.julia-version
  configs/smoke_expected.json
  reports/manifests
  reports/environment_report.md
)

check_deterministic_outputs() {
  local dirty
  # Untracked files (??) are ignored: on a first run the artifacts do not exist yet.
  dirty="$(git status --porcelain -- "${DETERMINISTIC_PATHS[@]}" | grep -v '^??' || true)"
  if [ -n "$dirty" ]; then
    echo "gate FAIL: tracked deterministic artifacts changed during the gate run:" >&2
    echo "$dirty" >&2
    return 1
  fi
  echo "deterministic artifacts unchanged (no git pollution)"
}

main() {
  echo "== gate start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  echo "-- 1/8 clean python environment --"
  rm -rf .venv
  "$UV" sync --frozen

  echo "-- 2/8 julia environment from lock --"
  "$JULIA_BIN" --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

  echo "-- 3/8 source manifest --"
  "$UV" run so-recon manifest

  echo "-- 4/8 environment report --"
  "$UV" run so-recon env-report

  echo "-- 5/8 end-to-end smoke --"
  "$UV" run so-recon smoke

  echo "-- 6/8 tests --"
  "$UV" run pytest -q

  echo "-- 7/8 lint, format and types --"
  "$UV" run ruff check .
  "$UV" run ruff format --check .
  "$UV" run mypy

  echo "-- 8/8 determinism of committed artifacts --"
  check_deterministic_outputs

  echo "== gate PASS $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
}

main 2>&1 | tee "$LOG_FILE"
