#!/usr/bin/env bash
# E01 gate: the legacy regression, then the registered suites, then the validators that read
# what those suites published.
#
# It deliberately does NOT call scripts/gate.sh. That script deletes .venv, re-instantiates
# the Julia depot and compares every lockfile with HEAD, which makes it a silent
# environment-recreation step in the middle of a stage that is adding a dependency
# (Matplotlib, plan 12.7). Plan 12.8 says the full foundation gate is run separately, in an
# agreed environment, AFTER the dependency change is committed.
#
# The deterministic E00 environment report is never rewritten here. Each E01 session records
# its own lock hashes and its own host in `e01_environment.json`, inside its run directory,
# which is untracked (plan 12.8).
#
# pipefail lives here because GNU Make 3.81 silently ignores .SHELLFLAGS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

UV="${UV:-uv}"
CONFIG="${E01_CONFIG:-configs/e01.yml}"
SUITES="${E01_SUITES:-p0 p1}"

LOG_DIR="artifacts/gate"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/e01-gate-$(date -u +%Y%m%dT%H%M%SZ).log"

# What must be byte-identical before and after this gate. The E01 tolerance block and the
# planned matrix are here because a gate that moved its own thresholds is not a gate; the
# legacy smoke expectations and the deterministic environment report are here because E01
# must not rebaseline E00 (plan 12.8).
FROZEN_PATHS=(
  configs/e01_tolerances.yml
  configs/e01_jobs.json
  configs/smoke_expected.json
  reports/environment_report.md
  reports/manifests
)

FROZEN_BEFORE=""

frozen_digests() {
  local path
  for path in "${FROZEN_PATHS[@]}"; do
    if [ -d "$path" ]; then
      find "$path" -type f -print0 | sort -z | xargs -0 shasum -a 256 2>/dev/null || true
    elif [ -f "$path" ]; then
      shasum -a 256 "$path"
    else
      echo "missing  $path"
    fi
  done
}

check_frozen() {
  local after
  after="$(frozen_digests)"
  if [ "$after" != "$FROZEN_BEFORE" ]; then
    echo "e01 gate FAIL: a frozen input changed during the gate run:" >&2
    diff <(printf '%s\n' "$FROZEN_BEFORE") <(printf '%s\n' "$after") >&2 || true
    return 1
  fi
  echo "frozen inputs unchanged (tolerances, job plan, smoke expectations, E00 env report)"
}

main() {
  echo "== e01 gate start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  FROZEN_BEFORE="$(frozen_digests)"
  echo "-- 1/7 locked python environment (no .venv deletion) --"
  "$UV" sync --frozen

  echo "-- 2/7 regression: unit suite and the absolute-path guard --"
  "$UV" run pytest tests/unit tests/test_no_absolute_paths.py -q

  echo "-- 3/7 lint, format and types --"
  "$UV" run ruff check .
  "$UV" run ruff format --check .
  "$UV" run mypy

  echo "-- 4/7 legacy end-to-end smoke (unchanged E00 path) --"
  "$UV" run so-recon smoke
  "$UV" run pytest tests/integration/test_julia_smoke.py -m julia -q

  echo "-- 5/7 registered E01 suites --"
  for suite in $SUITES; do
    echo "---- verify-physics --suite ${suite} ----"
    "$UV" run so-recon --config "$CONFIG" verify-physics --suite "$suite"
  done

  echo "-- 6/7 stage validators, on the artifacts those suites saved --"
  "$UV" run pytest tests/integration/test_e01_stage.py --run-e01-physics -q

  echo "-- 7/7 frozen inputs --"
  check_frozen

  echo "== e01 gate PASS $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
}

main 2>&1 | tee "$LOG_FILE"
