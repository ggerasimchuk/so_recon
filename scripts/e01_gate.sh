#!/usr/bin/env bash
# E01 gate: the legacy regression, then the registered suites, then the validators that read
# what those suites published, then the stage report built from the same artifacts.
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
# Three rules this script exists to hold, each of them a defect it used to have.
#
#   1. A SKIP IS NOT A PASS. `pytest -q` exits 0 with skips, so the validator step used to
#      greenlight while `test_a_published_balance_table_carries_both_statements` had skipped
#      itself for want of the artifact it was supposed to read. Plan 12.9 is normative:
#      «Stage gate требует присутствия всех mandatory checks и нуля skipped/unrun в его
#      собственной matrix; обычный pytest PASS с skips недостаточен.» Hence --fail-on-skip.
#
#   2. THE FROZEN INPUTS ARE CHECKED AFTER A FAILING STEP, NOT ONLY AFTER A PASSING ONE.
#      `check_frozen` used to run only when every earlier step had passed, so a tolerance
#      block or a job plan mutated BY a failing step — the one case worth reporting — was
#      never reported. `main` now RETURNS rather than exiting, and `report_outcome` runs
#      after it whatever that return was, including the early `|| return` of a failed
#      fail-fast step; the status of the whole run is then taken from `PIPESTATUS[0]`.
#
#      There is NO `trap` in this file, and the guarantee is bounded accordingly: anything
#      that terminates the shell without returning from `main` — a SIGINT or SIGTERM, or a
#      `set -u` abort — skips the frozen check, and such a run exits non-zero reporting
#      nothing rather than reporting a pass. A previous revision DID use `trap … EXIT` and
#      claimed "on every path out of this script"; on this host (bash 3.2.57) an EXIT trap
#      set inside a function that runs as the left side of a pipeline never fires at all, so
#      `check_frozen` ran on NO path and the gate returned `tee`'s status. That is why the
#      mechanism is plain control flow and why this paragraph states its edge rather than
#      claiming not to have one. `tests/unit/test_e01_gate.py` executes both halves: the
#      frozen check after a failed step 1, and the absence of the trap.
#
#   3. THE STAGE REPORT IS REBUILT FROM THE ARTIFACTS THIS RUN PRODUCED. Otherwise the
#      committed `reports/stages/E01.md` can drift from `artifacts/runs/` with nothing
#      noticing. It is rebuilt here and its own exit code is one of the gate's.
#
# The physics steps do NOT fail fast: a suite that failed still published a record, and the
# validators and the report have to run on it — that is how a failure gets described rather
# than merely detected. The environment and regression steps do fail fast, because nothing
# downstream of a broken environment means anything.
#
# `set -e` is deliberately ABSENT: steps 5 onwards must record a failure and carry on to the
# validators and the report, which is the paragraph above. `set -u` is the load-bearing one —
# an unset variable here is a bug, not a default. `pipefail` is belt-and-braces only: the one
# pipeline whose status matters is the `{ … } | tee` at the foot of this file, and that
# status is read from `PIPESTATUS[0]` explicitly rather than from the pipeline. Both flags
# are set here rather than in the Makefile because GNU Make 3.81 silently ignores
# .SHELLFLAGS.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

UV="${UV:-uv}"
CONFIG="${E01_CONFIG:-configs/e01.yml}"
# The oil-water matrix. The black-oil capability runs in a SESSION OF ITS OWN (plan 13.4) —
# `E01_SUITES="bo" scripts/e01_gate.sh`, or `so-recon verify-physics --suite bo` directly —
# because its early compilation belongs to that session's budget and not to this one's, and
# because its verdict is a separate gate that must not move the oil-water one.
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
  # Task 13: the black-oil capability's own tolerance block. It is a separate gate (plan
  # 13.5) with a separate frozen block, and a gate that moved its own thresholds is not a
  # gate whichever block they live in.
  configs/e01_blackoil_tolerances.yml
  configs/e01_jobs.json
  configs/smoke_expected.json
  reports/environment_report.md
  reports/manifests
)

FROZEN_BEFORE=""
FAILURES=()

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

# Run one step, remember whether it failed, and return its code so the caller can decide
# whether to go on.
#
# The status is captured from the command ITSELF and never from an `if` around it. An `if`
# whose condition fails and which has no `else` has exit status 0, so `local code=$?` after
# `fi` read 0 for every failing step: the log said "exited 0", the FAILURES entry said
# "exit 0", and every `step ... || return` below was inert. Reproduced on this host
# (bash 3.2.57) and pinned by
# `tests/unit/test_e01_gate.py::test_step_returns_the_command_s_real_exit_code`, which
# executes THIS function out of THIS file rather than grepping it.
step() {
  local label="$1"
  shift
  echo "-- ${label} --"
  "$@"
  local code=$?
  if [ "$code" -eq 0 ]; then
    return 0
  fi
  echo "e01 gate: step '${label}' exited ${code}" >&2
  FAILURES+=("${label} (exit ${code})")
  return "$code"
}

# The newest published run directory of each suite, in suite order. These are the ACTUAL
# directories the commands above wrote (plan 12.10: the report reads real run directories,
# never an invented run id).
latest_run_dirs() {
  "$UV" run python - <<'PY'
import json
import pathlib

latest: dict[str, str] = {}
runs = pathlib.Path("artifacts/runs")
for run_dir in sorted(runs.iterdir()) if runs.is_dir() else []:
    record = run_dir / "e01_suite.json"
    if not record.is_file():
        continue
    try:
        suite = json.loads(record.read_text(encoding="utf-8"))["suite"]
    except Exception:
        continue
    latest[str(suite)] = str(run_dir)
print(" ".join(latest[name] for name in sorted(latest)))
PY
}

report_outcome() {
  local steps_rc="$1"
  echo "-- frozen inputs (checked whatever happened above, including a failed step) --"
  if ! check_frozen; then
    FAILURES+=("frozen inputs changed during the gate")
  fi
  if [ "$steps_rc" -ne 0 ] || [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "== e01 gate FAIL $(date -u +%Y-%m-%dT%H:%M:%SZ) ==" >&2
    if [ "${#FAILURES[@]}" -gt 0 ]; then
      printf 'failed step: %s\n' "${FAILURES[@]}" >&2
    fi
    return 1
  fi
  echo "== e01 gate PASS $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  return 0
}

main() {
  echo "== e01 gate start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  FROZEN_BEFORE="$(frozen_digests)"

  # Fail fast: nothing below a broken environment or a broken regression means anything,
  # and launching the P0 and P1 physics suites on top of one is the expensive mistake this
  # ordering exists to avoid.
  step "1/8 locked python environment (no .venv deletion)" "$UV" sync --frozen || return
  step "2/8 regression: unit suite and the absolute-path guard" \
    "$UV" run pytest tests/unit tests/test_no_absolute_paths.py -q || return
  step "3/8 lint" "$UV" run ruff check . || return
  step "3/8 format" "$UV" run ruff format --check . || return
  step "3/8 types" "$UV" run mypy || return
  step "4/8 legacy end-to-end smoke (unchanged E00 path)" "$UV" run so-recon smoke || return
  step "4/8 julia smoke integration" \
    "$UV" run pytest tests/integration/test_julia_smoke.py -m julia -q || return

  # From here on, a failure is DESCRIBED rather than fled: a suite that failed still
  # published a record, and the validators and the stage report have to read it.
  echo "-- 5/8 registered E01 suites --"
  for suite in $SUITES; do
    step "verify-physics --suite ${suite}" \
      "$UV" run so-recon --config "$CONFIG" verify-physics --suite "$suite"
  done

  # The black-oil validators are added only when THIS invocation ran the black-oil suite.
  # They read a published `bo` session and `--fail-on-skip` turns a missing one into a
  # failure, so listing them unconditionally would make an oil-water gate fail for want of a
  # capability plan 13.5 keeps deliberately separate from it. `${a[@]+"${a[@]}"}` is the
  # expansion that is safe for an empty array under `set -u` on bash 3.2.
  local validators=(tests/integration/test_e01_stage.py)
  case " $SUITES " in
    *" bo "*) validators+=(tests/integration/test_e01_blackoil.py) ;;
  esac
  step "6/8 stage validators on the artifacts those suites saved (no skips allowed)" \
    "$UV" run pytest ${validators[@]+"${validators[@]}"} --run-e01-physics --fail-on-skip -q

  echo "-- 7/8 locating the run directories those suites published --"
  local report_runs
  report_runs="$(latest_run_dirs)"
  if [ -z "$report_runs" ]; then
    echo "e01 gate: no published E01 suite record under artifacts/runs" >&2
    FAILURES+=("no suite record to build the stage report from")
    return 1
  fi
  echo "citing: $report_runs"
  # Unquoted on purpose: one word per run directory.
  # shellcheck disable=SC2086
  step "8/8 stage report rebuilt from those artifacts" \
    "$UV" run so-recon --config "$CONFIG" e01-report --runs $report_runs
  return 0
}

# One subshell for the whole run, and no EXIT trap. `main` used to set `trap finish EXIT`
# and then run as the left side of a pipeline: on bash 3.2.57 — the shell this host has —
# an EXIT trap set inside a pipeline's subshell does not fire, so `check_frozen` never ran
# and the gate's status was `tee`'s. Here `main` RETURNS, `report_outcome` runs
# unconditionally after it whatever that return was, and the status of the group itself is
# taken from `PIPESTATUS` rather than from the pipe's last stage.
{
  main
  main_rc=$?
  report_outcome "$main_rc"
} 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
