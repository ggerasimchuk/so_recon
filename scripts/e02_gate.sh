#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

UV="${UV:-uv}"
CONFIG="${E02_CONFIG:-configs/e02.yml}"
SUITE=""
RESUME=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --suite)
      SUITE="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown e02 gate argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$SUITE" in
  math)
    "$UV" run pytest tests/unit tests/test_no_absolute_paths.py -q
    "$UV" run ruff check .
    "$UV" run ruff format --check .
    "$UV" run mypy
    "$UV" run so-recon --config "$CONFIG" verify-inverse --suite math
    ;;
  reduced)
    : "${E02_E01_REPORT:?set E02_E01_REPORT to fresh accepted E01.json}"
    "$UV" run pytest tests/integration/test_e02_reduced_inverse.py -m julia -q
    "$UV" run so-recon --config "$CONFIG" verify-inverse --suite reduced
    ;;
  p1)
    : "${E02_E01_REPORT:?set E02_E01_REPORT to fresh accepted E01.json}"
    if [ -n "$RESUME" ]; then
      "$UV" run so-recon --config "$CONFIG" inverse-resume --checkpoint "$RESUME"
    else
      echo "p1 gate requires --resume <actual checkpoint> or an explicit inverse-p1 command" >&2
      exit 2
    fi
    ;;
  *)
    echo "usage: scripts/e02_gate.sh --suite math|reduced|p1 [--resume PATH]" >&2
    exit 2
    ;;
esac

echo "E02 ${SUITE} gate PASS"
