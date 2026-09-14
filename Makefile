UV ?= uv
JULIA ?= julia
export UV
export JULIA

.PHONY: setup setup-julia test lint format smoke manifest env-report gate gate-clean \
	e01-gate e01-p0 e01-p1 e01-report

setup:
	$(UV) sync --frozen

setup-julia:
	$(JULIA) --project=julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

format:
	$(UV) run ruff format .

manifest:
	$(UV) run so-recon manifest

env-report:
	$(UV) run so-recon env-report

smoke:
	$(UV) run so-recon smoke

# Full E00 gate. Logic lives in the script: GNU Make 3.81 ignores .SHELLFLAGS.
gate:
	./scripts/gate.sh

# Gate with a throwaway Julia depot (clean install of the locked Julia environment).
gate-clean:
	./scripts/gate_clean.sh

# ---- E01 (plan Task 12) ----------------------------------------------------------------
# The E01 gate never calls scripts/gate.sh: that one deletes .venv and compares every
# lockfile with HEAD, which would make it a silent environment-recreation step inside a
# stage that adds a dependency. The full foundation gate is run separately, afterwards.
e01-gate:
	./scripts/e01_gate.sh

# One registered suite at a time, with the budgets the plan fixed: 10 minutes for P0 and
# one hour for P1. Each prints the run directory and the ledger path the next command needs.
e01-p0:
	$(UV) run so-recon --config configs/e01.yml verify-physics --suite p0

e01-p1:
	$(UV) run so-recon --config configs/e01.yml verify-physics --suite p1

# RUNS is the ACTUAL run directories the commands above printed, never an invented run id.
e01-report:
	$(UV) run so-recon --config configs/e01.yml e01-report --runs $(RUNS)
