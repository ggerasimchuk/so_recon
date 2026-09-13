UV ?= uv
JULIA ?= julia
export UV
export JULIA

.PHONY: setup setup-julia test lint format smoke manifest env-report gate gate-clean

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
