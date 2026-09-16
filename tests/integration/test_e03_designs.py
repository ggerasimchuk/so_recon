"""One REAL T3 truth forward through JutulDarcy (plan E03 Task 02 «Готовно»).

This is the native gate of the T3 design: balance, controls and finite bounded states on
the actual solver, not a toy. It is deliberately cheap (one forward of a 16x16x2 case)
and never part of the scientific corpus, whose budget runs through the E03 commands.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from so_recon.config.resources import P1_LOOP_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.geology.renderer import build_inverse_case, render_theta
from so_recon.inference.contracts import ThetaRecord
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.forward import (
    BASE_MAX_NONLINEAR_ITERATIONS,
    DEFAULT_MAX_TIMESTEP_DAYS,
    SolverConfig,
    simulate,
)
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.simulator.contracts import OutputRequest
from so_recon.synthetic.loop_designs import (
    T3_DESIGN_ID,
    make_t3_world,
    render_t3_coefficients,
    t3_design,
    t3_prior_context,
)
from so_recon.validation.physical_smc import PHYSICAL_STATE_MONTHS, _truth_checks

ROOT = Path(__file__).resolve().parents[2]

# The native gate runs against the repository julia project, exactly like the E02
# integration tests; its artifacts land under the gitignored artifacts/ tree.
REPO_PATHS = ProjectPaths.default(ROOT)


@pytest.mark.julia
def test_one_t3_truth_forward_passes_physical_checks() -> None:
    paths = REPO_PATHS
    paths.ensure_dirs()
    try:
        julia = find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"the T3 native gate requires Julia: {exc}")

    design = t3_design()
    rng = np.random.default_rng(20260916)
    coefficients = rng.standard_normal(design.n_geology)
    state = float(rng.standard_normal())
    arrays = render_t3_coefficients(
        coefficients, design, family=1, state_coordinate=state
    )
    context = t3_prior_context(design, arrays, np.random.default_rng(4))

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command,
            argv=(),
            cfg=None,
            paths=paths,
            parent_run_ids=parent_run_ids,
            now=datetime.now(UTC),
        )

    ctx = run_factory("e03-t3-truth-forward", ())
    physical_whitened = np.linalg.solve(context.chol, coefficients - context.mean)
    whitened = context.rotation.T @ physical_whitened
    theta = ThetaRecord(
        schema_id=context.density_schema.schema_id,
        s=1,
        v=tuple(float(x) for x in (*whitened[:8], 0.05, -0.05, 0.0)),
        z_perp=tuple(float(x) for x in (*whitened[8:], state)),
        basis_hash=context.density_schema.basis_hash,
    )
    rendered = render_theta(theta, context)
    case = build_inverse_case(rendered, context, paths, ctx)

    session = ctx.run_dir / "session"
    session.mkdir(parents=True, exist_ok=True)
    probe: Callable[[], ResourceSnapshot] = lambda: probe_resources(None, session)
    ledger = BudgetLedger.start(
        profile=P1_LOOP_PROFILE,
        path=ctx.run_dir / "ledger.json",
        session_id=ctx.run_id,
        probe=probe,
    )
    edges = case.report_edges_s
    request = OutputRequest(
        state_times_s=tuple(
            float(edges[min(month, len(edges) - 1)]) for month in PHYSICAL_STATE_MONTHS
        ),
        keep_native_restart=False,
    )
    with PersistentJuliaWorker(
        executable=julia,
        project=paths.julia,
        session_dir=session,
        profile=P1_LOOP_PROFILE,
        paths=paths,
        probe=probe,
    ) as worker:
        result = simulate(
            case,
            request,
            worker=worker,
            ctx=ctx,
            ledger=ledger,
            solver_config=SolverConfig(
                max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
                max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
            ),
        )

    assert result.status == "COMPLETE", result.reason
    checks = _truth_checks(result, paths)
    assert checks["status"] == "PASS", checks["checks"]


@pytest.mark.julia
def test_make_t3_world_truth_case_runs_one_forward() -> None:
    """The published world's own case (truth theta, all residual blocks) is solvable."""
    paths = REPO_PATHS
    paths.ensure_dirs()
    try:
        julia = find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"the T3 native gate requires Julia: {exc}")

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command,
            argv=(),
            cfg=None,
            paths=paths,
            parent_run_ids=parent_run_ids,
            now=datetime.now(UTC),
        )

    ctx = run_factory("e03-t3-world-forward", ())
    context, _pending, truth_ref = make_t3_world(T3_DESIGN_ID, 4242, paths, ctx)
    import json

    truth = json.loads(paths.resolve(truth_ref.path).read_text(encoding="utf-8"))
    theta = ThetaRecord.model_validate(truth["theta"])
    rendered = render_theta(theta, context)
    case = build_inverse_case(rendered, context, paths, ctx)

    session = ctx.run_dir / "session"
    session.mkdir(parents=True, exist_ok=True)
    probe: Callable[[], ResourceSnapshot] = lambda: probe_resources(None, session)
    ledger = BudgetLedger.start(
        profile=P1_LOOP_PROFILE,
        path=ctx.run_dir / "ledger.json",
        session_id=ctx.run_id,
        probe=probe,
    )
    edges = case.report_edges_s
    request = OutputRequest(
        state_times_s=tuple(
            float(edges[min(month, len(edges) - 1)]) for month in PHYSICAL_STATE_MONTHS
        ),
        keep_native_restart=False,
    )
    with PersistentJuliaWorker(
        executable=julia,
        project=paths.julia,
        session_dir=session,
        profile=P1_LOOP_PROFILE,
        paths=paths,
        probe=probe,
    ) as worker:
        result = simulate(
            case,
            request,
            worker=worker,
            ctx=ctx,
            ledger=ledger,
            solver_config=SolverConfig(
                max_timestep_days=DEFAULT_MAX_TIMESTEP_DAYS,
                max_nonlinear_iterations=BASE_MAX_NONLINEAR_ITERATIONS,
            ),
        )
    assert result.status == "COMPLETE", result.reason
    checks = _truth_checks(result, paths)
    assert checks["status"] == "PASS", checks["checks"]
