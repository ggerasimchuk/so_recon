"""Observation extraction and physical forward failure semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from so_recon.geology.conditional import LOG_K_QUANTITY, p1_prior_context
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.geology.renderer import build_inverse_case, render_theta
from so_recon.inference.contracts import (
    HistoryRow,
    LogRow,
    ObservationBundle,
    ObservationSupportMismatch,
    ThetaRecord,
)
from so_recon.inference.reduced_density import ReducedGaussianPrior
from so_recon.inference.target import (
    ForwardEvaluationError,
    PhysicalTarget,
    ReducedPhysicalTarget,
    _grids,
    _output_request,
    require_complete_forward,
)
from so_recon.observation.predict import extract_monthly_watercut, predict_observations
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.case_io import write_arrays
from so_recon.simulator.contracts import TIME_CELL_AXES, CostRecord, ForwardResult
from so_recon.simulator.results import MONTHLY_SCHEMA
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.p1 import P1Design, render_p1
from so_recon.synthetic.reduced_inverse import ReducedDesign


class _CaseContext:
    def __init__(self, root: Path) -> None:
        self.run_id = "case-test"
        self.run_dir = root / "artifacts" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.outputs: dict[str, ArtifactRef] = {}

    def add_output(self, key: str, ref: ArtifactRef) -> None:
        self.outputs[key] = ref


class _TargetContext(_CaseContext):
    def __init__(self, root: Path, index: int) -> None:
        self.run_id = f"target-test-{index}"
        self.run_dir = root / "artifacts" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.outputs = {}
        self.record = SimpleNamespace(status="RUNNING")

    def finish(
        self,
        status: str,
        outputs: dict[str, ArtifactRef] | None = None,
        notes: tuple[str, ...] | list[str] = (),
    ) -> None:
        del outputs, notes
        self.record.status = status


def monthly_row(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "well_id": "P1",
        "month_index": 0,
        "start_s": 0.0,
        "end_s": 1.0,
        "oil_prod_m3_sc": 1.0,
        "water_prod_m3_sc": 1.0,
        "water_inj_m3_sc": 1000.0,
        "liquid_prod_m3_sc": 2.0,
        "flowing_s": 1.0,
        "fw": 0.5,
        "fw_valid": True,
    }
    base.update(changes)
    return base


def test_monthly_watercut_uses_produced_water_not_injection() -> None:
    values = extract_monthly_watercut(pa.Table.from_pylist([monthly_row()], schema=MONTHLY_SCHEMA))
    assert values[("P1", 0)] == 0.5


def test_monthly_watercut_parity_with_the_published_value_is_checked() -> None:
    table = pa.Table.from_pylist([monthly_row(fw=0.9)], schema=MONTHLY_SCHEMA)
    with pytest.raises(ObservationSupportMismatch, match="parity"):
        extract_monthly_watercut(table)


def test_a_dry_month_stays_missing_instead_of_becoming_zero() -> None:
    table = pa.Table.from_pylist(
        [
            monthly_row(
                oil_prod_m3_sc=0.0,
                water_prod_m3_sc=0.0,
                liquid_prod_m3_sc=0.0,
                fw=None,
                fw_valid=False,
            )
        ],
        schema=MONTHLY_SCHEMA,
    )
    assert extract_monthly_watercut(table)[("P1", 0)] is None


def _bundle() -> ObservationBundle:
    return ObservationBundle(
        history=(
            HistoryRow(
                well_id="P1",
                month_index=0,
                raw_value=0.5,
                bin_index=1,
                quality_group="metered",
                observed_valid=True,
                reset=False,
            ),
        ),
        logs=(
            LogRow(
                observation_id="L1",
                support_cell_ids=(0, 1),
                support_weights=(0.25, 0.75),
                bin_index=1,
                date_times_s=(0.0, 2.0),
                date_weights=(0.5, 0.5),
                date_group="D1",
                bias_group="logs",
                valid=True,
                use="likelihood",
            ),
        ),
        bin_edges_by_group={
            "metered": (0.0, 0.25, 0.75, 1.0),
            "logs": (0.0, 0.25, 0.75, 1.0),
        },
        cutoff_s=2.0,
        information_hash="c" * 64,
        observation_hash="d" * 64,
    )


def _result(root: Path, *, times: tuple[float, ...] = (0.0, 2.0)) -> ForwardResult:
    paths = ProjectPaths.default(root)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    monthly = root / "artifacts" / "monthly.parquet"
    pq.write_table(pa.Table.from_pylist([monthly_row()], schema=MONTHLY_SCHEMA), monthly)
    refs = write_arrays(
        root / "artifacts" / "states.h5",
        {
            "so": (
                np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float64)[: len(times)],
                "1",
                TIME_CELL_AXES,
            )
        },
        paths=paths,
    )
    return ForwardResult(
        job_id="job-1",
        case_sha256="a" * 64,
        model_hash="b" * 64,
        physics_class="OW",
        status="COMPLETE",
        reason=None,
        completed_time_s=times[-1],
        times_s=times,
        states={"so": refs["so"]},
        monthly_path="artifacts/monthly.parquet",
        connections_path="artifacts/connections.parquet",
        balances_path="artifacts/balances.parquet",
        solver_metadata={},
        cost=CostRecord(
            wall_s=1.0,
            cpu_s=1.0,
            peak_rss_bytes=1,
            output_bytes=1,
            accepted_steps=1,
            cut_steps=0,
            nonlinear_iterations=1,
            retry_count=0,
            measurement_method="test",
        ),
        parent_attempt_ids=(),
    )


def test_prediction_extracts_exact_saved_support_states(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    result = predict_observations(_result(root), _bundle(), ProjectPaths.default(root))
    assert result.fw == {("P1", 0): 0.5}
    assert result.so_support[("L1", 0.0)] == pytest.approx(0.5)
    assert result.so_support[("L1", 2.0)] == pytest.approx(0.7)


def test_physical_target_can_request_registered_report_states_beside_log_dates() -> None:
    request = _output_request(_bundle(), (0.0, 1.0, 2.0))
    assert request.state_times_s == (0.0, 1.0, 2.0)


def test_prediction_refuses_a_log_date_the_forward_did_not_save(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    result = _result(root, times=(0.0,))
    with pytest.raises(ObservationSupportMismatch, match="requested state time"):
        predict_observations(result, _bundle(), ProjectPaths.default(root))


def test_nonuniform_report_centres_are_reconstructed_from_their_edges() -> None:
    bundle = _bundle().model_copy(
        update={
            "bin_edges_by_group": {
                "metered": (0.0, 0.1, 0.6, 1.0),
                "logs": (0.0, 0.1, 0.6, 1.0),
            }
        }
    )
    np.testing.assert_allclose(_grids(bundle)["metered"].centers, [0.0, 0.2, 1.0])


def test_edges_that_do_not_encode_zero_to_one_report_centres_are_refused() -> None:
    bundle = _bundle().model_copy(
        update={
            "bin_edges_by_group": {
                "metered": (0.0, 0.2, 0.6, 1.0),
                "logs": (0.0, 0.25, 0.75, 1.0),
            }
        }
    )
    with pytest.raises(ValueError, match="final centre"):
        _grids(bundle)


def _prior_context() -> object:
    design = P1Design()
    observations = {
        str(item["observation_id"]): float(item["value"])
        for item in render_p1(41, design).static_observations.to_pylist()
        if item["quantity"] == LOG_K_QUANTITY
    }
    return p1_prior_context(design, observations)


def _theta(context: object, *, log_bias: float = 0.0, geology: float = 0.0) -> ThetaRecord:
    from so_recon.inference.contracts import PriorContext

    assert isinstance(context, PriorContext)
    schema = context.density_schema
    v = [0.0] * schema.n_v
    v[0] = geology
    v[10] = log_bias
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=0,
        v=tuple(v),
        z_perp=(0.0,) * schema.n_residual,
        basis_hash=schema.basis_hash,
    )


def test_inverse_case_uses_rendered_arrays_without_a_truth_reference(tmp_path: Path) -> None:
    from so_recon.inference.contracts import PriorContext

    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    context = _prior_context()
    assert isinstance(context, PriorContext)
    ctx = _CaseContext(root)
    case = build_inverse_case(
        render_theta(_theta(context), context), context, paths, cast(RunContext, ctx)
    )
    referenced_paths = [
        case.grid.cell_centers_m.path,
        case.grid.cell_volume_m3.path,
        case.grid.neighbors.path,
        case.rock.porosity.path,
        case.rock.permeability_m2.path,
        case.initial.pressure_pa.path if case.initial.pressure_pa else "",
        case.initial.sw.path if case.initial.sw else "",
        case.observations.table_path or "",
    ]
    assert all("/truth/" not in path for path in referenced_paths)
    assert set(case.source_hashes) == {
        "conditional_information",
        "renderer",
        "physical_arrays",
    }
    assert case.model_hash != "e" * 64
    assert all("inverse_cases" in ref.path for ref in ctx.outputs.values())


def test_noise_only_change_reuses_the_same_physical_case(tmp_path: Path) -> None:
    from so_recon.inference.contracts import PriorContext

    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    context = _prior_context()
    assert isinstance(context, PriorContext)
    ctx = _CaseContext(root)
    first = build_inverse_case(
        render_theta(_theta(context, log_bias=0.0), context),
        context,
        paths,
        cast(RunContext, ctx),
    )
    second = build_inverse_case(
        render_theta(_theta(context, log_bias=2.0), context),
        context,
        paths,
        cast(RunContext, ctx),
    )
    changed_rock = build_inverse_case(
        render_theta(_theta(context, geology=1.0), context),
        context,
        paths,
        cast(RunContext, ctx),
    )
    assert first.model_hash == second.model_hash
    assert changed_rock.model_hash != first.model_hash


def test_physical_target_reuses_f_but_recomputes_l_for_noise_only_change(tmp_path: Path) -> None:
    from so_recon.inference.contracts import PriorContext

    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    context = _prior_context()
    assert isinstance(context, PriorContext)
    prior = GaussianConditionalPrior(context)
    worker = SimpleNamespace(paths=paths, environment_lock_hash="a" * 64)
    contexts: list[_TargetContext] = []
    forwards: list[ForwardResult] = []

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        assert command == "inverse-evaluation"
        assert parent_run_ids == ()
        ctx = _TargetContext(root, len(contexts))
        contexts.append(ctx)
        return cast(RunContext, ctx)

    def fake_simulate(
        case: object,
        request: object,
        **kwargs: Any,
    ) -> ForwardResult:
        del case, request, kwargs
        result = _result(root)
        forwards.append(result)
        return result

    target = PhysicalTarget(
        prior,
        prior,
        context,
        _bundle(),
        cast(PersistentJuliaWorker, worker),
        cast(BudgetLedger, object()),
        run_factory,
        simulate_fn=fake_simulate,
        load_forward_fn=lambda path, project_paths: forwards[0],
        adapter_hash="b" * 64,
    )
    first = target.evaluate(_theta(context, log_bias=0.0))
    second = target.evaluate(_theta(context, log_bias=2.0))

    # Keep the externally supplied observation_hash unchanged on purpose.  The cache must
    # still cover the complete semantic bundle, including use/mask fields.
    heldout = _bundle().model_copy(
        update={"logs": (_bundle().logs[0].model_copy(update={"use": "heldout"}),)}
    )
    masked_target = PhysicalTarget(
        prior,
        prior,
        context,
        heldout,
        cast(PersistentJuliaWorker, worker),
        cast(BudgetLedger, object()),
        run_factory,
        simulate_fn=fake_simulate,
        load_forward_fn=lambda path, project_paths: forwards[0],
        adapter_hash="b" * 64,
    )
    masked = masked_target.evaluate(_theta(context, log_bias=0.0))

    assert len(forwards) == 1
    assert first.forward_ref == second.forward_ref == masked.forward_ref
    assert first.cache_key != second.cache_key
    assert first.cache_key != masked.cache_key
    assert first.log_l != second.log_l
    assert all(ctx.record.status == "PASS" for ctx in contexts)


def test_empty_observations_do_not_call_the_physical_forward(tmp_path: Path) -> None:
    from so_recon.inference.contracts import PriorContext

    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    context = _prior_context()
    assert isinstance(context, PriorContext)
    prior = GaussianConditionalPrior(context)
    worker = SimpleNamespace(paths=paths, environment_lock_hash="a" * 64)
    called = False

    def forbidden_forward(*args: object, **kwargs: object) -> ForwardResult:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("an empty likelihood does not need F")

    empty = ObservationBundle(
        history=(),
        logs=(),
        bin_edges_by_group={},
        cutoff_s=1.0,
        information_hash="c" * 64,
        observation_hash="d" * 64,
    )
    target = PhysicalTarget(
        prior,
        prior,
        context,
        empty,
        cast(PersistentJuliaWorker, worker),
        cast(BudgetLedger, object()),
        lambda command, parents: cast(RunContext, _TargetContext(root, 0)),
        simulate_fn=forbidden_forward,
        adapter_hash="b" * 64,
    )
    result = target.evaluate(_theta(context))
    assert called is False
    assert result.log_l == 0.0
    assert result.forward_ref is None


def test_reduced_target_reuses_identical_physics_and_likelihood(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.default(root)
    design = ReducedDesign()
    prior = ReducedGaussianPrior(design)
    worker = SimpleNamespace(paths=paths, environment_lock_hash="a" * 64)
    contexts: list[_TargetContext] = []
    forwards: list[ForwardResult] = []

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        assert command == "e02-reduced-smc-evaluation"
        assert parent_run_ids == ("reference-run",)
        ctx = _TargetContext(root, len(contexts))
        contexts.append(ctx)
        return cast(RunContext, ctx)

    def fake_simulate(case: object, request: object, **kwargs: Any) -> ForwardResult:
        del case, request, kwargs
        result = _result(root)
        forwards.append(result)
        return result

    target = ReducedPhysicalTarget(
        prior,
        prior,
        design,
        _bundle().model_copy(update={"logs": ()}),
        cast(PersistentJuliaWorker, worker),
        cast(BudgetLedger, object()),
        run_factory,
        parent_run_ids=("reference-run",),
        simulate_fn=fake_simulate,
        load_forward_fn=lambda path, project_paths: forwards[0],
        adapter_hash="b" * 64,
    )
    theta = prior.sample(1, np.random.default_rng(3))[0]
    first = target.evaluate(theta)
    second = target.evaluate(theta)

    assert len(forwards) == 1
    assert first.forward_ref == second.forward_ref
    assert first.cache_key == second.cache_key
    assert first.log_l == second.log_l
    assert target.checkpoint_hashes["basis_hash"] == prior.schema.basis_hash
    assert all(ctx.record.status == "PASS" for ctx in contexts)


@pytest.mark.parametrize("status", ["TIMEOUT", "RESOURCE_FAILURE", "NUMERICAL_FAILURE"])
def test_forward_failure_is_an_exception_not_zero_likelihood(status: str) -> None:
    result = ForwardResult(
        job_id="job-1",
        case_sha256="a" * 64,
        model_hash="b" * 64,
        physics_class="OW",
        status=status,
        reason="injected failure",
        completed_time_s=0.0,
        times_s=(),
        states={},
        solver_metadata={},
        cost=CostRecord(
            wall_s=1.0,
            cpu_s=1.0,
            peak_rss_bytes=1,
            output_bytes=0,
            accepted_steps=0,
            cut_steps=0,
            nonlinear_iterations=0,
            retry_count=0,
            measurement_method="test",
        ),
        parent_attempt_ids=(),
    )
    with pytest.raises(ForwardEvaluationError, match=status):
        require_complete_forward(result)
