"""Corpus builder contracts (plan E03 Task 03) on stubbed physics.

The physics seams (`run_parent_forward`, `history_prediction`) are replaced by fakes:
these tests pin the LEAKAGE and ACCOUNTING properties of the builder — splits, labels,
noise provenance, failures, immutability — which are exactly the properties a real
forward must not change. The native thin slice runs in the integration suite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import so_recon.synthetic.inverse_corpus as corpus_module
from so_recon.config.learning import LearningConfig, FamilyPlan
from so_recon.geology.renderer import render_theta
from so_recon.inference.contracts import NoiseTheta, PriorContext, ThetaRecord
from so_recon.ml.dataset import CorpusDataset, theta_arrays
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactImmutabilityError, register_artifact
from so_recon.registry.run import RunContext
from so_recon.simulator.contracts import ControlSegment
from so_recon.synthetic.inverse_corpus import (
    CORPUS_TRUTH_TAG,
    TRUTH_RATE_CONTROL_RELATIVE_MAX,
    CorpusForwardError,
    build_learning_corpus,
    history_seed,
    load_corpus_manifest,
    truth_latent_rng,
)


def _learning() -> LearningConfig:
    return LearningConfig(
        corpus=(FamilyPlan(design_id="e02-t2-v2", train=2, development=1, evaluation=1),),
    )


@pytest.fixture()
def paths(tmp_path: Path) -> ProjectPaths:
    project = ProjectPaths.default(tmp_path)
    project.ensure_dirs()
    return project


@pytest.fixture()
def ctx(paths: ProjectPaths) -> RunContext:
    return RunContext.start(
        command="test-corpus",
        argv=(),
        cfg=None,
        paths=paths,
        now=datetime.now(UTC),
    )


def _run_factory(paths: ProjectPaths):
    def factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command,
            argv=(),
            cfg=None,
            paths=paths,
            parent_run_ids=parent_run_ids,
            now=datetime.now(UTC),
        )

    return factory


class FakePhysics:
    """Records calls and returns plausible forward outcomes without any solver."""

    def __init__(self, paths: ProjectPaths, ctx: RunContext) -> None:
        self.paths = paths
        self.ctx = ctx
        self.calls: list[dict[str, object]] = []
        self.fail_for: set[str] = set()

    def forward(self, rendered, case, *, ctx, worker, ledger, paths, run_factory):
        theta_json = json.loads(json.dumps([]))  # placeholder to keep json import honest
        del theta_json
        self.calls.append({"case_id": case.case_id})
        if case.case_id in self.fail_for:
            raise CorpusForwardError("simulated NUMERICAL_FAILURE after retries")
        run = run_factory("fake-corpus-forward", (ctx.run_id,))
        path = run.run_dir / "forward_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "COMPLETE", "fake": True}), encoding="utf-8")
        ref = register_artifact(
            path,
            paths,
            schema_version="forward-1",
            producer_run_id=run.run_id,
            media_type="application/json",
            now=datetime.now(UTC),
        )
        from so_recon.synthetic.inverse_corpus import ParentForwardOutcome

        return ParentForwardOutcome(
            forward_ref=ref,
            model_hash="f" * 64,
            checks={"status": "PASS"},
        )

    def prediction(self, truth, template, paths):
        wells = sorted({row.well_id for row in template.history})
        fw = {
            (well, month): 0.3 + 0.01 * month
            for well in wells
            for month in range(36)
        }
        from so_recon.inference.contracts import ModelObservations

        return ModelObservations(fw=fw, so_support={}, model_hash=truth.model_hash)


@pytest.fixture()
def physics(paths: ProjectPaths, ctx: RunContext, monkeypatch: pytest.MonkeyPatch) -> FakePhysics:
    fake = FakePhysics(paths, ctx)
    monkeypatch.setattr(corpus_module, "run_parent_forward", fake.forward)
    monkeypatch.setattr(corpus_module, "history_prediction", fake.prediction)
    return fake


def test_thin_corpus_publishes_manifest_labels_and_contexts(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
    )
    manifest = load_corpus_manifest(ref, paths)
    assert manifest.complete == 4
    assert manifest.failed == 0
    assert manifest.totals == {"train": 2, "development": 1, "evaluation": 1}

    dataset = CorpusDataset.open(ref, paths)
    rows = dataset.parents()
    assert len(rows) == 4
    splits = {row.parent_id: row.split for row in rows}
    assert list(splits.values()).count("train") == 2

    labels = dataset.labels()
    assert set(labels) == {row.parent_id for row in rows}
    for row in rows:
        label = labels[row.parent_id]
        assert tuple(label["v"]) == row.theta.v
        assert label["s"] == row.theta.s
        assert label["noise_sigma"] == row.noise.sigma


def test_minibatches_are_deterministic_and_parent_aligned(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    dataset = CorpusDataset.open(ref, paths)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    batches_a = [tuple(item.parent_id for item in b) for b in dataset.batches(batch_size=2, split="train", rng=rng_a)]
    batches_b = [tuple(item.parent_id for item in b) for b in dataset.batches(batch_size=2, split="train", rng=rng_b)]
    assert batches_a == batches_b
    assert all(len(batch) <= 2 for batch in batches_a)
    assert sum(len(batch) for batch in batches_a) == 2


def test_history_is_generated_with_the_noise_of_the_recorded_theta(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []
    real = corpus_module.generate_dynamic_history

    def spy(context, prediction, *, seed, noise=corpus_module.generate_dynamic_history.__defaults__):
        captured.append({"seed": seed, "noise": noise})
        return real(context, prediction, seed=seed, noise=noise)

    monkeypatch.setattr(corpus_module, "generate_dynamic_history", spy)
    ref = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    manifest = load_corpus_manifest(ref, paths)
    assert captured, "the history generator was never called"
    for call, parent in zip(captured, manifest.parents, strict=True):
        # plan 0.2 item 1: the noise passed is the noise OF the labelled theta
        assert call["noise"] == parent.noise
        assert call["noise"] != NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0) or (
            parent.noise == NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0)
        )
        assert call["seed"] == parent.history_seed


def test_a_parent_with_a_failed_forward_is_recorded_not_replaced(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref_ok = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    manifest_ok = load_corpus_manifest(ref_ok, paths)
    physics.fail_for = {call["case_id"] for call in physics.calls[:1]}
    ctx2 = RunContext.start(
        command="test-corpus-fail", argv=(), cfg=None, paths=paths, now=datetime.now(UTC)
    )
    ref_fail = build_learning_corpus(
        ctx=ctx2,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        experiment_id="e03-corpus-fail-1",
    )
    manifest = load_corpus_manifest(ref_fail, paths)
    assert manifest.failed == 1
    assert manifest.complete == 3
    assert manifest.failures[0].stage == "forward"
    assert "NUMERICAL_FAILURE" in manifest.failures[0].reason
    # the failed parent keeps its identity: same parent_id set minus the failed one
    ok_ids = {row.parent_id for row in manifest_ok.parents}
    fail_ids = {row.parent_id for row in manifest.parents}
    assert len(ok_ids - fail_ids) == 1
    del manifest_ok


def test_context_files_carry_no_truth_and_split_matches_the_manifest(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    dataset = CorpusDataset.open(ref, paths)
    for parent in dataset.parents():
        payload = dataset.load_context(parent)
        for key in ("truth", "theta", "seed", "full_so", "full_permeability"):
            assert key not in json.dumps(payload), f"forbidden key {key!r} leaked into context"
        assert payload["split"] == parent.split
        assert payload["parent_id"] == parent.parent_id
        # the canonical inference input keeps the E02 six-key allowlist shape
        assert set(payload["inference_input"]) == {
            "context",
            "G",
            "U",
            "observations",
            "density_schema",
            "basis",
        }
        assert payload["inference_input"]["observations"]["information_hash"]


def test_republication_with_different_content_is_refused(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    other = LearningConfig(
        corpus=(FamilyPlan(design_id="e02-t2-v2", train=1, development=1, evaluation=1),),
    )
    ctx2 = RunContext.start(
        command="test-corpus-repub", argv=(), cfg=None, paths=paths, now=datetime.now(UTC)
    )
    with pytest.raises(ArtifactImmutabilityError):
        build_learning_corpus(
            ctx=ctx2,
            learning=other,
            paths=paths,
            run_factory=_run_factory(paths),
            thin=False,
        )


def test_corpus_truth_stream_is_independent_of_the_world_streams() -> None:
    seed = 1234
    world_truth = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[0])
    corpus_rng = truth_latent_rng(seed)
    a = world_truth.standard_normal(11)
    b = corpus_rng.standard_normal(11)
    assert not np.allclose(a, b), "the corpus draw must not repeat the world's truth stream"
    # and the derivation is stable
    assert np.allclose(truth_latent_rng(seed).standard_normal(11), b)
    assert history_seed(seed) != seed
    assert CORPUS_TRUTH_TAG != history_seed(seed)


def test_labels_stack_into_arrays_consistent_with_the_schemas(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    dataset = CorpusDataset.open(ref, paths)
    s, v, z = theta_arrays(dataset.parents())
    assert s.shape == (4,)
    assert v.shape == (4, 11)
    assert z.shape == (4, 4)


# --------------------------------------------------------------------------------------
# per-stage failure recording (review finding 1)
# --------------------------------------------------------------------------------------


def test_a_parent_with_a_failed_prediction_stage_is_recorded_not_dropped(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = corpus_module.history_prediction
    state = {"n": 0}

    def failing(truth, template, paths_arg):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("prediction transport broke")
        return inner(truth, template, paths_arg)

    monkeypatch.setattr(corpus_module, "history_prediction", failing)
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        experiment_id="e03-corpus-stage-prediction-1",
    )
    manifest = load_corpus_manifest(ref, paths)
    assert manifest.expected == 4
    assert manifest.complete == 3
    assert manifest.failed == 1
    row = manifest.failures[0]
    assert row.parent_id == "e02-t2-v2-0000"
    assert row.stage == "prediction"
    assert "prediction transport broke" in row.reason
    assert row.attempts == 1


def test_a_parent_with_a_failed_history_stage_is_recorded(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = corpus_module.generate_dynamic_history
    state = {"n": 0}

    def failing(context, prediction, *, seed, noise=real.__defaults__):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("history generator exploded")
        return real(context, prediction, seed=seed, noise=noise)

    monkeypatch.setattr(corpus_module, "generate_dynamic_history", failing)
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        experiment_id="e03-corpus-stage-history-1",
    )
    manifest = load_corpus_manifest(ref, paths)
    assert manifest.complete == 3
    assert manifest.failed == 1
    assert manifest.failures[0].stage == "history"
    assert "history generator exploded" in manifest.failures[0].reason


def test_a_parent_with_a_failed_context_stage_is_recorded(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = corpus_module._world_for
    state = {"n": 0}

    def failing(design_id, seed, paths_arg, ctx_arg):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("generator registry missing")
        return real(design_id, seed, paths_arg, ctx_arg)

    monkeypatch.setattr(corpus_module, "_world_for", failing)
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        experiment_id="e03-corpus-stage-context-1",
    )
    manifest = load_corpus_manifest(ref, paths)
    assert manifest.complete == 3
    assert manifest.failed == 1
    assert manifest.failures[0].stage == "context"
    assert "generator registry missing" in manifest.failures[0].reason


# --------------------------------------------------------------------------------------
# build-time verification and the label→render round-trip (review finding 2)
# --------------------------------------------------------------------------------------


def test_label_to_render_round_trip_through_the_published_artifacts(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, np.ndarray]] = []
    inner = corpus_module.run_parent_forward

    def capturing(rendered, case, **kwargs):
        captured.append(
            {name: np.array(values, copy=True) for name, values in rendered.arrays.items()}
        )
        return inner(rendered, case, **kwargs)

    monkeypatch.setattr(corpus_module, "run_parent_forward", capturing)
    ref = build_learning_corpus(
        ctx=ctx, learning=_learning(), paths=paths, run_factory=_run_factory(paths), thin=False
    )
    dataset = CorpusDataset.open(ref, paths)
    assert len(captured) == len(dataset.parents())
    for parent, arrays in zip(dataset.parents(), captured, strict=True):
        label = dataset.labels()[parent.parent_id]
        payload = dataset.load_context(parent)
        context = PriorContext.model_validate(payload["inference_input"]["context"])
        theta = ThetaRecord(
            schema_id=label["schema_id"],
            s=label["s"],
            v=tuple(label["v"]),
            z_perp=tuple(label["z_perp"]),
            basis_hash=label["basis_hash"],
        )
        rerendered = render_theta(theta, context)
        for name, values in arrays.items():
            assert np.allclose(rerendered.arrays[name], values, rtol=0.0, atol=1.0e-12), (
                f"parent {parent.parent_id}: array {name} does not round-trip"
            )
        assert rerendered.noise == parent.noise
        assert label["noise_sigma"] == rerendered.noise.sigma


def test_truth_payload_records_the_render_round_trip(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        max_parents=1,
    )
    manifest = load_corpus_manifest(ref, paths)
    root = paths.resolve(ref.path).parent
    payload = json.loads(
        (root / "truth" / f"{manifest.parents[0].parent_id}.json").read_text(encoding="utf-8")
    )
    checks = payload["renderer_checks"]
    assert checks["arrays_reproduce"] is True
    assert checks["geology_round_trip_max_abs"] <= checks["tolerance"] == 1.0e-12


def test_tampered_labels_parquet_is_refused_at_load(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        max_parents=1,
    )
    dataset = CorpusDataset.open(ref, paths)
    labels_path = paths.resolve(dataset.manifest.labels_ref.path)
    schema = pq.read_table(labels_path).schema
    rows = pq.read_table(labels_path).to_pylist()
    rows[0]["v"] = [value + 1.0 for value in rows[0]["v"]]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), labels_path)
    tampered = CorpusDataset.open(ref, paths)
    with pytest.raises(ValueError, match="fails its published digest"):
        tampered.labels()


def test_tampered_context_index_is_refused_at_load(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        max_parents=1,
    )
    dataset = CorpusDataset.open(ref, paths)
    index_path = paths.resolve(dataset.manifest.context_dir_ref.path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["files"] = {}
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stream index"):
        dataset.load_context(dataset.parents()[0])


def test_published_streams_are_reverified_against_their_refs_at_build_time(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = corpus_module._verify_published_streams

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(corpus_module, "_verify_published_streams", spy)
    build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        max_parents=1,
    )
    assert len(calls) == 1


def test_stream_verification_refuses_a_tampered_truth_file(
    paths: ProjectPaths, ctx: RunContext, physics: FakePhysics
) -> None:
    ref = build_learning_corpus(
        ctx=ctx,
        learning=_learning(),
        paths=paths,
        run_factory=_run_factory(paths),
        thin=False,
        max_parents=1,
    )
    manifest = load_corpus_manifest(ref, paths)
    root = paths.resolve(ref.path).parent
    victim = root / "truth" / f"{manifest.parents[0].parent_id}.json"
    victim.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="index digest"):
        corpus_module._verify_published_streams(
            root,
            manifest.parents,
            labels_ref=manifest.labels_ref,
            context_ref=manifest.context_dir_ref,
            truth_ref=manifest.truth_dir_ref,
            paths=paths,
        )


# --------------------------------------------------------------------------------------
# the controls truth check (review finding 3)
# --------------------------------------------------------------------------------------

_DAY_S = 86400.0


def _segment(
    well_id: str,
    role: str,
    target: str,
    value: float,
    *,
    start_s: float = 0.0,
    end_s: float = 60.0 * _DAY_S,
) -> ControlSegment:
    return ControlSegment(
        start_s=start_s,
        end_s=end_s,
        well_id=well_id,
        role=role,
        target=target,
        value=value,
        bhp_limit_pa=None,
        connection_open=(True,),
    )


def _monthly_path(paths: ProjectPaths, rows: list[dict[str, object]]) -> str:
    table = pa.table(
        {
            "well_id": [row["well_id"] for row in rows],
            "month_index": [row["month_index"] for row in rows],
            "start_s": [row["start_s"] for row in rows],
            "end_s": [row["end_s"] for row in rows],
            "water_inj_m3_sc": [row["water_inj_m3_sc"] for row in rows],
            "liquid_prod_m3_sc": [row["liquid_prod_m3_sc"] for row in rows],
        }
    )
    path = paths.artifacts / "controls-check" / "monthly.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return paths.relative(path)


def _two_rate_months(inject: float, produce: float) -> list[dict[str, object]]:
    rows = []
    for month, (start_s, end_s) in enumerate(
        [(0.0, 30.0 * _DAY_S), (30.0 * _DAY_S, 60.0 * _DAY_S)]
    ):
        rows.append(
            {
                "well_id": "I1",
                "month_index": month,
                "start_s": start_s,
                "end_s": end_s,
                "water_inj_m3_sc": inject * 30.0,
                "liquid_prod_m3_sc": 0.0,
            }
        )
        rows.append(
            {
                "well_id": "P1",
                "month_index": month,
                "start_s": start_s,
                "end_s": end_s,
                "water_inj_m3_sc": 0.0,
                "liquid_prod_m3_sc": produce * 30.0,
            }
        )
    return rows


def test_controls_check_scores_rate_targets_against_the_published_monthly_table(
    paths: ProjectPaths,
) -> None:
    controls = (
        _segment("I1", "injector", "water_rate", 100.0),
        _segment("P1", "producer", "liquid_rate", 50.0),
    )
    result = SimpleNamespace(
        job_id="job-controls-test",
        monthly_path=_monthly_path(paths, _two_rate_months(100.0, 50.0)),
    )
    report = corpus_module._corpus_controls_check(controls, result, paths)
    assert report["n_rate_targets"] == 4
    assert report["rate_control_relative"] == pytest.approx(0.0)
    assert report["rate_controlled_wells"] == ["I1", "P1"]
    assert report["bhp_controlled_wells"] == []
    assert report["controls_months"] == 2


def test_controls_check_records_bhp_wells_without_a_rate_claim(paths: ProjectPaths) -> None:
    controls = (
        _segment("I1", "injector", "bhp", 2.5e7),
        _segment("P1", "producer", "liquid_rate", 50.0),
    )
    result = SimpleNamespace(
        job_id="job-controls-test",
        monthly_path=_monthly_path(paths, _two_rate_months(0.0, 50.0)),
    )
    report = corpus_module._corpus_controls_check(controls, result, paths)
    assert report["bhp_controlled_wells"] == ["I1"]
    assert report["n_rate_targets"] == 2
    assert report["rate_control_relative"] == pytest.approx(0.0)


def test_controls_check_measures_a_missed_rate(paths: ProjectPaths) -> None:
    controls = (
        _segment("I1", "injector", "water_rate", 100.0),
        _segment("P1", "producer", "liquid_rate", 50.0),
    )
    rows = _two_rate_months(100.0, 50.0)
    rows[3]["liquid_prod_m3_sc"] = 50.0 * 30.0 * 1.001  # the producer's month 1, 0.1% short
    result = SimpleNamespace(job_id="job-controls-test", monthly_path=_monthly_path(paths, rows))
    report = corpus_module._corpus_controls_check(controls, result, paths)
    assert report["rate_control_relative"] > TRUTH_RATE_CONTROL_RELATIVE_MAX


def test_truth_checks_fail_a_forward_that_missed_its_rate_targets(
    paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = (
        _segment("I1", "injector", "water_rate", 100.0),
        _segment("P1", "producer", "liquid_rate", 50.0),
    )
    rows = _two_rate_months(100.0 * 1.01, 50.0)  # every injector month 1% over
    balances = paths.artifacts / "controls-check" / "balances.parquet"
    balances.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "cumulative_relative": [1.0e-9, 1.0e-9],
                "median_step_relative": [1.0e-9, 1.0e-9],
            }
        ),
        balances,
    )
    result = SimpleNamespace(
        status="COMPLETE",
        job_id="job-controls-test",
        balances_path=paths.relative(balances),
        monthly_path=_monthly_path(paths, rows),
        states={},
    )
    monkeypatch.setattr(corpus_module, "_so_states", lambda result, paths: np.zeros((4, 512)))
    checks = corpus_module._corpus_truth_checks(result, SimpleNamespace(controls=controls), paths)
    assert checks["status"] == "FAIL"
    assert checks["checks"]["controls_rate"] is False
    assert checks["rate_control_relative"] > TRUTH_RATE_CONTROL_RELATIVE_MAX


def test_controls_check_refuses_a_missing_monthly_row(paths: ProjectPaths) -> None:
    controls = (
        _segment("I1", "injector", "water_rate", 100.0),
        _segment("P1", "producer", "liquid_rate", 50.0),
    )
    rows = [
        row
        for row in _two_rate_months(100.0, 50.0)
        if not (row["well_id"] == "P1" and row["month_index"] == 1)
    ]
    result = SimpleNamespace(job_id="job-controls-test", monthly_path=_monthly_path(paths, rows))
    with pytest.raises(CorpusForwardError, match="no monthly row"):
        corpus_module._corpus_controls_check(controls, result, paths)


# --------------------------------------------------------------------------------------
# context creation reads no truth stream (review finding 4)
# --------------------------------------------------------------------------------------


def test_context_creation_reads_no_truth_stream(
    paths: ProjectPaths, ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth_root = paths.artifacts / "corpus" / "e03-corpus-context-read-1" / "truth"
    truth_root.mkdir(parents=True, exist_ok=True)
    sentinel = truth_root / "e02-t2-v2-0000.json"
    sentinel.write_text('{"poison": true}', encoding="utf-8")
    denied: list[str] = []
    real_open = Path.open

    def denying_open(self, *args, **kwargs):
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        if "r" in mode and self.parent == truth_root:
            denied.append(str(self))
            raise PermissionError(
                f"the truth stream file {self} was read while the context was created"
            )
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denying_open)
    context, _pending, _generator = corpus_module._world_for("e02-t2-v2", 4242, paths, ctx)
    assert denied == []
    assert context.information_hash
    # the context is conditioned on the PUBLIC static data only, never on the truth file
    assert context.design["log_k_observations"]
