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

import numpy as np
import pytest

import so_recon.synthetic.inverse_corpus as corpus_module
from so_recon.config.learning import LearningConfig, FamilyPlan
from so_recon.inference.contracts import NoiseTheta
from so_recon.ml.dataset import CorpusDataset, theta_arrays
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactImmutabilityError, register_artifact
from so_recon.registry.run import RunContext
from so_recon.synthetic.inverse_corpus import (
    CORPUS_TRUTH_TAG,
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
