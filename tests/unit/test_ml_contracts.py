"""E03 ML record contracts (plan §4.2): the refusals that catch mismatched provenance."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from so_recon.inference.contracts import NoiseTheta, ThetaRecord
from so_recon.ml.contracts import (
    FORBIDDEN_CONTEXT_FEATURES,
    ComparisonMethod,
    ComparisonParent,
    ComparisonProtocol,
    ContextBatch,
    ContextSpec,
    CorpusManifest,
    FailureRow,
    LayoutSupport,
    ParentRow,
    ProposalManifest,
    TrainingManifest,
    ViewRow,
)
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_json

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _ref(artifact_id: str = A, path: str = "artifacts/runs/x/y.json") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        path=path,
        sha256=artifact_id,
        size_bytes=1,
        media_type="application/json",
        schema_version="test-1",
        producer_run_id="run",
        parent_artifact_ids=[],
        created_at="2026-09-16T00:00:00+00:00",
    )


def _theta(schema_id: str = "e02-t1-symmetric-12") -> ThetaRecord:
    return ThetaRecord(
        schema_id=schema_id, s=0, v=(0.1,) * 11, z_perp=(0.2,) * 4, basis_hash=B
    )


def _parent(**over: object) -> ParentRow:
    base: dict[str, object] = {
        "parent_id": "t1-000",
        "design_id": "e02-t1-v1",
        "split": "train",
        "truth_seed": 17,
        "history_seed": 18,
        "theta": _theta(),
        "noise": NoiseTheta(sigma=0.03, rho=0.4, nu=5.0, log_bias=0.0),
        "schema_id": "e02-t1-symmetric-12",
        "basis_hash": B,
        "observation_hash": C,
        "model_hash": A,
        "context_ref": _ref(),
        "truth_ref": _ref(),
        "views": (ViewRow(view_id="m36-copy0", prefix_months=36, noise_copy=0, observation_hash=C, weight=1.0),),
    }
    base.update(over)
    return ParentRow(**base)  # type: ignore[arg-type]


def _manifest(parents: tuple[ParentRow, ...], **over: object) -> CorpusManifest:
    base: dict[str, object] = {
        "experiment_id": "e03-thin",
        "config_version": "E03.1",
        "namespace": "thin_slice",
        "learning_config_hash": A,
        "design_distribution": {"e02-t1-v1": len(parents)},
        "totals": {"train": 1, "development": 0, "evaluation": 0},
        "expected": max(len(parents), 1),
        "complete": len(parents),
        "failed": 0,
        "parents": parents,
        "failures": (),
        "labels_ref": _ref(),
        "context_dir_ref": _ref(),
        "truth_dir_ref": _ref(),
        "schemas": {"e02-t1-symmetric-12": B},
        "basis_hashes": {"e02-t1-symmetric-12": B},
        "source_commit": "0" * 40,
        "source_dirty": False,
    }
    base.update(over)
    return CorpusManifest(**base)  # type: ignore[arg-type]


def test_a_train_parent_without_views_is_refused() -> None:
    with pytest.raises(ValidationError, match="no view"):
        _parent(views=())


def test_view_identity_must_name_prefix_and_copy_as_exact_segments() -> None:
    ViewRow(view_id="m36-copy0", prefix_months=36, noise_copy=0, observation_hash=C, weight=1.0)
    with pytest.raises(ValidationError, match="must name its prefix"):
        ViewRow(view_id="m36-copy0", prefix_months=3, noise_copy=0, observation_hash=C, weight=1.0)
    with pytest.raises(ValidationError, match="must name its prefix"):
        ViewRow(view_id="m36", prefix_months=36, noise_copy=1, observation_hash=C, weight=1.0)
    with pytest.raises(ValidationError, match="must name its prefix"):
        ViewRow(view_id="copy0", prefix_months=36, noise_copy=0, observation_hash=C, weight=1.0)


def test_view_weights_must_sum_to_one() -> None:
    views = (
        ViewRow(view_id="m36-copy0", prefix_months=36, noise_copy=0, observation_hash=C, weight=0.9),
        ViewRow(view_id="m36-copy1", prefix_months=36, noise_copy=1, observation_hash=C, weight=0.2),
    )
    with pytest.raises(ValidationError, match="sum to 1"):
        _parent(views=views)


def test_label_schema_mismatch_is_refused() -> None:
    with pytest.raises(ValidationError, match="label names"):
        _parent(schema_id="e02-t4-17d")


def test_manifest_refuses_lost_and_invented_parents() -> None:
    parent = _parent()
    with pytest.raises(ValidationError, match="written together"):
        _manifest((parent,), complete=2)
    failure = FailureRow(
        parent_id="t1-009",
        design_id="e02-t1-v1",
        split="train",
        truth_seed=99,
        stage="forward",
        reason="NUMERICAL_FAILURE",
        attempts=2,
    )
    with pytest.raises(ValidationError, match="rows appeared from nowhere"):
        _manifest(
            (parent,),
            failures=(failure,),
            failed=1,
            expected=1,
            design_distribution={"e02-t1-v1": 1},
        )


def test_manifest_accounts_failures_without_hiding_them() -> None:
    failure = FailureRow(
        parent_id="t1-009",
        design_id="e02-t1-v1",
        split="train",
        truth_seed=99,
        stage="forward",
        reason="NUMERICAL_FAILURE after registered retries",
        attempts=2,
    )
    manifest = _manifest((), failures=(failure,), failed=1, expected=1, complete=0)
    assert manifest.failures[0].stage == "forward"


def test_context_batch_shapes_are_validated() -> None:
    spec = dict(
        parent_id="t1-000",
        design_id="e02-t1-v1",
        split="train",
        spec_hash=A,
        well_time=np.zeros((2, 3, 4)),
        well_mask=np.ones((2, 3), dtype=bool),
        static=np.zeros(5),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_attr=np.zeros((2, 6)),
        well_ids=("I1", "P1"),
    )
    ContextBatch(**spec)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="well_mask shape"):
        ContextBatch(**{**spec, "well_mask": np.ones((3, 3), dtype=bool)})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="not in the batch"):
        ContextBatch(**{**spec, "edge_index": np.array([[0, 9], [1, 0]], dtype=np.int64)})  # type: ignore[arg-type]


def test_context_arrays_cannot_be_mutated_after_construction() -> None:
    batch = ContextBatch(
        parent_id="t1-000",
        design_id="e02-t1-v1",
        split="train",
        spec_hash=A,
        well_time=np.zeros((2, 3, 4)),
        well_mask=np.ones((2, 3), dtype=bool),
        static=np.zeros(5),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_attr=np.zeros((1, 6)),
        well_ids=("I1", "P1"),
    )
    with pytest.raises(ValueError, match="read-only"):
        batch.well_time[0, 0, 0] = 1.0


def test_forbidden_features_are_named_for_the_allowlist_tests() -> None:
    assert "truth" in FORBIDDEN_CONTEXT_FEATURES
    assert "full_so" in FORBIDDEN_CONTEXT_FEATURES
    assert "s" in FORBIDDEN_CONTEXT_FEATURES


def test_training_manifest_refuses_a_checkpoint_from_the_future() -> None:
    base: dict[str, object] = {
        "run_id": "run-1",
        "corpus_manifest_ref": _ref(),
        "corpus_manifest_sha256": A,
        "learning_config_hash": A,
        "model_config_hash": A,
        "scaler_hash": A,
        "context_spec_hash": A,
        "train_seed": 9101,
        "optimizer": "adamw",
        "device": "cpu",
        "dtype": "float32",
        "encoder_variant": "graph",
        "best_epoch": 3,
        "selected_epoch": 4,
        "epochs_run": 3,
        "metrics": {"dev_parent_nll": 1.5},
        "costs": {"wall_s": 10.0},
        "checkpoint_ref": _ref(),
        "proposal_manifest_ref": _ref(),
        "torch_version": "2",
        "nflows_version": "0.14",
    }
    with pytest.raises(ValidationError, match="future state"):
        TrainingManifest(**base)  # type: ignore[arg-type]
    ok = {**base, "selected_epoch": 3}
    TrainingManifest(**ok)  # type: ignore[arg-type]


def test_proposal_manifest_needs_at_least_one_layout_and_unique_pairs() -> None:
    layout = LayoutSupport(
        schema_id="e02-t1-symmetric-12", n_v=11, n_residual=4, families=(0, 1), basis_hash=B
    )
    base: dict[str, object] = {
        "weights_ref": _ref(),
        "weights_sha256": A,
        "flow_config_ref": _ref(),
        "flow_config_sha256": A,
        "scaler_ref": _ref(),
        "scaler_sha256": A,
        "supported_layouts": (layout,),
        "operational_dtype": "float64",
        "operational_backend": "cpu",
        "architecture_version": "e03-conditional-nsf-1",
        "context_builder_version": "e03-context-1",
    }
    ProposalManifest(**base)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="at least one"):
        ProposalManifest(**{**base, "supported_layouts": ()})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="repeat"):
        ProposalManifest(**{**base, "supported_layouts": (layout, layout)})  # type: ignore[arg-type]


def test_raw_proposal_method_never_claims_a_posterior() -> None:
    with pytest.raises(ValidationError, match="never carries a posterior"):
        ComparisonMethod(method_id="Q", start="raw_proposal", physical_runs=True, posterior_claim=True)


def test_preregistration_hash_locks_the_protocol() -> None:
    parent = ComparisonParent(
        parent_id="t1-eval-0",
        design_id="e02-t1-v1",
        split="evaluation",
        observation_hash=C,
        truth_ref=_ref(),
    )
    methods = (
        ComparisonMethod(method_id="B0", start="static_prior", physical_runs=True, posterior_claim=False),
        ComparisonMethod(method_id="M", start="defensive_mixture", physical_runs=True, posterior_claim=True),
    )
    protocol = ComparisonProtocol.preregister(
        protocol_id="e03-comparison-1",
        parents=(parent,),
        methods=methods,
        particle_counts=(32, 64),
        inference_seeds=(11, 12),
        primary_month=36,
        gates={"rmse_relative_improvement_min": 0.10},
    )
    assert protocol.preregistration_hash == sha256_json(
        {
            key: value
            for key, value in protocol.model_dump(mode="json").items()
            if key != "preregistration_hash"
        }
    )
    edited = protocol.model_dump(mode="json")
    edited["particle_counts"] = [32, 64, 128]
    with pytest.raises(ValidationError, match="edited after registration"):
        ComparisonProtocol(**edited)  # type: ignore[arg-type]
