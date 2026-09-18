"""E03 Task 10 Part A — the development-set diagnostics that gate a checkpoint freeze.

The five diagnostics the plan's Task 10 line names are held to the properties that make
them decisions rather than decoration:

* **dev NLL** is parent-averaged with equal world weight (§10.2), decomposed into the
  categorical and continuous factors §7.4 mandates, and reported beside the operational
  log density's difference from the conditional prior. The per-parent number is the
  view-weighted one, hand-checkable from the two views of one parent.
* **categorical support** carries EVERY declared family, including one the training
  corpus never labelled (§7.3): a diagnostic that dropped it would hide exactly the
  defect the check exists for.
* **residual sensitivity** is derived from what the SMC engine DOES publish — the pCN
  move records, whose acceptance ratio reduces exactly to `beta * (log L' - log L)` —
  and its value is hand-computed here from a two-level move log.
* **shuffle/prefix controls** must degrade the diagnostic, and a model built to ignore
  its context makes both of them fail to degrade, which the report must say out loud.
* **epoch cost** is the measured wall/RSS of the published `TrainingManifest`, reached
  through the run record's declared output — and a resumed session, whose session wall
  does not cover its cumulative epochs, refuses to publish a per-epoch number.

§6.3 split isolation is a test, not a comment: an evaluation parent is refused by name.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

from so_recon.config.learning import (
    EncoderParams,
    FamilyPlan,
    FlowParams,
    LearningConfig,
    TrainingHyperParams,
)
from so_recon.inference.contracts import DensitySchema, NoiseTheta, PriorContext, ThetaRecord
from so_recon.ml.checkpoint import load_checkpoint, save_checkpoint
from so_recon.ml.contracts import (
    CorpusManifest,
    ParentRow,
    ProposalManifest,
    TrainingManifest,
    ViewRow,
)
from so_recon.ml.dataset import CorpusDataset
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact
from so_recon.registry.hashing import sha256_bytes, sha256_json
from so_recon.registry.run import RunRecord
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import default_context_spec

WELLS = [("I1", 2, 2, 0), ("I2", 2, 13, 0), ("P1", 13, 2, 1), ("P2", 13, 13, 1)]
N_MONTHS = 36
EDGES = month_edges_s(date(2000, 1, 1), N_MONTHS)
CUTOFF_S = float(EDGES[-1])

#: Three declared families; the corpus labels only 0 and 1, so family 2 is the
#: "declared but never in train" category §7.3 forbids a diagnostic from dropping.
FAMILIES = (0, 1, 2)
LABELLED_FAMILIES = (0, 1)
BASIS_SEED = 7


def _prior_context() -> PriorContext:
    """One shared conditional prior: every parent of the corpus lives on one basis."""
    rng = np.random.default_rng(BASIS_SEED)
    n_geology = 12
    mean = 0.3 * rng.standard_normal(n_geology)
    rotation, _ = np.linalg.qr(rng.standard_normal((n_geology, n_geology)))
    rotation = rotation * np.sign(np.diag(rotation))
    root = rotation @ np.diag(np.linspace(0.6, 1.4, n_geology))
    schema = DensitySchema(
        schema_id="p1-conditional-12",
        n_v=11,
        n_residual=4,
        families=FAMILIES,
        basis_hash=sha256_json({"schema_id": "p1-conditional-12", "seed": BASIS_SEED}),
        transform_version="e02-geology-conditional-1",
    )
    return PriorContext(
        density_schema=schema,
        n_geology=n_geology,
        n_state_residual=0,
        mean=mean,
        chol=root,
        rotation=rotation,
        design={},
        g_hash=sha256_json({"g": BASIS_SEED}),
        information_hash=sha256_json({"information": BASIS_SEED}),
    )


def _theta(rng: np.random.Generator, context: PriorContext) -> ThetaRecord:
    schema = context.density_schema
    coordinates = rng.standard_normal(schema.n_v + schema.n_residual)
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=int(LABELLED_FAMILIES[int(rng.integers(0, len(LABELLED_FAMILIES)))]),
        v=tuple(coordinates[: schema.n_v].tolist()),
        z_perp=tuple(coordinates[schema.n_v :].tolist()),
        basis_hash=schema.basis_hash,
    )


def _payload(parent_id: str, context: PriorContext, seed: int, split: str) -> dict[str, Any]:
    design = {
        "design_id": "e02-t2-v2",
        "shape": [16, 16, 2],
        "n_months": N_MONTHS,
        "start_date": "2000-01-01",
        "well_columns": [
            {"well_id": name, "column": [i, j], "role": "producer" if role else "injector"}
            for name, i, j, role in WELLS
        ],
    }
    controls = []
    for name, _i, _j, role in WELLS:
        for month in range(N_MONTHS):
            controls.append(
                {
                    "well_id": name,
                    "start_s": EDGES[month],
                    "end_s": EDGES[month + 1],
                    "role": "producer" if role else "injector",
                    "target": "liquid_rate" if role else "water_rate",
                    "value": 20.0 + 0.5 * (seed % 5),
                    "bhp_limit_pa": None,
                    "connection_open": (True, True),
                }
            )
    history = []
    for name, _i, _j, role in WELLS:
        if role == 0:
            continue
        for month in range(N_MONTHS):
            history.append(
                {
                    "well_id": name,
                    "month_index": month,
                    "raw_value": 0.2 + 0.01 * ((seed + month) % 7),
                    "bin_index": 20 + ((seed + month) % 11),
                    "quality_group": "watercut-0.01",
                    "observed_valid": True,
                    "reset": False,
                    "sigma_multiplier": 1.0,
                }
            )
    grid_edges = [round(value, 4) for value in np.linspace(0.0, 1.0, 101)]
    context_payload = context.model_dump(mode="python")
    for name in ("mean", "chol", "rotation"):
        context_payload[name] = np.asarray(context_payload[name], dtype=np.float64).tolist()
    context_payload["design"] = {"inverse_design": design}
    return {
        "schema_version": "e03-parent-context-1",
        "parent_id": parent_id,
        "design_id": "e02-t2-v2",
        "split": split,
        "cutoff_s": CUTOFF_S,
        "well_ids": ["P1", "P2"],
        "inference_input": {
            "context": context_payload,
            "G": [],
            "U": controls,
            "observations": {
                "history": history,
                "logs": [],
                "bin_edges_by_group": {"watercut-0.01": grid_edges},
                "cutoff_s": CUTOFF_S,
                "information_hash": "a" * 64,
                "observation_hash": "b" * 64,
            },
        },
    }


@dataclass(frozen=True)
class Published:
    manifest_ref: ArtifactRef
    spec: Any


def _views(specs: tuple[tuple[int, float], ...], observation_hash: str) -> tuple[ViewRow, ...]:
    return tuple(
        ViewRow(
            view_id=f"m{months}-copy0",
            prefix_months=months,
            noise_copy=0,
            observation_hash=observation_hash,
            weight=weight,
        )
        for months, weight in specs
    )


def _publish_corpus(
    paths: ProjectPaths,
    root: Path,
    *,
    n_train: int = 3,
    n_dev: int = 2,
    n_eval: int = 1,
    dev_views: tuple[tuple[int, float], ...] = ((36, 1.0),),
) -> Published:
    """A real digest-verified corpus on disk: labels parquet, context stream, manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    context_dir = root / "context"
    truth_dir = root / "truth"
    context_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "index.json").write_bytes(b'{"files": {}}\n')
    context_obj = _prior_context()
    parents: list[ParentRow] = []
    labels_rows: list[dict[str, Any]] = []
    index_files: dict[str, str] = {}
    counts = {"train": n_train, "development": n_dev, "evaluation": n_eval}
    counter = 0
    for split, total in counts.items():
        for _ in range(total):
            theta = _theta(np.random.default_rng(900 + counter), context_obj)
            parent_id = f"e02-t2-v2-{counter:04d}"
            payload = _payload(parent_id, context_obj, 700 + counter, split)
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            (context_dir / f"{parent_id}.json").write_bytes(body)
            index_files[f"{parent_id}.json"] = sha256_bytes(body)
            observation_hash = payload["inference_input"]["observations"]["observation_hash"]
            ref = ArtifactRef(
                artifact_id=sha256_bytes(body),
                path=f"artifacts/corpus/{root.name}/context/{parent_id}.json",
                sha256=sha256_bytes(body),
                size_bytes=len(body),
                media_type="application/json",
                schema_version="e03-context-1",
                producer_run_id="test",
                parent_artifact_ids=[],
                created_at="2026-09-17T00:00:00",
            )
            view_specs = dev_views if split == "development" else ((36, 1.0),)
            parents.append(
                ParentRow(
                    parent_id=parent_id,
                    design_id="e02-t2-v2",
                    split=split,
                    truth_seed=counter,
                    history_seed=1000 + counter,
                    theta=theta,
                    noise=NoiseTheta(sigma=0.05, rho=0.3, nu=5.0, log_bias=0.01),
                    schema_id=theta.schema_id,
                    basis_hash=theta.basis_hash,
                    observation_hash=observation_hash,
                    model_hash="f" * 64,
                    context_ref=ref,
                    truth_ref=ref,
                    views=_views(view_specs, observation_hash),
                )
            )
            labels_rows.append(
                {
                    "parent_id": parent_id,
                    "split": split,
                    "s": theta.s,
                    "v": list(theta.v),
                    "z_perp": list(theta.z_perp),
                }
            )
            counter += 1

    labels_path = root / "labels.parquet"
    table = pa.table(
        {
            "parent_id": [row["parent_id"] for row in labels_rows],
            "split": [row["split"] for row in labels_rows],
            "s": [row["s"] for row in labels_rows],
            "v": [row["v"] for row in labels_rows],
            "z_perp": [row["z_perp"] for row in labels_rows],
        }
    )
    pq.write_table(table, labels_path)
    labels_ref = register_artifact(
        labels_path,
        paths,
        schema_version="e03-labels-1",
        producer_run_id="test",
        media_type="application/octet-stream",
        now=datetime.now(UTC),
    )
    index_body = json.dumps({"files": index_files}, indent=2, sort_keys=True).encode("utf-8")
    (context_dir / "index.json").write_bytes(index_body)
    context_ref = register_artifact(
        context_dir / "index.json",
        paths,
        schema_version="e03-context-index-1",
        producer_run_id="test",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    truth_ref = register_artifact(
        truth_dir / "index.json",
        paths,
        schema_version="e03-truth-index-1",
        producer_run_id="test",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    manifest = CorpusManifest(
        experiment_id=root.name,
        config_version="e03-learning-1",
        namespace="thin_slice",
        learning_config_hash="a" * 64,
        design_distribution={"e02-t2-v2": counter},
        totals={"train": n_train, "development": n_dev, "evaluation": n_eval},
        expected=counter,
        complete=counter,
        failed=0,
        parents=tuple(parents),
        labels_ref=labels_ref,
        context_dir_ref=context_ref,
        truth_dir_ref=truth_ref,
        schemas={p.schema_id: p.basis_hash for p in parents},
        basis_hashes={p.schema_id: p.basis_hash for p in parents},
        source_commit="test",
        source_dirty=False,
    )
    manifest_body = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode(
        "utf-8"
    )
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_bytes(manifest_body)
    manifest_ref = register_artifact(
        manifest_path,
        paths,
        schema_version="e03-corpus-manifest-1",
        producer_run_id="test",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    spec = default_context_spec(cutoff_s=CUTOFF_S, max_wells=4, max_months=N_MONTHS)
    return Published(manifest_ref=manifest_ref, spec=spec)


def _learning() -> LearningConfig:
    return LearningConfig(
        corpus=(FamilyPlan(design_id="e02-t2-v2", train=3, development=2, evaluation=1),),
        encoder=EncoderParams(
            variant="summary", width=8, heads=2, context_dim=8, max_wells=4, max_months=N_MONTHS
        ),
        flow=FlowParams(n_transforms=2, n_bins=4),
        training=TrainingHyperParams(batch_size=3, max_epochs=5, patience=5),
    )


@dataclass(frozen=True)
class Frozen:
    """A published proposal: the three Float64 streams plus the manifest vouching them."""

    checkpoint: Any
    manifest: ProposalManifest
    spec: Any


def _publish_proposal(
    corpus: Published, paths: ProjectPaths, root: Path, *, context_blind: bool = False
) -> Frozen:
    """Export a checkpoint the way `ml.commands` does, with no training run needed."""
    from so_recon.ml.train import build_proposal_model, prepare_training

    learning = _learning()
    dataset = CorpusDataset.open(corpus.manifest_ref, paths)
    prepared = prepare_training(dataset, learning, spec=corpus.spec)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    if context_blind:
        # A model whose pooled context is a constant: the encoder head's first linear
        # layer sees nothing of its input, so every world gets the same context vector.
        with torch.no_grad():
            model.encoder.head[0].weight.zero_()
    hashes = save_checkpoint(
        root,
        module=model,
        flow_config=prepared.flow_config,
        scaler_payload=prepared.scaler_payload,
    )
    refs = {}
    for key, filename, schema in (
        ("weights", "weights.safetensors", "e03-weights-1"),
        ("flow_config", "flow_config.json", "e03-config-1"),
        ("scaler", "scaler.json", "e03-scaler-1"),
    ):
        refs[key] = register_artifact(
            root / filename,
            paths,
            schema_version=schema,
            producer_run_id="test",
            media_type="application/octet-stream",
            now=datetime.now(UTC),
        )
    manifest = ProposalManifest(
        weights_ref=refs["weights"],
        weights_sha256=hashes.weights_sha256,
        flow_config_ref=refs["flow_config"],
        flow_config_sha256=hashes.flow_config_sha256,
        scaler_ref=refs["scaler"],
        scaler_sha256=hashes.scaler_sha256,
        supported_layouts=prepared.layouts,
        operational_dtype="float64",
        operational_backend="cpu",
        architecture_version=learning.flow.architecture_version,
        context_builder_version=corpus.spec.builder_version,
    )
    return Frozen(checkpoint=load_checkpoint(root), manifest=manifest, spec=corpus.spec)


@pytest.fixture()
def paths(tmp_path: Path) -> ProjectPaths:
    project = ProjectPaths.default(tmp_path)
    project.ensure_dirs()
    return project


@pytest.fixture()
def corpus(paths: ProjectPaths, tmp_path: Path) -> Published:
    return _publish_corpus(paths, tmp_path / "artifacts" / "corpus" / "diag-corpus-1")


@pytest.fixture()
def frozen(corpus: Published, paths: ProjectPaths, tmp_path: Path) -> Frozen:
    return _publish_proposal(corpus, paths, tmp_path / "artifacts" / "proposals" / "diag-1")


@pytest.fixture()
def dataset(corpus: Published, paths: ProjectPaths) -> CorpusDataset:
    return CorpusDataset.open(corpus.manifest_ref, paths)


# --------------------------------------------------------------------------------------
# §6.3 split isolation
# --------------------------------------------------------------------------------------


def test_the_diagnostics_never_return_a_parent_outside_the_development_split(
    dataset: CorpusDataset,
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds

    worlds = development_worlds(dataset)
    assert [world.parent_id for world in worlds] == [
        row.parent_id for row in dataset.parents("development")
    ]
    assert worlds and all(world.split == "development" for world in worlds)


def test_naming_an_evaluation_parent_is_refused_by_name(dataset: CorpusDataset) -> None:
    from so_recon.validation.proposal_diagnostics import (
        EvaluationSplitRefused,
        development_worlds,
    )

    evaluation_id = dataset.parents("evaluation")[0].parent_id
    with pytest.raises(EvaluationSplitRefused) as raised:
        development_worlds(dataset, parent_ids=(evaluation_id,))
    message = str(raised.value)
    assert evaluation_id in message
    assert "evaluation" in message and "development" in message


def test_the_train_family_census_refuses_the_evaluation_split(dataset: CorpusDataset) -> None:
    from so_recon.validation.proposal_diagnostics import EvaluationSplitRefused, families_in_split

    assert families_in_split(dataset, "train") == frozenset(LABELLED_FAMILIES)
    with pytest.raises(EvaluationSplitRefused):
        families_in_split(dataset, "evaluation")


# --------------------------------------------------------------------------------------
# dev NLL: decomposition, prior difference, per-parent and world-equal averaging
# --------------------------------------------------------------------------------------


def test_dev_nll_decomposes_the_two_factors_and_the_prior_difference(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.geology.density import GaussianConditionalPrior
    from so_recon.ml.context import build_context_batch
    from so_recon.ml.proposal import bind_frozen_proposal
    from so_recon.validation.proposal_diagnostics import development_nll, development_worlds

    worlds = development_worlds(dataset)
    report = development_nll(
        worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
    )
    assert report.n_parents == len(worlds) == 2

    for world, row in zip(worlds, report.parents, strict=True):
        assert row.parent_id == world.parent_id
        law = bind_frozen_proposal(
            checkpoint=frozen.checkpoint,
            manifest=frozen.manifest,
            spec=frozen.spec,
            batch=build_context_batch(world.context_payload, spec=frozen.spec, prefix_months=36),
            schema=world.schema,
        )
        log_s, log_v, log_z = law.log_prob_parts(world.theta)
        assert row.log_q_s == pytest.approx(log_s, abs=1e-12)
        assert row.log_q_v == pytest.approx(log_v, abs=1e-12)
        assert row.nll == pytest.approx(-(log_s + log_v), abs=1e-12)
        # the operational density of §5.2 carries the residual factor; the NLL does not
        assert row.log_q == pytest.approx(log_s + log_v + log_z, abs=1e-12)
        prior = GaussianConditionalPrior(
            PriorContext.model_validate(world.context_payload["inference_input"]["context"])
        )
        assert row.log_prior == pytest.approx(prior.log_prob(world.theta), abs=1e-12)
        assert row.log_q_minus_prior == pytest.approx(row.log_q - row.log_prior, abs=1e-12)

    # §10.2: equal weight per world, never a pooled pseudo-sample
    assert report.nll == pytest.approx(
        float(np.mean([row.nll for row in report.parents])), abs=1e-12
    )
    assert report.log_q_minus_prior == pytest.approx(
        float(np.mean([row.log_q_minus_prior for row in report.parents])), abs=1e-12
    )


def test_a_parents_views_are_weighted_by_their_published_weights(
    paths: ProjectPaths, tmp_path: Path, frozen: Frozen
) -> None:
    from so_recon.ml.context import build_context_batch
    from so_recon.ml.proposal import bind_frozen_proposal
    from so_recon.validation.proposal_diagnostics import development_nll, development_worlds

    # The published proposal is the `frozen` fixture's: one basis, one context spec, so a
    # corpus whose development parents carry two prefix views is scored by the same law.
    two_view = _publish_corpus(
        paths,
        tmp_path / "artifacts" / "corpus" / "two-view-1",
        dev_views=((12, 0.25), (36, 0.75)),
    )
    dataset = CorpusDataset.open(two_view.manifest_ref, paths)
    worlds = development_worlds(dataset)
    report = development_nll(
        worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
    )

    world = worlds[0]
    by_prefix: dict[int, float] = {}
    for months in (12, 36):
        law = bind_frozen_proposal(
            checkpoint=frozen.checkpoint,
            manifest=frozen.manifest,
            spec=frozen.spec,
            batch=build_context_batch(
                world.context_payload, spec=frozen.spec, prefix_months=months
            ),
            schema=world.schema,
        )
        log_s, log_v, _log_z = law.log_prob_parts(world.theta)
        by_prefix[months] = -(log_s + log_v)
    assert by_prefix[12] != by_prefix[36]
    assert report.parents[0].nll == pytest.approx(
        0.25 * by_prefix[12] + 0.75 * by_prefix[36], abs=1e-12
    )


def test_a_development_parent_without_a_view_is_refused_naming_the_field(
    dataset: CorpusDataset, paths: ProjectPaths
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds

    stripped = dataset.manifest.model_copy(
        update={
            "parents": tuple(
                row.model_copy(update={"views": ()}) if row.split == "development" else row
                for row in dataset.manifest.parents
            )
        }
    )
    broken = CorpusDataset(stripped, dataset.manifest_ref, paths)
    with pytest.raises(ValueError, match="views"):
        development_worlds(broken)


# --------------------------------------------------------------------------------------
# §7.3 categorical support over ALL declared families
# --------------------------------------------------------------------------------------


def test_categorical_support_reports_a_declared_family_the_corpus_never_labelled(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.validation.proposal_diagnostics import categorical_support, development_worlds

    worlds = development_worlds(dataset)
    report = categorical_support(
        worlds,
        checkpoint=frozen.checkpoint,
        manifest=frozen.manifest,
        spec=frozen.spec,
        families_seen_in_train=frozenset(LABELLED_FAMILIES),
    )
    assert report.declared_families == FAMILIES
    assert report.families_absent_from_train == (2,)
    for row in report.parents:
        assert tuple(sorted(row.probabilities)) == FAMILIES
        assert sum(row.probabilities.values()) == pytest.approx(1.0, abs=1e-10)
        # the point of the check: the unseen category carries mass, it is not dropped
        assert row.probabilities[2] > 0.0
    assert set(report.mean_probability) == set(FAMILIES)
    assert report.mean_probability[2] == pytest.approx(
        float(np.mean([row.probabilities[2] for row in report.parents])), abs=1e-12
    )
    assert report.min_probability_absent_from_train == pytest.approx(
        min(row.probabilities[2] for row in report.parents), abs=1e-12
    )


# --------------------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------------------


def test_a_control_that_does_not_degrade_is_reported_as_a_failure() -> None:
    from so_recon.validation.proposal_diagnostics import control_outcome

    degraded = control_outcome("shuffled_context", baseline_nll=2.0, control_nll=3.5, margin=1.0)
    assert degraded.delta == pytest.approx(1.5)
    assert degraded.degraded is True

    flat = control_outcome("shuffled_context", baseline_nll=2.0, control_nll=2.0, margin=0.0)
    assert flat.delta == pytest.approx(0.0)
    assert flat.degraded is False

    short = control_outcome("prefix_12", baseline_nll=2.0, control_nll=2.5, margin=1.0)
    assert short.degraded is False, "a control must clear the declared margin, not merely move"


def test_a_context_blind_model_makes_both_controls_fail_to_degrade(
    corpus: Published, paths: ProjectPaths, tmp_path: Path
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds, negative_controls

    blind = _publish_proposal(
        corpus, paths, tmp_path / "artifacts" / "proposals" / "blind-1", context_blind=True
    )
    dataset = CorpusDataset.open(corpus.manifest_ref, paths)
    worlds = development_worlds(dataset)
    report = negative_controls(
        worlds, checkpoint=blind.checkpoint, manifest=blind.manifest, spec=blind.spec
    )
    names = {outcome.name for outcome in report.controls}
    assert names == {"shuffled_context", "prefix_12"}
    for outcome in report.controls:
        assert outcome.delta == pytest.approx(0.0, abs=1e-9), outcome
        assert outcome.degraded is False
    assert report.all_degraded is False
    assert set(report.failed) == names


def test_the_controls_actually_perturb_a_context_sensitive_model(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds, negative_controls

    worlds = development_worlds(dataset)
    report = negative_controls(
        worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
    )
    for outcome in report.controls:
        assert outcome.delta != pytest.approx(0.0, abs=1e-9), outcome
    assert report.all_degraded == (not report.failed)


def test_a_prefix_control_no_shorter_than_the_published_view_is_refused(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds, negative_controls

    worlds = development_worlds(dataset)
    with pytest.raises(ValueError, match="prefix"):
        negative_controls(
            worlds,
            checkpoint=frozen.checkpoint,
            manifest=frozen.manifest,
            spec=frozen.spec,
            prefix_months=36,
        )


def test_the_shuffled_control_needs_more_than_one_world(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.validation.proposal_diagnostics import development_worlds, negative_controls

    worlds = development_worlds(dataset)[:1]
    with pytest.raises(ValueError, match="two development parents"):
        negative_controls(
            worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
        )


# --------------------------------------------------------------------------------------
# residual sensitivity, derived from the published pCN move records
# --------------------------------------------------------------------------------------


def _schema() -> DensitySchema:
    return _prior_context().density_schema


def _move(level: int, kernel: str, log_alpha: float | None, accepted: bool) -> dict[str, Any]:
    return {
        "level": level,
        "particle_index": 0,
        "kernel": kernel,
        "accepted": accepted,
        "log_alpha": log_alpha,
        "log_alpha_in_support": log_alpha is not None,
    }


def _checkpoint_manifest(moves: list[dict[str, Any]], beta_history: list[float]) -> dict[str, Any]:
    schema = _schema()
    theta = ThetaRecord(
        schema_id=schema.schema_id,
        s=0,
        v=tuple([0.0] * schema.n_v),
        z_perp=tuple([0.0] * schema.n_residual),
        basis_hash=schema.basis_hash,
    )
    return {
        "schema_version": "e02-smc-checkpoint-1",
        "state": {
            "particles": [
                {
                    "particle_id": 0,
                    "ancestor_id": 0,
                    "evaluation": {"theta": theta.model_dump(mode="json")},
                }
            ],
            "diagnostics": {"moves": moves, "beta_history": beta_history},
        },
    }


def test_residual_sensitivity_is_the_likelihood_drop_the_pcn_moves_record() -> None:
    from so_recon.validation.proposal_diagnostics import residual_l_sensitivity

    # beta_history[level] is the beta in force while that level rejuvenates.
    # For the pCN kernel the proposal ratio cancels the prior/bridge residual factor
    # exactly, so log_alpha = min(0, beta * (log L' - log L)).
    moves = [
        _move(1, "pcn", -1.0, accepted=False),  # beta 0.5 -> drop 2.0
        _move(2, "pcn", -0.25, accepted=False),  # beta 1.0 -> drop 0.25
        _move(2, "pcn", 0.0, accepted=True),  # uphill, recorded as drop 0.0
        _move(2, "pcn", None, accepted=False),  # out of support, not a drop
        _move(2, "rw", -8.0, accepted=False),  # another block entirely
        _move(0, "pcn", -3.0, accepted=False),  # beta 0.0: L is not in the target yet
    ]
    report = residual_l_sensitivity(_checkpoint_manifest(moves, [0.0, 0.5, 1.0]), schema=_schema())
    assert report.n_residual_moves == 5
    assert report.n_scored == 3
    assert report.n_out_of_support == 1
    assert report.n_before_tempering == 1
    assert report.log_l_drops == (0.0, 0.25, 2.0)
    assert report.median_log_l_drop == pytest.approx(0.25)
    assert report.max_log_l_drop == pytest.approx(2.0)
    assert report.acceptance_rate == pytest.approx(1.0 / 5.0)
    assert "pcn" in report.derivation and "log L" in report.derivation


def test_residual_sensitivity_without_a_residual_block_is_refused() -> None:
    from so_recon.validation.proposal_diagnostics import residual_l_sensitivity

    schema = _schema().model_copy(update={"n_residual": 0})
    with pytest.raises(ValueError, match="n_residual"):
        residual_l_sensitivity(_checkpoint_manifest([], [0.0]), schema=schema)


def test_residual_sensitivity_refuses_a_checkpoint_that_records_no_moves() -> None:
    from so_recon.validation.proposal_diagnostics import residual_l_sensitivity

    manifest = _checkpoint_manifest([], [0.0])
    del manifest["state"]["diagnostics"]["moves"]
    with pytest.raises(ValueError, match=r"state\.diagnostics\.moves"):
        residual_l_sensitivity(manifest, schema=_schema())


def test_residual_sensitivity_refuses_a_move_at_a_level_no_beta_covers() -> None:
    from so_recon.validation.proposal_diagnostics import residual_l_sensitivity

    with pytest.raises(ValueError, match="beta_history"):
        residual_l_sensitivity(
            _checkpoint_manifest([_move(4, "pcn", -1.0, accepted=False)], [0.0, 1.0]),
            schema=_schema(),
        )


def test_residual_sensitivity_refuses_a_checkpoint_of_another_layout() -> None:
    from so_recon.validation.proposal_diagnostics import residual_l_sensitivity

    manifest = _checkpoint_manifest([_move(1, "pcn", -1.0, accepted=False)], [0.0, 1.0])
    manifest["state"]["particles"][0]["evaluation"]["theta"]["schema_id"] = "other-layout"
    with pytest.raises(ValueError, match="other-layout"):
        residual_l_sensitivity(manifest, schema=_schema())


# --------------------------------------------------------------------------------------
# measured epoch cost, read through the run record's declared output
# --------------------------------------------------------------------------------------


def _digest(label: str) -> str:
    """A realistic digest: the contract refuses placeholders, and so does a real manifest."""
    return sha256_json({"e03-task-10-fixture": label})


def _training_manifest(*, epochs_run: int, costs: dict[str, float]) -> TrainingManifest:
    weights = b"safetensors-fixture"
    ref = ArtifactRef(
        artifact_id=sha256_bytes(weights),
        path="artifacts/runs/train/weights.safetensors",
        sha256=sha256_bytes(weights),
        size_bytes=len(weights),
        media_type="application/octet-stream",
        schema_version="e03-weights-1",
        producer_run_id="train-1",
        parent_artifact_ids=[],
        created_at="2026-09-17T00:00:00",
    )
    return TrainingManifest(
        run_id="train-1",
        corpus_manifest_ref=ref,
        corpus_manifest_sha256=_digest("corpus_manifest"),
        learning_config_hash=_digest("learning_config"),
        model_config_hash=_digest("model_config"),
        scaler_hash=_digest("scaler"),
        context_spec_hash=_digest("context_spec"),
        train_seed=0,
        optimizer="adamw",
        device="cpu",
        dtype="float32",
        encoder_variant="summary",
        best_epoch=3,
        selected_epoch=3,
        epochs_run=epochs_run,
        metrics={"dev_nll_at_best": 1.25},
        costs=costs,
        checkpoint_ref=ref,
        proposal_manifest_ref=ref,
        torch_version="2.0.0",
        nflows_version="0.14",
    )


def _run_record(paths: ProjectPaths, manifest: TrainingManifest, argv: list[str]) -> RunRecord:
    body = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True).encode("utf-8")
    path = paths.artifacts / "runs" / "train-1" / "training_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    ref = register_artifact(
        path,
        paths,
        schema_version="e03-training-manifest-1",
        producer_run_id="train-1",
        media_type="application/json",
        now=datetime.now(UTC),
    )
    return RunRecord(
        run_id="train-1",
        command="train-proposal",
        argv=argv,
        created_at="2026-09-17T00:00:00",
        finished_at="2026-09-17T01:00:00",
        git_commit="abc",
        git_dirty=False,
        spec_version="1",
        config_version="e03-learning-1",
        resolved_config_hash=_digest("resolved_config"),
        environment_lock_hash=_digest("environment_lock"),
        python_version="3.13.0",
        schema_versions={"e03_training_manifest": "e03-training-manifest-1"},
        raw_input_hashes={"corpus_manifest": _digest("corpus_manifest")},
        parent_run_ids=[],
        status="PASS",
        outputs={"training_manifest": ref},
        notes=[],
    )


def test_epoch_cost_is_the_measured_wall_and_rss_of_the_published_manifest(
    paths: ProjectPaths,
) -> None:
    from so_recon.validation.proposal_diagnostics import epoch_costs

    manifest = _training_manifest(
        epochs_run=4,
        costs={"wall_s": 120.0, "peak_rss_bytes": 2.0e9, "torch_threads": 2.0},
    )
    record = _run_record(paths, manifest, ["train-proposal", "--corpus", "x"])
    costs = epoch_costs(record, paths)
    assert costs.run_id == "train-1"
    assert costs.epochs_run == 4
    assert costs.wall_s == pytest.approx(120.0)
    assert costs.wall_s_per_epoch == pytest.approx(30.0)
    assert costs.peak_rss_bytes == 2_000_000_000
    assert costs.torch_threads == pytest.approx(2.0)
    assert costs.note is None


def test_a_resumed_session_publishes_no_per_epoch_wall(paths: ProjectPaths) -> None:
    from so_recon.validation.proposal_diagnostics import epoch_costs

    manifest = _training_manifest(epochs_run=9, costs={"wall_s": 60.0})
    record = _run_record(
        paths, manifest, ["train-proposal", "--corpus", "x", "--resume", "state.pt"]
    )
    costs = epoch_costs(record, paths)
    assert costs.wall_s == pytest.approx(60.0)
    assert costs.wall_s_per_epoch is None
    assert costs.note is not None and "--resume" in costs.note
    assert costs.peak_rss_bytes is None


def test_a_run_record_without_the_declared_training_manifest_is_refused(
    paths: ProjectPaths,
) -> None:
    from so_recon.validation.proposal_diagnostics import epoch_costs

    manifest = _training_manifest(epochs_run=4, costs={"wall_s": 120.0})
    record = _run_record(paths, manifest, ["train-proposal"])
    stripped = record.model_copy(update={"outputs": {}})
    with pytest.raises(ValueError, match="training_manifest"):
        epoch_costs(stripped, paths)


def test_a_training_manifest_whose_bytes_moved_is_refused(paths: ProjectPaths) -> None:
    from so_recon.validation.proposal_diagnostics import epoch_costs

    manifest = _training_manifest(epochs_run=4, costs={"wall_s": 120.0})
    record = _run_record(paths, manifest, ["train-proposal"])
    paths.resolve(record.outputs["training_manifest"].path).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        epoch_costs(record, paths)


def test_a_training_manifest_without_a_measured_wall_is_refused(paths: ProjectPaths) -> None:
    from so_recon.validation.proposal_diagnostics import epoch_costs

    manifest = _training_manifest(epochs_run=4, costs={"torch_threads": 1.0})
    record = _run_record(paths, manifest, ["train-proposal"])
    with pytest.raises(ValueError, match="wall_s"):
        epoch_costs(record, paths)


# --------------------------------------------------------------------------------------
# §12: this module reads artifacts and starts nothing
# --------------------------------------------------------------------------------------


def test_a_full_diagnostics_pass_spawns_no_native_process(
    dataset: CorpusDataset, frozen: Frozen
) -> None:
    from so_recon.validation.proposal_diagnostics import (
        categorical_support,
        development_nll,
        development_worlds,
        negative_controls,
    )

    with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("no native job")):
        worlds = development_worlds(dataset)
        assert (
            development_nll(
                worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
            ).n_parents
            == 2
        )
        categorical_support(
            worlds,
            checkpoint=frozen.checkpoint,
            manifest=frozen.manifest,
            spec=frozen.spec,
            families_seen_in_train=frozenset(LABELLED_FAMILIES),
        )
        negative_controls(
            worlds, checkpoint=frozen.checkpoint, manifest=frozen.manifest, spec=frozen.spec
        )
