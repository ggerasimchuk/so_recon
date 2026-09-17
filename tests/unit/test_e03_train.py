"""E03 Task 06 — bounded training and the CPU64 checkpoint export (plan §5.3, §7.4–§7.5).

The tests hold the loop to the properties the plan names, not to any number a good run
would produce:

* the loss is J(phi) of §5.3 — log q(s|C) + log q(v|s,C), categorical restricted to the
  declared families, per-parent view weights normalising the views of one world;
* a tiny controlled overfit moves the loss materially and every gradient is finite;
* resume restores optimizer/RNG/minibatch cursor so a 3+2 session equals a 5 session
  parameter-for-parameter, and the epoch after the join is still identical;
* selection is development parent-averaged NLL only — the evaluation context stream is
  never even read (a poisoned evaluation file cannot stop training);
* a corrupt training state is refused, and so is a state from another corpus;
* importing the commands and validating the smoke config spawns no native process, and
* a full train+export session leaves no global torch state behind it.

The corpus fixture publishes a real digest-verified corpus on disk (labels parquet,
context stream + index, manifest) with no Julia anywhere.
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
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.inference.contracts import DensitySchema, NoiseTheta, PriorContext, ThetaRecord
from so_recon.ml.contracts import CorpusManifest, ParentRow, ViewRow
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact
from so_recon.registry.hashing import sha256_bytes, sha256_json
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import default_context_spec

ROOT = Path(__file__).resolve().parents[2]

WELLS = [("I1", 2, 2, 0), ("I2", 2, 13, 0), ("P1", 13, 2, 1), ("P2", 13, 13, 1)]
N_MONTHS = 36
EDGES = month_edges_s(date(2000, 1, 1), N_MONTHS)
CUTOFF_S = float(EDGES[-1])


def _prior_context(seed: int) -> PriorContext:
    rng = np.random.default_rng(seed)
    n_geology = 12
    mean = 0.3 * rng.standard_normal(n_geology)
    rotation, _ = np.linalg.qr(rng.standard_normal((n_geology, n_geology)))
    rotation = rotation * np.sign(np.diag(rotation))
    root = rotation @ np.diag(np.linspace(0.6, 1.4, n_geology))
    schema = DensitySchema(
        schema_id="p1-conditional-12",
        n_v=11,
        n_residual=4,
        families=(0, 1),
        basis_hash=sha256_json({"schema_id": "p1-conditional-12", "seed": seed}),
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
        g_hash=sha256_json({"g": seed}),
        information_hash=sha256_json({"information": seed}),
    )


def _theta(rng: np.random.Generator, context: PriorContext) -> ThetaRecord:
    schema = context.density_schema
    coordinates = rng.standard_normal(schema.n_v + schema.n_residual)
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=int(rng.integers(0, len(schema.families))),
        v=tuple(coordinates[: schema.n_v].tolist()),
        z_perp=tuple(coordinates[schema.n_v :].tolist()),
        basis_hash=schema.basis_hash,
    )


def _payload(parent_id: str, context: PriorContext, seed: int) -> dict[str, Any]:
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
                    "value": 20.0,
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
                    "bin_index": 31,
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
        "split": "train",
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
    root: Path


def _publish_corpus(
    paths: ProjectPaths, root: Path, *, n_train: int, n_dev: int, n_eval: int
) -> Published:
    """A real digest-verified corpus on disk: labels parquet, context stream, manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    context_dir = root / "context"
    truth_dir = root / "truth"
    context_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "index.json").write_bytes(b'{"files": {}}\n')
    parents: list[ParentRow] = []
    labels_rows: list[dict[str, Any]] = []
    index_files: dict[str, str] = {}
    counts = {"train": n_train, "development": n_dev, "evaluation": n_eval}
    counter = 0
    for split, total in counts.items():
        for _ in range(total):
            context_obj = _prior_context(100 + counter)
            theta = _theta(np.random.default_rng(900 + counter), context_obj)
            parent_id = f"e02-t2-v2-{counter:04d}"
            payload = _payload(parent_id, context_obj, 700 + counter)
            payload["split"] = split
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
                    views=(
                        ViewRow(
                            view_id="m36-copy0",
                            prefix_months=36,
                            noise_copy=0,
                            observation_hash=observation_hash,
                            weight=1.0,
                        ),
                    ),
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
    spec = default_context_spec(cutoff_s=CUTOFF_S, max_wells=4, max_months=36)
    return Published(manifest_ref=manifest_ref, spec=spec, root=root)


def _learning(*, max_epochs: int = 200, patience: int = 20, batch_size: int = 4) -> LearningConfig:
    return LearningConfig(
        corpus=(FamilyPlan(design_id="e02-t2-v2", train=4, development=2, evaluation=1),),
        encoder=EncoderParams(
            variant="summary", width=8, heads=2, context_dim=8, max_wells=4, max_months=36
        ),
        flow=FlowParams(n_transforms=2, n_bins=4),
        training=TrainingHyperParams(
            batch_size=batch_size, max_epochs=max_epochs, patience=patience
        ),
    )


@pytest.fixture()
def paths(tmp_path: Path) -> ProjectPaths:
    project = ProjectPaths.default(tmp_path)
    project.ensure_dirs()
    return project


@pytest.fixture()
def corpus(paths: ProjectPaths, tmp_path: Path) -> Published:
    root = tmp_path / "artifacts" / "corpus" / "tiny-corpus-1"
    return _publish_corpus(paths, root, n_train=4, n_dev=2, n_eval=1)


def _prepared(corpus: Published, paths: ProjectPaths, learning: LearningConfig) -> Any:
    from so_recon.ml.dataset import CorpusDataset
    from so_recon.ml.train import prepare_training

    dataset = CorpusDataset.open(corpus.manifest_ref, paths)
    return prepare_training(dataset, learning, spec=corpus.spec)


# --------------------------------------------------------------------------------------
# §5.3 the loss itself
# --------------------------------------------------------------------------------------


def test_the_loss_is_the_section_53_objective_with_family_masking(
    corpus: Published, paths: ProjectPaths
) -> None:
    from so_recon.ml.train import build_proposal_model, view_nll

    learning = _learning()
    prepared = _prepared(corpus, paths, learning)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)

    parent = prepared.train_parents[0]
    view = parent.views[0]
    schema = view.schema
    assert schema.families == (0, 1)
    # the categorical is the softmax over the DECLARED families of this world's schema
    masses = []
    for s in schema.families:
        theta = ThetaRecord(
            schema_id=parent.theta.schema_id,
            s=s,
            v=parent.theta.v,
            z_perp=parent.theta.z_perp,
            basis_hash=parent.theta.basis_hash,
        )
        log_s, _log_v = view_nll(model, prepared, view, theta=theta)
        masses.append(float(np.exp(float(log_s.detach()))))
    assert float(np.sum(masses)) == pytest.approx(1.0, abs=1e-5)
    # the prior comparison is the parent's own conditional prior on its own label
    log_s, log_v = view_nll(model, prepared, view)
    assert float(log_s.detach()) == pytest.approx(
        float(np.log(masses[schema.families.index(parent.theta.s)]))
    )
    assert view.prior_log_prob == pytest.approx(
        GaussianConditionalPrior(PriorContext.model_validate(view.prior_context_payload)).log_prob(
            parent.theta
        ),
        abs=1e-12,
    )
    assert np.isfinite(float(log_v.detach()))


def test_controlled_overfit_on_a_tiny_batch_decreases_the_loss_materially(
    corpus: Published, paths: ProjectPaths
) -> None:
    from so_recon.ml.train import build_proposal_model, train_proposal

    learning = _learning(max_epochs=40, patience=40, batch_size=4)
    prepared = _prepared(corpus, paths, learning)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    outcome = train_proposal(model, prepared, learning, session_epoch_limit=40)
    first = outcome.history[0]["train_nll"]
    last = outcome.history[-1]["train_nll"]
    assert last < first - 0.25, (first, last)
    # every recorded gradient norm is finite: no step ever ran on a nonfinite batch
    assert all(np.isfinite(row["grad_norm_max"]) for row in outcome.history)
    assert all(row["failed_batches_delta"] == 0 for row in outcome.history)


def test_gradients_of_one_batch_are_finite(corpus: Published, paths: ProjectPaths) -> None:
    from so_recon.ml.train import build_proposal_model, view_nll

    learning = _learning()
    prepared = _prepared(corpus, paths, learning)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    loss = torch.zeros((), dtype=torch.float32)
    for parent in prepared.train_parents:
        for view in parent.views:
            log_s, log_v = view_nll(model, prepared, view)
            loss = loss + view.weight * (-(log_s + log_v))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_a_nonfinite_batch_is_skipped_and_counted(
    corpus: Published, paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    import so_recon.ml.train as train_module
    from so_recon.ml.train import build_proposal_model, train_proposal

    learning = _learning(max_epochs=5, patience=5, batch_size=4)
    prepared = _prepared(corpus, paths, learning)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)

    real_view_nll = train_module.view_nll
    poison_parent = prepared.train_parents[0].parent_id

    def poisoned(model_arg: Any, prepared_arg: Any, view: Any, theta: Any = None) -> Any:
        if view.parent_id == poison_parent:
            return torch.tensor(float("inf")), torch.tensor(0.0)
        return real_view_nll(model_arg, prepared_arg, view, theta=theta)

    monkeypatch.setattr(train_module, "view_nll", poisoned)
    outcome = train_proposal(model, prepared, learning, session_epoch_limit=3)
    assert outcome.failed_batches == 3  # one skipped batch per epoch
    assert all(row["failed_batches_delta"] == 1 for row in outcome.history)
    assert outcome.stop_reason == "session_limit"


# --------------------------------------------------------------------------------------
# resume: optimizer / RNG / minibatch cursor
# --------------------------------------------------------------------------------------


def _flat(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_resume_reproduces_the_uninterrupted_run_exactly(
    corpus: Published, paths: ProjectPaths, tmp_path: Path
) -> None:
    from so_recon.ml.train import (
        build_proposal_model,
        load_training_state,
        save_training_state,
        train_proposal,
    )

    learning = _learning(max_epochs=8, patience=8, batch_size=3)
    prepared = _prepared(corpus, paths, learning)

    straight = build_proposal_model(prepared, learning, seed=learning.training_seed)
    straight_outcome = train_proposal(straight, prepared, learning, session_epoch_limit=5)

    split = build_proposal_model(prepared, learning, seed=learning.training_seed)
    first = train_proposal(split, prepared, learning, session_epoch_limit=3)
    state_path = tmp_path / "training_state.pt"
    save_training_state(state_path, first.state)
    resumed_model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    state = load_training_state(state_path, prepared=prepared, learning=learning)
    assert state.epoch == 3 and state.cursor == 0
    second = train_proposal(resumed_model, prepared, learning, state=state, session_epoch_limit=2)

    assert [row["epoch"] for row in second.history[-2:]] == [4.0, 5.0]
    assert [row["dev_nll"] for row in second.history[-2:]] == [
        row["dev_nll"] for row in straight_outcome.history[3:5]
    ]
    for name, tensor in _flat(resumed_model).items():
        assert torch.equal(tensor, _flat(straight)[name]), name
    # the join is invisible to the FUTURE too: one more epoch stays identical
    more_straight = train_proposal(
        straight, prepared, learning, state=straight_outcome.state, session_epoch_limit=1
    )
    more_resumed = train_proposal(
        resumed_model, prepared, learning, state=second.state, session_epoch_limit=1
    )
    assert [row["train_nll"] for row in more_resumed.history] == [
        row["train_nll"] for row in more_straight.history
    ]
    assert more_resumed.state.epoch == more_straight.state.epoch
    for name, tensor in _flat(resumed_model).items():
        assert torch.equal(tensor, _flat(straight)[name]), name


def test_a_state_from_another_corpus_is_refused(
    corpus: Published, paths: ProjectPaths, tmp_path: Path
) -> None:
    from so_recon.ml.train import (
        build_proposal_model,
        load_training_state,
        save_training_state,
        train_proposal,
    )

    learning = _learning(max_epochs=4, patience=4)
    prepared = _prepared(corpus, paths, learning)
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    outcome = train_proposal(model, prepared, learning, session_epoch_limit=1)
    state_path = tmp_path / "training_state.pt"
    save_training_state(state_path, outcome.state)

    other_root = tmp_path / "artifacts" / "corpus" / "other-corpus-1"
    other = _publish_corpus(paths, other_root, n_train=4, n_dev=2, n_eval=1)
    other_prepared = _prepared(other, paths, learning)
    with pytest.raises(ValueError, match="corpus"):
        load_training_state(state_path, prepared=other_prepared, learning=learning)


def test_a_corrupt_state_file_is_refused(tmp_path: Path) -> None:
    from so_recon.ml.train import CorruptTrainingState, load_training_state

    path = tmp_path / "training_state.pt"
    path.write_bytes(b"this is not a torch archive")
    with pytest.raises(CorruptTrainingState):
        load_training_state(path, prepared=None, learning=_learning())


# --------------------------------------------------------------------------------------
# split discipline
# --------------------------------------------------------------------------------------


def test_the_evaluation_stream_is_never_read_and_selection_follows_dev_nll(
    corpus: Published, paths: ProjectPaths
) -> None:
    from so_recon.ml.dataset import CorpusDataset
    from so_recon.ml.train import build_proposal_model, prepare_training, train_proposal

    learning = _learning(max_epochs=6, patience=6, batch_size=4)
    dataset = CorpusDataset.open(corpus.manifest_ref, paths)
    prepared = prepare_training(dataset, learning, spec=corpus.spec)
    assert len(prepared.train_parents) == 4
    assert len(prepared.dev_parents) == 2

    # poison the evaluation context AFTER preparation: training may not read it at all
    eval_rows = dataset.parents("evaluation")
    assert len(eval_rows) == 1
    eval_file = corpus.root / "context" / f"{eval_rows[0].parent_id}.json"
    poisoned = eval_file.read_bytes()
    eval_file.write_bytes(b'{"parent_id": "poisoned"}')

    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    outcome = train_proposal(model, prepared, learning, session_epoch_limit=4)
    dev_values = [row["dev_nll"] for row in outcome.history]
    # selection is the development parent-averaged NLL: the chosen epoch holds the
    # minimum over the epoch-0 initial state and every epoch that ran
    if outcome.selected_epoch == 0:
        assert outcome.dev_nll_at_best <= min(dev_values)
    else:
        assert dev_values[outcome.selected_epoch - 1] == min(dev_values)
    assert outcome.best_epoch == outcome.selected_epoch
    assert outcome.stop_reason == "session_limit"

    eval_file.write_bytes(poisoned)


def test_epoch_permutations_are_derived_distinct_and_reproducible() -> None:
    from so_recon.ml.train import epoch_permutation

    first = epoch_permutation(9101, 0, 8)
    second = epoch_permutation(9101, 1, 8)
    assert sorted(first.tolist()) == list(range(8))
    assert not np.array_equal(first, second)
    assert np.array_equal(epoch_permutation(9101, 0, 8), first)
    assert not np.array_equal(epoch_permutation(9102, 0, 8), first)


def test_every_view_carries_its_own_split_and_train_only_scaler(
    corpus: Published, paths: ProjectPaths
) -> None:
    prepared = _prepared(corpus, paths, _learning())
    for parent in (*prepared.train_parents, *prepared.dev_parents):
        for view in parent.views:
            assert view.batch.split == parent.split
    assert prepared.scaler_payload["schema_version"] == "e03-scaler-1"


# --------------------------------------------------------------------------------------
# commands: no native job, manifest honesty, export reload identity
# --------------------------------------------------------------------------------------


@pytest.fixture()
def ml_project(tmp_project: Path) -> Path:
    config = tmp_project / "configs" / "e03_smoke.yml"
    config.write_text((ROOT / "configs" / "e03_smoke.yml").read_text(encoding="utf-8"))
    return tmp_project


def _no_native_jobs() -> Any:
    return mock.patch.object(subprocess, "Popen", side_effect=AssertionError("no native job"))


def _no_julia_jobs() -> Any:
    """Refuse any julia process spawn while letting the run registry run git."""
    real_popen = subprocess.Popen

    def guard(*args: Any, **kwargs: Any) -> Any:
        command = args[0] if args else kwargs.get("args")
        text = " ".join(map(str, command)) if isinstance(command, (list, tuple)) else str(command)
        if "julia" in text.lower():
            raise AssertionError(f"no native julia job may spawn: {text[:120]}")
        return real_popen(*args, **kwargs)

    return mock.patch.object(subprocess, "Popen", side_effect=guard)


def test_import_and_config_validation_spawn_no_native_job(ml_project: Path) -> None:
    with _no_native_jobs():
        import so_recon.cli  # noqa: F401
        from so_recon.config.load import load_project_config

        cfg = load_project_config(ml_project / "configs" / "e03_smoke.yml")
        assert cfg.learning is not None
        from so_recon.ml.commands import run_export_proposal, run_train_proposal  # noqa: F401


def test_train_and_export_cli_publish_a_reloadable_frozen_proposal(
    ml_project: Path, tmp_path: Path
) -> None:
    from so_recon.cli import main

    paths = ProjectPaths.default(ml_project)
    root = ml_project / "artifacts" / "corpus" / "tiny-corpus-1"
    published = _publish_corpus(paths, root, n_train=4, n_dev=2, n_eval=1)
    with _no_julia_jobs():
        code = main(
            [
                "--root",
                str(ml_project),
                "--config",
                str(ml_project / "configs" / "e03_smoke.yml"),
                "train-proposal",
                "--corpus",
                str(root / "corpus_manifest.json"),
                "--epochs",
                "2",
            ]
        )
    assert code == 0
    run_dirs = sorted((ml_project / "artifacts" / "runs").iterdir())
    training_dirs = [d for d in run_dirs if (d / "training_manifest.json").is_file()]
    assert len(training_dirs) == 1
    manifest = json.loads((training_dirs[0] / "training_manifest.json").read_text())
    assert manifest["selected_epoch"] <= manifest["epochs_run"]
    assert manifest["device"] == "cpu" and manifest["dtype"] == "float32"
    assert manifest["metrics"]["n_train_parents"] == 4
    assert manifest["metrics"]["n_development_parents"] == 2
    epochs_log = json.loads((training_dirs[0] / "epochs.json").read_text())
    assert len(epochs_log["epochs"]) == 2
    assert epochs_log["stop_reason"] == "session_limit"
    assert {
        "train_nll",
        "train_nll_s",
        "train_nll_v",
        "dev_nll",
        "dev_nll_s",
        "dev_nll_v",
        "dev_log_q_minus_prior",
        "grad_norm_mean",
        "grad_norm_max",
    } <= set(epochs_log["epochs"][0])

    out = tmp_path / "exported"
    with _no_julia_jobs():
        code = main(
            [
                "--root",
                str(ml_project),
                "--config",
                str(ml_project / "configs" / "e03_smoke.yml"),
                "export-proposal",
                "--training-manifest",
                str(training_dirs[0] / "training_manifest.json"),
                "--out",
                str(out),
            ]
        )
    assert code == 0

    from safetensors.numpy import load as load_tensors

    tensors = load_tensors((out / "weights.safetensors").read_bytes())
    floating = [array for array in tensors.values() if np.issubdtype(array.dtype, np.floating)]
    assert tensors and floating
    # Float32 training exports AS Float64 (plan §7.5); integer buffers stay integers
    assert all(array.dtype == np.float64 for array in floating)
    proposal_manifest = json.loads((out / "proposal_manifest.json").read_text())
    assert proposal_manifest["operational_dtype"] == "float64"
    assert proposal_manifest["operational_backend"] == "cpu"
    assert proposal_manifest["supported_layouts"]

    # reload identity through the Task 05 machinery: the re-export is the same law
    from so_recon.ml.checkpoint import load_checkpoint
    from so_recon.ml.context import build_context_batch
    from so_recon.ml.contracts import ProposalManifest
    from so_recon.ml.dataset import CorpusDataset
    from so_recon.ml.proposal import bind_frozen_proposal

    checkpoint_a = load_checkpoint(training_dirs[0] / "proposal")
    checkpoint_b = load_checkpoint(out)
    manifest_b = ProposalManifest.model_validate(proposal_manifest)
    dataset = CorpusDataset.open(published.manifest_ref, paths)
    dev_row = next(row for row in dataset.parents() if row.split == "development")
    payload = json.loads((root / "context" / f"{dev_row.parent_id}.json").read_text())
    # the spec the SESSION anchored: encoder dims come from the smoke config, not the
    # fixture's smaller test spec (the manifest's context_spec_hash is its fingerprint)
    from so_recon.config.load import load_project_config

    session_cfg = load_project_config(ml_project / "configs" / "e03_smoke.yml")
    assert session_cfg.learning is not None
    bind_spec = default_context_spec(
        cutoff_s=CUTOFF_S,
        max_wells=session_cfg.learning.encoder.max_wells,
        max_months=session_cfg.learning.encoder.max_months,
    )
    batch = build_context_batch(payload, spec=bind_spec)
    schema = DensitySchema.model_validate(payload["inference_input"]["context"]["density_schema"])
    frozen_a = bind_frozen_proposal(
        checkpoint=checkpoint_a, manifest=manifest_b, spec=bind_spec, batch=batch, schema=schema
    )
    frozen_b = bind_frozen_proposal(
        checkpoint=checkpoint_b, manifest=manifest_b, spec=bind_spec, batch=batch, schema=schema
    )
    draws_a = frozen_a.sample(8, np.random.default_rng(5))
    draws_b = frozen_b.sample(8, np.random.default_rng(5))
    assert draws_a == draws_b
    theta = draws_a[0]
    assert frozen_a.log_prob(theta) == frozen_b.log_prob(theta)
    assert frozen_a.fingerprint == frozen_b.fingerprint


def test_train_cli_refuses_a_missing_corpus_with_a_fail_record(ml_project: Path) -> None:
    from so_recon.cli import main

    with _no_julia_jobs():
        code = main(
            [
                "--root",
                str(ml_project),
                "--config",
                str(ml_project / "configs" / "e03_smoke.yml"),
                "train-proposal",
                "--corpus",
                str(ml_project / "artifacts" / "corpus" / "missing" / "corpus_manifest.json"),
                "--epochs",
                "1",
            ]
        )
    assert code == 1
    run_dirs = sorted((ml_project / "artifacts" / "runs").iterdir())
    record = json.loads((run_dirs[-1] / "run.json").read_text())
    assert record["status"] == "FAIL"
    assert record["command"] == "train-proposal"


def test_a_training_session_leaves_no_global_torch_state_behind(
    corpus: Published, paths: ProjectPaths
) -> None:
    from so_recon.ml.train import build_proposal_model, train_proposal

    learning = _learning(max_epochs=2, patience=2)
    prepared = _prepared(corpus, paths, learning)
    threads_before = torch.get_num_threads()
    rng_state_before = torch.random.get_rng_state()
    model = build_proposal_model(prepared, learning, seed=learning.training_seed)
    train_proposal(model, prepared, learning, session_epoch_limit=2)
    assert torch.get_num_threads() == threads_before
    assert torch.equal(torch.random.get_rng_state(), rng_state_before)
