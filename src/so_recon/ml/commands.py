"""E03 Task 06 — the train-proposal and export-proposal command bodies (plan §14).

Both commands are pure Python on published artifacts: no Julia worker is created, directly
or through config validation, because neither the corpus read path nor the training loop
nor the export touches the native bridge at all.

`train-proposal` runs ONE bounded session against a published corpus manifest, selects
the checkpoint by development parent-averaged NLL, exports the selected weights through
the Task 05 machinery (Float64 safetensors + canonical JSON) and writes the
`TrainingManifest` that ties corpus, scaler, config, costs and checkpoint together. A
session stopped by `--epochs` is resumable: the runtime state beside the artifacts is
`training_state.pt`, never part of the frozen checkpoint.

`export-proposal` re-verifies a published training manifest (digests of all three streams
against the proposal manifest), rebuilds the model strictly, and re-exports the identical
Float64 streams into a target directory — the idempotence of the bytes is itself checked,
so «the same law» is a statement about digests, not about intentions.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from so_recon.config.schema import ProjectConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef, register_artifact, write_json_artifact
from so_recon.registry.run import RunContext, RunStatus
from so_recon.runner import execute_run

EPOCHS_LOG_SCHEMA = "e03-epochs-log-1"

STOP_REASONS = ("session_limit", "early_stop", "max_epochs")


def _now() -> datetime:
    return datetime.now(UTC)


def _register(
    path: Path, paths: ProjectPaths, ctx: RunContext, *, schema_version: str
) -> ArtifactRef:
    return register_artifact(
        path,
        paths,
        schema_version=schema_version,
        producer_run_id=ctx.run_id,
        media_type="application/octet-stream",
        now=_now(),
    )


def _corpus_spec(dataset: Any, learning: Any) -> Any:
    """The one context spec of the corpus, anchored on the first train parent's cutoff."""
    from so_recon.synthetic.inverse_corpus import default_context_spec

    rows = dataset.parents("train")
    if not rows:
        raise ValueError("the corpus carries no train parent to anchor a context spec on")
    payload = dataset.load_context(rows[0])
    return default_context_spec(
        cutoff_s=float(payload["cutoff_s"]),
        max_wells=learning.encoder.max_wells,
        max_months=learning.encoder.max_months,
    )


def _write_proposal(
    ctx: RunContext,
    paths: ProjectPaths,
    *,
    module: Any,
    flow_config: dict[str, Any],
    scaler_payload: dict[str, Any],
    layouts: tuple[Any, ...],
    architecture_version: str,
    context_builder_version: str,
    root: Path,
) -> tuple[ArtifactRef, Any]:
    """Export the Float64 frozen streams and the proposal manifest that vouches for them."""
    from so_recon.ml.checkpoint import (
        FLOW_CONFIG_FILENAME,
        SCALER_FILENAME,
        WEIGHTS_FILENAME,
        save_checkpoint,
    )
    from so_recon.ml.contracts import ProposalManifest

    hashes = save_checkpoint(
        root, module=module, flow_config=flow_config, scaler_payload=scaler_payload
    )
    weights_ref = _register(root / WEIGHTS_FILENAME, paths, ctx, schema_version="e03-weights-1")
    config_ref = _register(root / FLOW_CONFIG_FILENAME, paths, ctx, schema_version="e03-config-1")
    scaler_ref = _register(root / SCALER_FILENAME, paths, ctx, schema_version="e03-scaler-1")
    manifest = ProposalManifest(
        weights_ref=weights_ref,
        weights_sha256=hashes.weights_sha256,
        flow_config_ref=config_ref,
        flow_config_sha256=hashes.flow_config_sha256,
        scaler_ref=scaler_ref,
        scaler_sha256=hashes.scaler_sha256,
        supported_layouts=layouts,
        operational_dtype="float64",
        operational_backend="cpu",
        architecture_version=architecture_version,
        context_builder_version=context_builder_version,
    )
    manifest_ref = write_json_artifact(
        root.parent / "proposal_manifest.json",
        manifest.model_dump(mode="json"),
        paths,
        schema_version="e03-proposal-manifest-1",
        producer_run_id=ctx.run_id,
        now=_now(),
    )
    return manifest_ref, manifest


def run_train_proposal(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    corpus_path: Path,
    epochs: int | None = None,
    resume: Path | None = None,
    argv: Sequence[str] = (),
) -> RunContext:
    """One bounded CPU training session over a published corpus (plan §7.4, §14)."""
    from so_recon.ml.checkpoint import WEIGHTS_FILENAME
    from so_recon.ml.contracts import TRAINING_MANIFEST_SCHEMA, TrainingManifest
    from so_recon.ml.dataset import CorpusDataset
    from so_recon.ml.train import (
        build_proposal_model,
        load_training_state,
        prepare_training,
        save_training_state,
        train_proposal,
    )

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        learning = cfg.learning
        if learning is None:
            raise ValueError("train-proposal requires the validated learning block")
        if epochs is not None and epochs < 1:
            raise ValueError(f"--epochs must be >= 1 when given, got {epochs}")
        source_path = Path(corpus_path)
        resolved = paths.relative(source_path)
        source = paths.resolve(resolved)
        if not source.is_file():
            raise ValueError(f"corpus manifest {resolved} does not exist")
        corpus_ref = _register(source, paths, ctx, schema_version="e03-corpus-manifest-1")
        ctx.update(raw_input_hashes={"corpus_manifest": corpus_ref.sha256})

        dataset = CorpusDataset.open(corpus_ref, paths)
        spec = _corpus_spec(dataset, learning)
        prepared = prepare_training(dataset, learning, spec=spec)
        log.info(
            "corpus %s: train=%d development=%d evaluation=%d views(train)=%d",
            dataset.manifest.experiment_id,
            len(prepared.train_parents),
            len(prepared.dev_parents),
            prepared.parent_counts["evaluation"],
            sum(len(parent.views) for parent in prepared.train_parents),
        )
        model = build_proposal_model(prepared, learning, seed=learning.training_seed)
        state = (
            load_training_state(resume, prepared=prepared, learning=learning)
            if resume is not None
            else None
        )
        if state is not None:
            log.info("resuming from epoch %d (best %d)", state.epoch, state.best_epoch)
        outcome = train_proposal(model, prepared, learning, state=state, session_epoch_limit=epochs)

        model.load_state_dict(outcome.state.best_model, strict=True)
        proposal_dir = ctx.run_dir / "proposal"
        manifest_ref, _proposal = _write_proposal(
            ctx,
            paths,
            module=model,
            flow_config=prepared.flow_config,
            scaler_payload=prepared.scaler_payload,
            layouts=prepared.layouts,
            architecture_version=learning.flow.architecture_version,
            context_builder_version=spec.builder_version,
            root=proposal_dir,
        )
        state_path = ctx.run_dir / "training_state.pt"
        save_training_state(state_path, outcome.state)
        epochs_ref = write_json_artifact(
            ctx.run_dir / "epochs.json",
            {
                "schema_version": EPOCHS_LOG_SCHEMA,
                "stop_reason": outcome.stop_reason,
                "best_epoch": outcome.best_epoch,
                "selected_epoch": outcome.selected_epoch,
                "epochs_run": outcome.epochs_run,
                "epochs": outcome.history,
            },
            paths,
            schema_version=EPOCHS_LOG_SCHEMA,
            producer_run_id=ctx.run_id,
            now=_now(),
        )
        if not outcome.history:
            raise ValueError(
                "the session recorded no epoch: refusing to publish a selection from nothing"
            )
        last = outcome.history[-1]
        metrics: dict[str, float] = {
            "epochs_run": float(outcome.epochs_run),
            "best_epoch": float(outcome.best_epoch),
            "dev_nll_at_best": float(outcome.dev_nll_at_best),
            "train_nll_last": last["train_nll"],
            "train_nll_s_last": last["train_nll_s"],
            "train_nll_v_last": last["train_nll_v"],
            "dev_nll_last": last["dev_nll"],
            "dev_nll_s_last": last["dev_nll_s"],
            "dev_nll_v_last": last["dev_nll_v"],
            "dev_log_q_minus_prior_last": last["dev_log_q_minus_prior"],
            "grad_norm_max_last": last["grad_norm_max"],
            "grad_norm_mean_last": last["grad_norm_mean"],
            "failed_batches": float(outcome.failed_batches),
            "n_train_parents": float(len(prepared.train_parents)),
            "n_development_parents": float(len(prepared.dev_parents)),
            "n_evaluation_parents": float(prepared.parent_counts["evaluation"]),
            "n_train_views": float(sum(len(p.views) for p in prepared.train_parents)),
        }
        manifest = TrainingManifest(
            run_id=ctx.run_id,
            corpus_manifest_ref=corpus_ref,
            corpus_manifest_sha256=corpus_ref.sha256,
            learning_config_hash=prepared.learning_config_hash,
            model_config_hash=prepared.flow_config_sha256,
            scaler_hash=prepared.scaler_sha256,
            context_spec_hash=spec.spec_hash,
            train_seed=learning.training_seed,
            optimizer=learning.training.optimizer,
            device=learning.training.device,
            dtype=learning.training.dtype,
            encoder_variant=learning.encoder.variant,
            best_epoch=outcome.best_epoch,
            selected_epoch=outcome.selected_epoch,
            epochs_run=outcome.epochs_run,
            metrics=metrics,
            costs=dict(outcome.costs),
            checkpoint_ref=_register(
                proposal_dir / WEIGHTS_FILENAME, paths, ctx, schema_version="e03-weights-1"
            ),
            proposal_manifest_ref=manifest_ref,
            torch_version=torch.__version__,
            nflows_version=importlib.metadata.version("nflows"),
        )
        training_ref = write_json_artifact(
            ctx.run_dir / "training_manifest.json",
            manifest.model_dump(mode="json"),
            paths,
            schema_version=TRAINING_MANIFEST_SCHEMA,
            producer_run_id=ctx.run_id,
            now=_now(),
        )
        state_ref = _register(state_path, paths, ctx, schema_version="e03-training-state-1")
        ctx.add_output("proposal_manifest", manifest_ref)
        ctx.add_output("checkpoint_weights", manifest.checkpoint_ref)
        ctx.add_output("training_manifest", training_ref)
        ctx.add_output("training_state", state_ref)
        ctx.add_output("epochs_log", epochs_ref)
        log.info(
            "session stopped by %s after %d epochs; selected epoch %d (dev NLL %.6f)",
            outcome.stop_reason,
            outcome.epochs_run,
            outcome.selected_epoch,
            outcome.dev_nll_at_best,
        )
        if outcome.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {outcome.stop_reason!r}")
        notes = [
            f"stop_reason={outcome.stop_reason}",
            f"epochs_run={outcome.epochs_run} selected_epoch={outcome.selected_epoch}",
            f"dev_nll_at_best={outcome.dev_nll_at_best:.6f}",
            f"failed_batches={outcome.failed_batches}",
            f"wall_s={outcome.costs['wall_s']:.1f}",
        ]
        if "peak_rss_bytes" in outcome.costs:
            notes.append(f"peak_rss_bytes={int(outcome.costs['peak_rss_bytes'])}")
        return "PASS", notes

    return execute_run(
        command="train-proposal",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={
            "e03_training_manifest": "e03-training-manifest-1",
            "e03_proposal_manifest": "e03-proposal-manifest-1",
            "e03_epochs_log": EPOCHS_LOG_SCHEMA,
        },
    )


def run_export_proposal(
    cfg: ProjectConfig,
    paths: ProjectPaths,
    *,
    training_manifest_path: Path,
    out: Path | None = None,
    argv: Sequence[str] = (),
) -> RunContext:
    """Verify a training manifest's checkpoint and re-export the frozen Float64 law."""
    from so_recon.ml.checkpoint import CheckpointHashes, load_checkpoint, save_checkpoint
    from so_recon.ml.contracts import (
        PROPOSAL_MANIFEST_SCHEMA,
        TRAINING_MANIFEST_SCHEMA,
        ProposalManifest,
        TrainingManifest,
    )
    from so_recon.ml.proposal import proposal_from_checkpoint

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        # cfg carries the learning block this command validated; the export itself reads
        # everything it needs from the published training manifest.
        _ = cfg
        manifest_path = Path(training_manifest_path)
        resolved = paths.relative(manifest_path)
        source = paths.resolve(resolved)
        if not source.is_file():
            raise ValueError(f"training manifest {resolved} does not exist")
        training = TrainingManifest.model_validate(json.loads(source.read_text(encoding="utf-8")))
        training_ref = _register(source, paths, ctx, schema_version=TRAINING_MANIFEST_SCHEMA)
        ctx.update(raw_input_hashes={"training_manifest": training_ref.sha256})
        proposal_manifest_path = paths.resolve(training.proposal_manifest_ref.path)
        proposal_manifest = ProposalManifest.model_validate(
            json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
        )
        if proposal_manifest.schema_version != PROPOSAL_MANIFEST_SCHEMA:
            raise ValueError("unexpected proposal manifest schema")
        checkpoint_root = paths.resolve(training.checkpoint_ref.path).parent
        checkpoint = load_checkpoint(
            checkpoint_root,
            expected=CheckpointHashes(
                weights_sha256=proposal_manifest.weights_sha256,
                flow_config_sha256=proposal_manifest.flow_config_sha256,
                scaler_sha256=proposal_manifest.scaler_sha256,
            ),
        )
        # a strict rebuild is part of the export: the streams must reconstruct the model
        module = proposal_from_checkpoint(checkpoint)

        target = out if out is not None else paths.artifacts / "proposals" / training.run_id
        if not target.is_absolute():
            target = paths.root / target
        hashes = save_checkpoint(
            target,
            module=module,
            flow_config=checkpoint.flow_config,
            scaler_payload=checkpoint.scaler_payload,
        )
        if (hashes.weights_sha256, hashes.flow_config_sha256, hashes.scaler_sha256) != (
            proposal_manifest.weights_sha256,
            proposal_manifest.flow_config_sha256,
            proposal_manifest.scaler_sha256,
        ):
            raise ValueError(
                "re-exporting the Float64 checkpoint produced different bytes: the "
                "frozen law is not the idempotent object it must be"
            )
        exported = proposal_manifest.model_copy(update={"training_manifest_ref": training_ref})
        manifest_ref = write_json_artifact(
            target / "proposal_manifest.json",
            exported.model_dump(mode="json"),
            paths,
            schema_version=PROPOSAL_MANIFEST_SCHEMA,
            producer_run_id=ctx.run_id,
            now=_now(),
        )
        ctx.add_output("training_manifest_input", training_ref)
        ctx.add_output("proposal_manifest", manifest_ref)
        log.info(
            "re-exported proposal of run %s to %s (weights %s)",
            training.run_id,
            paths.relative(target),
            hashes.weights_sha256[:12],
        )
        return "PASS", [
            f"training_run={training.run_id}",
            f"selected_epoch={training.selected_epoch}",
            f"weights_sha256={hashes.weights_sha256}",
        ]

    return execute_run(
        command="export-proposal",
        argv=argv,
        cfg=cfg,
        paths=paths,
        body=body,
        schema_versions={"e03_proposal_manifest": "e03-proposal-manifest-1"},
    )


__all__ = [
    "EPOCHS_LOG_SCHEMA",
    "STOP_REASONS",
    "run_export_proposal",
    "run_train_proposal",
]
