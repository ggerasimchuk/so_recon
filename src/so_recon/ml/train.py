"""E03 Task 06 — the bounded training loop of the learned proposal (plan §5.3, §7.4).

The loop optimises exactly J(phi) of §5.3,

    J(phi) = -(1/J) sum_j [log q_phi(s_j|C_j) + log q_phi(v_j|s_j,C_j)],

per training EXAMPLE j = one view of one parent, with the views of one parent weighted
by their published `ViewRow.weight` (they sum to 1, which IS the plan's normalisation by
view count: more noise copies of a world cannot inflate its weight). The residual
N(0, I) block is never a network target, but it appears in every «difference vs prior»
log through the operational log density `log q(s,v,z|C)`.

Checkpoint selection is development parent-averaged NLL and nothing else; the evaluation
split is not read by anything here (neither the loader call graph nor the selection), so
§6.3 holds by construction on top of the train-only scaler fit of `ml.normalization`.

Resume restores the model, the optimizer, the torch RNG state and the minibatch cursor
(epoch index + position inside the epoch's permutation). Per-epoch permutations are
DERIVED from (training seed, epoch index) rather than consumed from one mutable stream,
so restoring the cursor restores the stream exactly, and a session boundary is invisible
both backwards (identical parameters) and forwards (the next epoch is identical too).

No global torch state leaks: thread count and the torch RNG are saved on entry and
restored on exit of every public entry point of this module.
"""

from __future__ import annotations

import io
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from so_recon.config.learning import LearningConfig, SplitName, TrainingHyperParams
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.inference.contracts import DensitySchema, PriorContext, ThetaRecord
from so_recon.ml.context import build_context_batch
from so_recon.ml.contracts import ContextBatch, ContextSpec, LayoutSupport, ParentRow
from so_recon.ml.dataset import CorpusDataset
from so_recon.ml.flow import DEFAULT_CONDITIONER_HIDDEN_FEATURES, ConditionalNSF
from so_recon.ml.normalization import FeatureScaler
from so_recon.ml.proposal import LearnedProposal, flow_config_payload
from so_recon.registry.atomic import write_bytes_atomic
from so_recon.registry.hashing import sha256_json

TRAINING_STATE_SCHEMA = "e03-training-state-1"

_LOG_TWO_PI = math.log(2.0 * math.pi)

RssSampler = Callable[[], float | None]


class CorruptTrainingState(ValueError):
    """A training state file that is not the torch archive it claims to be."""


def _default_rss_sampler() -> float | None:
    from so_recon.environment.resources import process_tree_usage

    usage = process_tree_usage(os.getpid())
    return None if usage.rss_bytes is None else float(usage.rss_bytes)


# --------------------------------------------------------------------------------------
# the training stream: parents, views, labels, prior references
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingView:
    """One training example: a view of a parent with its tensors and its label context."""

    parent_id: str
    view_id: str
    weight: float
    batch: ContextBatch
    schema: DensitySchema
    theta: ThetaRecord
    prior_log_prob: float
    prior_context_payload: dict[str, Any]


@dataclass(frozen=True)
class TrainingParent:
    """One corpus parent with its views; batches are cut on this boundary."""

    parent_id: str
    split: SplitName
    theta: ThetaRecord
    views: tuple[TrainingView, ...]


def _load_parents(
    dataset: CorpusDataset, spec: ContextSpec, split: SplitName
) -> tuple[TrainingParent, ...]:
    rows: Sequence[ParentRow] = dataset.parents(split)
    parents: list[TrainingParent] = []
    for row in rows:
        payload = dataset.load_context(row)
        context_payload = payload["inference_input"]["context"]
        schema = DensitySchema.model_validate(context_payload["density_schema"])
        schema.validate_theta(row.theta)
        prior = GaussianConditionalPrior(PriorContext.model_validate(context_payload))
        views: list[TrainingView] = []
        for view in row.views:
            if view.noise_copy != 0:
                raise ValueError(
                    f"parent {row.parent_id!r} view {view.view_id!r} is noise copy "
                    f"{view.noise_copy}: the published context stream carries copy 0 only, "
                    "and training on an unpublished copy would silently mislabel it"
                )
            batch = build_context_batch(payload, spec=spec, prefix_months=view.prefix_months)
            views.append(
                TrainingView(
                    parent_id=row.parent_id,
                    view_id=view.view_id,
                    weight=view.weight,
                    batch=batch,
                    schema=schema,
                    theta=row.theta,
                    prior_log_prob=prior.log_prob(row.theta),
                    prior_context_payload=dict(context_payload),
                )
            )
        parents.append(
            TrainingParent(
                parent_id=row.parent_id, split=row.split, theta=row.theta, views=tuple(views)
            )
        )
    return tuple(parents)


@dataclass(frozen=True)
class PreparedTraining:
    """Everything the loop needs that is not the model, the optimizer or the state."""

    spec: ContextSpec
    scaler: FeatureScaler
    scaler_payload: dict[str, Any]
    train_parents: tuple[TrainingParent, ...]
    dev_parents: tuple[TrainingParent, ...]
    n_v: int
    head_families: int
    flow_config: dict[str, Any]
    layouts: tuple[LayoutSupport, ...]
    corpus_manifest_sha256: str
    learning_config_hash: str
    parent_counts: dict[str, int]

    @property
    def scaler_sha256(self) -> str:
        return sha256_json(self.scaler_payload)

    @property
    def flow_config_sha256(self) -> str:
        return sha256_json(self.flow_config)


def prepare_training(
    dataset: CorpusDataset, learning: LearningConfig, *, spec: ContextSpec
) -> PreparedTraining:
    """Load the train/development streams, fit the scaler train-only, freeze the layout.

    The evaluation split is never touched: neither its context files nor its labels are
    read, so it cannot influence scaling, batching or any later choice made here.
    """
    train_parents = _load_parents(dataset, spec, "train")
    dev_parents = _load_parents(dataset, spec, "development")
    if not train_parents:
        raise ValueError("the corpus carries no train parent: there is nothing to fit")
    if not dev_parents:
        raise ValueError(
            "the corpus carries no development parent: checkpoint selection is development "
            "parent-averaged NLL and cannot happen without development parents"
        )
    cutoffs = {
        float(view.batch.well_time.shape[1]) for parent in train_parents for view in parent.views
    } | {float(view.batch.well_time.shape[1]) for parent in dev_parents for view in parent.views}
    if len(cutoffs) > 1:
        raise ValueError(f"parents disagree about the month axis: {sorted(cutoffs)}")

    widths = {view.schema.n_v for parent in train_parents for view in parent.views}
    widths |= {view.schema.n_v for parent in dev_parents for view in parent.views}
    if len(widths) > 1:
        raise ValueError(
            f"the corpus mixes latent widths {sorted(widths)}: one conditional NSF serves one "
            "n_v, so a mixed-layout corpus needs one proposal per layout"
        )
    n_v = widths.pop()
    head_families = (
        max(
            max(view.schema.families)
            for parent in (*train_parents, *dev_parents)
            for view in parent.views
        )
        + 1
    )
    scaler = FeatureScaler.fit(
        [view.batch for parent in train_parents for view in parent.views], spec
    )
    flow_config = flow_config_payload(
        flow=learning.flow,
        encoder_variant=learning.encoder.variant,
        encoder_params=learning.encoder,
        n_well_time_features=len(spec.well_time_features),
        n_edge_features=len(spec.edge_features),
        n_static_features=len(spec.static_features),
        n_v=n_v,
        context_dim=learning.encoder.context_dim,
        hidden_features=DEFAULT_CONDITIONER_HIDDEN_FEATURES,
        n_families=head_families,
    )
    layouts: dict[tuple[str, str], LayoutSupport] = {}
    for parent in (*train_parents, *dev_parents):
        for view in parent.views:
            schema = view.schema
            layouts.setdefault(
                (schema.schema_id, schema.basis_hash),
                LayoutSupport(
                    schema_id=schema.schema_id,
                    n_v=schema.n_v,
                    n_residual=schema.n_residual,
                    families=schema.families,
                    basis_hash=schema.basis_hash,
                ),
            )
    manifest = dataset.manifest
    return PreparedTraining(
        spec=spec,
        scaler=scaler,
        scaler_payload=scaler.payload(),
        train_parents=train_parents,
        dev_parents=dev_parents,
        n_v=n_v,
        head_families=head_families,
        flow_config=flow_config,
        layouts=tuple(layouts.values()),
        corpus_manifest_sha256=dataset.manifest_ref.sha256,
        learning_config_hash=sha256_json(learning.model_dump(mode="json")),
        parent_counts={
            "train": manifest.totals.get("train", 0),
            "development": manifest.totals.get("development", 0),
            "evaluation": manifest.totals.get("evaluation", 0),
        },
    )


def build_proposal_model(
    prepared: PreparedTraining, learning: LearningConfig, *, seed: int
) -> LearnedProposal:
    """The trainable composite, initialised from `seed` without moving torch globals."""
    from so_recon.ml.encoders import build_encoder

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        encoder = build_encoder(
            learning.encoder.variant,
            learning.encoder,
            len(prepared.spec.well_time_features),
            len(prepared.spec.edge_features),
            len(prepared.spec.static_features),
        )
        flow = ConditionalNSF(
            learning.flow,
            n_v=prepared.n_v,
            context_dim=learning.encoder.context_dim,
            n_families=prepared.head_families,
            hidden_features=DEFAULT_CONDITIONER_HIDDEN_FEATURES,
        )
        return LearnedProposal(encoder, flow, n_families=prepared.head_families)
    finally:
        torch.random.set_rng_state(rng_state)


# --------------------------------------------------------------------------------------
# the §5.3 objective on one view
# --------------------------------------------------------------------------------------


def view_nll(
    model: LearnedProposal,
    prepared: PreparedTraining,
    view: TrainingView,
    *,
    theta: ThetaRecord | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(log q(s|C), log q(v|s,C))` of one view's label, as differentiable scalars.

    The categorical factor is the softmax over the families the view's own schema
    DECLARES — exactly the restriction the frozen operational law applies at binding — so
    the number trained is the number the exported proposal reports.
    """
    label = view.theta if theta is None else theta
    scaled = prepared.scaler.transform(view.batch)
    context = model.context_vector(scaled)
    logits = model.head(context)
    families = list(view.schema.families)
    allowed = logits[0, families]
    log_probs = torch.log_softmax(allowed, dim=-1)
    log_s = log_probs[families.index(label.s)]
    dtype = next(model.parameters()).dtype
    v = torch.tensor([tuple(label.v)], dtype=dtype)
    s = torch.tensor([label.s], dtype=torch.int64)
    condition = model.condition_for(context, s)
    log_v = model.flow.log_prob(v, condition)[0]
    return log_s, log_v


def _parent_nll(
    model: LearnedProposal, prepared: PreparedTraining, parent: TrainingParent
) -> torch.Tensor:
    """The view-weighted `-log q` of one parent; view weights sum to one (plan §5.3)."""
    total = torch.zeros((), dtype=next(model.parameters()).dtype)
    for view in parent.views:
        log_s, log_v = view_nll(model, prepared, view)
        total = total + view.weight * (-(log_s + log_v))
    return total


def _log_residual(z_perp: tuple[float, ...]) -> float:
    z = np.fromiter(z_perp, dtype=np.float64)
    return float(-0.5 * float(z @ z) - 0.5 * z.size * _LOG_TWO_PI)


def _evaluate(
    model: LearnedProposal, prepared: PreparedTraining, parents: Sequence[TrainingParent]
) -> dict[str, float]:
    """Parent-averaged NLL decomposition and the operational-vs-prior log density gap."""
    was_training = model.training
    model.eval()
    nll: list[float] = []
    nll_s: list[float] = []
    nll_v: list[float] = []
    gap: list[float] = []
    try:
        with torch.inference_mode():
            for parent in parents:
                parts_s = 0.0
                parts_v = 0.0
                for view in parent.views:
                    log_s, log_v = view_nll(model, prepared, view)
                    parts_s += view.weight * float(log_s)
                    parts_v += view.weight * float(log_v)
                nll_s.append(parts_s)
                nll_v.append(parts_v)
                nll.append(-(parts_s + parts_v))
                log_q = parts_s + parts_v + _log_residual(parent.theta.z_perp)
                gap.append(log_q - parent.views[0].prior_log_prob)
    finally:
        model.train(was_training)
    return {
        "nll": float(np.mean(nll)),
        "nll_s": float(np.mean(nll_s)),
        "nll_v": float(np.mean(nll_v)),
        "log_q_minus_prior": float(np.mean(gap)),
    }


def epoch_permutation(seed: int, epoch: int, n: int) -> np.ndarray:
    """The deterministic minibatch order of one epoch, derived — never carried."""
    return np.random.default_rng([seed, epoch]).permutation(n)


# --------------------------------------------------------------------------------------
# the resumable state
# --------------------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Everything a continuation needs: weights, optimizer, RNG, cursor, selection."""

    model: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    torch_rng: torch.Tensor
    epoch: int
    cursor: int
    best_epoch: int
    best_dev_nll: float
    best_model: dict[str, torch.Tensor]
    epochs_since_best: int
    failed_batches: int
    history: list[dict[str, float]] = field(default_factory=list)
    identity: dict[str, str] = field(default_factory=dict)


def state_identity(prepared: PreparedTraining, learning: LearningConfig) -> dict[str, str]:
    return {
        "schema_version": TRAINING_STATE_SCHEMA,
        "corpus_manifest_sha256": prepared.corpus_manifest_sha256,
        "learning_config_hash": prepared.learning_config_hash,
        "flow_config_sha256": prepared.flow_config_sha256,
        "scaler_sha256": prepared.scaler_sha256,
        "train_seed": str(learning.training_seed),
    }


def save_training_state(path: Path, state: TrainingState) -> None:
    """Write the resumable state atomically; it is a runtime artifact, never a checkpoint."""
    buffer = io.BytesIO()
    torch.save(
        {
            "identity": state.identity,
            "model": state.model,
            "optimizer": state.optimizer,
            "torch_rng": state.torch_rng,
            "epoch": state.epoch,
            "cursor": state.cursor,
            "best_epoch": state.best_epoch,
            "best_dev_nll": state.best_dev_nll,
            "best_model": state.best_model,
            "epochs_since_best": state.epochs_since_best,
            "failed_batches": state.failed_batches,
            "history": state.history,
        },
        buffer,
    )
    write_bytes_atomic(path, buffer.getvalue())


def load_training_state(
    path: Path,
    *,
    prepared: PreparedTraining | None,
    learning: LearningConfig,
) -> TrainingState:
    """Restore a state, refusing a corrupt file and a state trained elsewhere."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CorruptTrainingState(
            f"training state {path} cannot be read as a {TRAINING_STATE_SCHEMA} archive: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or "identity" not in payload:
        raise CorruptTrainingState(f"training state {path} is missing its identity block")
    if prepared is not None:
        expected = state_identity(prepared, learning)
        actual = dict(payload["identity"])
        for key in ("corpus_manifest_sha256", "learning_config_hash", "flow_config_sha256"):
            if actual.get(key) != expected[key]:
                raise ValueError(
                    f"the training state was made against another {key.removesuffix('_sha256')}"
                    f" ({actual.get(key)}, this session would use {expected[key]}): resuming "
                    "across corpora or configs would silently mix two trainings"
                )
    try:
        return TrainingState(
            model=dict(payload["model"]),
            optimizer=dict(payload["optimizer"]),
            torch_rng=payload["torch_rng"],
            epoch=int(payload["epoch"]),
            cursor=int(payload["cursor"]),
            best_epoch=int(payload["best_epoch"]),
            best_dev_nll=float(payload["best_dev_nll"]),
            best_model=dict(payload["best_model"]),
            epochs_since_best=int(payload["epochs_since_best"]),
            failed_batches=int(payload["failed_batches"]),
            history=list(payload["history"]),
            identity=dict(payload["identity"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptTrainingState(
            f"training state {path} does not carry the {TRAINING_STATE_SCHEMA} fields: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# --------------------------------------------------------------------------------------
# the bounded loop
# --------------------------------------------------------------------------------------


@dataclass
class TrainingOutcome:
    """The end of one bounded session: the selected model and the §7.4 logs."""

    state: TrainingState
    history: list[dict[str, float]]
    best_epoch: int
    selected_epoch: int
    epochs_run: int
    stop_reason: str
    failed_batches: int
    costs: dict[str, float]
    dev_nll_at_best: float


def train_proposal(
    model: LearnedProposal,
    prepared: PreparedTraining,
    learning: LearningConfig,
    *,
    state: TrainingState | None = None,
    session_epoch_limit: int | None = None,
    rss_sampler: RssSampler | None = None,
) -> TrainingOutcome:
    """Run bounded epochs of §5.3 with §7.4 settings; leave no global torch state behind.

    `session_epoch_limit` caps THIS session's epochs (the operator's smoke bound); the
    loop's own bounds are `max_epochs` and `patience` of the config. Selection tracks the
    development parent-averaged NLL; the exported checkpoint is always the best epoch.
    """
    hyper: TrainingHyperParams = learning.training
    if hyper.device != "cpu" or hyper.dtype != "float32":
        raise ValueError(
            f"training is CPU Float32 by plan §7.4; the config declares {hyper.device}/"
            f"{hyper.dtype}"
        )
    sample_rss = rss_sampler or _default_rss_sampler
    threads_before = torch.get_num_threads()
    rng_before = torch.random.get_rng_state()
    torch.set_num_threads(hyper.torch_threads)
    started = time.monotonic()
    rss_peak: float | None = sample_rss()

    def observe_rss() -> float | None:
        nonlocal rss_peak
        current = sample_rss()
        if current is not None and (rss_peak is None or current > rss_peak):
            rss_peak = current
        return rss_peak

    try:
        return _run_loop(
            model,
            prepared,
            learning,
            hyper,
            state=state,
            session_epoch_limit=session_epoch_limit,
            rss=observe_rss,
            started=started,
            threads_measured=torch.get_num_threads(),
        )
    finally:
        torch.set_num_threads(threads_before)
        torch.random.set_rng_state(rng_before)


def _cloned_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _run_loop(
    model: LearnedProposal,
    prepared: PreparedTraining,
    learning: LearningConfig,
    hyper: TrainingHyperParams,
    *,
    state: TrainingState | None,
    session_epoch_limit: int | None,
    rss: Callable[[], float | None],
    started: float,
    threads_measured: int,
) -> TrainingOutcome:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyper.learning_rate, weight_decay=hyper.weight_decay
    )
    history: list[dict[str, float]]
    if state is not None:
        model.load_state_dict(state.model, strict=True)
        optimizer.load_state_dict(state.optimizer)
        torch.random.set_rng_state(state.torch_rng)
        best_epoch = state.best_epoch
        best_dev_nll = state.best_dev_nll
        best_model = dict(state.best_model)
        epochs_since_best = state.epochs_since_best
        failed_batches = state.failed_batches
        history = list(state.history)
        start_epoch = state.epoch
        cursor = state.cursor
    else:
        dev_first = _evaluate(model, prepared, prepared.dev_parents)
        best_epoch = 0
        best_dev_nll = dev_first["nll"]
        best_model = _cloned_state(model)
        epochs_since_best = 0
        failed_batches = 0
        history = []
        start_epoch = 0
        cursor = 0

    parents = prepared.train_parents
    stop_reason = "session_limit"
    epochs_this_session = 0
    model.train()
    for epoch in range(start_epoch, hyper.max_epochs):
        if session_epoch_limit is not None and epochs_this_session >= session_epoch_limit:
            stop_reason = "session_limit"
            break
        permutation = epoch_permutation(learning.training_seed, epoch, len(parents))
        epoch_failed = 0
        grad_norms: list[float] = []
        for start in range(cursor * hyper.batch_size, len(parents), hyper.batch_size):
            chunk = [parents[int(index)] for index in permutation[start : start + hyper.batch_size]]
            if not chunk:
                continue
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=next(model.parameters()).dtype)
            for parent in chunk:
                loss = loss + _parent_nll(model, prepared, parent)
            loss = loss / len(chunk)
            if not bool(torch.isfinite(loss)):
                failed_batches += 1
                epoch_failed += 1
                optimizer.zero_grad()
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hyper.gradient_norm_cap)
            if not bool(torch.isfinite(total_norm)):
                failed_batches += 1
                epoch_failed += 1
                optimizer.zero_grad()
                continue
            optimizer.step()
            grad_norms.append(float(total_norm))
        cursor = 0

        dev = _evaluate(model, prepared, prepared.dev_parents)
        train = _evaluate(model, prepared, prepared.train_parents)
        if dev["nll"] < best_dev_nll:
            best_epoch = epoch + 1
            best_dev_nll = dev["nll"]
            best_model = _cloned_state(model)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        epochs_this_session += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_nll": train["nll"],
                "train_nll_s": train["nll_s"],
                "train_nll_v": train["nll_v"],
                "dev_nll": dev["nll"],
                "dev_nll_s": dev["nll_s"],
                "dev_nll_v": dev["nll_v"],
                "dev_log_q_minus_prior": dev["log_q_minus_prior"],
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
                "failed_batches_delta": float(epoch_failed),
                "best_dev_nll": best_dev_nll,
            }
        )
        if epochs_since_best >= hyper.patience:
            stop_reason = "early_stop"
            break
        if epoch + 1 >= hyper.max_epochs:
            stop_reason = "max_epochs"
            break

    wall = time.monotonic() - started
    final_rss = rss()
    final_state = TrainingState(
        model=_cloned_state(model),
        optimizer=optimizer.state_dict(),
        torch_rng=torch.random.get_rng_state(),
        epoch=start_epoch + epochs_this_session,
        cursor=0,
        best_epoch=best_epoch,
        best_dev_nll=best_dev_nll,
        best_model=best_model,
        epochs_since_best=epochs_since_best,
        failed_batches=failed_batches,
        history=history,
        identity=state_identity(prepared, learning),
    )
    costs: dict[str, float] = {
        "wall_s": float(wall),
        "torch_threads": float(threads_measured),
    }
    if final_rss is not None:
        costs["peak_rss_bytes"] = float(final_rss)
    return TrainingOutcome(
        state=final_state,
        history=history,
        best_epoch=best_epoch,
        selected_epoch=best_epoch,
        epochs_run=start_epoch + epochs_this_session,
        stop_reason=stop_reason,
        failed_batches=failed_batches,
        costs=costs,
        dev_nll_at_best=best_dev_nll,
    )


__all__ = [
    "CorruptTrainingState",
    "PreparedTraining",
    "TRAINING_STATE_SCHEMA",
    "TrainingOutcome",
    "TrainingParent",
    "TrainingState",
    "TrainingView",
    "build_proposal_model",
    "epoch_permutation",
    "load_training_state",
    "prepare_training",
    "save_training_state",
    "train_proposal",
    "view_nll",
]
