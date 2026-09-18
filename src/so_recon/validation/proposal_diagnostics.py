"""E03 Task 10 — the development-set diagnostics that gate freezing a checkpoint.

Five numbers decide whether an architecture/checkpoint may be frozen before evaluation,
and this module produces exactly those five from ALREADY PUBLISHED artifacts:

1. **dev NLL** (plan §7.4): the parent-averaged negative log-likelihood of the label
   under the frozen operational law, decomposed into the categorical `log q(s|C)` and
   continuous `log q(v|s,C)` factors, reported beside `log q(s,v,z|C) - log p0` — the
   difference against the conditional prior the plan's training logs mandate.
2. **categorical support** (plan §7.3): the learned s-probabilities over ALL families
   the world's schema DECLARES. A family the training corpus never labelled is present
   with its probability; dropping it is the defect this diagnostic exists to catch, so
   `families_absent_from_train` names it explicitly.
3. **residual sensitivity** (plan §10.3): how much of the likelihood's response sits on
   the residual coordinates raw q draws from the prior. See the derivation below.
4. **shuffle/prefix controls**: a shuffled-context and a truncated-history variant must
   degrade the dev NLL by a declared margin. `ControlReport.failed` names the controls
   that did not — a control that cannot fail is not a control.
5. **epoch cost**: the MEASURED wall and peak RSS of the published `TrainingManifest`,
   reached through the training run record's declared `outputs["training_manifest"]`.

**§6.3 split isolation.** Nothing here reads an evaluation parent. `development_worlds`
returns development parents only and refuses an evaluation parent NAMED by a caller;
`families_in_split` refuses the evaluation split outright. Truth files are never opened:
the label these diagnostics score is `ParentRow.theta`, the training label of §4.1.

**§10.2 averaging.** Every number is per parent first — view-weighted inside the parent
by the published `ViewRow.weight` — and then averaged over parents with equal world
weight. Views and parents are never pooled into one pseudo-sample.

**§12.** This module starts no solver and no training run. It reads JSON, parquet and
safetensors, and evaluates a CPU Float64 model. The corpus context spec is a parameter,
never built here, so importing this module pulls in no corpus builder.

**The residual derivation (the number Task 08 left absent).** The SMC engine records no
per-move `z_perp`, but it does record every pCN move's `log_alpha`, and for that kernel
the acceptance ratio collapses to the likelihood alone. A pCN move changes only `z`; for
the current whitened layouts (§5.2) both `p0` and the defensive mixture `r` carry the
residual as the SAME factor `N(z; 0, I)`, so the bridge target moves by
`-0.5(|z'|^2 - |z|^2) + beta (log L' - log L)`, while `log_reverse_minus_forward` of the
pCN proposal is exactly `+0.5(|z'|^2 - |z|^2)`. The two cancel, leaving

    log_alpha = min(0, beta * (log L' - log L)).

So `-log_alpha / beta` is the likelihood DROP a prior-scale residual perturbation caused,
read from what the engine already publishes. The measurement is one-sided: an uphill move
is capped at `log_alpha = 0` and enters as a drop of 0.0, so the statistic is the drop
distribution, not the signed response. Levels at `beta = 0` carry no likelihood at all
and are counted, not scored.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from so_recon.config.learning import SplitName
from so_recon.geology.density import GaussianConditionalPrior
from so_recon.inference.contracts import DensitySchema, PriorContext, ThetaRecord
from so_recon.ml.checkpoint import FrozenCheckpoint
from so_recon.ml.context import build_context_batch
from so_recon.ml.contracts import TRAINING_MANIFEST_SCHEMA, ContextSpec, ProposalManifest
from so_recon.ml.contracts import TrainingManifest as TrainingManifestRecord
from so_recon.ml.dataset import CorpusDataset
from so_recon.ml.proposal import FrozenConditionalNSF, bind_frozen_proposal
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_file
from so_recon.registry.run import RunRecord

#: The kernel whose recorded acceptance ratio is the residual likelihood response.
RESIDUAL_KERNEL = "pcn"

#: What `ResidualSensitivity.derivation` states on its face.
RESIDUAL_DERIVATION = (
    "pcn move records: the pCN proposal ratio cancels the residual factor shared by p0 "
    "and r, so log_alpha = min(0, beta * (log L' - log L)) and the drop is -log_alpha/beta "
    "(one-sided: an uphill move is capped at 0)"
)


class EvaluationSplitRefused(ValueError):
    """A diagnostic was pointed at an evaluation parent (plan §6.3)."""


# --------------------------------------------------------------------------------------
# the development set
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevelopmentView:
    """One published view of a development parent: its history prefix and its weight."""

    view_id: str
    prefix_months: int
    weight: float


@dataclass(frozen=True)
class DevelopmentWorld:
    """One development parent with everything a diagnostic needs and nothing more."""

    parent_id: str
    design_id: str
    split: SplitName
    theta: ThetaRecord
    schema: DensitySchema
    context_payload: dict[str, Any]
    views: tuple[DevelopmentView, ...]

    @property
    def prior_log_prob(self) -> float:
        """`log p0(theta | G)` under this world's own published conditional prior."""
        prior = GaussianConditionalPrior(
            PriorContext.model_validate(self.context_payload["inference_input"]["context"])
        )
        return float(prior.log_prob(self.theta))


def development_worlds(
    dataset: CorpusDataset, *, parent_ids: Sequence[str] | None = None
) -> tuple[DevelopmentWorld, ...]:
    """The development parents of a published corpus, in manifest order.

    With `parent_ids`, exactly those parents are loaded and every one of them must be a
    development parent: a train or evaluation parent named here is refused by name, so
    §6.3 cannot be crossed by a typo in a parent list.
    """
    rows = dataset.parents("development")
    if parent_ids is not None:
        by_id = {row.parent_id: row for row in dataset.parents()}
        selected = []
        for parent_id in parent_ids:
            row = by_id.get(parent_id)
            if row is None:
                raise ValueError(
                    f"parent {parent_id!r} is not in corpus {dataset.manifest.experiment_id!r}"
                )
            if row.split != "development":
                raise EvaluationSplitRefused(
                    f"parent {parent_id!r} is a {row.split!r} parent: these diagnostics read "
                    "the development split only, because development artifacts determine "
                    "checkpoint selection (plan §6.3)"
                )
            selected.append(row)
        rows = tuple(selected)
    worlds: list[DevelopmentWorld] = []
    for row in rows:
        if not row.views:
            raise ValueError(
                f"development parent {row.parent_id!r} carries no ParentRow.views: without "
                "a published view there is no history prefix to score it on"
            )
        payload = dataset.load_context(row)
        schema = DensitySchema.model_validate(
            payload["inference_input"]["context"]["density_schema"]
        )
        schema.validate_theta(row.theta)
        worlds.append(
            DevelopmentWorld(
                parent_id=row.parent_id,
                design_id=row.design_id,
                split=row.split,
                theta=row.theta,
                schema=schema,
                context_payload=payload,
                views=tuple(
                    DevelopmentView(
                        view_id=view.view_id,
                        prefix_months=view.prefix_months,
                        weight=view.weight,
                    )
                    for view in row.views
                ),
            )
        )
    return tuple(worlds)


def families_in_split(dataset: CorpusDataset, split: SplitName) -> frozenset[int]:
    """The categories the labels of `split` actually contain; never the evaluation split."""
    if split == "evaluation":
        raise EvaluationSplitRefused(
            "the evaluation split's labels are not a development diagnostic: the census of "
            "categories the corpus labelled is taken on train (plan §6.3)"
        )
    return frozenset(int(row.theta.s) for row in dataset.parents(split))


# --------------------------------------------------------------------------------------
# binding the frozen operational law to one world
# --------------------------------------------------------------------------------------


def _bound(
    world: DevelopmentWorld,
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    prefix_months: int,
    context_from: DevelopmentWorld | None = None,
) -> FrozenConditionalNSF:
    """Freeze the law against one world's context at one prefix.

    `context_from` swaps ONLY the context tensors — the schema, and therefore the latent
    layout every theta is validated against, stays the scored world's own. That is what
    makes the shuffled-context control a context control and not a layout mismatch.
    """
    source = world if context_from is None else context_from
    batch = build_context_batch(source.context_payload, spec=spec, prefix_months=prefix_months)
    return bind_frozen_proposal(
        checkpoint=checkpoint,
        manifest=manifest,
        spec=spec,
        batch=batch,
        schema=world.schema,
    )


@dataclass(frozen=True)
class _Parts:
    """The view-weighted log density parts of one parent."""

    log_q_s: float
    log_q_v: float
    log_q_z: float

    @property
    def nll(self) -> float:
        return -(self.log_q_s + self.log_q_v)


def _view_laws(
    world: DevelopmentWorld,
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    prefix_months: int | None = None,
    context_from: DevelopmentWorld | None = None,
) -> tuple[tuple[float, FrozenConditionalNSF], ...]:
    """One frozen law per published view, paired with that view's weight."""
    return tuple(
        (
            view.weight,
            _bound(
                world,
                checkpoint=checkpoint,
                manifest=manifest,
                spec=spec,
                prefix_months=view.prefix_months if prefix_months is None else prefix_months,
                context_from=context_from,
            ),
        )
        for view in world.views
    )


def _weighted_parts(
    laws: Sequence[tuple[float, FrozenConditionalNSF]], theta: ThetaRecord
) -> _Parts:
    """`(log q(s|C), log q(v|s,C), log p0(z))` averaged over the parent's own views.

    The weights are the published `ViewRow.weight`, which sum to 1 by contract: more
    noise copies or prefixes of one world can never inflate that world (plan §5.3).
    """
    totals = [0.0, 0.0, 0.0]
    for weight, law in laws:
        for index, value in enumerate(law.log_prob_parts(theta)):
            totals[index] += weight * value
    return _Parts(log_q_s=totals[0], log_q_v=totals[1], log_q_z=totals[2])


def _parent_parts(
    world: DevelopmentWorld,
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    prefix_months: int | None = None,
    context_from: DevelopmentWorld | None = None,
) -> _Parts:
    """The view-weighted parts of one parent's own label."""
    return _weighted_parts(
        _view_laws(
            world,
            checkpoint=checkpoint,
            manifest=manifest,
            spec=spec,
            prefix_months=prefix_months,
            context_from=context_from,
        ),
        world.theta,
    )


# --------------------------------------------------------------------------------------
# 1. development NLL
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentNLL:
    """One development parent's view-weighted numbers. Every field is a log density."""

    parent_id: str
    log_q_s: float
    log_q_v: float
    log_q: float
    log_prior: float

    @property
    def nll(self) -> float:
        """The §5.3 objective on this parent: the residual block is not a network target."""
        return -(self.log_q_s + self.log_q_v)

    @property
    def log_q_minus_prior(self) -> float:
        """The operational density's difference from the conditional prior (plan §7.4)."""
        return self.log_q - self.log_prior


@dataclass(frozen=True)
class DevelopmentNLL:
    """The parent-averaged development NLL and its §7.4 decomposition."""

    parents: tuple[ParentNLL, ...]
    nll: float
    nll_categorical: float
    nll_continuous: float
    log_q_minus_prior: float

    @property
    def n_parents(self) -> int:
        return len(self.parents)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("there is no development parent to average over")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def development_nll(
    worlds: Sequence[DevelopmentWorld],
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
) -> DevelopmentNLL:
    """Per-parent dev NLL, then the equal-world-weight average (plan §10.2)."""
    rows: list[ParentNLL] = []
    for world in worlds:
        parts = _parent_parts(world, checkpoint=checkpoint, manifest=manifest, spec=spec)
        rows.append(
            ParentNLL(
                parent_id=world.parent_id,
                log_q_s=parts.log_q_s,
                log_q_v=parts.log_q_v,
                log_q=parts.log_q_s + parts.log_q_v + parts.log_q_z,
                log_prior=world.prior_log_prob,
            )
        )
    return DevelopmentNLL(
        parents=tuple(rows),
        nll=_mean([row.nll for row in rows]),
        nll_categorical=-_mean([row.log_q_s for row in rows]),
        nll_continuous=-_mean([row.log_q_v for row in rows]),
        log_q_minus_prior=_mean([row.log_q_minus_prior for row in rows]),
    )


# --------------------------------------------------------------------------------------
# 2. categorical support over every DECLARED family
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentCategoricalSupport:
    """One parent's learned s-probabilities, one entry per DECLARED family."""

    parent_id: str
    label: int
    probabilities: dict[int, float]


@dataclass(frozen=True)
class CategoricalSupportReport:
    """The §7.3 support check: declared categories, including ones train never held."""

    parents: tuple[ParentCategoricalSupport, ...]
    declared_families: tuple[int, ...]
    families_absent_from_train: tuple[int, ...]
    mean_probability: dict[int, float]
    min_probability_absent_from_train: float | None


def categorical_support(
    worlds: Sequence[DevelopmentWorld],
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    families_seen_in_train: frozenset[int],
) -> CategoricalSupportReport:
    """`q(s|C)` over every family the schema declares, per parent then world-averaged.

    The probabilities are read by scoring the parent's own label RELABELLED to each
    declared family in turn, so the reported vector is the frozen law's own categorical
    factor and cannot silently omit a family that never appeared in training.
    """
    if not worlds:
        raise ValueError("categorical support needs at least one development parent")
    declared = tuple(worlds[0].schema.families)
    rows: list[ParentCategoricalSupport] = []
    for world in worlds:
        if tuple(world.schema.families) != declared:
            raise ValueError(
                f"parent {world.parent_id!r} declares families {tuple(world.schema.families)} "
                f"but {worlds[0].parent_id!r} declares {declared}: one support report "
                "describes one declared support"
            )
        laws = _view_laws(world, checkpoint=checkpoint, manifest=manifest, spec=spec)
        # The view weights average PROBABILITIES here, not log probabilities: a support
        # report whose entries did not sum to one would not be a categorical support.
        probabilities = {int(family): 0.0 for family in declared}
        for weight, law in laws:
            for family in declared:
                relabelled = world.theta.model_copy(update={"s": int(family)})
                log_s, _log_v, _log_z = law.log_prob_parts(relabelled)
                probabilities[int(family)] += weight * float(math.exp(log_s))
        rows.append(
            ParentCategoricalSupport(
                parent_id=world.parent_id,
                label=int(world.theta.s),
                probabilities=probabilities,
            )
        )
    absent = tuple(family for family in declared if family not in families_seen_in_train)
    mean_probability = {
        family: _mean([row.probabilities[family] for row in rows]) for family in declared
    }
    minimum = (
        min(row.probabilities[family] for row in rows for family in absent) if absent else None
    )
    return CategoricalSupportReport(
        parents=tuple(rows),
        declared_families=declared,
        families_absent_from_train=absent,
        mean_probability=mean_probability,
        min_probability_absent_from_train=minimum,
    )


# --------------------------------------------------------------------------------------
# 3. residual sensitivity, from the published pCN move records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualSensitivity:
    """The likelihood response to the residual block raw q draws from the prior."""

    n_residual_moves: int
    n_scored: int
    n_out_of_support: int
    n_before_tempering: int
    log_l_drops: tuple[float, ...]
    median_log_l_drop: float | None
    max_log_l_drop: float | None
    acceptance_rate: float | None
    derivation: str


def _diagnostics_block(checkpoint_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint_manifest.get("state")
    if not isinstance(state, Mapping):
        raise ValueError(
            "the checkpoint manifest carries no 'state' block: residual sensitivity is "
            "read from state.diagnostics.moves of an e02-smc-checkpoint-1 manifest"
        )
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("the checkpoint state carries no 'diagnostics' block")
    return diagnostics


def residual_l_sensitivity(
    checkpoint_manifest: Mapping[str, Any], *, schema: DensitySchema
) -> ResidualSensitivity:
    """The residual likelihood-drop distribution of one SMC run (plan §10.3).

    Reads only what the engine publishes: `state.diagnostics.moves` (kernel, level,
    accepted, log_alpha) and `state.diagnostics.beta_history`. See the module docstring
    for why a pCN `log_alpha` is `min(0, beta * (log L' - log L))` and nothing else.
    """
    if schema.n_residual < 1:
        raise ValueError(
            f"schema {schema.schema_id!r} declares n_residual={schema.n_residual}: with no "
            "residual block there is no residual sensitivity to measure"
        )
    diagnostics = _diagnostics_block(checkpoint_manifest)
    if "moves" not in diagnostics:
        raise ValueError(
            "the checkpoint publishes no state.diagnostics.moves: residual sensitivity is "
            "derived from the recorded pcn acceptance ratios and cannot be invented"
        )
    if "beta_history" not in diagnostics:
        raise ValueError(
            "the checkpoint publishes no state.diagnostics.beta_history: a move's level "
            "cannot be turned into the beta that was in force"
        )
    state = checkpoint_manifest["state"]
    for particle in state.get("particles", ()):
        theta = particle["evaluation"]["theta"]
        if str(theta["schema_id"]) != schema.schema_id:
            raise ValueError(
                f"the checkpoint's particles carry layout {theta['schema_id']!r} but the "
                f"schema passed declares {schema.schema_id!r}: the residual block of one "
                "run cannot be read against another layout"
            )
        if len(theta["z_perp"]) != schema.n_residual:
            raise ValueError(
                f"the checkpoint's particles carry {len(theta['z_perp'])} residual "
                f"coordinates, the schema declares {schema.n_residual}"
            )
    beta_history = [float(value) for value in diagnostics["beta_history"]]
    drops: list[float] = []
    accepted = 0
    out_of_support = 0
    before_tempering = 0
    residual_moves = [
        move for move in diagnostics["moves"] if str(move["kernel"]) == RESIDUAL_KERNEL
    ]
    for move in residual_moves:
        level = int(move["level"])
        if level >= len(beta_history):
            raise ValueError(
                f"a pcn move records level {level} but beta_history has "
                f"{len(beta_history)} entries: the beta in force at that level is unknown"
            )
        if bool(move["accepted"]):
            accepted += 1
        log_alpha = move["log_alpha"]
        if log_alpha is None:
            out_of_support += 1
            continue
        beta = beta_history[level]
        if beta <= 0.0:
            before_tempering += 1
            continue
        drops.append(-float(log_alpha) / beta)
    ordered = tuple(sorted(drops))
    return ResidualSensitivity(
        n_residual_moves=len(residual_moves),
        n_scored=len(ordered),
        n_out_of_support=out_of_support,
        n_before_tempering=before_tempering,
        log_l_drops=ordered,
        median_log_l_drop=float(np.median(ordered)) if ordered else None,
        max_log_l_drop=float(max(ordered)) if ordered else None,
        acceptance_rate=(accepted / len(residual_moves)) if residual_moves else None,
        derivation=RESIDUAL_DERIVATION,
    )


# --------------------------------------------------------------------------------------
# 4. shuffle and prefix controls
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlOutcome:
    """One negative control and whether it actually degraded the diagnostic."""

    name: str
    baseline_nll: float
    control_nll: float
    margin: float

    @property
    def delta(self) -> float:
        """Control minus baseline: positive means the control made the fit worse."""
        return self.control_nll - self.baseline_nll

    @property
    def degraded(self) -> bool:
        return self.delta > self.margin


@dataclass(frozen=True)
class ControlReport:
    """The controls of one checkpoint, with the ones that did not bite named."""

    controls: tuple[ControlOutcome, ...]
    baseline_nll: float

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(outcome.name for outcome in self.controls if not outcome.degraded)

    @property
    def all_degraded(self) -> bool:
        return not self.failed


def control_outcome(
    name: str, *, baseline_nll: float, control_nll: float, margin: float
) -> ControlOutcome:
    """One control's verdict. A control that merely moves has not cleared its margin."""
    if margin < 0.0:
        raise ValueError(f"a control margin is a non-negative NLL gap, got {margin!r}")
    return ControlOutcome(
        name=name, baseline_nll=baseline_nll, control_nll=control_nll, margin=margin
    )


def negative_controls(
    worlds: Sequence[DevelopmentWorld],
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    prefix_months: int = 12,
    margin: float = 0.0,
) -> ControlReport:
    """The shuffled-context and truncated-prefix controls against the dev NLL.

    Shuffled context pairs each world's label with the NEXT world's context tensors; the
    scored world keeps its own schema, so only the evidence moves. The prefix control
    scores every world on a shorter history than its published views declare. Both are
    averaged over parents exactly as the baseline is, and both must clear `margin`.
    """
    if len(worlds) < 2:
        raise ValueError(
            "the shuffled-context control needs at least two development parents: with one "
            "world there is no other context to pair its label with"
        )
    shortest = min(view.prefix_months for world in worlds for view in world.views)
    if prefix_months >= shortest:
        raise ValueError(
            f"the prefix control asks for {prefix_months} months but the shortest published "
            f"view is {shortest}: a control prefix that truncates nothing cannot degrade"
        )
    baseline = development_nll(worlds, checkpoint=checkpoint, manifest=manifest, spec=spec).nll
    shuffled = _mean(
        [
            _parent_parts(
                world,
                checkpoint=checkpoint,
                manifest=manifest,
                spec=spec,
                context_from=worlds[(index + 1) % len(worlds)],
            ).nll
            for index, world in enumerate(worlds)
        ]
    )
    truncated = _mean(
        [
            _parent_parts(
                world,
                checkpoint=checkpoint,
                manifest=manifest,
                spec=spec,
                prefix_months=prefix_months,
            ).nll
            for world in worlds
        ]
    )
    return ControlReport(
        controls=(
            control_outcome(
                "shuffled_context",
                baseline_nll=baseline,
                control_nll=shuffled,
                margin=margin,
            ),
            control_outcome(
                f"prefix_{prefix_months}",
                baseline_nll=baseline,
                control_nll=truncated,
                margin=margin,
            ),
        ),
        baseline_nll=baseline,
    )


# --------------------------------------------------------------------------------------
# 5. the measured epoch cost of the published training run
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochCosts:
    """What one training session actually cost, as its manifest measured it (plan §7.4)."""

    run_id: str
    epochs_run: int
    wall_s: float
    wall_s_per_epoch: float | None
    peak_rss_bytes: int | None
    torch_threads: float | None
    note: str | None


def _declared_output(record: RunRecord, key: str, paths: ProjectPaths) -> Any:
    ref = record.outputs.get(key)
    if ref is None:
        raise ValueError(
            f"run {record.run_id!r} declares no output {key!r}: it publishes "
            f"{sorted(record.outputs)}. A published artifact is addressed by the run "
            "record's declared output, never by a guessed sibling filename"
        )
    path = paths.resolve(ref.path)
    if not path.is_file():
        raise ValueError(f"the artifact run {record.run_id!r} declares at {ref.path} is missing")
    digest = sha256_file(path)
    if digest != ref.sha256:
        raise ValueError(
            f"{ref.path} fails its declared sha256 (got {digest}, the run record vouches "
            f"for {ref.sha256}): the artifact changed after publication"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def epoch_costs(record: RunRecord, paths: ProjectPaths) -> EpochCosts:
    """The measured per-epoch wall and peak RSS of a published training run.

    Read from `TrainingManifest.costs` — the numbers the session sampled — and never
    estimated. A RESUMED session is refused a per-epoch wall: `costs['wall_s']` covers
    that session alone while `epochs_run` counts every epoch since the first, so the
    ratio would not be a per-epoch time. The run record's `argv` is what says so.
    """
    payload = _declared_output(record, "training_manifest", paths)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"the training_manifest run {record.run_id!r} declares is a "
            f"{type(payload).__name__}, not a {TRAINING_MANIFEST_SCHEMA!r} object"
        )
    if payload.get("schema_version") != TRAINING_MANIFEST_SCHEMA:
        raise ValueError(
            f"run {record.run_id!r} declares a training_manifest of schema "
            f"{payload.get('schema_version')!r}, expected {TRAINING_MANIFEST_SCHEMA!r}"
        )
    manifest = TrainingManifestRecord.model_validate(payload)
    if "wall_s" not in manifest.costs:
        raise ValueError(
            f"training manifest of run {manifest.run_id!r} carries no costs['wall_s']: the "
            f"session published {sorted(manifest.costs)}, and an epoch time is measured or "
            "not reported"
        )
    wall_s = float(manifest.costs["wall_s"])
    resumed = "--resume" in record.argv
    note: str | None = None
    per_epoch: float | None = None
    if resumed:
        note = (
            f"run {record.run_id!r} was invoked with --resume: costs['wall_s'] measures this "
            f"session while epochs_run={manifest.epochs_run} counts every epoch since the "
            "first, so no per-epoch wall is derivable from this manifest alone"
        )
    elif manifest.epochs_run < 1:
        note = (
            f"run {record.run_id!r} recorded epochs_run={manifest.epochs_run}: there is no "
            "epoch to divide the session wall by"
        )
    else:
        per_epoch = wall_s / manifest.epochs_run
    rss = manifest.costs.get("peak_rss_bytes")
    threads = manifest.costs.get("torch_threads")
    return EpochCosts(
        run_id=manifest.run_id,
        epochs_run=manifest.epochs_run,
        wall_s=wall_s,
        wall_s_per_epoch=per_epoch,
        peak_rss_bytes=None if rss is None else int(rss),
        torch_threads=None if threads is None else float(threads),
        note=note,
    )


__all__ = [
    "RESIDUAL_DERIVATION",
    "RESIDUAL_KERNEL",
    "CategoricalSupportReport",
    "ControlOutcome",
    "ControlReport",
    "DevelopmentNLL",
    "DevelopmentView",
    "DevelopmentWorld",
    "EpochCosts",
    "EvaluationSplitRefused",
    "ParentCategoricalSupport",
    "ParentNLL",
    "ResidualSensitivity",
    "categorical_support",
    "control_outcome",
    "development_nll",
    "development_worlds",
    "epoch_costs",
    "families_in_split",
    "residual_l_sensitivity",
]
