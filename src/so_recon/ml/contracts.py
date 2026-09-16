"""E03 ML contracts (plan §4.2): the five records the learned loop adds.

Nothing here duplicates an existing type: `ArtifactRef`, `RunContext`, `ThetaRecord`
and the registry stay the single sources they already are. What E03 needs and E02 did
not have is a provenance chain for a TRAINED object — corpus, features, training run,
frozen proposal, comparison protocol — and it is typed here so a mismatched pair (a
checkpoint scored against the context of another world, a proposal bound to another
basis) is refused at construction instead of discovered in the numbers.

The three data streams of plan §4.1 stay separate in these records: `ContextBatch`
carries only inference inputs, `ParentRow.theta` is the training label, and truth
fields live on the corpus parent row for the evaluator alone.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, Field, field_validator, model_validator

from so_recon.config.learning import SplitName
from so_recon.config.schema import StrictModel
from so_recon.inference.contracts import NoiseTheta, ThetaRecord
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_json
from so_recon.simulator.contracts import Sha256

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]

CORPUS_MANIFEST_SCHEMA = "e03-corpus-manifest-1"
LABELS_SCHEMA = "e03-labels-1"
CONTEXT_SPEC_SCHEMA = "e03-context-spec-1"
TRAINING_MANIFEST_SCHEMA = "e03-training-manifest-1"
PROPOSAL_MANIFEST_SCHEMA = "e03-proposal-manifest-1"
COMPARISON_PROTOCOL_SCHEMA = "e03-comparison-protocol-1"


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------


class ViewRow(StrictModel):
    """One derived view of a parent (plan §6.3): a history prefix and/or noise copy."""

    view_id: str = Field(min_length=1)
    prefix_months: int = Field(ge=1)
    noise_copy: int = Field(ge=0)
    observation_hash: Sha256
    weight: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _identity_matches_content(self) -> ViewRow:
        segments = self.view_id.split("-")
        for token in (f"m{self.prefix_months}", f"copy{self.noise_copy}"):
            if token not in segments:
                raise ValueError(
                    f"view_id {self.view_id!r} must name its prefix and noise copy so a row "
                    "cannot be re-labelled after the fact"
                )
        return self


class ParentRow(StrictModel):
    """One corpus parent: identity, split, labels, noise and artifact references.

    `theta` is the TRAINING LABEL of plan §4.1 — the draw the network is fit to. It is
    stored here (and in `labels.parquet`) for the training loader; the evaluator reads
    the full physical truth from `truth_ref` instead. `noise` must be the noise the
    history was GENERATED with, which plan §0.2 makes the noise of this very theta.
    """

    parent_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    split: SplitName
    truth_seed: int = Field(ge=0)
    history_seed: int = Field(ge=0)
    theta: ThetaRecord
    noise: NoiseTheta
    schema_id: str = Field(min_length=1)
    basis_hash: Sha256
    observation_hash: Sha256
    model_hash: Sha256
    context_ref: ArtifactRef
    truth_ref: ArtifactRef
    views: tuple[ViewRow, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _label_and_schema_agree(self) -> ParentRow:
        if self.theta.schema_id != self.schema_id or self.theta.basis_hash != self.basis_hash:
            raise ValueError(
                f"parent {self.parent_id!r}: label names {self.theta.schema_id!r}/"
                f"{self.theta.basis_hash} but the row declares {self.schema_id!r}/"
                f"{self.basis_hash}"
            )
        if not self.views and self.split == "train":
            raise ValueError(
                f"train parent {self.parent_id!r} carries no view: without at least one "
                "view row a training loader cannot know what history it owns"
            )
        total = float(sum(view.weight for view in self.views))
        if self.views and abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"parent {self.parent_id!r}: view weights must sum to 1 within 1e-9 so "
                "noise copies cannot inflate a world (plan §5.3), got "
                f"{total!r}"
            )
        return self


class FailureRow(StrictModel):
    """A world that did not complete, with its reason. Failed draws never disappear."""

    parent_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    split: SplitName
    truth_seed: int = Field(ge=0)
    stage: Literal[
        "context",
        "draw",
        "render",
        "forward",
        "prediction",
        "history",
        "publish",
    ]
    reason: str = Field(min_length=1)
    attempts: int = Field(ge=1)


class CorpusManifest(StrictModel):
    """The published identity of one corpus build (plan §4.2).

    `expected/complete/failed` keep the accounting honest: a corpus that lost worlds
    silently cannot happen, because the counts and the per-parent rows are written
    together and re-validated on load.
    """

    schema_version: Literal[CORPUS_MANIFEST_SCHEMA] = CORPUS_MANIFEST_SCHEMA
    experiment_id: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    namespace: Literal["scientific", "thin_slice"]
    learning_config_hash: Sha256
    design_distribution: dict[str, int]
    totals: dict[str, int]
    expected: int = Field(ge=1)
    complete: int = Field(ge=0)
    failed: int = Field(ge=0)
    parents: tuple[ParentRow, ...]
    failures: tuple[FailureRow, ...] = Field(default_factory=tuple)
    labels_ref: ArtifactRef
    context_dir_ref: ArtifactRef
    truth_dir_ref: ArtifactRef
    schemas: dict[str, Sha256]
    basis_hashes: dict[str, Sha256]
    source_commit: str = Field(min_length=1)
    source_dirty: bool

    @model_validator(mode="after")
    def _counts_account_for_every_parent(self) -> CorpusManifest:
        if self.complete != len(self.parents):
            raise ValueError(
                f"complete={self.complete} but the manifest lists {len(self.parents)} "
                "parents: the count and the rows are written together"
            )
        if self.failed != len(self.failures):
            raise ValueError(
                f"failed={self.failed} but the manifest lists {len(self.failures)} "
                "failure rows"
            )
        if self.expected < self.complete + self.failed:
            raise ValueError(
                f"expected={self.expected} is below complete+failed="
                f"{self.complete + self.failed}: rows appeared from nowhere"
            )
        splits: dict[str, int] = {"train": 0, "development": 0, "evaluation": 0}
        for parent in self.parents:
            splits[parent.split] += 1
        for name, count in splits.items():
            if count > self.totals.get(name, 0):
                raise ValueError(
                    f"{count} {name} parents exceed the planned total "
                    f"{self.totals.get(name, 0)}"
                )
        return self


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------


class ContextSpec(StrictModel):
    """The typed allowlist of encoder inputs (plan §4.1, §7.1), frozen per corpus.

    `well_time_features`, `static_features` and `edge_features` are ordered tuples:
    `ContextBatch` tensors follow exactly this order, and the encoder hash covers the
    spec hash so a model trained on another feature contract cannot be bound.
    """

    schema_version: Literal[CONTEXT_SPEC_SCHEMA] = CONTEXT_SPEC_SCHEMA
    builder_version: str = Field(min_length=1)
    well_time_features: tuple[str, ...]
    static_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    control_kinds: tuple[str, ...]
    units: dict[str, str]
    cutoff_s: float = Field(gt=0.0)
    max_wells: int = Field(ge=1)
    max_months: int = Field(ge=1)
    allowed_context_keys: tuple[str, ...]

    @field_validator("well_time_features", "static_features", "edge_features")
    @classmethod
    def _features_are_unique_and_ordered(
        cls, value: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} must not repeat a feature name")
        return value

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


FORBIDDEN_CONTEXT_FEATURES: frozenset[str] = frozenset(
    {
        "truth",
        "theta",
        "seed",
        "basis_hash",
        "s",
        "v",
        "z_perp",
        "full_permeability",
        "full_porosity",
        "full_sw",
        "full_so",
        "full_pressure",
        "achieved_hidden_rate",
        "true_connections",
    }
)


class ContextBatch(StrictModel):
    """One world's inference inputs as tensors (plan §4.1): nothing else travels here.

    Identity (`parent_id`, `split`, `design_id`, hashes) stays OUTSIDE the tensors —
    per-world metadata is metadata, never an embedding (plan §7.1).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    split: SplitName
    spec_hash: str = Field(min_length=1)
    well_time: F64
    well_mask: BOOL
    static: F64
    edge_index: I64
    edge_attr: F64
    well_ids: tuple[str, ...]

    @field_validator("well_time", "well_mask", "static", "edge_index", "edge_attr")
    @classmethod
    def _arrays_are_read_only(cls, value: npt.NDArray[Any], info: Any) -> npt.NDArray[Any]:
        array = np.array(value, copy=True)
        if info.field_name in ("well_time", "static", "edge_attr"):
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"{info.field_name} must be finite everywhere")
        array.setflags(write=False)
        return array

    @model_validator(mode="after")
    def _shapes_agree_with_the_spec_and_each_other(self) -> ContextBatch:
        n_wells, n_months, n_features = self.well_time.shape
        if n_wells == 0 or n_months == 0 or n_features == 0:
            raise ValueError("well_time must be non-empty in every dimension")
        if self.well_mask.shape != (n_wells, n_months):
            raise ValueError(
                f"well_mask shape {self.well_mask.shape} does not match well_time "
                f"{(n_wells, n_months)}"
            )
        if self.well_ids and len(self.well_ids) != n_wells:
            raise ValueError(
                f"{len(self.well_ids)} well ids for {n_wells} tensor rows"
            )
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError(
                f"edge_attr has {self.edge_attr.shape[0]} rows for "
                f"{self.edge_index.shape[1]} edges"
            )
        if self.edge_index.size and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n_wells
        ):
            raise ValueError("edge_index refers to a well that is not in the batch")
        return self


# --------------------------------------------------------------------------------------
# training and the frozen proposal
# --------------------------------------------------------------------------------------


class TrainingManifest(StrictModel):
    """One bounded training run and the checkpoint it selected (plan §7.4)."""

    schema_version: Literal[TRAINING_MANIFEST_SCHEMA] = TRAINING_MANIFEST_SCHEMA
    run_id: str = Field(min_length=1)
    corpus_manifest_ref: ArtifactRef
    corpus_manifest_sha256: Sha256
    learning_config_hash: Sha256
    model_config_hash: Sha256
    scaler_hash: Sha256
    context_spec_hash: Sha256
    train_seed: int = Field(ge=0)
    optimizer: str = Field(min_length=1)
    device: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    encoder_variant: str = Field(min_length=1)
    best_epoch: int = Field(ge=0)
    selected_epoch: int = Field(ge=0)
    epochs_run: int = Field(ge=0)
    metrics: dict[str, float]
    costs: dict[str, float]
    checkpoint_ref: ArtifactRef
    proposal_manifest_ref: ArtifactRef
    torch_version: str = Field(min_length=1)
    nflows_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _selection_is_within_what_ran(self) -> TrainingManifest:
        if self.selected_epoch > self.epochs_run or self.best_epoch > self.epochs_run:
            raise ValueError(
                f"selected/best epoch {self.selected_epoch}/{self.best_epoch} beyond the "
                f"{self.epochs_run} epochs that ran: a checkpoint cannot be chosen from "
                "a future state"
            )
        for name, value in self.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"metrics[{name!r}] must be finite, got {value}")
        return self


class LayoutSupport(StrictModel):
    """One latent layout a frozen proposal may serve (plan §3.4)."""

    schema_id: str = Field(min_length=1)
    n_v: int = Field(ge=1)
    n_residual: int = Field(ge=0)
    families: tuple[int, ...]
    basis_hash: Sha256


class ProposalManifest(StrictModel):
    """The frozen operational density: weights, config, scaler and allowed layouts."""

    schema_version: Literal[PROPOSAL_MANIFEST_SCHEMA] = PROPOSAL_MANIFEST_SCHEMA
    weights_ref: ArtifactRef
    weights_sha256: Sha256
    flow_config_ref: ArtifactRef
    flow_config_sha256: Sha256
    scaler_ref: ArtifactRef
    scaler_sha256: Sha256
    supported_layouts: tuple[LayoutSupport, ...]
    operational_dtype: Literal["float64"]
    operational_backend: Literal["cpu"]
    architecture_version: str = Field(min_length=1)
    context_builder_version: str = Field(min_length=1)
    training_manifest_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def _layouts_are_distinguishable(self) -> ProposalManifest:
        keys = [
            (layout.schema_id, layout.basis_hash) for layout in self.supported_layouts
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("supported_layouts repeat a (schema_id, basis_hash) pair")
        if not self.supported_layouts:
            raise ValueError("a proposal must declare at least one supported layout")
        return self


# --------------------------------------------------------------------------------------
# comparison protocol
# --------------------------------------------------------------------------------------

MethodId = Literal["B0", "B1", "Q", "M", "A_summary", "A_set"]
METHOD_IDS: tuple[MethodId, ...] = ("B0", "B1", "Q", "M", "A_summary", "A_set")


class ComparisonMethod(StrictModel):
    """One row of the §10.1 method table: what starts the SMC and what it claims."""

    method_id: MethodId
    start: Literal["static_prior", "raw_proposal", "defensive_mixture"]
    physical_runs: bool
    posterior_claim: bool

    @model_validator(mode="after")
    def _claim_matches_the_method(self) -> ComparisonMethod:
        raw_q_alone = self.method_id == "Q"
        if raw_q_alone and self.posterior_claim:
            raise ValueError(
                "method Q is a raw conditional proposal: it never carries a posterior "
                "claim (plan §4.4)"
            )
        if self.method_id in ("B1", "M", "A_summary", "A_set") and not self.physical_runs:
            raise ValueError(
                f"method {self.method_id} is an SMC run and requires physical evaluations"
            )
        return self


class ComparisonParent(StrictModel):
    """An evaluation parent of the comparison matrix, named before any result exists."""

    parent_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    split: Literal["evaluation"]
    observation_hash: Sha256
    truth_ref: ArtifactRef


class ComparisonProtocol(StrictModel):
    """The preregistered comparison (plan §10): parents, methods, budgets, gates.

    `preregistration_hash` covers everything except itself and is computed once, BEFORE
    any evaluation run: the unit test holds that re-hashing the protocol reproduces it,
    so a matrix edited after seeing truth no longer validates.
    """

    schema_version: Literal[COMPARISON_PROTOCOL_SCHEMA] = COMPARISON_PROTOCOL_SCHEMA
    protocol_id: str = Field(min_length=1)
    parents: tuple[ComparisonParent, ...]
    methods: tuple[ComparisonMethod, ...]
    particle_counts: tuple[int, ...]
    inference_seeds: tuple[int, ...]
    primary_support: Literal["eight_quadrants"] = "eight_quadrants"
    estimator: Literal["posterior_mean"] = "posterior_mean"
    primary_month: int = Field(ge=1)
    gates: dict[str, float]
    preregistration_hash: Sha256

    @model_validator(mode="after")
    def _matrix_is_nonempty_and_hashed(self) -> ComparisonProtocol:
        if not self.parents or not self.methods:
            raise ValueError("a comparison protocol names at least one parent and method")
        method_ids = [method.method_id for method in self.methods]
        if len(set(method_ids)) != len(method_ids):
            raise ValueError(f"methods repeat an id: {method_ids}")
        recomputed = sha256_json(
            {
                key: value
                for key, value in self.model_dump(mode="json").items()
                if key != "preregistration_hash"
            }
        )
        if recomputed != self.preregistration_hash:
            raise ValueError(
                "preregistration_hash does not cover this protocol as it stands: the "
                "matrix was edited after registration"
            )
        return self

    @classmethod
    def preregister(
        cls,
        *,
        protocol_id: str,
        parents: tuple[ComparisonParent, ...],
        methods: tuple[ComparisonMethod, ...],
        particle_counts: tuple[int, ...],
        inference_seeds: tuple[int, ...],
        primary_month: int,
        gates: dict[str, float],
    ) -> ComparisonProtocol:
        """Build the protocol with its hash computed over the registration content."""
        payload = {
            "schema_version": COMPARISON_PROTOCOL_SCHEMA,
            "protocol_id": protocol_id,
            "parents": [parent.model_dump(mode="json") for parent in parents],
            "methods": [method.model_dump(mode="json") for method in methods],
            "particle_counts": list(particle_counts),
            "inference_seeds": list(inference_seeds),
            "primary_support": "eight_quadrants",
            "estimator": "posterior_mean",
            "primary_month": primary_month,
            "gates": dict(sorted(gates.items())),
        }
        return cls(
            protocol_id=protocol_id,
            parents=parents,
            methods=methods,
            particle_counts=particle_counts,
            inference_seeds=inference_seeds,
            primary_month=primary_month,
            gates=gates,
            preregistration_hash=sha256_json(payload),
        )


__all__ = [
    "BOOL",
    "COMPARISON_PROTOCOL_SCHEMA",
    "CONTEXT_SPEC_SCHEMA",
    "CORPUS_MANIFEST_SCHEMA",
    "ContextBatch",
    "ContextSpec",
    "CorpusManifest",
    "ComparisonMethod",
    "ComparisonParent",
    "ComparisonProtocol",
    "FailureRow",
    "F64",
    "FORBIDDEN_CONTEXT_FEATURES",
    "I64",
    "LABELS_SCHEMA",
    "LayoutSupport",
    "METHOD_IDS",
    "MethodId",
    "ParentRow",
    "ProposalManifest",
    "PROPOSAL_MANIFEST_SCHEMA",
    "TRAINING_MANIFEST_SCHEMA",
    "TrainingManifest",
    "ViewRow",
]
