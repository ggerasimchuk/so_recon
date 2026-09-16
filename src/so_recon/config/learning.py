"""E03 learned-proposal configuration (plan E03 §6–§7, frozen before evaluation).

These are the EXPERIMENT settings of the first learned inverse loop: corpus sizes and
splits, seeds, the estimator, network/flow/training hyperparameters and the defensive
epsilon. They are project choices, not scientific thresholds — the acceptance numbers
live in `configs/e03_tolerances.yml` and are never read from here.

Two invariants this module owns and the tests hold:

* **Splits are disjoint by construction.** A parent is the pair (family, index); its
  split label and its truth seed are pure functions of the corpus master seed, so two
  parents cannot share a seed or drift into another split. E01 seeds 41–45 and E02
  seeds 141–144 are never produced because every parent seed comes from the E03 master
  stream.
* **Family layouts are read from the design registry, not from names.** The corpus plan
  stores design ids only; `n_v`/`n_residual` are resolved through the synthetic design
  registry at build time (plan §3.1).
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """The frozen/forbid contract of `config.schema.StrictModel`, spelled out locally.

    `config.schema` imports this module for the `learning` field, so importing
    `StrictModel` from there would be a cycle. `config.resources` does the same thing
    for the same reason; the two definitions are identical by construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

#: Version of the learning block itself. A change of semantics needs a new version.
LEARNING_CONFIG_VERSION = "e03-learning-1"

#: Split labels of the corpus protocol (plan §6.2–§6.3). Assigned to the PARENT before
#: simulation; every child view (prefix, noise copy, fine/coarse) inherits it.
SplitName = Literal["train", "development", "evaluation"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "development", "evaluation")


class FamilyPlan(_StrictModel):
    """One design family's share of the corpus (plan §6.2).

    `train`/`development`/`evaluation` count INDEPENDENT parents, not a partition of one
    pool: 128 + 32 + 8 worlds are simulated in total for the scientific corpus.
    """

    design_id: str = Field(min_length=1)
    train: int = Field(ge=0)
    development: int = Field(ge=0)
    evaluation: int = Field(ge=0)
    split: SplitName = "train"

    @property
    def n_parents(self) -> int:
        return self.train + self.development + self.evaluation


class TrainingHyperParams(_StrictModel):
    """Optimizer/schedule settings of plan §7.4. Experiment settings, not thresholds."""

    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=8, ge=1)
    max_epochs: int = Field(default=200, ge=1)
    patience: int = Field(default=20, ge=1)
    gradient_norm_cap: float = Field(default=5.0, gt=0.0)
    # CPU Float32 is the default training dtype; the operational density is CPU Float64
    # regardless of this setting (plan §7.4–§7.5).
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu"] = "cpu"
    num_workers: int = Field(default=0, ge=0)
    torch_threads: int = Field(default=1, ge=1)


class EncoderParams(_StrictModel):
    """The shared trunk parameters of the three encoders (plan §7.2)."""

    variant: Literal["graph", "temporal_set", "summary"] = "graph"
    width: int = Field(default=64, ge=8)
    heads: int = Field(default=2, ge=1)
    context_dim: int = Field(default=128, ge=8)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    max_wells: int = Field(default=8, ge=1)
    max_months: int = Field(default=36, ge=1)


class FlowParams(_StrictModel):
    """NSF head settings (plan §7.3). Fixed project choices for the first version."""

    n_transforms: int = Field(default=6, ge=1)
    n_bins: int = Field(default=8, ge=1)
    tail_bound: float = Field(default=4.0, gt=0.0)
    min_bin_width: float = Field(default=1e-3, gt=0.0)
    min_bin_height: float = Field(default=1e-3, gt=0.0)
    min_derivative: float = Field(default=1e-3, gt=0.0)
    base: Literal["normal"] = "normal"
    tails: Literal["linear"] = "linear"
    architecture_version: str = "e03-conditional-nsf-1"


class LearningConfig(_StrictModel):
    """The `learning` block of an E03 `ProjectConfig`."""

    config_version: str = Field(default=LEARNING_CONFIG_VERSION, min_length=1)
    # Master seed of the corpus. Parent truth/history seeds are derived from it; the
    # training seed and its replica stay separate streams (plan §6.1, §10.1).
    corpus_seed: int = Field(default=3117, ge=0)
    training_seed: int = Field(default=9101, ge=0)
    training_seed_replica: int = Field(default=9102, ge=0)
    corpus: tuple[FamilyPlan, ...]
    thin_slice: tuple[FamilyPlan, ...] = Field(default_factory=tuple)
    defensive_epsilon: float = Field(default=0.10, gt=0.0, le=1.0)
    estimator: Literal["posterior_mean"] = "posterior_mean"
    primary_support: Literal["eight_quadrants"] = "eight_quadrants"
    primary_month: int = Field(default=36, ge=1)
    diagnostic_months: tuple[Annotated[int, Field(ge=1)], ...] = (12, 24)
    prefix_months: tuple[Annotated[int, Field(ge=1)], ...] = (12, 24, 36)
    max_noise_copies: int = Field(default=2, ge=1)
    inference_seeds: tuple[Annotated[int, Field(ge=0)], ...] = (11, 12)
    particle_counts: tuple[Annotated[int, Field(ge=2)], ...] = (32, 64)
    training: TrainingHyperParams = TrainingHyperParams()
    encoder: EncoderParams = EncoderParams()
    flow: FlowParams = FlowParams()
    context_builder_version: str = Field(default="e03-context-1", min_length=1)
    operational_dtype: Literal["float64"] = "float64"
    operational_backend: Literal["cpu"] = "cpu"

    @model_validator(mode="after")
    def _corpus_is_well_formed(self) -> LearningConfig:
        if not self.corpus:
            raise ValueError("the corpus needs at least one family plan")
        design_ids = [plan.design_id for plan in self.corpus]
        duplicates = sorted({d for d in design_ids if design_ids.count(d) > 1})
        if duplicates:
            raise ValueError(f"duplicate design_id in corpus plans: {duplicates}")
        if self.training_seed == self.training_seed_replica:
            raise ValueError(
                "training_seed and training_seed_replica must differ: the second seed "
                "exists to probe initialization stability, not to repeat one run"
            )
        for name, value in zip(
            ("primary_month", *self.diagnostic_months, *self.prefix_months),
            (self.primary_month, *self.diagnostic_months, *self.prefix_months),
            strict=True,
        ):
            if value > self.encoder.max_months:
                raise ValueError(
                    f"{name}={value} exceeds encoder.max_months={self.encoder.max_months}"
                )
        return self

    def plans(self, *, thin: bool) -> tuple[FamilyPlan, ...]:
        """The scientific corpus plans, or the one-shot smoke namespace."""
        return self.thin_slice if thin else self.corpus

    def parent_seed(self, plan: FamilyPlan, index: int, *, thin: bool) -> int:
        """The truth seed of parent `index` inside `plan`.

        Split layout inside a family: indices `[0, train)` are train parents,
        `[train, train+development)` development, the rest evaluation. The seed is a pure
        function of (master seed, design id, index, namespace), so the same corpus always
        draws the same parents and no parent can appear in two splits.
        """
        if not 0 <= index < plan.n_parents:
            raise ValueError(f"parent index {index} outside 0..{plan.n_parents - 1}")
        design_tag = hash_typed(plan.design_id)
        state = np.random.SeedSequence(
            [self.corpus_seed, design_tag, int(index), 1 if thin else 0]
        ).generate_state(1, dtype=np.uint32)[0]
        return int(state) % (2**31)

    def split_of(self, plan: FamilyPlan, index: int) -> SplitName:
        """The split label of parent `index` (assigned before simulation, plan §6.3)."""
        if not 0 <= index < plan.n_parents:
            raise ValueError(f"parent index {index} outside 0..{plan.n_parents - 1}")
        if index < plan.train:
            return "train"
        if index < plan.train + plan.development:
            return "development"
        return "evaluation"

    def corpus_parent_seeds(self, *, thin: bool) -> dict[tuple[str, int], int]:
        """Every parent seed of the namespace, for uniqueness checks and manifests."""
        return {
            (plan.design_id, index): self.parent_seed(plan, index, thin=thin)
            for plan in self.plans(thin=thin)
            for index in range(plan.n_parents)
        }


def hash_typed(text: str) -> int:
    """A stable, platform-independent tag of a design id (zlib.crc32 of its utf-8 bytes).

    `hash()` is salted per process; a salted value would make parent seeds depend on the
    interpreter run, which is exactly what a corpus seed must never do.
    """
    import zlib

    return zlib.crc32(text.encode("utf-8"))


#: Historical parent seeds of earlier stages. E03 evaluation parents are new draws and a
#: test holds them apart from these, so an old world cannot return as a blind parent.
RESERVED_HISTORICAL_SEEDS: frozenset[int] = frozenset(
    {41, 42, 43, 44, 45, 141, 142, 143, 144}
)


def corpus_totals(plans: tuple[FamilyPlan, ...]) -> dict[str, int]:
    """Total parent counts per split — the 128/32/8 of plan §6.2 in machine form."""
    totals = {name: 0 for name in SPLIT_NAMES}
    for plan in plans:
        totals["train"] += plan.train
        totals["development"] += plan.development
        totals["evaluation"] += plan.evaluation
    if sum(totals.values()) != sum(plan.n_parents for plan in plans):
        raise AssertionError("split totals must account for every parent")
    return totals


def scientific_corpus_plan() -> tuple[FamilyPlan, ...]:
    """The registered 128/32/8 distribution of plan §6.2."""
    return (
        FamilyPlan(design_id="e02-t1-v1", train=64, development=16, evaluation=4),
        FamilyPlan(design_id="e02-t2-v2", train=24, development=6, evaluation=1),
        FamilyPlan(design_id="e03-t3-v1", train=16, development=4, evaluation=1),
        FamilyPlan(design_id="e02-t4-v1", train=24, development=6, evaluation=2),
    )


def thin_slice_plan() -> tuple[FamilyPlan, ...]:
    """The one-shot technical smoke namespace (plan §6.2): 8/2/1 T1 parents.

    This namespace is not part of any scientific gate; it exists so the first full loop
    can be exercised end to end before the corpus budget is spent.
    """
    return (FamilyPlan(design_id="e02-t1-v1", train=8, development=2, evaluation=1),)


def default_learning_config() -> LearningConfig:
    """The `configs/e03.yml` learning block as a single source for tests and commands."""
    return LearningConfig(
        corpus=scientific_corpus_plan(),
        thin_slice=thin_slice_plan(),
    )


__all__ = [
    "RESERVED_HISTORICAL_SEEDS",
    "SPLIT_NAMES",
    "EncoderParams",
    "FamilyPlan",
    "FlowParams",
    "LEARNING_CONFIG_VERSION",
    "LearningConfig",
    "SplitName",
    "TrainingHyperParams",
    "corpus_totals",
    "default_learning_config",
    "hash_typed",
    "scientific_corpus_plan",
    "thin_slice_plan",
]
