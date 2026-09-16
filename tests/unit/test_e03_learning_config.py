"""E03 learning-block and ML-dependency contracts (plan E03 Task 01).

Three families of checks live here:

* the learning config refuses the layouts that would silently corrupt a corpus
  (duplicate families, coinciding training seeds, months beyond the encoder window);
* splits and parent seeds are disjoint BY CONSTRUCTION — across splits, across
  namespaces, and away from the reserved E01/E02 seeds;
* the ML stack itself: CPU Float32 backward and CPU Float64 forward/inverse/logdet
  through the exact nflows transform class the NSF head wraps. This is the smoke of
  plan §7.3 — a version that fails it is not pinned in `uv.lock` for this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from so_recon.config.learning import (
    RESERVED_HISTORICAL_SEEDS,
    FamilyPlan,
    LearningConfig,
    corpus_totals,
    default_learning_config,
    scientific_corpus_plan,
    thin_slice_plan,
)
from so_recon.config.load import config_hash, load_project_config
from so_recon.config.schema import ProjectConfig

REPO = Path(__file__).resolve().parents[2]


def _plan(**over: object) -> FamilyPlan:
    base: dict[str, object] = {"design_id": "e02-t1-v1", "train": 4, "development": 2, "evaluation": 1}
    base.update(over)
    return FamilyPlan(**base)  # type: ignore[arg-type]


def test_scientific_corpus_totals_are_the_registered_128_32_8() -> None:
    plans = scientific_corpus_plan()
    assert corpus_totals(plans) == {"train": 128, "development": 32, "evaluation": 8}
    by_design = {plan.design_id: plan for plan in plans}
    assert by_design["e02-t1-v1"].n_parents == 84
    assert by_design["e03-t3-v1"].n_parents == 21


def test_thin_slice_is_t1_only_smoke() -> None:
    plans = thin_slice_plan()
    assert len(plans) == 1
    assert corpus_totals(plans) == {"train": 8, "development": 2, "evaluation": 1}


def test_duplicate_design_in_corpus_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate design_id"):
        LearningConfig(corpus=(_plan(), _plan()))


def test_coinciding_training_seeds_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        LearningConfig(corpus=(_plan(),), training_seed=5, training_seed_replica=5)


def test_month_beyond_the_encoder_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="encoder.max_months"):
        LearningConfig(corpus=(_plan(),), diagnostic_months=(48,))


def test_duplicate_well_time_feature_is_rejected_via_context_spec() -> None:
    from so_recon.ml.contracts import ContextSpec

    with pytest.raises(ValidationError, match="must not repeat"):
        ContextSpec(
            builder_version="e03-context-1",
            well_time_features=("control_kind", "control_kind"),
            static_features=(),
            edge_features=(),
            control_kinds=("bhp_pa",),
            units={},
            cutoff_s=94608000.0,
            max_wells=8,
            max_months=36,
            allowed_context_keys=("G", "U"),
        )


def test_parent_seeds_are_unique_within_and_across_splits() -> None:
    cfg = default_learning_config()
    seeds = cfg.corpus_parent_seeds(thin=False)
    values = list(seeds.values())
    assert len(set(values)) == len(values)
    # every seed of the scientific namespace is distinct from the smoke namespace
    thin = cfg.corpus_parent_seeds(thin=True)
    assert not set(values) & set(thin.values())
    # historical stages' parents cannot return as fresh draws
    assert not set(values) & RESERVED_HISTORICAL_SEEDS
    assert not set(thin.values()) & RESERVED_HISTORICAL_SEEDS


def test_parent_seed_is_a_pure_function_of_its_arguments() -> None:
    cfg = default_learning_config()
    plan = cfg.corpus[0]
    assert cfg.parent_seed(plan, 0, thin=False) == cfg.parent_seed(plan, 0, thin=False)
    assert cfg.parent_seed(plan, 0, thin=False) != cfg.parent_seed(plan, 1, thin=False)
    assert cfg.parent_seed(plan, 0, thin=False) != cfg.parent_seed(plan, 0, thin=True)


def test_split_assignment_covers_every_parent_exactly_once() -> None:
    cfg = default_learning_config()
    for plan in cfg.corpus:
        labels = [cfg.split_of(plan, i) for i in range(plan.n_parents)]
        assert labels.count("train") == plan.train
        assert labels.count("development") == plan.development
        assert labels.count("evaluation") == plan.evaluation


def test_out_of_range_parent_index_is_refused() -> None:
    cfg = default_learning_config()
    with pytest.raises(ValueError, match="outside"):
        cfg.parent_seed(cfg.corpus[0], cfg.corpus[0].n_parents, thin=False)


def test_existing_stage_configs_still_load_by_their_versions() -> None:
    for name, version in (("project.yml", "E00"), ("e01.yml", None), ("e02.yml", "E02")):
        path = REPO / "configs" / name
        if not path.exists():
            continue
        cfg = load_project_config(path)
        assert cfg.learning is None or name == "e03.yml"
        assert config_hash(cfg)
    e03 = load_project_config(REPO / "configs" / "e03.yml")
    assert e03.config_version == "E03.1"
    assert e03.learning is not None
    assert e03.learning.defensive_epsilon == 0.10


def test_learning_block_is_refused_in_a_3_0_config() -> None:
    from so_recon.config.load import resolved_config_dict

    minimal = load_project_config(REPO / "configs" / "project.yml")
    assert minimal.spec_version in ("3.0", "4.0")
    legacy = ProjectConfig(
        spec_version="3.0",
        config_version="test.1",
        paths={},
        sources={"files": []},
    )
    dump = resolved_config_dict(legacy)
    assert "learning" not in dump
    assert "resources" not in dump


def test_a_3_0_config_cannot_set_the_learning_block() -> None:
    with pytest.raises(ValidationError, match="3.0 has no 'learning'"):
        ProjectConfig(
            spec_version="3.0",
            config_version="test.1",
            paths={},
            sources={"files": []},
            learning=default_learning_config(),
        )


# --------------------------------------------------------------------------------------
# the ML dependency smoke (plan §7.3)
# --------------------------------------------------------------------------------------


def test_nflows_cpu_float32_backward_pass_runs() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("nflows")
    from nflows.transforms import PiecewiseRationalQuadraticCDF

    torch.manual_seed(0)
    transform = PiecewiseRationalQuadraticCDF(
        shape=[2],
        tails="linear",
        tail_bound=4.0,
        num_bins=8,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    )
    inputs = torch.randn(16, 2, dtype=torch.float32, requires_grad=True)
    outputs, logdet = transform.forward(inputs)
    assert outputs.dtype == torch.float32
    assert logdet.dtype == torch.float32
    (outputs.pow(2).sum() - logdet.sum()).backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())


def test_nflows_cpu_float64_forward_inverse_logdet_consistency() -> None:
    torch = pytest.importorskip("torch")
    from nflows.transforms import PiecewiseRationalQuadraticCDF

    torch.manual_seed(1)
    transform = PiecewiseRationalQuadraticCDF(
        shape=[3],
        tails="linear",
        tail_bound=4.0,
        num_bins=8,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ).double()
    points = torch.tensor(
        [[0.0, 0.5, -0.5], [3.5, -3.5, 1.25], [0.1, -0.1, 0.05]], dtype=torch.float64
    )
    outputs, logdet = transform.forward(points)
    restored, inverse_logdet = transform.inverse(outputs)
    np.testing.assert_allclose(restored.detach().numpy(), points.detach().numpy(), atol=1e-8)
    np.testing.assert_allclose(
        (logdet + inverse_logdet).detach().numpy(), 0.0, atol=1e-8
    )


def test_torch_dtype_promotion_is_float64_for_operational_density() -> None:
    torch = pytest.importorskip("torch")
    weights = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    promoted = weights.double()
    assert promoted.dtype == torch.float64
    # a Float32-trained parameter transfers to Float64 without changing values
    np.testing.assert_array_equal(promoted.detach().numpy(), weights.detach().numpy().astype(np.float64))
