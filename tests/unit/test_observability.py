"""Pair diagnostics refuse missing shared data and report state ambiguity separately."""

from __future__ import annotations

import numpy as np
import pytest

from so_recon.inference.contracts import ModelObservations
from so_recon.validation.observability import ambiguity_metrics, compare_pair


def _prediction(values: dict[tuple[str, int], float | None], model: str) -> ModelObservations:
    return ModelObservations(fw=values, so_support={}, model_hash=model * 64)


def test_ambiguity_metrics_keep_observation_and_state_gaps_separate() -> None:
    metrics = ambiguity_metrics(
        np.array([0.1, 0.2]),
        np.array([0.11, 0.19]),
        np.array([0.2, 0.8]),
        np.array([0.3, 0.6]),
    )
    assert metrics["mean_absolute_fw_gap"] == pytest.approx(0.01)
    assert metrics["max_absolute_so_gap"] == pytest.approx(0.2)


def test_pair_uses_only_mutually_observed_watercut_rows() -> None:
    out = compare_pair(
        _prediction({("P1", 0): 0.1, ("P1", 1): None}, "a"),
        _prediction({("P1", 0): 0.11, ("P1", 1): 0.9}, "b"),
        np.array([0.2, 0.8]),
        np.array([0.3, 0.6]),
        np.eye(2),
    )
    assert out["status"] == "AMBIGUITY_DEMONSTRATED"
    assert out["n_common_fw"] == 1
    assert out["mean_absolute_fw_gap"] == pytest.approx(0.01)
    assert out["max_absolute_so_gap"] == pytest.approx(0.2)


def test_pair_with_no_common_watercut_is_no_data_not_nan() -> None:
    out = compare_pair(
        _prediction({("P1", 0): None}, "a"),
        _prediction({("P1", 0): 0.1}, "b"),
        np.array([0.2]),
        np.array([0.8]),
        np.eye(1),
    )
    assert out == {"status": "NO_COMMON_DATA", "n_common_fw": 0}
