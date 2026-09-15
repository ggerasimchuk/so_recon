"""Stable, proposal-corrected SMC weight algebra."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

from so_recon.inference.weights import (
    CessSupportDiscontinuity,
    NoTargetSupport,
    bridge_log_target,
    cess,
    ess,
    next_beta,
    normalize_log_weights,
)


def test_normalization_keeps_zero_weight_zero_and_returns_the_constant() -> None:
    normalized, constant = normalize_log_weights(np.array([1000.0, 999.0, -math.inf]))
    assert constant == pytest.approx(float(logsumexp([1000.0, 999.0])))
    np.testing.assert_allclose(np.exp(normalized[:2]), [1 / (1 + math.exp(-1)), 1 / (1 + math.e)])
    assert normalized[2] == -math.inf
    assert np.isclose(float(logsumexp(normalized)), 0.0, atol=1e-14)


def test_all_zero_target_is_not_reset_to_uniform() -> None:
    with pytest.raises(NoTargetSupport):
        normalize_log_weights(np.array([-math.inf, -math.inf]))


@pytest.mark.parametrize("bad", [[0.0, math.nan], [0.0, math.inf], [], [[0.0]]])
def test_invalid_weight_vectors_are_refused(bad: object) -> None:
    with pytest.raises(ValueError):
        normalize_log_weights(np.asarray(bad, dtype=np.float64))


def test_ess_uses_normalized_weights_and_counts_only_their_mass() -> None:
    logw = np.log([0.5, 0.3, 0.2])
    assert ess(logw) == pytest.approx(1.0 / (0.5**2 + 0.3**2 + 0.2**2))
    with pytest.raises(ValueError, match="normalized"):
        ess(logw + 2.0)


def test_cess_at_zero_is_n_even_for_nonuniform_weights() -> None:
    logw = np.log([0.9, 0.1])
    assert np.isclose(cess(logw, np.array([-100.0, 30.0]), 0.0), 2.0)
    assert bridge_log_target(0.0, -math.inf, -math.inf, -2.0) == -2.0


def test_shifting_log_likelihood_changes_evidence_not_relative_cess() -> None:
    logw = np.log([0.7, 0.2, 0.1])
    ell = np.array([-2.0, 0.5, 3.0])
    delta = 0.37
    assert cess(logw, ell, delta) == pytest.approx(cess(logw, ell + 10_000.0, delta))
    a = float(logsumexp(logw + delta * ell))
    shifted = float(logsumexp(logw + delta * (ell + 10_000.0)))
    assert shifted - a == pytest.approx(delta * 10_000.0)


def test_beta_schedule_is_monotone_and_hits_one_exactly_when_allowed() -> None:
    logw = np.full(4, -math.log(4.0))
    flat = np.zeros(4)
    assert next_beta(0.3, logw, flat, 0.8) == 1.0

    ell = np.array([-8.0, -1.0, 1.0, 8.0])
    beta = next_beta(0.0, logw, ell, 0.8)
    assert 0.0 < beta < 1.0
    assert cess(logw, ell, beta) == pytest.approx(3.2, abs=1e-8)
    assert next_beta(beta, logw, ell, 0.8) > beta


def test_no_remaining_target_support_is_diagnostic() -> None:
    logw = np.log([0.5, 0.5])
    with pytest.raises(NoTargetSupport):
        cess(logw, np.array([-math.inf, -math.inf]), 0.1)


def test_a_cess_jump_at_zero_is_not_an_infinite_beta_zero_loop() -> None:
    logw = np.log([0.99, 0.01])
    ell = np.array([-math.inf, 0.0])
    with pytest.raises(CessSupportDiscontinuity):
        next_beta(0.0, logw, ell, 0.8)


def test_bridge_target_has_exact_endpoints_and_proposal_correction() -> None:
    assert bridge_log_target(0.0, -10.0, -20.0, -3.0) == -3.0
    assert bridge_log_target(1.0, -10.0, -20.0, -3.0) == -30.0
    assert bridge_log_target(0.25, -10.0, -20.0, -3.0) == pytest.approx(-9.75)
    assert bridge_log_target(0.4, -math.inf, -2.0, -3.0) == -math.inf
    with pytest.raises(ValueError):
        bridge_log_target(1.1, -1.0, -1.0, -1.0)
