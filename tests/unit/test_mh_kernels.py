"""Gradient-free proposal ratios and complete Metropolis-Hastings acceptance."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from so_recon.inference.contracts import DensitySchema, TargetEvaluation, ThetaRecord
from so_recon.inference.kernels import (
    Move,
    kernel_probabilities,
    mh_log_accept,
    pcn_reverse_minus_forward,
    propose_family,
    propose_global,
    propose_pcn,
    propose_rw,
)

SCHEMA = DensitySchema(
    schema_id="kernel-test",
    n_v=2,
    n_residual=2,
    families=(0, 1, 2),
    basis_hash="a" * 64,
    transform_version="identity-1",
)


def theta(
    *, s: int = 0, v: tuple[float, ...] = (0.0, 0.0), z: tuple[float, ...] = (0.0, 0.0)
) -> ThetaRecord:
    return ThetaRecord(schema_id=SCHEMA.schema_id, s=s, v=v, z_perp=z, basis_hash=SCHEMA.basis_hash)


def evaluation(
    value: ThetaRecord, *, log_p0: float, log_l: float, log_r: float
) -> TargetEvaluation:
    return TargetEvaluation(
        theta=value,
        log_p0=log_p0,
        log_p0_in_support=log_p0 != -math.inf,
        log_l=log_l,
        log_l_in_support=log_l != -math.inf,
        log_r=log_r,
        log_r_in_support=log_r != -math.inf,
        forward_ref=None,
        cache_key=f"toy-{value.s}-{value.v}-{value.z_perp}",
    )


def test_pcn_proposal_ratio_has_correct_sign() -> None:
    assert np.isclose(pcn_reverse_minus_forward(np.array([0.0]), np.array([2.0])), 2.0)


def test_global_move_draws_the_whole_theta_and_uses_the_density_ratio() -> None:
    class PointMass:
        fingerprint = "b" * 64

        def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
            del rng
            assert n == 1
            return (theta(s=2, v=(3.0, 4.0), z=(5.0, 6.0)),)

        def log_prob(self, value: ThetaRecord) -> float:
            return -1.0 if value.s == 0 else -4.0

    move = propose_global(theta(), PointMass(), np.random.default_rng(1))
    assert move.proposed == theta(s=2, v=(3.0, 4.0), z=(5.0, 6.0))
    assert move.log_reverse_minus_forward == 3.0
    assert move.kernel == "global"


def test_rw_and_pcn_change_only_their_declared_blocks() -> None:
    old = theta(s=1, v=(1.0, 2.0), z=(3.0, 4.0))
    rw = propose_rw(old, 0.2, np.random.default_rng(3))
    assert rw.proposed.s == old.s and rw.proposed.z_perp == old.z_perp
    assert rw.proposed.v != old.v
    assert rw.log_reverse_minus_forward == 0.0

    pcn = propose_pcn(old, 0.3, np.random.default_rng(3))
    assert pcn.proposed.s == old.s and pcn.proposed.v == old.v
    assert pcn.proposed.z_perp != old.z_perp
    assert pcn.log_reverse_minus_forward == pytest.approx(
        pcn_reverse_minus_forward(np.array(old.z_perp), np.array(pcn.proposed.z_perp))
    )


def test_family_switch_validates_dimensions_and_changes_family() -> None:
    move = propose_family(theta(s=1), SCHEMA, np.random.default_rng(2))
    assert move.proposed.s in {0, 2}
    assert move.proposed.v == theta().v and move.proposed.z_perp == theta().z_perp
    assert move.log_reverse_minus_forward == 0.0
    malformed = theta().model_copy(update={"v": (0.0,)})
    with pytest.raises(ValueError, match="dimension"):
        propose_family(malformed, SCHEMA, np.random.default_rng(2))


def test_fixed_kernel_probabilities_are_normalized_per_schema() -> None:
    full = kernel_probabilities(SCHEMA)
    assert full == {"global": 0.25, "rw": 0.35, "pcn": 0.25, "family": 0.15}
    reduced = kernel_probabilities(SCHEMA.model_copy(update={"n_residual": 0, "families": (0,)}))
    assert set(reduced) == {"global", "rw"}
    assert sum(reduced.values()) == pytest.approx(1.0)


def test_detailed_balance_holds_for_an_asymmetric_three_state_move() -> None:
    pi = np.array([0.2, 0.3, 0.5])
    q = np.array([[0.0, 0.7, 0.3], [0.2, 0.0, 0.8], [0.6, 0.4, 0.0]])
    states = [theta(s=i) for i in range(3)]
    values = [
        evaluation(state, log_p0=math.log(pi[i]), log_l=0.0, log_r=0.0)
        for i, state in enumerate(states)
    ]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            forward = Move(states[j], math.log(q[j, i]) - math.log(q[i, j]), "test")
            reverse = Move(states[i], math.log(q[i, j]) - math.log(q[j, i]), "test")
            alpha_ij = math.exp(mh_log_accept(values[i], values[j], 1.0, forward))
            alpha_ji = math.exp(mh_log_accept(values[j], values[i], 1.0, reverse))
            assert pi[i] * q[i, j] * alpha_ij == pytest.approx(
                pi[j] * q[j, i] * alpha_ji, abs=1e-12
            )


def test_pcn_at_beta_zero_corrects_for_a_nonreference_proposal_density() -> None:
    old = evaluation(theta(z=(0.0, 0.0)), log_p0=0.0, log_l=0.0, log_r=0.0)
    new = evaluation(theta(z=(2.0, 0.0)), log_p0=0.0, log_l=0.0, log_r=-8.0)
    move = Move(new.theta, pcn_reverse_minus_forward(np.zeros(2), np.array([2.0, 0.0])), "pcn")
    assert mh_log_accept(old, new, 0.0, move) == pytest.approx(-6.0)


def test_rw_chain_preserves_a_gaussian_target() -> None:
    rng = np.random.default_rng(43)
    current_x = 0.0
    current = evaluation(
        theta(v=(current_x, 0.0)),
        log_p0=norm.logpdf(current_x, 1.2, math.sqrt(0.7)) + norm.logpdf(0.0),
        log_l=0.0,
        log_r=0.0,
    )
    draws: list[float] = []
    for step in range(30_000):
        move = propose_rw(current.theta, 1.0, rng)
        x = move.proposed.v[0]
        proposed = evaluation(
            move.proposed,
            log_p0=norm.logpdf(x, 1.2, math.sqrt(0.7)) + norm.logpdf(move.proposed.v[1]),
            log_l=0.0,
            log_r=0.0,
        )
        if math.log(rng.random()) < mh_log_accept(current, proposed, 1.0, move):
            current = proposed
        if step >= 5_000:
            draws.append(current.theta.v[0])
    assert np.mean(draws) == pytest.approx(1.2, abs=0.06)
    assert np.var(draws) == pytest.approx(0.7, abs=0.08)


def test_rw_chain_preserves_a_correlated_tempered_bridge() -> None:
    rng = np.random.default_rng(44)
    beta = 0.37
    mean_r = np.array([-1.0, 0.5])
    cov_r = np.array([[1.4, -0.3], [-0.3, 0.8]])
    mean_p = np.array([1.2, -0.7])
    cov_p = np.array([[0.6, 0.25], [0.25, 1.1]])
    precision = (1.0 - beta) * np.linalg.inv(cov_r) + beta * np.linalg.inv(cov_p)
    expected_cov = np.linalg.inv(precision)
    expected_mean = expected_cov @ (
        (1.0 - beta) * np.linalg.solve(cov_r, mean_r) + beta * np.linalg.solve(cov_p, mean_p)
    )

    point = mean_r.copy()

    def scored(value: np.ndarray) -> TargetEvaluation:
        record = theta(v=(float(value[0]), float(value[1])))
        return evaluation(
            record,
            log_p0=float(multivariate_normal.logpdf(value, mean=mean_p, cov=cov_p)),
            log_l=0.0,
            log_r=float(multivariate_normal.logpdf(value, mean=mean_r, cov=cov_r)),
        )

    current = scored(point)
    draws: list[tuple[float, float]] = []
    for step in range(35_000):
        move = propose_rw(current.theta, 0.8, rng)
        proposed = scored(np.asarray(move.proposed.v))
        if math.log(rng.random()) < mh_log_accept(current, proposed, beta, move):
            current = proposed
        if step >= 5_000:
            draws.append(current.theta.v)
    sample = np.asarray(draws)
    np.testing.assert_allclose(sample.mean(axis=0), expected_mean, atol=0.06)
    np.testing.assert_allclose(np.cov(sample.T), expected_cov, atol=0.08)
