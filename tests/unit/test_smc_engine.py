"""The SMC engine is one resumable phase machine, not a restartable approximation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from so_recon.config.inference import InferenceConfig
from so_recon.geology.density import Density
from so_recon.inference.contracts import (
    DensitySchema,
    Particle,
    SMCState,
    TargetEvaluation,
    ThetaRecord,
)
from so_recon.inference.smc import SMCBudgetStop, continue_inference, infer
from so_recon.registry.hashing import sha256_json
from so_recon.validation.toy_inverse import GaussianToyTarget

SCHEMA = DensitySchema(
    schema_id="gaussian-toy-1",
    n_v=1,
    n_residual=0,
    families=(0,),
    basis_hash="a" * 64,
    transform_version="identity-1",
)


class GaussianDensity:
    def __init__(self, mean: float, variance: float) -> None:
        self.mean = mean
        self.variance = variance
        self.fingerprint = sha256_json(
            {"kind": "test-gaussian", "mean": mean, "variance": variance}
        )

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        values = rng.normal(self.mean, math.sqrt(self.variance), size=n)
        return tuple(
            ThetaRecord(
                schema_id=SCHEMA.schema_id,
                s=0,
                v=(float(value),),
                z_perp=(),
                basis_hash=SCHEMA.basis_hash,
            )
            for value in values
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        SCHEMA.validate_theta(theta)
        return float(norm.logpdf(theta.v[0], self.mean, math.sqrt(self.variance)))


def problem() -> tuple[GaussianToyTarget, Density, InferenceConfig]:
    proposal = GaussianDensity(-0.5, 2.0)
    target = GaussianToyTarget(0.0, 1.0, 1.5, 0.7, proposal)
    config = InferenceConfig(
        n_particles=12,
        cess_fraction=0.8,
        resample_fraction=0.9,
        max_beta_steps=24,
        moves_per_level=2,
        seed=73,
        rw_scale=0.7,
        pcn_scale=0.2,
    )
    return target, proposal, config


def never_stop() -> bool:
    return False


def stop_on(call: int):
    seen = 0

    def requested() -> bool:
        nonlocal seen
        seen += 1
        return seen == call

    return requested


def test_uninterrupted_engine_reaches_the_real_target(tmp_path: Path) -> None:
    target, proposal, config = problem()
    state = infer(target, proposal, SCHEMA, config, tmp_path / "checkpoint", never_stop)
    assert state.algorithm_status == "COMPLETE"
    assert state.beta == 1.0
    assert state.phase == "finalize"
    assert len(state.particles) == config.n_particles
    assert np.isclose(np.exp(state.log_weights).sum(), 1.0)
    assert state.log_evidence != 0.0
    assert (tmp_path / "checkpoint" / "latest.json").is_file()


@pytest.mark.parametrize("boundary", [1, 2, 15, 25, 26, 27, 35, 60])
def test_stop_and_resume_is_bitwise_the_same_phase_machine(tmp_path: Path, boundary: int) -> None:
    target, proposal, config = problem()
    uninterrupted = infer(target, proposal, SCHEMA, config, tmp_path / "full", never_stop)
    interrupted = infer(
        target, proposal, SCHEMA, config, tmp_path / f"cut-{boundary}", stop_on(boundary)
    )
    if interrupted.algorithm_status == "COMPLETE":
        pytest.skip(f"boundary {boundary} is beyond this deterministic run")
    resumed = continue_inference(
        interrupted,
        target,
        proposal,
        SCHEMA,
        config,
        tmp_path / f"cut-{boundary}",
        never_stop,
    )
    assert resumed.model_dump(mode="json") == uninterrupted.model_dump(mode="json")


def test_initial_pending_draw_is_saved_before_evaluation(tmp_path: Path) -> None:
    target, proposal, config = problem()
    state = infer(target, proposal, SCHEMA, config, tmp_path / "checkpoint", stop_on(2))
    assert state.phase == "initialize"
    assert state.particles == ()
    assert state.pending_proposal is not None
    assert state.pending_log_u is None
    resumed = continue_inference(
        state, target, proposal, SCHEMA, config, tmp_path / "checkpoint", never_stop
    )
    full = infer(target, proposal, SCHEMA, config, tmp_path / "full", never_stop)
    assert resumed.model_dump(mode="json") == full.model_dump(mode="json")


def test_target_failure_suspends_the_engine_instead_of_becoming_mh_rejection(
    tmp_path: Path,
) -> None:
    target, proposal, config = problem()

    class FailingTarget:
        fingerprint = "f" * 64

        def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
            del theta
            raise RuntimeError("injected physical failure")

    state = infer(FailingTarget(), proposal, SCHEMA, config, tmp_path / "failure", never_stop)
    assert state.algorithm_status == "EVALUATION_FAILURE"
    assert state.phase == "initialize"
    assert state.pending_proposal is not None
    assert "injected physical failure" in str(state.diagnostics["evaluation_failure"])


def test_budget_stop_preserves_pending_evaluation_for_a_later_session(tmp_path: Path) -> None:
    target, proposal, config = problem()

    class OnceBudgetedTarget:
        fingerprint = target.fingerprint
        checkpoint_hashes: dict[str, str] = {}

        def __init__(self) -> None:
            self.stop = True

        def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
            if self.stop:
                raise SMCBudgetStop("P1 session exhausted")
            return target.evaluate(theta)

    bounded = OnceBudgetedTarget()
    state = infer(bounded, proposal, SCHEMA, config, tmp_path / "budget-stop", never_stop)
    assert state.algorithm_status == "INCOMPLETE_BUDGET"
    assert state.phase == "initialize"
    assert state.pending_proposal is not None
    assert state.diagnostics["stop_reason"] == "P1 session exhausted"

    bounded.stop = False
    resumed = continue_inference(
        state, bounded, proposal, SCHEMA, config, tmp_path / "budget-stop", never_stop
    )
    uninterrupted = infer(target, proposal, SCHEMA, config, tmp_path / "full", never_stop)
    assert resumed.model_dump(mode="json") == uninterrupted.model_dump(mode="json")


def test_initial_ensemble_is_persisted_before_rejuvenation(tmp_path: Path) -> None:
    target, proposal, config = problem()
    state = infer(target, proposal, SCHEMA, config, tmp_path / "initial", stop_on(26))
    initial = state.diagnostics["initial_evaluations"]
    assert len(initial) == config.n_particles
    assert initial == [particle.evaluation.model_dump(mode="json") for particle in state.particles]


def test_all_zero_target_support_has_its_own_status(tmp_path: Path) -> None:
    _, proposal, config = problem()

    class ZeroTarget:
        fingerprint = "e" * 64

        def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
            return TargetEvaluation(
                theta=theta,
                log_p0=-math.inf,
                log_p0_in_support=False,
                log_l=-math.inf,
                log_l_in_support=False,
                log_r=proposal.log_prob(theta),
                log_r_in_support=True,
                forward_ref=None,
                cache_key=sha256_json(theta.model_dump(mode="json")),
            )

    state = infer(ZeroTarget(), proposal, SCHEMA, config, tmp_path / "zero", never_stop)
    assert state.algorithm_status == "NO_TARGET_SUPPORT"
    assert state.beta == 0.0


def test_cess_support_jump_is_an_evaluation_failure_not_no_support(tmp_path: Path) -> None:
    target, proposal, config = problem()
    ready = infer(target, proposal, SCHEMA, config, tmp_path / "ready", stop_on(25))
    assert ready.phase == "reweight" and ready.beta == 0.0
    particles: list[Particle] = []
    for index, particle in enumerate(ready.particles):
        evaluation_payload = particle.evaluation.model_dump()
        if index:
            evaluation_payload.update(
                {
                    "log_p0": -math.inf,
                    "log_p0_in_support": False,
                    "log_l": -math.inf,
                    "log_l_in_support": False,
                }
            )
        particles.append(
            particle.model_copy(
                update={"evaluation": TargetEvaluation.model_validate(evaluation_payload)}
            )
        )
    discontinuous = SMCState.model_validate({**ready.model_dump(), "particles": tuple(particles)})
    result = continue_inference(
        discontinuous,
        target,
        proposal,
        SCHEMA,
        config,
        tmp_path / "ready",
        never_stop,
    )
    assert result.algorithm_status == "EVALUATION_FAILURE"
    assert "CESS_SUPPORT_DISCONTINUITY" in str(result.diagnostics["evaluation_failure"])


def test_resume_refuses_incompatible_algorithm_identity(tmp_path: Path) -> None:
    target, proposal, config = problem()
    stopped = infer(target, proposal, SCHEMA, config, tmp_path / "checkpoint", stop_on(2))
    incompatible = GaussianDensity(9.0, 1.0)
    with pytest.raises(ValueError, match="proposal hash"):
        continue_inference(
            stopped,
            target,
            incompatible,
            SCHEMA,
            config,
            tmp_path / "checkpoint",
            never_stop,
        )


def test_beta_budget_stops_honestly_before_one(tmp_path: Path) -> None:
    target, proposal, config = problem()
    tiny_budget = config.model_copy(update={"max_beta_steps": 1})
    state = infer(target, proposal, SCHEMA, tiny_budget, tmp_path / "budget", never_stop)
    assert state.algorithm_status == "INCOMPLETE_BUDGET"
    assert state.beta < 1.0
    assert state.diagnostics["stop_reason"] == "MAX_BETA_STEPS"


def test_zero_mass_mh_proposal_is_a_recorded_rejection_not_an_engine_failure(
    tmp_path: Path,
) -> None:
    class PointStartDensity(GaussianDensity):
        def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
            del rng
            return tuple(
                ThetaRecord(
                    schema_id=SCHEMA.schema_id,
                    s=0,
                    v=(0.0,),
                    z_perp=(),
                    basis_hash=SCHEMA.basis_hash,
                )
                for _ in range(n)
            )

    proposal = PointStartDensity(0.0, 1.0)

    class BoundedTarget:
        fingerprint = "d" * 64

        def evaluate(self, theta: ThetaRecord) -> TargetEvaluation:
            supported = abs(theta.v[0]) <= 0.1
            return TargetEvaluation(
                theta=theta,
                log_p0=0.0 if supported else -math.inf,
                log_p0_in_support=supported,
                log_l=0.0,
                log_l_in_support=True,
                log_r=proposal.log_prob(theta),
                log_r_in_support=True,
                forward_ref=None,
                cache_key=sha256_json(theta.model_dump(mode="json")),
            )

    config = InferenceConfig(
        n_particles=4,
        cess_fraction=0.8,
        resample_fraction=0.5,
        max_beta_steps=2,
        moves_per_level=20,
        seed=17,
        rw_scale=2.0,
        pcn_scale=0.2,
    )
    state = infer(BoundedTarget(), proposal, SCHEMA, config, tmp_path / "bounded", never_stop)
    assert state.algorithm_status == "COMPLETE"
    rejected = [
        move for move in state.diagnostics["moves"] if move["log_alpha_in_support"] is False
    ]
    assert rejected
    assert all(move["accepted"] is False and move["log_alpha"] is None for move in rejected)
