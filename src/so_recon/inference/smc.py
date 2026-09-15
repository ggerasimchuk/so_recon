"""A resumable sequential Monte Carlo phase machine for the E02 latent measure."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from so_recon.config.inference import InferenceConfig
from so_recon.geology.density import Density
from so_recon.inference.contracts import (
    DensitySchema,
    Particle,
    SMCState,
    TargetEvaluation,
    ThetaRecord,
)
from so_recon.inference.kernels import (
    Move,
    kernel_probabilities,
    mh_log_accept,
    propose_family,
    propose_global,
    propose_pcn,
    propose_rw,
)
from so_recon.inference.resampling import systematic_resample
from so_recon.inference.weights import (
    CessSupportDiscontinuity,
    NoTargetSupport,
    ess,
    next_beta,
    normalize_log_weights,
)
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.hashing import sha256_json

LATEST_STATE_FILENAME = "latest.json"


class SMCBudgetStop(RuntimeError):
    """A target could not book more work in this session and may be retried unchanged."""


class Target(Protocol):
    fingerprint: str

    def evaluate(self, theta: ThetaRecord) -> TargetEvaluation: ...


def _target_components(target: Target) -> dict[str, str]:
    raw = getattr(target, "checkpoint_hashes", {})
    if not isinstance(raw, dict) or any(
        not isinstance(name, str) or not isinstance(value, str) for name, value in raw.items()
    ):
        raise ValueError("target checkpoint_hashes must be a string-to-string mapping")
    return dict(raw)


def _replace(state: SMCState, **changes: object) -> SMCState:
    return SMCState.model_validate({**state.model_dump(), **changes})


def _rng_from(state: SMCState) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state
    return rng


def _persist(state: SMCState, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(checkpoint_dir / LATEST_STATE_FILENAME, state.model_dump(mode="json"))


def _diagnostics(state: SMCState, **changes: object) -> dict[str, object]:
    return {**state.diagnostics, **changes}


def _evaluation(target: Target, proposal: Density, theta: ThetaRecord) -> TargetEvaluation:
    result = target.evaluate(theta)
    if result.theta != theta:
        raise ValueError("target evaluation returned a different theta")
    declared = proposal.log_prob(theta)
    if declared == -math.inf:
        raise ValueError(
            "proposal.sample returned a theta with zero density under proposal.log_prob"
        )
    if not (
        result.log_r == declared
        or (
            math.isfinite(result.log_r)
            and math.isfinite(declared)
            and math.isclose(result.log_r, declared, rel_tol=0.0, abs_tol=1e-12)
        )
    ):
        raise ValueError(
            f"target log_r {result.log_r!r} does not match proposal density {declared!r}"
        )
    return result


def _initial_state(
    target: Target,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
) -> SMCState:
    rng = np.random.default_rng(config.seed)
    probabilities = kernel_probabilities(schema)
    return SMCState(
        particles=(),
        log_weights=(),
        log_weights_in_support=(),
        beta=0.0,
        level=0,
        log_evidence=0.0,
        log_evidence_in_support=True,
        phase="initialize",
        cursor=0,
        rng_state=rng.bit_generator.state,
        proposal_hash=proposal.fingerprint,
        target_hash=target.fingerprint,
        pending_proposal=None,
        pending_log_u=None,
        algorithm_status="INCOMPLETE_BUDGET",
        diagnostics={
            "schema_id": schema.schema_id,
            "basis_hash": schema.basis_hash,
            "config_hash": sha256_json(config.model_dump(mode="json")),
            "kernel_probabilities": probabilities,
            "kernel_hash": sha256_json(probabilities),
            "target_components": _target_components(target),
            "consumed_cost": {},
            "pending_jobs": [],
            "beta_history": [0.0],
            "ess_history": [],
            "resampling": [],
            "moves": [],
        },
    )


def _validate_resume(
    state: SMCState,
    target: Target,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
) -> None:
    if state.proposal_hash != proposal.fingerprint:
        raise ValueError(
            f"proposal hash mismatch: checkpoint {state.proposal_hash}, current "
            f"{proposal.fingerprint}"
        )
    if state.target_hash != target.fingerprint:
        raise ValueError(
            f"target hash mismatch: checkpoint {state.target_hash}, current {target.fingerprint}"
        )
    if state.diagnostics.get("basis_hash") != schema.basis_hash:
        raise ValueError("basis hash mismatch")
    config_hash = sha256_json(config.model_dump(mode="json"))
    if state.diagnostics.get("config_hash") != config_hash:
        raise ValueError("config hash mismatch")
    probabilities = kernel_probabilities(schema)
    if state.diagnostics.get("kernel_hash") != sha256_json(probabilities):
        raise ValueError("kernel law mismatch")
    if state.diagnostics.get("target_components") != _target_components(target):
        raise ValueError("target component hashes mismatch")
    for particle in state.particles:
        schema.validate_theta(particle.evaluation.theta)
    if state.pending_proposal is not None:
        schema.validate_theta(state.pending_proposal)


def _fail_evaluation(state: SMCState, exc: Exception, checkpoint_dir: Path) -> SMCState:
    failed = _replace(
        state,
        algorithm_status="EVALUATION_FAILURE",
        diagnostics=_diagnostics(state, evaluation_failure=f"{type(exc).__name__}: {exc}"),
    )
    _persist(failed, checkpoint_dir)
    return failed


def _stop_budget(state: SMCState, exc: SMCBudgetStop, checkpoint_dir: Path) -> SMCState:
    stopped = _replace(
        state,
        algorithm_status="INCOMPLETE_BUDGET",
        diagnostics=_diagnostics(state, stop_reason=str(exc)),
    )
    _persist(stopped, checkpoint_dir)
    return stopped


def _fail_support(state: SMCState, exc: Exception, checkpoint_dir: Path) -> SMCState:
    failed = _replace(
        state,
        algorithm_status="NO_TARGET_SUPPORT",
        diagnostics=_diagnostics(state, no_target_support=f"{type(exc).__name__}: {exc}"),
    )
    _persist(failed, checkpoint_dir)
    return failed


def _choose_move(
    theta: ThetaRecord,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
    probabilities: dict[str, float],
    rng: np.random.Generator,
) -> Move:
    names = tuple(probabilities)
    selected = str(rng.choice(names, p=[probabilities[name] for name in names]))
    if selected == "global":
        return propose_global(theta, proposal, rng)
    if selected == "rw":
        return propose_rw(theta, config.rw_scale, rng)
    if selected == "pcn":
        return propose_pcn(theta, config.pcn_scale, rng)
    if selected == "family":
        return propose_family(theta, schema, rng)
    raise RuntimeError(f"unknown fixed kernel {selected!r}")


def _drive(
    state: SMCState,
    target: Target,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
    checkpoint_dir: Path,
    stop_requested: Callable[[], bool],
) -> SMCState:
    rng = _rng_from(state)
    diagnostics = dict(state.diagnostics)
    diagnostics.pop("stop_reason", None)
    state = _replace(
        state,
        algorithm_status="INCOMPLETE_BUDGET",
        diagnostics=diagnostics,
    )
    probabilities = kernel_probabilities(schema)
    while True:
        if stop_requested():
            _persist(state, checkpoint_dir)
            return state

        if state.phase == "initialize":
            if state.pending_proposal is None:
                pending = proposal.sample(1, rng)[0]
                schema.validate_theta(pending)
                state = _replace(
                    state,
                    pending_proposal=pending,
                    pending_log_u=None,
                    rng_state=rng.bit_generator.state,
                )
                _persist(state, checkpoint_dir)
                continue
            try:
                result = _evaluation(target, proposal, state.pending_proposal)
            except SMCBudgetStop as exc:
                return _stop_budget(state, exc, checkpoint_dir)
            except Exception as exc:
                return _fail_evaluation(state, exc, checkpoint_dir)
            particle = Particle(
                particle_id=len(state.particles),
                ancestor_id=len(state.particles),
                evaluation=result,
            )
            particles = (*state.particles, particle)
            if len(particles) == config.n_particles:
                uniform = tuple([-math.log(config.n_particles)] * config.n_particles)
                state = _replace(
                    state,
                    particles=particles,
                    log_weights=uniform,
                    log_weights_in_support=tuple([True] * config.n_particles),
                    phase="reweight",
                    cursor=0,
                    pending_proposal=None,
                    pending_log_u=None,
                    diagnostics=_diagnostics(
                        state,
                        initial_evaluations=[
                            item.evaluation.model_dump(mode="json") for item in particles
                        ],
                    ),
                )
            else:
                state = _replace(
                    state,
                    particles=particles,
                    cursor=len(particles),
                    pending_proposal=None,
                    pending_log_u=None,
                )
            _persist(state, checkpoint_dir)
            continue

        if state.phase == "reweight":
            if state.beta == 1.0:
                state = _replace(state, phase="finalize", cursor=0)
                _persist(state, checkpoint_dir)
                continue
            if state.level >= config.max_beta_steps:
                state = _replace(
                    state,
                    algorithm_status="INCOMPLETE_BUDGET",
                    diagnostics=_diagnostics(state, stop_reason="MAX_BETA_STEPS"),
                )
                _persist(state, checkpoint_dir)
                return state
            ell = np.asarray(
                [
                    -math.inf
                    if particle.evaluation.log_p0 == -math.inf
                    or particle.evaluation.log_l == -math.inf
                    else particle.evaluation.log_p0
                    + particle.evaluation.log_l
                    - particle.evaluation.log_r
                    for particle in state.particles
                ],
                dtype=np.float64,
            )
            old_logw = np.asarray(state.log_weights, dtype=np.float64)
            try:
                new_beta = next_beta(state.beta, old_logw, ell, config.cess_fraction)
                new_logw, log_increment = normalize_log_weights(
                    old_logw + (new_beta - state.beta) * ell
                )
            except CessSupportDiscontinuity as exc:
                return _fail_evaluation(state, exc, checkpoint_dir)
            except NoTargetSupport as exc:
                return _fail_support(state, exc, checkpoint_dir)
            beta_history = [*state.diagnostics["beta_history"], new_beta]
            state = _replace(
                state,
                log_weights=tuple(new_logw.tolist()),
                log_weights_in_support=tuple(np.isfinite(new_logw).tolist()),
                beta=new_beta,
                level=state.level + 1,
                log_evidence=state.log_evidence + log_increment,
                log_evidence_in_support=True,
                phase="resample",
                cursor=0,
                diagnostics=_diagnostics(state, beta_history=beta_history),
            )
            _persist(state, checkpoint_dir)
            continue

        if state.phase == "resample":
            logw = np.asarray(state.log_weights, dtype=np.float64)
            pre_ess = ess(logw)
            particles = state.particles
            resampling = list(state.diagnostics["resampling"])
            if pre_ess < config.resample_fraction * config.n_particles:
                indices = systematic_resample(logw, rng)
                particles = tuple(
                    Particle(
                        particle_id=index,
                        ancestor_id=state.particles[int(parent)].ancestor_id,
                        evaluation=state.particles[int(parent)].evaluation,
                    )
                    for index, parent in enumerate(indices)
                )
                logw = np.full(config.n_particles, -math.log(config.n_particles))
                resampling.append(
                    {
                        "level": state.level,
                        "pre_ess": pre_ess,
                        "indices": indices.tolist(),
                        "ancestor_ids": [particle.ancestor_id for particle in particles],
                    }
                )
            else:
                resampling.append({"level": state.level, "pre_ess": pre_ess, "indices": []})
            state = _replace(
                state,
                particles=particles,
                log_weights=tuple(logw.tolist()),
                log_weights_in_support=tuple(np.isfinite(logw).tolist()),
                phase="rejuvenate",
                cursor=0,
                rng_state=rng.bit_generator.state,
                diagnostics=_diagnostics(
                    state,
                    ess_history=[*state.diagnostics["ess_history"], pre_ess],
                    resampling=resampling,
                ),
            )
            _persist(state, checkpoint_dir)
            continue

        if state.phase == "rejuvenate":
            total = config.moves_per_level * config.n_particles
            if state.cursor >= total:
                state = _replace(
                    state,
                    phase="finalize" if state.beta == 1.0 else "reweight",
                    cursor=0,
                    pending_proposal=None,
                    pending_log_u=None,
                )
                _persist(state, checkpoint_dir)
                continue
            particle_index = state.cursor % config.n_particles
            current = state.particles[particle_index]
            pending_meta = state.diagnostics.get("pending_move")
            if state.pending_proposal is None:
                move = _choose_move(
                    current.evaluation.theta,
                    proposal,
                    schema,
                    config,
                    probabilities,
                    rng,
                )
                schema.validate_theta(move.proposed)
                log_u = math.log(float(rng.random()))
                pending_meta = {
                    "kernel": move.kernel,
                    "log_reverse_minus_forward": move.log_reverse_minus_forward,
                    "particle_index": particle_index,
                }
                state = _replace(
                    state,
                    pending_proposal=move.proposed,
                    pending_log_u=log_u,
                    rng_state=rng.bit_generator.state,
                    diagnostics=_diagnostics(state, pending_move=pending_meta),
                )
                _persist(state, checkpoint_dir)
                continue
            if not isinstance(pending_meta, dict):
                raise ValueError("pending proposal has no persisted kernel metadata")
            if int(pending_meta["particle_index"]) != particle_index:
                raise ValueError("pending proposal belongs to another particle cursor")
            move = Move(
                proposed=state.pending_proposal,
                log_reverse_minus_forward=float(pending_meta["log_reverse_minus_forward"]),
                kernel=str(pending_meta["kernel"]),
            )
            try:
                proposed = _evaluation(target, proposal, move.proposed)
            except SMCBudgetStop as exc:
                return _stop_budget(state, exc, checkpoint_dir)
            except Exception as exc:
                return _fail_evaluation(state, exc, checkpoint_dir)
            assert state.pending_log_u is not None
            log_alpha = mh_log_accept(current.evaluation, proposed, state.beta, move)
            accepted = state.pending_log_u < log_alpha
            moved_particles = list(state.particles)
            if accepted:
                moved_particles[particle_index] = current.model_copy(
                    update={"evaluation": proposed}
                )
            moves = [
                *state.diagnostics["moves"],
                {
                    "level": state.level,
                    "particle_index": particle_index,
                    "kernel": move.kernel,
                    "accepted": accepted,
                    "log_alpha": log_alpha if math.isfinite(log_alpha) else None,
                    "log_alpha_in_support": math.isfinite(log_alpha),
                },
            ]
            diagnostics = dict(state.diagnostics)
            diagnostics.pop("pending_move", None)
            diagnostics["moves"] = moves
            state = _replace(
                state,
                particles=tuple(moved_particles),
                cursor=state.cursor + 1,
                pending_proposal=None,
                pending_log_u=None,
                diagnostics=diagnostics,
            )
            _persist(state, checkpoint_dir)
            continue

        if state.phase == "finalize":
            state = _replace(state, algorithm_status="COMPLETE")
            _persist(state, checkpoint_dir)
            return state

        raise RuntimeError(f"unknown SMC phase {state.phase!r}")


def infer(
    target: Target,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
    checkpoint_dir: Path,
    stop_requested: Callable[[], bool],
) -> SMCState:
    """Start a new SMC state and drive it until completion, stop or failure."""
    latest = checkpoint_dir / LATEST_STATE_FILENAME
    if latest.exists():
        raise FileExistsError(
            f"working checkpoint {latest} already exists; load and continue it or choose "
            "a new run directory"
        )
    state = _initial_state(target, proposal, schema, config)
    _persist(state, checkpoint_dir)
    return _drive(state, target, proposal, schema, config, checkpoint_dir, stop_requested)


def continue_inference(
    state: SMCState,
    target: Target,
    proposal: Density,
    schema: DensitySchema,
    config: InferenceConfig,
    checkpoint_dir: Path,
    stop_requested: Callable[[], bool],
) -> SMCState:
    """Continue the exact stored phase/cursor/RNG state; never rerun initialization."""
    _validate_resume(state, target, proposal, schema, config)
    if state.algorithm_status == "COMPLETE":
        return state
    return _drive(state, target, proposal, schema, config, checkpoint_dir, stop_requested)


__all__ = ["SMCBudgetStop", "Target", "continue_inference", "infer"]
