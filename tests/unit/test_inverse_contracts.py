"""E02.1: the typed inverse input, checked before any of it can reach a solver.

Every record E02 passes between prior, observation model, SMC engine and report is defined
once, in `so_recon.inference.contracts`, and every one of them refuses a malformed value at
construction. The cases here are the ones that would otherwise be silent: a latent vector
that is shorter than the schema it claims (truncation, not an error), a family the schema
does not carry, a mixture whose weights do not sum to one (renormalised behind the reader's
back), a NaN coordinate, and an unknown field that a typo would otherwise create.

The configuration half is the same rule one level up: `inference` is a 4.0 field, a 3.0
configuration may not set one, and a 4.0 configuration without one is legal — absence is
how every E01 command that needs no inference settings keeps working unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from so_recon.config.inference import InferenceConfig
from so_recon.config.load import SPEC_4_0_ONLY_FIELDS, ConfigError, load_project_config
from so_recon.inference.contracts import (
    LOG_BIAS_SCALE,
    NU_FIXED,
    OBSERVATION_ROLES,
    PROBABILITY_SUM_TOLERANCE,
    RHO_MAX,
    SIGMA_MIN,
    SIGMA_SPAN,
    DensitySchema,
    E01DependencyError,
    HistoryRow,
    LikelihoodNumericalError,
    LoglikResult,
    LogRow,
    ModelObservations,
    NoiseTheta,
    ObservationBundle,
    ObservationSupportMismatch,
    Particle,
    PosteriorBundle,
    PriorContext,
    RendererNumericalError,
    SMCState,
    TargetEvaluation,
    ThetaRecord,
    UnsupportedObservationError,
    validate_probability_weights,
)
from so_recon.registry.artifact import ArtifactRef

BASIS = "a" * 64
OTHER_BASIS = "b" * 64


# --------------------------------------------------------------------------------------
# the latent record and its schema
# --------------------------------------------------------------------------------------


def _schema(**over: object) -> DensitySchema:
    base: dict[str, object] = {
        "schema_id": "test",
        "n_v": 2,
        "n_residual": 1,
        "families": (0, 1),
        "basis_hash": BASIS,
        "transform_version": "identity-1",
    }
    base.update(over)
    return DensitySchema(**base)  # type: ignore[arg-type]


def _theta(**over: object) -> ThetaRecord:
    base: dict[str, object] = {
        "schema_id": "test",
        "s": 0,
        "v": (0.0, 0.0),
        "z_perp": (1.0,),
        "basis_hash": BASIS,
    }
    base.update(over)
    return ThetaRecord(**base)  # type: ignore[arg-type]


def test_dimension_is_not_silently_truncated() -> None:
    schema = DensitySchema(
        schema_id="test",
        n_v=2,
        n_residual=1,
        families=(0, 1),
        basis_hash="a" * 64,
        transform_version="identity-1",
    )
    theta = ThetaRecord(schema_id="test", s=0, v=(0.0,), z_perp=(1.0,), basis_hash="a" * 64)
    with pytest.raises(ValueError, match="dimension"):
        schema.validate_theta(theta)


def test_residual_dimension_is_not_silently_truncated() -> None:
    """SPEC 7.5: the remaining modes are kept. A short residual is a dropped mode."""
    with pytest.raises(ValueError, match="dimension"):
        _schema().validate_theta(_theta(z_perp=()))


def test_family_outside_the_declared_support_is_refused() -> None:
    with pytest.raises(ValueError, match="family outside support"):
        _schema().validate_theta(_theta(s=2))


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"schema_id": "other"}, "schema/basis mismatch"),
        ({"basis_hash": OTHER_BASIS}, "schema/basis mismatch"),
    ],
)
def test_a_theta_from_another_schema_is_refused(over: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _schema().validate_theta(_theta(**over))


def test_a_valid_theta_passes_its_schema() -> None:
    assert _schema().validate_theta(_theta()) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_nonfinite_coordinate_is_refused(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        _theta(v=(0.0, bad))
    with pytest.raises(ValidationError, match="finite"):
        _theta(z_perp=(bad,))


def test_a_negative_family_index_is_refused() -> None:
    with pytest.raises(ValidationError):
        _theta(s=-1)


@pytest.mark.parametrize(
    "over",
    [
        {"families": ()},
        {"families": (1, 0)},
        {"families": (0, 0)},
        {"families": (-1, 0)},
        {"n_v": 0},
        {"n_residual": -1},
    ],
)
def test_a_malformed_density_schema_is_refused(over: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _schema(**over)


def test_unknown_fields_are_refused_by_every_record() -> None:
    """`extra='forbid'`: a typo must be an error, not a field nobody reads."""
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        _schema(n_latent=3)
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        _theta(log_prob=0.0)


def test_the_latent_measure_is_named_and_fixed() -> None:
    assert _schema().measure == "counting_x_latent_lebesgue"
    with pytest.raises(ValidationError):
        _schema(measure="lebesgue")


# --------------------------------------------------------------------------------------
# probability weights
# --------------------------------------------------------------------------------------


def test_probability_weights_are_checked_and_never_renormalised() -> None:
    weights = (0.25, 0.75)
    assert validate_probability_weights(weights, label="date_weights") == weights
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probability_weights((0.5, 0.6), label="date_weights")
    with pytest.raises(ValueError, match="positive"):
        validate_probability_weights((0.0, 1.0), label="date_weights")
    with pytest.raises(ValueError, match="positive"):
        validate_probability_weights((-0.1, 1.1), label="date_weights")


def test_the_probability_tolerance_is_the_one_the_plan_fixes() -> None:
    assert PROBABILITY_SUM_TOLERANCE == 1e-12
    off = 1.0 + 5e-12
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probability_weights((off,), label="date_weights")


# --------------------------------------------------------------------------------------
# the noise law
# --------------------------------------------------------------------------------------


def test_noise_theta_is_the_declared_function_of_the_nuisance_coordinates() -> None:
    at_zero = NoiseTheta.from_latent((0.0,) * 11)
    assert at_zero.nu == NU_FIXED == 5.0
    assert at_zero.sigma == pytest.approx(SIGMA_MIN + SIGMA_SPAN * 0.5)
    assert at_zero.rho == pytest.approx(RHO_MAX * 0.5)
    assert at_zero.log_bias == 0.0
    high = NoiseTheta.from_latent((0.0,) * 8 + (8.0, 8.0, 2.0))
    assert high.sigma == pytest.approx(SIGMA_MIN + SIGMA_SPAN, abs=1e-12)
    assert high.rho == pytest.approx(RHO_MAX, abs=1e-12)
    assert high.log_bias == pytest.approx(2.0 * LOG_BIAS_SCALE)


def test_a_latent_vector_without_the_nuisance_block_is_refused() -> None:
    with pytest.raises(ValueError, match="nuisance"):
        NoiseTheta.from_latent((0.0,) * 8)


@pytest.mark.parametrize(
    "over",
    [{"sigma": 0.0}, {"sigma": -0.01}, {"rho": 1.0}, {"rho": -0.1}, {"nu": 2.0}],
)
def test_a_noise_law_outside_its_domain_is_refused(over: dict[str, float]) -> None:
    base = {"sigma": 0.03, "rho": 0.4, "nu": 5.0, "log_bias": 0.0}
    base.update(over)
    with pytest.raises(ValidationError):
        NoiseTheta(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------


def _history(**over: object) -> HistoryRow:
    base: dict[str, object] = {
        "well_id": "P1",
        "month_index": 0,
        "raw_value": 0.3,
        "bin_index": 1,
        "quality_group": "metered",
        "observed_valid": True,
        "reset": False,
    }
    base.update(over)
    return HistoryRow(**base)  # type: ignore[arg-type]


def _log(**over: object) -> LogRow:
    base: dict[str, object] = {
        "observation_id": "L1",
        "support_cell_ids": (3, 4),
        "support_weights": (0.5, 0.5),
        "bin_index": 0,
        "date_times_s": (0.0, 86400.0),
        "date_weights": (0.25, 0.75),
        "date_group": "D1",
        "bias_group": "logs",
        "valid": True,
        "use": "condition_prior",
    }
    base.update(over)
    return LogRow(**base)  # type: ignore[arg-type]


def _bundle(**over: object) -> ObservationBundle:
    base: dict[str, object] = {
        "history": (_history(), _history(month_index=1)),
        "logs": (_log(),),
        "bin_edges_by_group": {
            "metered": (0.0, 0.25, 0.75, 1.0),
            "logs": (-40.0, -30.0, -20.0),
        },
        "cutoff_s": 86400.0,
        "information_hash": "c" * 64,
        "observation_hash": "d" * 64,
    }
    base.update(over)
    return ObservationBundle(**base)  # type: ignore[arg-type]


def test_the_observation_role_registry_is_the_declared_one() -> None:
    assert OBSERVATION_ROLES == ("condition_prior", "heldout_diagnostic", "likelihood")
    with pytest.raises(ValidationError):
        _log(use="scoring")


def test_a_history_row_that_claims_a_value_names_its_bin() -> None:
    with pytest.raises(ValidationError, match="bin"):
        _history(bin_index=None)
    with pytest.raises(ValidationError, match="bin"):
        _history(observed_valid=False, raw_value=None)
    assert _history(observed_valid=False, raw_value=None, bin_index=None).bin_index is None


def test_a_history_row_multiplier_comes_from_observed_quality() -> None:
    assert _history().sigma_multiplier == 1.0
    with pytest.raises(ValidationError):
        _history(sigma_multiplier=0.0)
    with pytest.raises(ValidationError):
        _history(sigma_multiplier=-2.0)


def test_a_log_row_keeps_its_support_and_its_date_mixture_aligned() -> None:
    with pytest.raises(ValidationError, match="support"):
        _log(support_weights=(1.0,))
    with pytest.raises(ValidationError, match="date"):
        _log(date_weights=(1.0,))
    with pytest.raises(ValidationError, match="sum to 1"):
        _log(date_weights=(0.5, 0.6))
    with pytest.raises(ValidationError, match="sum to 1"):
        _log(support_weights=(0.5, 0.6))


def test_a_log_row_refuses_a_repeated_or_unordered_date() -> None:
    with pytest.raises(ValidationError, match="increasing"):
        _log(date_times_s=(86400.0, 0.0))
    with pytest.raises(ValidationError, match="increasing"):
        _log(date_times_s=(0.0, 0.0))
    with pytest.raises(ValidationError, match="unique"):
        _log(support_cell_ids=(3, 3))


def test_a_bundle_keys_history_by_sorted_unique_well_and_month() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        _bundle(history=(_history(month_index=1), _history(month_index=0)))
    with pytest.raises(ValidationError, match="unique|sorted"):
        _bundle(history=(_history(), _history()))


def test_a_bundle_refuses_a_bin_index_outside_its_group_edges() -> None:
    with pytest.raises(ValidationError, match="bin"):
        _bundle(history=(_history(bin_index=3),))
    with pytest.raises(ValidationError, match="group"):
        _bundle(history=(_history(quality_group="unmetered"),))


def test_a_bundle_carries_one_bias_group() -> None:
    """One bias group per bundle: a second one needs its own anchored latent, not a share."""
    edges = {
        "metered": (0.0, 0.25, 0.75, 1.0),
        "logs": (-40.0, -30.0, -20.0),
        "cores": (-40.0, -30.0, -20.0),
    }
    with pytest.raises(ValidationError, match="bias_group"):
        _bundle(
            logs=(_log(), _log(observation_id="L2", bias_group="cores")),
            bin_edges_by_group=edges,
        )


def test_a_bundle_refuses_unordered_bin_edges() -> None:
    with pytest.raises(ValidationError, match="increasing"):
        _bundle(bin_edges_by_group={"metered": (1.0, 0.5, 0.0), "logs": (-40.0, -30.0, -20.0)})
    with pytest.raises(ValidationError, match="edges"):
        _bundle(bin_edges_by_group={"metered": (0.5,), "logs": (-40.0, -30.0, -20.0)})


def test_model_observations_stay_inside_their_physical_range() -> None:
    obs = ModelObservations(
        fw={("P1", 0): 0.25, ("P1", 1): None},
        so_support={("L1", 0.0): 0.6},
        model_hash="e" * 64,
    )
    assert obs.fw[("P1", 1)] is None
    with pytest.raises(ValidationError):
        ModelObservations(fw={("P1", 0): 1.5}, so_support={}, model_hash="e" * 64)
    with pytest.raises(ValidationError):
        ModelObservations(fw={}, so_support={("L1", 0.0): -0.1}, model_hash="e" * 64)


def test_an_empty_observation_set_gives_a_zero_loglikelihood() -> None:
    empty = LoglikResult(value=0.0, terms=(), n_used=0, observation_hash="d" * 64)
    assert (empty.value, empty.terms, empty.n_used) == (0.0, (), 0)
    with pytest.raises(ValidationError, match="n_used"):
        LoglikResult(value=0.0, terms=(-1.0,), n_used=0, observation_hash="d" * 64)
    with pytest.raises(ValidationError, match="no observations"):
        LoglikResult(value=-3.0, terms=(), n_used=0, observation_hash="d" * 64)


def test_a_loglikelihood_may_be_minus_infinity_but_never_nan() -> None:
    outside = LoglikResult(value=-math.inf, terms=(-math.inf,), n_used=1, observation_hash="d" * 64)
    assert outside.value == -math.inf
    with pytest.raises(ValidationError):
        LoglikResult(value=math.nan, terms=(math.nan,), n_used=1, observation_hash="d" * 64)
    with pytest.raises(ValidationError):
        LoglikResult(value=math.inf, terms=(math.inf,), n_used=1, observation_hash="d" * 64)


# --------------------------------------------------------------------------------------
# the conditional prior context
# --------------------------------------------------------------------------------------


def _prior(**over: object) -> PriorContext:
    base: dict[str, object] = {
        "density_schema": _schema(schema_id="p1", n_v=11, n_residual=4),
        "n_geology": 3,
        "n_state_residual": 0,
        "mean": np.zeros(3),
        "chol": np.eye(3),
        "rotation": np.eye(3),
        "design": {"wells": 4},
        "g_hash": "f" * 64,
        "information_hash": "c" * 64,
    }
    base.update(over)
    return PriorContext(**base)  # type: ignore[arg-type]


def test_a_prior_context_checks_the_shapes_it_declares() -> None:
    ctx = _prior()
    assert ctx.mean.shape == (3,)
    with pytest.raises(ValidationError, match="shape"):
        _prior(mean=np.zeros(4))
    with pytest.raises(ValidationError, match="shape"):
        _prior(chol=np.eye(4))
    with pytest.raises(ValidationError, match="shape"):
        _prior(rotation=np.zeros((3, 4)))


def test_a_prior_context_refuses_a_degenerate_or_nonfinite_factor() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _prior(mean=np.array([0.0, math.nan, 0.0]))
    chol = np.eye(3)
    chol[1, 1] = 0.0
    with pytest.raises(ValidationError, match="diagonal"):
        _prior(chol=chol)


def test_a_prior_context_arrays_cannot_be_mutated_afterwards() -> None:
    source = np.zeros(3)
    ctx = _prior(mean=source)
    source[0] = 5.0
    assert ctx.mean[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        ctx.mean[0] = 5.0


# --------------------------------------------------------------------------------------
# the SMC state and what it is allowed to call a posterior
# --------------------------------------------------------------------------------------


def _artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id="1" * 64,
        path=f"artifacts/{name}.json",
        sha256="1" * 64,
        size_bytes=2,
        media_type="application/json",
        schema_version="smc-1",
        producer_run_id="run",
        parent_artifact_ids=[],
        created_at="2026-09-15T00:00:00+00:00",
    )


def _evaluation(**over: object) -> TargetEvaluation:
    base: dict[str, object] = {
        "theta": _theta(),
        "log_p0": -1.0,
        "log_l": -2.0,
        "log_r": -1.0,
        "forward_ref": None,
        "cache_key": "k0",
    }
    base.update(over)
    return TargetEvaluation(**base)  # type: ignore[arg-type]


def _state(**over: object) -> SMCState:
    base: dict[str, object] = {
        "particles": (Particle(particle_id=0, ancestor_id=0, evaluation=_evaluation()),),
        "log_weights": (0.0,),
        "beta": 0.5,
        "level": 1,
        "log_evidence": -0.5,
        "phase": "reweight",
        "cursor": 0,
        "rng_state": {"bit_generator": "PCG64"},
        "proposal_hash": "2" * 64,
        "target_hash": "3" * 64,
        "pending_proposal": None,
        "pending_log_u": None,
        "algorithm_status": "INCOMPLETE_BUDGET",
        "diagnostics": {},
    }
    base.update(over)
    return SMCState(**base)  # type: ignore[arg-type]


def test_log_prior_parts_are_diagnostics_on_the_particle() -> None:
    particle = Particle(
        particle_id=0,
        ancestor_id=0,
        evaluation=_evaluation(),
        log_prior_parts={"geology": -0.5, "nuisance": -0.5},
    )
    assert particle.log_prior_parts["geology"] == -0.5
    # The trusted number stays the evaluation's own; the parts never replace it.
    assert particle.evaluation.log_p0 == -1.0


def test_an_smc_state_weighs_exactly_its_particles() -> None:
    assert _state().level == 1
    with pytest.raises(ValidationError, match="log_weights"):
        _state(log_weights=(0.0, 0.0))
    with pytest.raises(ValidationError, match="beta"):
        _state(beta=1.5)
    with pytest.raises(ValidationError):
        _state(phase="polish")


def test_a_pending_metropolis_proposal_keeps_its_threshold() -> None:
    with pytest.raises(ValidationError, match="pending"):
        _state(pending_proposal=_theta())
    with pytest.raises(ValidationError, match="pending"):
        _state(pending_log_u=-0.3)
    paired = _state(pending_proposal=_theta(), pending_log_u=-0.3)
    assert paired.pending_log_u == -0.3


def _posterior(**over: object) -> PosteriorBundle:
    base: dict[str, object] = {
        "state_ref": _artifact("state"),
        "particles_ref": _artifact("particles"),
        "diagnostics_ref": _artifact("diagnostics"),
        "ledger_ref": _artifact("ledger"),
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
        "convergence_status": "CONVERGED",
        "physical_state_refs": (),
        "parent_run_ids": ("run",),
    }
    base.update(over)
    return PosteriorBundle(**base)  # type: ignore[arg-type]


def test_an_unfinished_tempering_is_never_called_a_posterior() -> None:
    """SPEC 13.4.2: beta<1 is an intermediate distribution, not a posterior."""
    with pytest.raises(ValidationError, match="beta"):
        _posterior(beta=0.4)
    stopped = _posterior(
        algorithm_status="INCOMPLETE_BUDGET", beta=0.4, convergence_status="NOT_ASSESSED"
    )
    assert stopped.beta == 0.4


def test_process_status_and_science_status_are_not_merged() -> None:
    with pytest.raises(ValidationError, match="convergence"):
        _posterior(convergence_status="NOT_CONVERGED")
    unconverged = _posterior(
        algorithm_status="POSTERIOR_NOT_CONVERGED", convergence_status="NOT_CONVERGED"
    )
    assert unconverged.beta == 1.0
    failed = _posterior(
        algorithm_status="EVALUATION_FAILURE", beta=0.2, convergence_status="NOT_ASSESSED"
    )
    assert failed.beta == 0.2


# --------------------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "base"),
    [
        (E01DependencyError, RuntimeError),
        (RendererNumericalError, RuntimeError),
        (LikelihoodNumericalError, RuntimeError),
        (ObservationSupportMismatch, ValueError),
        (UnsupportedObservationError, ValueError),
    ],
)
def test_every_refusal_keeps_its_reason(error: type[Exception], base: type[Exception]) -> None:
    raised = error("because the support does not match")
    assert isinstance(raised, base)
    assert raised.reason == "because the support does not match"  # type: ignore[attr-defined]
    assert str(raised) == "because the support does not match"


# --------------------------------------------------------------------------------------
# the configuration block
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

INFERENCE_YAML = """
inference:
  seed: 20260915
  n_particles: 32
"""


def test_inference_settings_carry_the_declared_defaults() -> None:
    cfg = InferenceConfig(seed=1)
    assert cfg.n_particles == 32
    assert cfg.cess_fraction == 0.8
    assert cfg.resample_fraction == 0.5
    assert cfg.max_beta_steps == 24
    assert cfg.moves_per_level == 2
    assert cfg.rw_scale == 0.25
    assert cfg.pcn_scale == 0.2


@pytest.mark.parametrize(
    "over",
    [
        {"n_particles": 1},
        {"cess_fraction": 0.0},
        {"cess_fraction": 1.5},
        {"resample_fraction": 0.0},
        {"max_beta_steps": 0},
        {"moves_per_level": 0},
        {"rw_scale": 0.0},
        {"pcn_scale": 1.0},
    ],
)
def test_inference_settings_outside_their_domain_are_refused(over: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(seed=1, **over)  # type: ignore[arg-type]


def test_a_3_0_config_may_not_declare_inference_settings(tmp_project: Path) -> None:
    path = tmp_project / "configs" / "project.yml"
    path.write_text(path.read_text(encoding="utf-8") + INFERENCE_YAML, encoding="utf-8")
    with pytest.raises(ConfigError, match="inference"):
        load_project_config(path)


def test_a_4_0_config_without_inference_settings_is_legal(tmp_project: Path) -> None:
    path = tmp_project / "configs" / "project.yml"
    path.write_text(path.read_text(encoding="utf-8").replace('"3.0"', '"4.0"'), encoding="utf-8")
    cfg = load_project_config(path)
    assert cfg.spec_version == "4.0"
    assert cfg.inference is None
    # The E01 configuration is unchanged by this addition and still reads.
    e01 = load_project_config(REPO_ROOT / "configs" / "e01.yml")
    assert e01.inference is None
    assert e01.resources is not None


def test_a_4_0_config_may_declare_inference_settings(tmp_project: Path) -> None:
    path = tmp_project / "configs" / "project.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace('"3.0"', '"4.0"') + INFERENCE_YAML,
        encoding="utf-8",
    )
    cfg = load_project_config(path)
    assert cfg.inference is not None
    assert cfg.inference.seed == 20260915
    assert cfg.inference.n_particles == 32


def test_inference_is_registered_as_a_4_0_only_field() -> None:
    """A 3.0 dump must not gain a key: every historical resolved_config_hash depends on it."""
    assert "inference" in SPEC_4_0_ONLY_FIELDS
