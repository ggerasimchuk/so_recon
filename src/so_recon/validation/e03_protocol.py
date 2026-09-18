"""E03 scientific-target identity and posterior-claim labelling (plan §4.3, §4.4).

The E02 `PhysicalTarget.fingerprint` is a RUNTIME identity: it deliberately includes the
proposal, so a checkpoint cannot be resumed against another sampler law. That strictness
makes it useless for the E03 question «do B1 and ML infer the SAME posterior?» — a learned
run and a prior-start run of one world must be comparable even though their runtime
fingerprints differ. §4.3 therefore fixes a second, SCIENTIFIC identity of the target:
everything that defines the mathematical posterior, nothing that defines how it was
sampled.

Included (§4.3): prior/renderer/latent semantics (the prior fingerprint plus the renderer
identity and the declared design), G/information/cutoff, observation values/masks/roles
(the observation semantic hash) with the likelihood/noise operator definitions, geometry/
controls/fluids/boundary (the design the renderer turns into a case), and the physical
solver/discretization (solver settings, adapter sources, environment lock).

Excluded (§4.3): the learned proposal and its epsilon, training seed, particle count N,
inference seed, and the choice of diagnostic outputs — a different `state_times_s` request
still changes the runtime fingerprint and the cache keys, but it cannot change the
posterior, and this identity says so.

`PhysicalTarget.fingerprint` and `checkpoint_hashes` are untouched and remain the only
identities a checkpoint resume accepts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml

from so_recon.geology.renderer import renderer_hash
from so_recon.registry.hashing import sha256_json
from so_recon.synthetic.inverse_corpus import (
    TRUTH_BALANCE_CUMULATIVE_MAX,
    TRUTH_BALANCE_STEP_MEDIAN_MAX,
)

SCIENTIFIC_TARGET_IDENTITY_KIND = "e03-scientific-target-identity-1"

#: §4.4: a raw draw set from the learned law is never a corrected ensemble. Any artifact
#: publishing one must say so with exactly this marker; only beta=1 COMPLETE runs may use
#: the posterior label.
RAW_PROPOSAL_ENSEMBLE_KIND = "raw_proposal"
#: §4.4/§10.1: B0 is the static conditional prior — p0 draws with proven provenance
#: (`log_r == log_p0`). It is a legitimate ensemble with state evidence, but it is the
#: UNCORRECTED prior: it never carries a posterior claim, and it is neither a raw learned
#: proposal (that marker is q's) nor a beta<1 diagnostic partial.
PRIOR_ENSEMBLE_KIND = "prior_ensemble"
POSTERIOR_ENSEMBLE_KIND = "posterior"
DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND = "diagnostic_partial"

#: §0.2 item 10: the E03 normative truth-balance thresholds, restated so a run payload can
#: cite the numbers its scope runs under without inheriting the E02 runner's 1e-4 step
#: allowance. The single source is `synthetic.inverse_corpus`; `configs/e03_tolerances.yml`
#: must agree with it (see `verify_balance_threshold_config`).
E03_BALANCE_THRESHOLDS: dict[str, float] = {
    "truth_balance_cumulative_max": TRUTH_BALANCE_CUMULATIVE_MAX,
    "truth_balance_step_median_max": TRUTH_BALANCE_STEP_MEDIAN_MAX,
}

E03_TOLERANCES_RELPATH = "configs/e03_tolerances.yml"


class ScientificTarget(Protocol):
    """The §4.3 inputs, as `PhysicalTarget` already carries them."""

    prior: Any
    context: Any
    observations: Any
    observation_semantic_hash: str
    operator_hash: str
    solver_hash: str
    adapter_hash: str
    worker: Any


def scientific_target_identity(target: ScientificTarget) -> str:
    """The identity of the mathematical posterior, blind to how it is sampled.

    Changing the proposal (learned or prior-start, any epsilon) must NOT change this value;
    changing physics or data MUST. Runtime/checkpoint fingerprints stay strict and separate.
    """
    return sha256_json(
        {
            "kind": SCIENTIFIC_TARGET_IDENTITY_KIND,
            # prior + latent semantics: schema, families, measure, transform, basis, and
            # the G/information identities the conditioning hangs on.
            "prior": target.prior.fingerprint,
            # renderer + design: geometry/controls/fluids/boundary semantics that turn a
            # theta into one physical case.
            "renderer": renderer_hash(target.context),
            "design": target.context.design,
            "g_hash": target.context.g_hash,
            "information_hash": target.context.information_hash,
            # observation values/masks/roles, the cutoff, and the likelihood/noise
            # operator definitions that score predictions against them.
            "cutoff_s": float(target.observations.cutoff_s),
            "observation_semantic_hash": target.observation_semantic_hash,
            "operator_hash": target.operator_hash,
            # the physical solver/discretization: settings, adapter sources, environment.
            "solver_hash": target.solver_hash,
            "adapter_hash": target.adapter_hash,
            "environment_lock_hash": target.worker.environment_lock_hash,
        }
    )


def completed_posterior(beta: float, algorithm_status: str) -> bool:
    """Only a beta=1 COMPLETE run has a posterior to claim (§4.4, ledger ruling)."""
    return algorithm_status == "COMPLETE" and float(beta) == 1.0


def ensemble_kind(beta: float, algorithm_status: str) -> str:
    """The §4.4 label: `posterior` only at beta=1 COMPLETE, else an explicit diagnostic."""
    if completed_posterior(beta, algorithm_status):
        return POSTERIOR_ENSEMBLE_KIND
    return DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND


def verify_balance_threshold_config(root: Path) -> dict[str, float]:
    """Refuse to run E03 inference unless the frozen config carries the §0.2 item 10 law.

    `configs/e03_tolerances.yml` is frozen BEFORE evaluation, so this is a cross-check of
    two published statements of one number, not a threshold read at run time: the learned
    loop starts only when the config's physics block equals the corpus constants (1e-3
    cumulative, 1e-5 median step — the SPEC number, not E02's inherited 1e-4).
    """
    path = root / E03_TOLERANCES_RELPATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("physics"), dict):
        raise ValueError(f"{E03_TOLERANCES_RELPATH} carries no physics thresholds block")
    physics = dict(payload["physics"])
    for key, expected in E03_BALANCE_THRESHOLDS.items():
        if key not in physics:
            raise ValueError(f"{E03_TOLERANCES_RELPATH} physics block omits {key!r}")
        if float(physics[key]) != expected:
            raise ValueError(
                f"{E03_TOLERANCES_RELPATH} declares {key}={physics[key]!r}, but the E03 "
                f"scope is fixed at {expected!r} (plan §0.2 item 10): the config and the "
                "code disagree on the normative balance law"
            )
    return dict(E03_BALANCE_THRESHOLDS)


__all__ = [
    "DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND",
    "E03_BALANCE_THRESHOLDS",
    "E03_TOLERANCES_RELPATH",
    "POSTERIOR_ENSEMBLE_KIND",
    "RAW_PROPOSAL_ENSEMBLE_KIND",
    "SCIENTIFIC_TARGET_IDENTITY_KIND",
    "completed_posterior",
    "ensemble_kind",
    "scientific_target_identity",
    "verify_balance_threshold_config",
]
