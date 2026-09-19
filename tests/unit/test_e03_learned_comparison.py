"""E03 Task 08 — the comparison skeleton (plan §4.4, §10.1, §10.2).

Rows this suite pins:

* B0 is the static conditional prior only: the selection reads prior-start initial
  draws whose sampling density EQUALS p0 (`log_r == log_p0`, the artifact-level
  provenance proof), demands their physical forwards, and never consults the learned
  proposal — so changing q cannot change the selected B0;
* a raw-q ensemble is labelled `ensemble_kind=raw_proposal` and B0's draws
  `ensemble_kind=prior_ensemble`; neither ever carries a posterior claim (plan §4.4);
* world averaging weights independent worlds equally — seed-averaged per world first,
  never a pooled pseudo-sample over rows or cells;
* run diagnostics (beta path, pre-resampling ESS, per-kernel acceptance, ancestors,
  physical states, runtime/disk/failures) are extracted from real run payload shapes;
* the artifact loader maps each particle to ITS OWN forward, dedupes shared forwards,
  and builds operational products without any truth input (the loader's signature is
  part of the contract: no truth parameter exists to misuse).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from so_recon.paths import ProjectPaths
from so_recon.validation.e03_protocol import (
    POSTERIOR_ENSEMBLE_KIND,
    PRIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
)
from so_recon.validation.ensemble_states import (
    ZoneSupport,
    aggregate_particle_zones,
    ensemble_state_products,
    score_ensemble_states,
)
from so_recon.validation.learned_comparison import (
    average_over_worlds,
    b0_from_prior_start_checkpoint,
    comparison_row,
    load_particle_states,
    operational_products_from_bundle,
    run_diagnostics_from_payloads,
)
from so_recon.validation.proposal_diagnostics import ResidualSensitivity

ONE_ZONE = ZoneSupport(
    names=("both-cells",), matrix=np.array([[1.0, 1.0]]), kind="primary_quadrants"
)


def _ref(index: int) -> dict[str, Any]:
    return {
        "artifact_id": f"{index:064d}",
        "created_at": "2026-09-18T00:00:00+00:00",
        "media_type": "application/json",
        "parent_artifact_ids": [],
        "path": f"artifacts/forwards/f{index}/forward_result.json",
        "producer_run_id": f"run-{index}",
        "schema_version": "forward-1",
        "sha256": f"{index:064d}",
        "size_bytes": 100,
    }


def _theta(s: int) -> dict[str, Any]:
    return {
        "schema_id": "p1-conditional-12",
        "s": s,
        "v": [0.1, 0.2],
        "z_perp": [0.3],
        "basis_hash": "b" * 64,
    }


def _evaluation(index: int, s: int, *, log_r: float | None = None) -> dict[str, Any]:
    log_p0 = -3.0 - index
    return {
        "cache_key": f"c{index}",
        "forward_ref": _ref(index),
        "log_l": -1.0,
        "log_l_in_support": True,
        "log_p0": log_p0,
        "log_p0_in_support": True,
        "log_r": log_p0 if log_r is None else log_r,
        "log_r_in_support": True,
        "theta": _theta(s),
    }


def _checkpoint_manifest(
    *, log_r: float | None = None, with_forwards: bool = True
) -> dict[str, Any]:
    def evaluation(index: int, s: int) -> dict[str, Any]:
        payload = _evaluation(index, s, log_r=log_r)
        if not with_forwards:
            payload["forward_ref"] = None
        return payload

    return {
        "state": {
            "particles": [
                {"particle_id": 0, "ancestor_id": 0, "evaluation": evaluation(0, 0)},
                {"particle_id": 1, "ancestor_id": 0, "evaluation": evaluation(1, 1)},
            ],
            "log_weights": [-0.5, -0.7],
            "diagnostics": {
                "initial_evaluations": [
                    evaluation(0, 0),
                    evaluation(1, 1),
                ],
            },
        }
    }


# --------------------------------------------------------------------------------------
# B0: the static conditional prior, provenance-proven and source-independent
# --------------------------------------------------------------------------------------


def test_b0_selects_prior_start_initial_draws_with_provenance() -> None:
    manifest = _checkpoint_manifest()
    b0 = b0_from_prior_start_checkpoint(manifest)
    assert b0.provenance == "prior_start_initial_draws_log_r_equals_log_p0"
    assert [eval_["theta"]["s"] for eval_ in b0.evaluations] == [0, 1]
    assert [eval_["forward_ref"]["path"] for eval_ in b0.evaluations] == [
        "artifacts/forwards/f0/forward_result.json",
        "artifacts/forwards/f1/forward_result.json",
    ]


def test_changing_q_does_not_change_the_selected_b0() -> None:
    base = _checkpoint_manifest()
    learned = {**base, "proposal": {"q_fingerprint": "a" * 64, "epsilon": 0.1}}
    other_q = {**base, "proposal": {"q_fingerprint": "c" * 64, "epsilon": 0.2}}
    no_proposal = {k: v for k, v in base.items() if k != "proposal"}
    first = b0_from_prior_start_checkpoint(learned)
    assert first == b0_from_prior_start_checkpoint(other_q)
    assert first == b0_from_prior_start_checkpoint(no_proposal)


def test_a_learned_initial_ensemble_is_not_b0() -> None:
    # the defensive mixture r != p0: log_r differs from log_p0, §0.2 item 6
    manifest = _checkpoint_manifest(log_r=-1.7)
    with pytest.raises(ValueError, match="log_r"):
        b0_from_prior_start_checkpoint(manifest)


def test_b0_requires_physical_forwards() -> None:
    manifest = _checkpoint_manifest(with_forwards=False)
    with pytest.raises(ValueError, match="forward"):
        b0_from_prior_start_checkpoint(manifest)


# --------------------------------------------------------------------------------------
# raw q never carries a posterior claim
# --------------------------------------------------------------------------------------


def _products_fixture() -> tuple:
    zones = aggregate_particle_zones(
        so=np.array([[0.2, 0.6], [0.4, 0.2]]),
        pv=np.array([[1.0, 3.0], [3.0, 1.0]]),
        bo=np.ones((2, 2)),
        support=ONE_ZONE,
    )
    weights = np.array([0.5, 0.5])
    products = ensemble_state_products(
        zones, weights, support=ONE_ZONE, s_labels=(0, 1), admissible_s=(0, 1)
    )
    scores = score_ensemble_states(
        zones,
        weights,
        truth_so=np.array([0.5, 0.5]),
        truth_pv=np.array([1.0, 1.0]),
        support=ONE_ZONE,
    )
    return products, scores


def test_comparison_row_refuses_a_raw_proposal_posterior_claim() -> None:
    products, scores = _products_fixture()
    with pytest.raises(ValueError, match="posterior"):
        comparison_row(
            parent_id="e02-t1-v1-0008",
            method_id="Q",
            inference_seed=11,
            ensemble_kind=RAW_PROPOSAL_ENSEMBLE_KIND,
            posterior_claim=True,
            products=products,
            scores=scores,
        )
    row = comparison_row(
        parent_id="e02-t1-v1-0008",
        method_id="Q",
        inference_seed=11,
        ensemble_kind=RAW_PROPOSAL_ENSEMBLE_KIND,
        posterior_claim=False,
        products=products,
        scores=scores,
    )
    assert row["posterior_claim"] is False
    assert row["ensemble_kind"] == RAW_PROPOSAL_ENSEMBLE_KIND


def test_comparison_row_refuses_a_prior_ensemble_posterior_claim() -> None:
    """B0's row exists, and it is the UNCORRECTED prior: never a posterior (plan §4.4)."""
    products, scores = _products_fixture()
    with pytest.raises(ValueError, match="posterior"):
        comparison_row(
            parent_id="e02-t1-v1-0008",
            method_id="B0",
            inference_seed=11,
            ensemble_kind=PRIOR_ENSEMBLE_KIND,
            posterior_claim=True,
            products=products,
            scores=scores,
        )
    row = comparison_row(
        parent_id="e02-t1-v1-0008",
        method_id="B0",
        inference_seed=11,
        ensemble_kind=PRIOR_ENSEMBLE_KIND,
        posterior_claim=False,
        products=products,
        scores=scores,
    )
    assert row["ensemble_kind"] == PRIOR_ENSEMBLE_KIND
    assert row["posterior_claim"] is False


def test_comparison_row_carries_the_primary_metrics_and_products() -> None:
    products, scores = _products_fixture()
    run = run_diagnostics_from_payloads(_smc_payload(), _moves_manifest())
    row = comparison_row(
        parent_id="e02-t1-v1-0008",
        method_id="M",
        inference_seed=11,
        ensemble_kind=POSTERIOR_ENSEMBLE_KIND,
        posterior_claim=True,
        products=products,
        scores=scores,
        run=run,
    )
    assert row["parent_id"] == "e02-t1-v1-0008"
    assert row["method_id"] == "M"
    assert row["inference_seed"] == 11
    assert row["estimator"] == "posterior_mean"
    assert row["mae"] == pytest.approx(0.075)
    assert row["rmse"] == pytest.approx(0.075)
    assert row["s_probabilities"] == {"0": 0.5, "1": 0.5}
    assert row["representative_rule"]
    assert row["beta_path"] == [0.0, 1.0]
    assert row["pre_resampling_ess"] == [25.6, 17.5]
    assert row["acceptance_by_kernel"] == {"global": 0.5, "family": 1.0}
    assert json.dumps(row, allow_nan=False)


# --------------------------------------------------------------------------------------
# world averaging
# --------------------------------------------------------------------------------------


def test_world_averaging_weights_worlds_equally_not_rows() -> None:
    rows = [
        {"parent_id": "A", "method_id": "M", "mae": 0.2},
        {"parent_id": "B", "method_id": "M", "mae": 0.4},
        {"parent_id": "B", "method_id": "M", "mae": 0.5},
    ]
    # per-world seed mean first: A -> 0.2, B -> 0.45; then equal world weight -> 0.325
    assert average_over_worlds(rows, "mae") == pytest.approx(0.325)
    with pytest.raises(ValueError, match="mae"):
        average_over_worlds([{"parent_id": "A", "method_id": "M"}], "mae")


def test_world_averaging_refuses_rows_without_method_id() -> None:
    # an unlabelled row cannot be grouped by method, so it is refused outright
    with pytest.raises(ValueError, match="method_id"):
        average_over_worlds([{"parent_id": "A", "mae": 0.2}], "mae")


def test_world_averaging_refuses_pooling_two_methods() -> None:
    rows = [
        {"parent_id": "A", "method_id": "q_learned", "mae": 0.2},
        {"parent_id": "A", "method_id": "defensive_mixture", "mae": 0.6},
    ]
    # the refusal names both methods, so the mix can never pass silently
    with pytest.raises(ValueError, match=r"defensive_mixture.*q_learned"):
        average_over_worlds(rows, "mae")
    # averaged per method in separate calls, the equal-world law still holds
    assert average_over_worlds(rows[:1], "mae") == pytest.approx(0.2)
    assert average_over_worlds(rows[1:], "mae") == pytest.approx(0.6)


# --------------------------------------------------------------------------------------
# run diagnostics from real payload shapes
# --------------------------------------------------------------------------------------


def _smc_payload() -> dict[str, Any]:
    return {
        "beta_history": [0.0, 1.0],
        "ess_history": [30.0, 28.0],
        "resampling": [
            {"level": 1, "pre_ess": 25.6, "indices": []},
            {"level": 2, "pre_ess": 17.5, "indices": [0, 1]},
        ],
        "budget": {"session": {"wall_s": 10.0, "cpu_s": 20.0, "output_bytes": 1000, "forwards": 3}},
        "failures": [],
    }


def _moves_manifest() -> dict[str, Any]:
    return {
        "state": {
            "particles": [
                {"ancestor_id": 0, "evaluation": {"forward_ref": _ref(0), "theta": _theta(0)}},
                {"ancestor_id": 0, "evaluation": {"forward_ref": _ref(0), "theta": _theta(0)}},
                {"ancestor_id": 1, "evaluation": {"forward_ref": _ref(1), "theta": _theta(1)}},
            ],
            "diagnostics": {
                "moves": [
                    {"kernel": "global", "accepted": True},
                    {"kernel": "global", "accepted": False},
                    {"kernel": "family", "accepted": True},
                ],
            },
        }
    }


def test_run_diagnostics_from_payloads() -> None:
    run = run_diagnostics_from_payloads(_smc_payload(), _moves_manifest())
    assert run.beta_path == (0.0, 1.0)
    assert run.pre_resampling_ess == (25.6, 17.5)
    assert run.n_unique_ancestors == 2
    assert run.n_physical_states == 2  # two distinct forward artifact ids
    assert run.acceptance_by_kernel == {"global": 0.5, "family": 1.0}
    assert run.runtime_wall_s == pytest.approx(10.0)
    assert run.cpu_s == pytest.approx(20.0)
    assert run.output_bytes == 1000
    assert run.failures == 0
    assert run.residual_movement is None
    assert "no residual diagnostic was supplied" in (run.residual_movement_note or "")


def test_run_diagnostics_carry_the_residual_movement_the_run_published() -> None:
    residual = ResidualSensitivity(
        config_hash="d" * 64,
        pcn_scale=0.3,
        n_residual_moves=5,
        n_scored=3,
        n_out_of_support=1,
        n_before_tempering=1,
        log_l_drops=(0.1, 0.4, 0.9),
        median_log_l_drop=0.4,
        max_log_l_drop=0.9,
        acceptance_rate_scored=2 / 3,
        acceptance_rate_all=0.8,
        derivation="pcn log_alpha = min(0, beta * (log L' - log L))",
    )
    run = run_diagnostics_from_payloads(_smc_payload(), _moves_manifest(), residual=residual)
    assert run.residual_movement == {
        "median_log_l_drop": 0.4,
        "max_log_l_drop": 0.9,
        "acceptance_rate_scored": pytest.approx(2 / 3),
        "n_scored": 3,
        "n_residual_moves": 5,
        "n_out_of_support": 1,
        "n_before_tempering": 1,
        "config_hash": "d" * 64,
        "pcn_scale": 0.3,
    }
    assert run.residual_movement_note == residual.derivation
    row = comparison_row(
        parent_id="w0",
        method_id="M",
        inference_seed=11,
        ensemble_kind=POSTERIOR_ENSEMBLE_KIND,
        posterior_claim=True,
        run=run,
    )
    assert row["residual_movement"]["median_log_l_drop"] == 0.4


def test_a_row_names_its_particle_count_and_scientific_target() -> None:
    row = comparison_row(
        parent_id="w0",
        method_id="B1",
        inference_seed=12,
        ensemble_kind=POSTERIOR_ENSEMBLE_KIND,
        posterior_claim=True,
        n_particles=64,
        scientific_target_identity="e" * 64,
    )
    assert row["n_particles"] == 64
    assert row["scientific_target_identity"] == "e" * 64


def test_a_row_refuses_a_particle_count_its_products_contradict() -> None:
    support = ONE_ZONE
    zones = aggregate_particle_zones(
        so=np.array([[0.5, 0.5], [0.7, 0.7]]),
        pv=np.ones((2, 2)),
        bo=np.ones((2, 2)),
        support=support,
    )
    products = ensemble_state_products(
        zones, np.array([0.5, 0.5]), support=support, s_labels=(0, 1)
    )
    with pytest.raises(ValueError, match="n_particles"):
        comparison_row(
            parent_id="w0",
            method_id="M",
            inference_seed=11,
            ensemble_kind=POSTERIOR_ENSEMBLE_KIND,
            posterior_claim=True,
            n_particles=64,
            products=products,
        )


# --------------------------------------------------------------------------------------
# the artifact loader: per-particle forwards, no truth anywhere
# --------------------------------------------------------------------------------------


def _publish_two_forwards(tmp_project: Path) -> tuple[ProjectPaths, list[dict[str, Any]]]:
    from tests.unit.test_forward_results import _cost, _extraction, _project

    paths, case, job_template = _project(tmp_project)
    from so_recon.simulator.contracts import JobDescriptor

    refs: list[dict[str, Any]] = []
    for index, pore_volume in enumerate((25000.0, 26000.0)):
        payload = _extraction()
        payload["states"]["pore_volume_m3"] = [
            [pore_volume] * 8 for _ in payload["states"]["times_s"]
        ]
        job = JobDescriptor(
            **{
                **job_template.model_dump(),
                "job_id": f"job-e03-loader-{index}",
                "result_dir": f"artifacts/results/job-e03-loader-{index}",
            }
        )
        from so_recon.simulator.results import publish_forward_result, write_forward_result

        result = publish_forward_result(
            job, case, payload, paths, cost=_cost(), solver_metadata={"protocol": "t"}
        )
        assert result.status == "COMPLETE", result.reason
        record = paths.resolve(job.result_dir) / "forward_result.json"
        digest = write_forward_result(result, record)
        refs.append(
            {
                "artifact_id": digest,
                "created_at": "2026-09-18T00:00:00+00:00",
                "media_type": "application/json",
                "parent_artifact_ids": [],
                "path": paths.relative(record),
                "producer_run_id": f"run-{index}",
                "schema_version": "forward-1",
                "sha256": digest,
                "size_bytes": record.stat().st_size,
            }
        )
    return paths, refs


def test_load_particle_states_reads_each_particles_own_forward(tmp_project: Path) -> None:
    from so_recon.registry.artifact import ArtifactRef

    paths, refs = _publish_two_forwards(tmp_project)
    ref0, ref1 = (ArtifactRef.model_validate(ref) for ref in refs)
    # particle 0 and particle 2 share one forward; particle 1 has its own
    so, pv, bo = load_particle_states([ref0, ref1, ref0], paths, time_index=-1)
    assert so.shape == (3, 8)
    assert bo[0, 0] == pytest.approx(0.98)
    np.testing.assert_allclose(pv[0], 25000.0)
    np.testing.assert_allclose(pv[1], 26000.0)
    np.testing.assert_allclose(pv[2], 25000.0)  # deduplicated read, own values


def test_operational_products_from_bundle_build_without_truth(tmp_project: Path) -> None:
    from so_recon.registry.artifact import ArtifactRef

    paths, refs = _publish_two_forwards(tmp_project)
    ref0, ref1 = (ArtifactRef.model_validate(ref) for ref in refs)
    two_zones = ZoneSupport(
        names=("cells-0-3", "cells-4-7"),
        matrix=np.vstack([np.pad(np.ones(4), (0, 4)), np.pad(np.ones(4), (4, 0))]),
        kind="primary_quadrants",
    )
    bundle = {
        "state_ref": _ref(99),
        "particles_ref": _ref(98),
        "diagnostics_ref": _ref(97),
        "ledger_ref": _ref(96),
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
        "convergence_status": "NOT_ASSESSED",
        "physical_state_refs": [refs[0], refs[1]],
        "parent_run_ids": ["run-x"],
    }
    manifest = {
        "state": {
            "particles": [
                {
                    "particle_id": 0,
                    "ancestor_id": 0,
                    "evaluation": {"forward_ref": refs[0], "theta": _theta(0)},
                },
                {
                    "particle_id": 1,
                    "ancestor_id": 0,
                    "evaluation": {"forward_ref": refs[1], "theta": _theta(1)},
                },
            ],
            "log_weights": [0.0, 0.0],
            "diagnostics": {},
        }
    }
    products = operational_products_from_bundle(
        bundle, manifest, paths, support=two_zones, time_index=-1
    )
    assert products.estimator == "posterior_mean"
    assert products.s_probabilities == {"0": 0.5, "1": 0.5}
    assert np.isfinite(products.posterior_mean_so).all()
    assert products.zone_weight_so_defined.tolist() == [1.0, 1.0]
    # the loader contract: no truth parameter exists to misuse
    signature = inspect.signature(operational_products_from_bundle)
    assert not any("truth" in name for name in signature.parameters)


def test_operational_products_refuse_a_forward_outside_the_bundle(tmp_project: Path) -> None:
    paths, refs = _publish_two_forwards(tmp_project)
    two_zones = ZoneSupport(
        names=("cells-0-3", "cells-4-7"),
        matrix=np.vstack([np.pad(np.ones(4), (0, 4)), np.pad(np.ones(4), (4, 0))]),
        kind="primary_quadrants",
    )
    bundle = {
        "physical_state_refs": [refs[0]],  # the bundle never saw refs[1]
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
    }
    manifest = {
        "state": {
            "particles": [
                {
                    "particle_id": 0,
                    "ancestor_id": 0,
                    "evaluation": {"forward_ref": refs[1], "theta": _theta(1)},
                },
            ],
            "log_weights": [0.0],
            "diagnostics": {},
        }
    }
    with pytest.raises(ValueError, match="bundle"):
        operational_products_from_bundle(bundle, manifest, paths, support=two_zones, time_index=-1)
