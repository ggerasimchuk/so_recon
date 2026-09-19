"""E03 Task 11 — the comparison-matrix assembler (plan §10, §5.5, §3.3).

What this suite pins is not new metric mathematics (that lives in `ensemble_states` and
`learned_comparison`) but the ASSEMBLY law of the matrix:

* the 64 main cells are enumerated FROM the preregistered protocol, never from whatever
  rows happen to exist;
* every planned cell leaves a mark: a result, an explicit incomplete, an explicit failure
  or an explicit NOT_RUN — a cell cannot vanish, and a partial matrix cannot call itself
  complete (§10.1 «partial execution остаётся partial evidence»);
* a beta<1 run is a diagnostic and never enters the accuracy curve (§10.4);
* averaging is seeds-within-world then equal worlds, and an average built on a matrix with
  holes reports PARTIAL evidence instead of a clean number (§5.5, §10.4);
* B1 and M are compared only when their scientific target identity is EQUAL (§4.3);
* the cold-start cost is the headline, a shared cache never makes the second method free,
  and the break-even exists only with positive online saving (§10.4);
* a T5 fine child is a paired view, not a ninth world, and an unrun fine child is
  NOT_RUN/resource-limited, never a PASS (§3.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from so_recon.ml.contracts import ComparisonMethod, ComparisonParent, ComparisonProtocol
from so_recon.validation.e03_protocol import (
    DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
    POSTERIOR_ENSEMBLE_KIND,
    PRIOR_ENSEMBLE_KIND,
)
from so_recon.validation.e03_report import (
    ColdStartCost,
    MatrixCell,
    MethodCampaignCost,
    T5Pair,
    assemble_comparison_matrix,
    break_even_runs,
    campaign_cost_report,
    cell_status_from_learned_smc_payload,
    independent_world_count,
    planned_main_cells,
    relative_improvement,
    t5_outcomes,
    verify_comparison_config,
    world_average,
)

REPO = Path(__file__).resolve().parents[2]
HASH = "a" * 64
IDENTITY = "b" * 64


def _ref() -> dict[str, Any]:
    return {
        "artifact_id": HASH,
        "created_at": "2026-09-18T00:00:00+00:00",
        "media_type": "application/json",
        "parent_artifact_ids": [],
        "path": "artifacts/runs/r/truth.json",
        "producer_run_id": "run-0",
        "schema_version": "truth-1",
        "sha256": HASH,
        "size_bytes": 10,
    }


def _protocol(parent_ids: tuple[str, ...]) -> ComparisonProtocol:
    parents = tuple(
        ComparisonParent(
            parent_id=parent_id,
            design_id="e03-t1-v1",
            split="evaluation",
            observation_hash=HASH,
            truth_ref=_ref(),
        )
        for parent_id in parent_ids
    )
    methods = (
        ComparisonMethod(
            method_id="B0", start="static_prior", physical_runs=True, posterior_claim=False
        ),
        ComparisonMethod(
            method_id="B1", start="static_prior", physical_runs=True, posterior_claim=True
        ),
        ComparisonMethod(
            method_id="M", start="defensive_mixture", physical_runs=True, posterior_claim=True
        ),
    )
    return ComparisonProtocol.preregister(
        protocol_id="e03-comparison-1",
        parents=parents,
        methods=methods,
        particle_counts=(32, 64),
        inference_seeds=(11, 12),
        primary_month=36,
        gates={"rmse_relative_improvement_min": 0.10},
    )


def _row(
    parent_id: str,
    method_id: str,
    n_particles: int,
    inference_seed: int,
    **overrides: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "parent_id": parent_id,
        "method_id": method_id,
        "n_particles": n_particles,
        "inference_seed": inference_seed,
        "ensemble_kind": POSTERIOR_ENSEMBLE_KIND,
        "scientific_target_identity": IDENTITY,
        "mae": 0.1,
        "rmse": 0.2,
    }
    row.update(overrides)
    return row


def _full_rows(parent_ids: tuple[str, ...], **overrides: Any) -> list[dict[str, Any]]:
    return [
        _row(parent_id, method_id, n_particles, seed, **overrides)
        for parent_id in parent_ids
        for method_id in ("B1", "M")
        for n_particles in (32, 64)
        for seed in (11, 12)
    ]


# --------------------------------------------------------------------------------------
# the planned matrix
# --------------------------------------------------------------------------------------


def test_the_main_matrix_is_the_declared_64_cells() -> None:
    cells = planned_main_cells(_protocol(tuple(f"t1-eval-{i}" for i in range(8))))
    assert len(cells) == 64
    assert len(set(cells)) == 64
    assert {cell.method_id for cell in cells} == {"B1", "M"}
    assert {cell.n_particles for cell in cells} == {32, 64}
    assert {cell.inference_seed for cell in cells} == {11, 12}


def test_the_main_matrix_refuses_a_method_the_protocol_never_registered() -> None:
    with pytest.raises(ValueError, match="A_set"):
        planned_main_cells(_protocol(("t1-eval-0",)), main_methods=("B1", "A_set"))


# --------------------------------------------------------------------------------------
# no cell can vanish
# --------------------------------------------------------------------------------------


def test_a_missing_cell_is_reported_as_not_run_and_the_matrix_is_not_complete() -> None:
    parents = ("w0", "w1")
    rows = _full_rows(parents)
    dropped = rows.pop()
    matrix = assemble_comparison_matrix(_protocol(parents), rows)
    assert len(matrix.outcomes) == 16
    missing = [outcome for outcome in matrix.outcomes if outcome.status == "NOT_RUN"]
    assert [outcome.cell.parent_id for outcome in missing] == [dropped["parent_id"]]
    assert missing[0].reason
    assert matrix.complete is False
    assert matrix.completeness == "PARTIAL"


def test_a_failed_cell_cannot_vanish_from_the_matrix() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    failed = rows.pop(0)
    cell = MatrixCell(
        parent_id=failed["parent_id"],
        method_id=failed["method_id"],
        n_particles=failed["n_particles"],
        inference_seed=failed["inference_seed"],
    )
    matrix = assemble_comparison_matrix(
        _protocol(parents), rows, failures={cell: "native solver aborted at level 3"}
    )
    failures = [outcome for outcome in matrix.outcomes if outcome.status == "FAILURE"]
    assert [outcome.cell for outcome in failures] == [cell]
    assert failures[0].reason == "native solver aborted at level 3"
    assert matrix.complete is False
    assert matrix.counts()["FAILURE"] == 1
    assert [outcome.cell for outcome in matrix.unfinished()] == [cell]


def test_a_campaign_whose_every_cell_failed_is_partial_evidence_not_not_run() -> None:
    parents = ("w0",)
    protocol = _protocol(parents)
    declared = {cell: "native solver aborted" for cell in planned_main_cells(protocol)}
    matrix = assemble_comparison_matrix(protocol, [], failures=declared)
    assert matrix.counts()["FAILURE"] == 8
    assert matrix.completeness == "PARTIAL"


def test_a_complete_matrix_says_complete() -> None:
    parents = ("w0", "w1")
    matrix = assemble_comparison_matrix(_protocol(parents), _full_rows(parents))
    assert matrix.complete is True
    assert matrix.completeness == "COMPLETE"
    assert matrix.counts() == {"RESULT": 16, "INCOMPLETE": 0, "FAILURE": 0, "NOT_RUN": 0}


def test_an_empty_campaign_is_not_run_and_still_enumerates_every_planned_cell() -> None:
    matrix = assemble_comparison_matrix(_protocol(("w0",)), [])
    assert matrix.completeness == "NOT_RUN"
    assert len(matrix.outcomes) == 8
    assert matrix.complete is False


def test_two_results_for_one_cell_are_refused() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    with pytest.raises(ValueError, match="twice"):
        assemble_comparison_matrix(_protocol(parents), [*rows, rows[0]])


def test_a_row_outside_the_planned_matrix_is_reported_not_silently_accepted() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    stray = _row("w-unplanned", "M", 64, 11)
    matrix = assemble_comparison_matrix(_protocol(parents), [*rows, stray])
    assert [row["parent_id"] for row in matrix.unplanned_rows] == ["w-unplanned"]
    assert matrix.complete is False
    assert matrix.completeness == "PARTIAL"


def test_a_main_cell_filled_with_a_prior_ensemble_is_refused() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    rows[0]["ensemble_kind"] = PRIOR_ENSEMBLE_KIND
    with pytest.raises(ValueError, match="prior_ensemble"):
        assemble_comparison_matrix(_protocol(parents), rows)


# --------------------------------------------------------------------------------------
# beta<1 and the accuracy curve
# --------------------------------------------------------------------------------------


def test_a_beta_below_one_run_is_incomplete_and_never_enters_the_accuracy_curve() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    for row in rows:
        if row["method_id"] == "M" and row["n_particles"] == 64 and row["inference_seed"] == 12:
            row.update(
                {
                    "ensemble_kind": DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
                    "beta": 0.72,
                    "algorithm_status": "INCOMPLETE_BUDGET",
                    "rmse": 0.0,
                }
            )
    matrix = assemble_comparison_matrix(_protocol(parents), rows)
    incomplete = [outcome for outcome in matrix.outcomes if outcome.status == "INCOMPLETE"]
    assert len(incomplete) == 1
    assert "0.72" in incomplete[0].reason
    assert matrix.complete is False
    kept = matrix.results_for("M", n_particles=64)
    assert [row["inference_seed"] for row in kept] == [11]
    average = world_average(matrix, method_id="M", metric="rmse", n_particles=64)
    assert average.value == pytest.approx(0.2)
    assert average.evidence == "PARTIAL"


# --------------------------------------------------------------------------------------
# aggregation law
# --------------------------------------------------------------------------------------


def test_the_frozen_main_average_is_seeds_within_a_world_then_equal_worlds() -> None:
    parents = ("w0", "w1")
    rows = _full_rows(parents)
    by_seed = {("w0", 11): 0.2, ("w0", 12): 0.4, ("w1", 11): 0.5, ("w1", 12): 0.5}
    for row in rows:
        if row["method_id"] == "M" and row["n_particles"] == 64:
            row["rmse"] = by_seed[(row["parent_id"], row["inference_seed"])]
    matrix = assemble_comparison_matrix(_protocol(parents), rows)
    average = world_average(matrix, method_id="M", metric="rmse", n_particles=64)
    # seed means 0.3 and 0.5, equal world weight -> 0.4; the pooled row mean is also 0.4
    # only because the seeds are balanced, so the world values are asserted too.
    assert average.world_values == {"w0": pytest.approx(0.3), "w1": pytest.approx(0.5)}
    assert average.value == pytest.approx(0.4)
    assert average.n_worlds == 2
    assert average.evidence == "COMPLETE"


def test_the_relative_gain_refuses_two_averages_over_different_worlds() -> None:
    parents = ("w0", "w1")
    rows = _full_rows(parents)
    for row in rows:
        if row["method_id"] == "M":
            row["rmse"] = 0.18
    matrix = assemble_comparison_matrix(_protocol(parents), rows)
    method = world_average(matrix, method_id="M", metric="rmse", n_particles=64)
    baseline = world_average(matrix, method_id="B1", metric="rmse", n_particles=64)
    gain = relative_improvement(method, baseline)
    assert gain.value == pytest.approx(0.1)
    assert gain.evidence == "COMPLETE"
    narrowed = world_average(
        assemble_comparison_matrix(_protocol(("w0",)), _full_rows(("w0",))),
        method_id="B1",
        metric="rmse",
        n_particles=64,
    )
    with pytest.raises(ValueError, match="different worlds"):
        relative_improvement(method, narrowed)
    at_n32 = world_average(matrix, method_id="B1", metric="rmse", n_particles=32)
    with pytest.raises(ValueError, match="one particle count"):
        relative_improvement(method, at_n32)


def test_averaging_refuses_to_pool_two_methods() -> None:
    parents = ("w0",)
    matrix = assemble_comparison_matrix(_protocol(parents), _full_rows(parents))
    with pytest.raises(ValueError, match="one method"):
        world_average(matrix, method_id=("B1", "M"), metric="rmse", n_particles=64)  # type: ignore[arg-type]


def test_unequal_scientific_target_identity_is_refused_before_any_comparison() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    for row in rows:
        if row["method_id"] == "M":
            row["scientific_target_identity"] = "c" * 64
    with pytest.raises(ValueError, match="2 scientific target identities"):
        assemble_comparison_matrix(_protocol(parents), rows)


def test_a_result_row_without_a_scientific_target_identity_is_refused() -> None:
    parents = ("w0",)
    rows = _full_rows(parents)
    del rows[0]["scientific_target_identity"]
    with pytest.raises(ValueError, match="scientific target identity"):
        assemble_comparison_matrix(_protocol(parents), rows)


# --------------------------------------------------------------------------------------
# the posterior_claim shape defect
# --------------------------------------------------------------------------------------


def test_the_cell_status_reads_beta_and_status_not_the_truthy_claim_string() -> None:
    refused = {
        "parent": {"parent_id": "w0"},
        "n_particles": 64,
        "seed": 11,
        "beta": 0.72,
        "algorithm_status": "INCOMPLETE_BUDGET",
        "ensemble_kind": DIAGNOSTIC_PARTIAL_ENSEMBLE_KIND,
        # The producer writes prose here; `bool(...)` of it is True for a REFUSAL.
        "posterior_claim": "REFUSED: beta<1 or incomplete output",
    }
    status, reason = cell_status_from_learned_smc_payload(refused)
    assert status == "INCOMPLETE"
    assert "0.72" in reason and "INCOMPLETE_BUDGET" in reason
    complete = {
        **refused,
        "beta": 1.0,
        "algorithm_status": "COMPLETE",
        "ensemble_kind": POSTERIOR_ENSEMBLE_KIND,
        "posterior_claim": "POSTERIOR",
    }
    assert cell_status_from_learned_smc_payload(complete)[0] == "RESULT"


def test_a_payload_whose_label_disagrees_with_its_beta_is_refused() -> None:
    lying = {
        "parent": {"parent_id": "w0"},
        "n_particles": 64,
        "seed": 11,
        "beta": 0.5,
        "algorithm_status": "COMPLETE",
        "ensemble_kind": POSTERIOR_ENSEMBLE_KIND,
        "posterior_claim": "POSTERIOR",
    }
    with pytest.raises(ValueError, match="ensemble_kind"):
        cell_status_from_learned_smc_payload(lying)


# --------------------------------------------------------------------------------------
# cost (§10.4)
# --------------------------------------------------------------------------------------


def test_cold_start_amortization_and_break_even() -> None:
    cost = ColdStartCost(
        context_corpus_s=100.0,
        preprocess_s=20.0,
        all_training_trials_s=60.0,
        online_s=10.0,
        verification_s=20.0,
    )
    assert cost.shared_s == pytest.approx(200.0)
    assert cost.cold_start_s == pytest.approx(210.0)
    assert cost.amortized_s(4) == pytest.approx(60.0)
    with pytest.raises(ValueError, match="at least one"):
        cost.amortized_s(0)
    assert break_even_runs(shared_s=200.0, b1_online_s=30.0, m_online_s=10.0) == 10
    assert break_even_runs(shared_s=201.0, b1_online_s=30.0, m_online_s=10.0) == 11


def test_break_even_does_not_exist_without_a_positive_online_saving() -> None:
    assert break_even_runs(shared_s=200.0, b1_online_s=10.0, m_online_s=10.0) is None
    assert break_even_runs(shared_s=200.0, b1_online_s=10.0, m_online_s=30.0) is None


def test_a_shared_cache_cannot_make_the_second_method_look_free() -> None:
    free_rider = MethodCampaignCost(
        method_id="M",
        actual_campaign_s=1.0,
        standalone_replay_s=None,
        cache_hits_from_other_methods=12,
    )
    with pytest.raises(ValueError, match="standalone"):
        campaign_cost_report((free_rider,))
    honest = MethodCampaignCost(
        method_id="M",
        actual_campaign_s=1.0,
        standalone_replay_s=140.0,
        cache_hits_from_other_methods=12,
    )
    report = campaign_cost_report((honest,))
    assert report["methods"]["M"]["actual_campaign_s"] == pytest.approx(1.0)
    assert report["methods"]["M"]["standalone_replay_s"] == pytest.approx(140.0)


# --------------------------------------------------------------------------------------
# T5 (§3.3)
# --------------------------------------------------------------------------------------


def test_a_t5_child_does_not_inflate_the_world_count() -> None:
    parents = ("w0", "w1")
    matrix = assemble_comparison_matrix(_protocol(parents), _full_rows(parents))
    pairs = (
        T5Pair(parent_id="w0", fine_status="RESULT", reason=None),
        T5Pair(parent_id="w1", fine_status="NOT_RUN", reason="fine run exceeded the budget"),
    )
    assert independent_world_count(matrix, pairs) == 2


def test_a_t5_pair_naming_a_parent_outside_the_matrix_is_refused() -> None:
    matrix = assemble_comparison_matrix(_protocol(("w0",)), _full_rows(("w0",)))
    with pytest.raises(ValueError, match="not an evaluation parent"):
        independent_world_count(matrix, (T5Pair(parent_id="w9", fine_status="RESULT"),))


def test_a_fine_child_that_did_not_run_is_not_run_never_a_pass() -> None:
    outcomes = t5_outcomes(
        (
            T5Pair(parent_id="w0", fine_status="NOT_RUN", reason="fine run exceeded the budget"),
            T5Pair(parent_id="w1", fine_status="RESULT"),
        )
    )
    assert outcomes[0]["status"] == "NOT_RUN"
    assert outcomes[0]["certifies_grid_accuracy"] is False
    assert "budget" in outcomes[0]["reason"]
    assert outcomes[1]["status"] == "PAIRED"
    assert outcomes[1]["certifies_grid_accuracy"] is True
    with pytest.raises(ValueError, match="reason"):
        t5_outcomes((T5Pair(parent_id="w0", fine_status="NOT_RUN"),))


# --------------------------------------------------------------------------------------
# the frozen configuration
# --------------------------------------------------------------------------------------


def test_the_repository_config_declares_a_matrix_that_matches_its_own_factors() -> None:
    payload = json.loads((REPO / "configs/e03_experiments.json").read_text(encoding="utf-8"))
    settings = verify_comparison_config(payload)
    assert settings["main_matrix_smc_runs"] == 64
    assert settings["main_methods"] == ("B1", "M")
    assert settings["primary_metric"] == "mae"
    assert settings["secondary_metric"] == "rmse"
    assert settings["primary_month"] == 36
    assert settings["registered_expansion"] == (128, 256)
    assert settings["additional_methods"] == ("B0", "Q", "A_summary", "A_set")


def test_a_main_method_counted_as_additional_too_is_refused() -> None:
    payload = json.loads((REPO / "configs/e03_experiments.json").read_text(encoding="utf-8"))
    payload["comparison"]["additional_methods"].append("M")
    with pytest.raises(ValueError, match="never both"):
        verify_comparison_config(payload)


def test_a_config_whose_declared_run_count_contradicts_its_factors_is_refused() -> None:
    payload = json.loads((REPO / "configs/e03_experiments.json").read_text(encoding="utf-8"))
    payload["comparison"]["main_matrix_smc_runs"] = 32
    with pytest.raises(ValueError, match="64"):
        verify_comparison_config(payload)


def test_a_config_whose_main_method_is_not_a_declared_method_is_refused() -> None:
    payload = json.loads((REPO / "configs/e03_experiments.json").read_text(encoding="utf-8"))
    payload["comparison"]["methods"] = ["B0", "M"]
    with pytest.raises(ValueError, match="B1"):
        verify_comparison_config(payload)
