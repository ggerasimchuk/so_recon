"""E03 Task 09 — the pure parts of the first-thin-cycle composition (plan §10.1, §4.4).

These run without Julia: they pin the two places where the cycle reads its own and its
producers' published shapes, because that is where it can silently disagree with them.

* the RUN WORLD is selected from fields `ParentRow` actually publishes — a corpus row
  carries no ordinal, so a selection made on one would always be empty;
* the raw-q ensemble's `posterior_claim` is a BOOLEAN that `cycle_checks` reads as a
  boolean; the refusal prose lives in its own field. A value whose type differs between
  the writer and the reader is exactly what §4.4's machine-checkable labelling exists to
  prevent, so the checks are exercised over the real published payload.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from so_recon.ml.contracts import ParentRow
from so_recon.validation.e03_protocol import (
    E03_BALANCE_THRESHOLDS,
    PRIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
)
from so_recon.validation.learned_comparison import B0_PROVENANCE, comparison_row
from tests.integration.e03_cycle import (
    CYCLE_DESIGN_ID,
    CYCLE_RUN_PARENT_INDEX,
    METHOD_IDS,
    assemble_cycle_report,
    cycle_checks,
    method_cost,
    raw_q_ensemble_payload,
    select_run_parent,
)

#: A corpus manifest row as the builder publishes it, reduced to the fields the selection
#: reads. The key set is asserted against `ParentRow` below, so this fixture cannot drift
#: away from the contract it stands for.
SELECTION_FIELDS = ("parent_id", "design_id", "split")


def _row(parent_id: str, design_id: str, split: str) -> dict[str, str]:
    return {"parent_id": parent_id, "design_id": design_id, "split": split}


def _corpus_rows() -> list[dict[str, str]]:
    """Five cycle parents plus decoys, deliberately out of parent-id order."""
    return [
        _row("e02-t1-v1-0003", CYCLE_DESIGN_ID, "train"),
        _row("e02-t1-v1-0004", CYCLE_DESIGN_ID, "development"),
        _row("e02-t1-v1-0000", CYCLE_DESIGN_ID, "train"),
        _row("e02-t2-v2-0000", "e02-t2-v2", "train"),
        _row("e02-t1-v1-0002", CYCLE_DESIGN_ID, "train"),
        _row("e02-t1-v1-0001", CYCLE_DESIGN_ID, "train"),
    ]


def test_the_selection_reads_only_fields_the_corpus_row_publishes() -> None:
    """`ParentRow` is strict and carries no ordinal, so no position can be selected on."""
    assert "parent_index" not in ParentRow.model_fields
    assert set(SELECTION_FIELDS) <= set(ParentRow.model_fields)


def test_select_run_parent_picks_one_train_parent_deterministically() -> None:
    rows = _corpus_rows()
    chosen = select_run_parent(rows)
    assert chosen["parent_id"] == "e02-t1-v1-0000"
    assert chosen["split"] == "train"
    assert chosen["design_id"] == CYCLE_DESIGN_ID
    # the same corpus always names the same world, whatever order the rows arrive in
    assert select_run_parent(list(reversed(rows))) == chosen
    assert CYCLE_RUN_PARENT_INDEX == 0


def test_select_run_parent_refuses_a_corpus_with_no_train_parent_of_the_design() -> None:
    rows = [
        _row("e02-t1-v1-0004", CYCLE_DESIGN_ID, "development"),
        _row("e02-t2-v2-0000", "e02-t2-v2", "train"),
    ]
    with pytest.raises(ValueError, match="no run world"):
        select_run_parent(rows)


# --------------------------------------------------------------------------------------
# the published raw-q ensemble, and the checks read over it
# --------------------------------------------------------------------------------------

N_PARTICLES = 16
EPSILON = 0.10


def _law() -> SimpleNamespace:
    return SimpleNamespace(
        q=SimpleNamespace(fingerprint="q-fingerprint"),
        mixture=SimpleNamespace(fingerprint="r-fingerprint"),
        epsilon=EPSILON,
        weights_sha256="0" * 64,
    )


def _q_payload() -> dict[str, Any]:
    return raw_q_ensemble_payload(
        evaluations=(),
        law=_law(),
        n_particles=N_PARTICLES,
        seed=137,
        parent_id="e02-t1-v1-0000",
    )


def _ledger_record(forwards: int) -> dict[str, Any]:
    entry = {"cost": {"wall_s": 20.0, "cpu_s": 19.0, "output_bytes": 4096, "peak_rss_bytes": 1024}}
    return {"entries": [dict(entry) for _ in range(forwards)], "session_id": "session"}


def _checks_inputs() -> dict[str, Any]:
    rows = [
        comparison_row(
            parent_id="e02-t1-v1-0000",
            method_id=method_id,
            inference_seed=1301,
            ensemble_kind=RAW_PROPOSAL_ENSEMBLE_KIND if method_id == "Q" else "posterior",
            posterior_claim=method_id in {"B1", "M"},
        )
        for method_id in METHOD_IDS
    ]
    return {
        "corpus": {"expected": 5, "complete": 5, "failed": 0},
        "truth_checks": {"status": "PASS"},
        "b1": {"algorithm_status": "COMPLETE", "beta": 1.0},
        "m": {
            "algorithm_status": "COMPLETE",
            "beta": 1.0,
            "balance_thresholds": dict(E03_BALANCE_THRESHOLDS),
            "proposal": {"epsilon": EPSILON},
        },
        "b0": {
            "provenance": B0_PROVENANCE,
            "n_draws": N_PARTICLES,
            "expected": N_PARTICLES,
            "ensemble_kind": PRIOR_ENSEMBLE_KIND,
            "posterior_claim": False,
        },
        "q": _q_payload(),
        "rows": rows,
        "costs": {
            "B0": method_cost({"entries": (), "session_id": None}, note="reused"),
            "B1": method_cost(_ledger_record(3)),
            "Q": method_cost(_ledger_record(2)),
            "M": method_cost(_ledger_record(4)),
        },
        "epsilon": EPSILON,
        "same_target_identity": True,
    }


def test_the_raw_q_ensemble_refuses_the_claim_as_a_value_not_as_prose() -> None:
    payload = _q_payload()
    assert payload["ensemble_kind"] == RAW_PROPOSAL_ENSEMBLE_KIND
    assert payload["posterior_claim"] is False
    assert "posterior" in payload["posterior_claim_note"]
    assert payload["epsilon"] == EPSILON
    assert payload["weights"] == [1.0 / N_PARTICLES] * N_PARTICLES


def test_cycle_checks_pass_over_the_published_payloads() -> None:
    """Every check is True for a cycle that did what the plan asks of it."""
    checks = cycle_checks(**_checks_inputs())
    assert all(checks.values()), sorted(name for name, ok in checks.items() if not ok)
    report = assemble_cycle_report(
        world={},
        knobs={},
        proposal_provenance={},
        methods={},
        rows=_checks_inputs()["rows"],
        figures={},
        costs=_checks_inputs()["costs"],
        checks=checks,
    )
    assert report["checks_pass"] is True
    assert report["scientific_status"] == "INCONCLUSIVE"


def test_a_prose_posterior_claim_does_not_pass_as_a_refusal() -> None:
    """The regression this file exists for: a truthy sentence is not `False`."""
    inputs = _checks_inputs()
    inputs["q"] = {
        **inputs["q"],
        "posterior_claim": "REFUSED: a raw proposal ensemble is never a posterior",
    }
    checks = cycle_checks(**inputs)
    assert checks["q_never_claims_posterior"] is False


def test_b0_must_be_the_prior_ensemble_without_a_claim() -> None:
    inputs = _checks_inputs()
    inputs["b0"] = {**inputs["b0"], "ensemble_kind": "posterior", "posterior_claim": True}
    checks = cycle_checks(**inputs)
    assert checks["b0_never_claims_posterior"] is False
