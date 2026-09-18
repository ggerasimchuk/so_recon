"""E03 Task 09 — the first thin full cycle: B0 / B1 / raw q / M on one new T1 world.

This is the native entry point of the composition in `tests.integration.e03_cycle`: it
loads `configs/e03_smoke.yml`, hands it to `run_e03_cycle_smoke` and then asserts the
contract of what came back. It is `julia`-marked, so an ordinary `pytest -m "not julia"`
cycle never starts it — executing a real corpus, a real training session and three real
SMC methods is an explicitly started human decision, not something a unit run stumbles
into.

What this test pins about the result (plan §10.1, §5.5, §1.3):

* FOUR methods, on ONE world, at the SAME smoke knobs: B0, B1, Q and M, each with its own
  comparison row and its own recorded cost;
* Q is published as `ensemble_kind=raw_proposal` and B0 as the uncorrected
  `prior_ensemble` with the `log_r == log_p0` provenance proof — neither ever carries a
  posterior claim (§4.4), and B0 spends no forwards of its own because it IS B1's
  prior-start initial draws;
* B1 and M both reach beta=1.0 COMPLETE against the SAME scientific target, M under the
  declared defensive epsilon and the frozen 1e-5 balance law (§9);
* the So/errors/cost report exists on disk with its figures, and its scientific status is
  INCONCLUSIVE by construction: technical completion of a smoke this small earns no
  PROMISING_STATE, PROMISING_ML or PASS label (§1.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from so_recon.config.load import load_project_config
from so_recon.paths import ProjectPaths
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.validation.e03_protocol import (
    E03_BALANCE_THRESHOLDS,
    POSTERIOR_ENSEMBLE_KIND,
    PRIOR_ENSEMBLE_KIND,
    RAW_PROPOSAL_ENSEMBLE_KIND,
)
from so_recon.validation.ensemble_states import INFERENCE_GRID_SHAPE
from so_recon.validation.learned_comparison import B0_PROVENANCE
from tests.integration.e03_cycle import (
    B0_COST_NOTE,
    CYCLE_DESIGN_ID,
    CYCLE_REPORT_SCHEMA,
    METHOD_IDS,
    SCIENTIFIC_STATUS_INCONCLUSIVE,
    SCIENTIFIC_STATUS_REASON,
    run_e03_cycle_smoke,
)

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "configs" / "e03_smoke.yml"


@pytest.mark.julia
def test_first_thin_cycle_publishes_four_methods_on_one_new_t1_world() -> None:
    cfg = load_project_config(SMOKE_CONFIG)
    paths = ProjectPaths.from_config(ROOT, cfg.paths)
    paths.ensure_dirs()
    if not (paths.julia / "Manifest.toml").is_file():
        pytest.fail("the E03 cycle needs the instantiated julia project; run make setup-julia")
    try:
        julia = find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"the E03 cycle is a native physics run and requires Julia: {exc}")

    assert cfg.inference is not None
    assert cfg.learning is not None
    outcome = run_e03_cycle_smoke(paths=paths, cfg=cfg, julia=julia)

    report = outcome.report
    assert report["schema_version"] == CYCLE_REPORT_SCHEMA
    assert paths.resolve(outcome.report_ref.path).is_file()

    # --- the world: one new T1 parent whose own truth forward passed its checks --------
    world = report["world"]
    assert world["design_id"] == CYCLE_DESIGN_ID
    assert world["truth_checks"]["status"] == "PASS"
    assert world["parent_id"]

    # --- the knobs actually used are the smoke config's, not a matrix cell's ----------
    knobs = report["knobs"]
    assert knobs["config_version"] == cfg.config_version
    assert knobs["inference"]["n_particles"] == cfg.inference.n_particles
    assert knobs["inference"]["seed"] == cfg.inference.seed
    assert knobs["inference"]["max_beta_steps"] == cfg.inference.max_beta_steps
    assert knobs["profile"]["profile"] == "P1_E03_SMOKE"

    # --- the proposal says on its face where it came from -----------------------------
    assert report["proposal_provenance"]["source"] in {"external", "cycle-smoke-training"}

    # --- four methods, one row each, in the fixed order --------------------------------
    rows = report["rows"]
    assert tuple(row["method_id"] for row in rows) == METHOD_IDS
    by_method: dict[str, dict[str, Any]] = {row["method_id"]: row for row in rows}
    for method_id, row in by_method.items():
        assert row["parent_id"] == world["parent_id"], method_id
        assert row["inference_seed"] == cfg.inference.seed, method_id
        assert row["n_particles"] == cfg.inference.n_particles, method_id
        # the So/errors side of the report (§5.5): every method scored against the truth
        for metric in ("mae", "rmse", "crps", "coverage", "mean_width"):
            assert row[metric] is not None, (method_id, metric)

    # --- Q: a raw proposal ensemble, never a posterior (§4.4) --------------------------
    assert by_method["Q"]["ensemble_kind"] == RAW_PROPOSAL_ENSEMBLE_KIND
    assert by_method["Q"]["posterior_claim"] is False

    # --- B0: the prior-start initial draws, provenance-proven and forward-free ---------
    b0 = report["methods"]["B0"]
    assert b0["provenance"] == B0_PROVENANCE
    assert b0["n_draws"] == cfg.inference.n_particles
    assert b0["ensemble_kind"] == PRIOR_ENSEMBLE_KIND
    assert b0["posterior_claim"] is False
    assert by_method["B0"]["ensemble_kind"] == PRIOR_ENSEMBLE_KIND
    assert by_method["B0"]["posterior_claim"] is False

    # --- B1 and M: both COMPLETE at beta=1.0; M under the declared epsilon and law -----
    b1 = report["methods"]["B1"]
    assert (b1["algorithm_status"], float(b1["beta"])) == ("COMPLETE", 1.0)
    m = report["methods"]["M"]
    assert (m["algorithm_status"], float(m["beta"])) == ("COMPLETE", 1.0)
    assert m["ensemble_kind"] == POSTERIOR_ENSEMBLE_KIND
    assert float(m["epsilon"]) == float(cfg.learning.defensive_epsilon)
    assert m["balance_thresholds"] == dict(E03_BALANCE_THRESHOLDS)
    assert by_method["M"]["posterior_claim"] is True
    assert by_method["M"]["beta_path"][-1] == 1.0

    # --- the cost side of the report (§10.4/§12): one accounted session per method -----
    costs = report["costs"]
    for method_id in METHOD_IDS:
        cost = costs[method_id]
        assert cost["wall_s"] is not None, method_id
        assert cost["forwards"] is not None, method_id
        assert cost["output_bytes"] is not None, method_id
    assert costs["B0"]["forwards"] == 0
    assert costs["B0"]["note"] == B0_COST_NOTE
    assert costs["B1"]["forwards"] > 0
    assert costs["Q"]["forwards"] > 0
    assert costs["M"]["forwards"] > 0
    assert "world" in costs

    # --- the figures: one published map per grid layer, per method ---------------------
    figures = report["figures"]
    for method_id in METHOD_IDS:
        rendered = figures[method_id]
        assert len(rendered) == INFERENCE_GRID_SHAPE[2], method_id
        for relative in rendered:
            assert paths.resolve(relative).is_file(), relative

    # --- the machine checks the cycle makes about its own artifacts --------------------
    checks = report["checks"]
    assert report["checks_pass"] is True
    for name in (
        "corpus_complete",
        "truth_checks_pass",
        "b1_beta_one_complete",
        "m_beta_one_complete",
        "b0_provenance_proven",
        "b0_never_claims_posterior",
        "q_never_claims_posterior",
        "method_ids_distinct",
        "every_method_cost_recorded",
        "balance_law_recorded_by_m",
        "defensive_epsilon_is_declared",
        "b1_and_m_share_the_scientific_target",
    ):
        assert checks[name] is True, name

    # --- §1.3: finishing is not a result --------------------------------------------
    assert report["scientific_status"] == SCIENTIFIC_STATUS_INCONCLUSIVE
    assert report["scientific_status_reason"] == SCIENTIFIC_STATUS_REASON
    # the report's own vocabulary is closed: there is no field a reader could mistake for
    # a scientific verdict on this smoke.
    assert set(report) == {
        "schema_version",
        "scientific_status",
        "scientific_status_reason",
        "world",
        "knobs",
        "proposal_provenance",
        "methods",
        "rows",
        "figures",
        "costs",
        "checks",
        "checks_pass",
    }
