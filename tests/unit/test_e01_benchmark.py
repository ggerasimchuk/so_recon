"""E01.12.4/12.5 — the benchmark block may not call a warm run a cold one.

12.4 is explicit: warm-up and cold costs are recorded separately, and the first cold run of
one regime is never compared with the warm of another. The P1 suite's block reported
`cold_s` as the wall of `world41` — but `world41_preflight` ran first in the same worker and
paid the import and the specialisation, so `world41` is a warm number wearing a cold label,
and a reader of the block concluded that specialisation costs about one second.
"""

from __future__ import annotations

from typing import Any

from so_recon.simulator.suite_record import JobOutcome
from so_recon.simulator.suites import first_cold_import_s, warm_block


def _job(job_id: str, wall_s: float, cold_import_s: float | None) -> JobOutcome:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "group": "worlds",
        "kind": "world",
        "profile": "P1_LOOP",
        "accounting": "ledger",
        "expected_outcome": "COMPLETE",
        "status": "COMPLETE",
        "wall_s": wall_s,
        "cpu_s": 1.0,
        "peak_rss_bytes": 1,
        "output_bytes": 1,
        "native_chunk_calls": 1,
        "accepted_steps": 1,
        "cut_steps": 0,
        "nonlinear_iterations": 1,
        "retry_count": 0,
        "cold_import_s": cold_import_s,
    }
    return JobOutcome.model_validate(payload)


def test_the_first_run_of_a_model_is_not_published_as_the_cold_cost() -> None:
    block = warm_block(
        warm_s=[4.0, 4.05, 3.98, 4.01, 4.03],
        failures=(),
        cold_import_s=7.458,
        first_run_of_model_s=5.062,
        first_job_id="world41_preflight",
        first_job_wall_s=28.808,
    )
    assert "cold_s" not in block, "a warm run must not be published under a cold name"
    assert block["cold_import_s"] == 7.458
    assert block["first_run_of_model_s"] == 5.062
    assert block["first_job_wall_s"] == 28.808
    assert block["first_job_id"] == "world41_preflight"
    assert block["p50_s"] > 0.0


def test_the_cold_import_comes_from_the_session_s_first_job() -> None:
    jobs = (
        _job("world41_preflight", 28.808, 7.458),
        _job("world41", 5.062, 7.458),
        _job("world42", 3.959, 7.458),
    )
    assert first_cold_import_s(jobs) == 7.458
    assert first_cold_import_s(()) is None
    assert first_cold_import_s((_job("x", 1.0, None),)) is None


def test_a_short_warm_sample_still_reports_what_it_has() -> None:
    block = warm_block(
        warm_s=[4.0],
        failures=("warm41_2: RESOURCE_FAILURE",),
        cold_import_s=None,
        first_run_of_model_s=None,
        first_job_id=None,
        first_job_wall_s=None,
    )
    assert block["n"] == 1
    assert block["failure_rate"] == 0.5
    assert "p50_s" not in block
    assert "cold_s" not in block
