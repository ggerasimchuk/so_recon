"""E01.13 — the black-oil capability validators. They READ a published BO session.

Every test here opens the newest `e01_suite.json` whose suite is `bo` and the artifacts it
names. Plan 12.9's rule applies unchanged: the gate has to cite numbers a real session
published, so a metric that exists only inside a test process is a metric the gate cannot
use, and re-running the capability here would score a second set of results against the
first's conclusions.

They carry the `e01_physics` marker Task 12 established and SKIP without `--run-e01-physics`,
so a plain `pytest` cycle does not depend on a session nobody asked for. The contract half of
13.1 — what a black-oil case may declare, and that an unknown physics class is INVALID_INPUT —
is in `tests/unit/test_e01_blackoil_contract.py` and needs no session at all.

13.1 is explicit that a BO test is not marked passed on one successful constructor. What is
checked here, on the real results: the three-phase saturation identity against the pinned
engine's own epsilon; positive densities, shrinkage factors and a non-negative dissolved
ratio on every published state; free gas appearing only below the bubble point; the component
gas balance closing WITH the dissolved term; the free/dissolved split recomputed from the
published states agreeing with the native component inventory; and the continuous trajectory
against the one a NEW worker continued, on So/Sw/Sg/p/Rs, on both gas inventories and on the
surface volumes of all three components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from so_recon.paths import ProjectPaths, find_repo_root
from so_recon.simulator.case_io import read_array
from so_recon.simulator.contracts import BlackOilFluidSpec, CaseBundle
from so_recon.simulator.results import (
    BO_COMPONENTS,
    GAS_MONTHLY_FILENAME,
    RESULT_FILENAME,
    load_forward_result,
)
from so_recon.simulator.suite_record import (
    BO_CHECKS,
    MANDATORY_CHECKS,
    SUITE_REPORT_FILENAME,
    SuiteReport,
)
from so_recon.simulator.suites import BO_RESTART_AFTER_STEP
from so_recon.validation.physics import (
    BLACKOIL_GATES,
    DEFAULT_BLACKOIL_TOLERANCES_RELPATH,
    load_blackoil_tolerances,
)

pytestmark = pytest.mark.e01_physics

ROOT = find_repo_root(Path(__file__).resolve().parents[2])
PATHS = ProjectPaths.default(ROOT)

#: The four jobs `configs/e01_jobs.json` declares for this suite.
BO_JOB_IDS = ("bo_closed", "bo_depletion", "bo_restart_prefix", "bo_restart_suffix_new_worker")


def _newest_bo_report() -> tuple[Path, SuiteReport] | None:
    found: tuple[Path, SuiteReport] | None = None
    if not PATHS.runs.is_dir():
        return None
    for run_dir in sorted(PATHS.runs.iterdir()):
        path = run_dir / SUITE_REPORT_FILENAME
        if not path.is_file():
            continue
        report = SuiteReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if report.suite == "bo":
            found = (run_dir, report)
    return found


@pytest.fixture(scope="module")
def bo() -> tuple[Path, SuiteReport]:
    found = _newest_bo_report()
    if found is None:
        pytest.skip(
            "no published black-oil suite under artifacts/runs; run `so-recon --config "
            "configs/e01.yml verify-physics --suite bo` first"
        )
    return found


@pytest.fixture(scope="module")
def results(bo: tuple[Path, SuiteReport]) -> dict[str, Any]:
    """The two published fixture results of the session, read back through their records."""
    _run_dir, report = bo
    out: dict[str, Any] = {}
    for job_id in ("bo_closed", "bo_depletion"):
        relative = report.artifacts.get(f"result.{job_id}")
        assert relative is not None, f"the session names no published result for {job_id}"
        out[job_id] = load_forward_result(PATHS.resolve(relative) / RESULT_FILENAME, PATHS)
    return out


def _states(result: Any) -> dict[str, NDArray[np.float64]]:
    return {
        name: np.asarray(read_array(ref, PATHS), dtype=np.float64)
        for name, ref in result.states.items()
    }


# --------------------------------------------------------------------------------------
# the session itself
# --------------------------------------------------------------------------------------


def test_all_four_declared_black_oil_jobs_ran_and_came_back_as_the_matrix_expects(
    bo: tuple[Path, SuiteReport],
) -> None:
    _run_dir, report = bo
    rows = {job.job_id: job for job in report.jobs}
    assert set(rows) == set(BO_JOB_IDS), sorted(rows)
    for job_id in BO_JOB_IDS:
        job = rows[job_id]
        assert job.status == job.expected_outcome, (job_id, job.status, job.reason)
    assert not report.remaining_job_ids, report.remaining_job_ids
    # 13.4: the restart pair runs through the production driver and is charged to a ledger;
    # the two fixtures run inside the diagnostic and are accounted by the launcher.
    assert rows["bo_restart_prefix"].accounting == "ledger"
    assert rows["bo_restart_suffix_new_worker"].accounting == "ledger"
    assert rows["bo_restart_suffix_new_worker"].parent_job_id == "bo_restart_prefix"


def test_the_capability_gate_is_separate_from_the_oil_water_one(
    bo: tuple[Path, SuiteReport],
) -> None:
    """13.5: the BO verdict is its own gate and its checks are not in the OW stage matrix."""
    _run_dir, report = bo
    names = {check.name for check in report.checks}
    assert names == set(BO_CHECKS), sorted(names)
    assert not (BO_CHECKS & MANDATORY_CHECKS), "a BO check in the OW stage matrix couples them"
    for check in report.checks:
        assert check.status == "PASS", (check.name, check.reason)
    # Scored against the black-oil block, and the block that is in force right now.
    assert report.artifacts["blackoil.tolerances"] == DEFAULT_BLACKOIL_TOLERANCES_RELPATH
    from so_recon.registry.hashing import sha256_file

    digest = sha256_file(PATHS.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH))
    for check in report.checks:
        declared = check.input_hashes.get("blackoil_tolerances")
        if declared is not None:
            assert declared == digest, (
                f"{check.name} was scored against a black-oil tolerance block that has changed "
                "since; rerun the suite rather than reading its old numbers"
            )


def test_the_published_pvt_is_the_academic_benchmark_and_says_so(
    bo: tuple[Path, SuiteReport],
) -> None:
    """13.3: exported numbers, hashed, and marked as a benchmark rather than a field PVT."""
    _run_dir, report = bo
    fixture = json.loads(
        PATHS.resolve(report.artifacts["blackoil.pvt_fixture"]).read_text(encoding="utf-8")
    )
    pvt = fixture["pvt"]
    assert fixture["academic_benchmark"] is True
    assert "Romashka" in fixture["note"]
    assert pvt["source"] == "JutulDarcy.blackoil_bench_pvt(:spe1)"
    assert pvt["phase_order"] == list(BO_COMPONENTS)
    # The pinned benchmark returns its densities in the ECLIPSE DENSITY order (oil, water,
    # gas) and the constructor wants phase order. The permutation is explicit and recorded.
    assert pvt["reference_densities_deck_order_kg_m3"] == [786.507, 1037.84, 0.969758]
    assert pvt["reference_densities_kg_m3"] == [1037.84, 786.507, 0.969758]
    assert pvt["density_permutation_deck_to_phase"] == [2, 1, 3]
    assert len(pvt["pvto"]["rs"]) == len(pvt["pvto"]["sat_pressure_pa"])
    assert fixture["relperm"]["hysteresis"] == "none"
    assert fixture["relperm"]["exponents"] == [2.0, 2.0, 2.0]
    assert fixture["relperm"]["residual_saturations"] == [0.1, 0.1, 0.0]
    assert fixture["relperm"]["endpoints"] == [1.0, 1.0, 1.0]


def test_the_published_case_declares_case_2_and_names_the_tables_it_was_built_from(
    bo: tuple[Path, SuiteReport],
) -> None:
    """The interface note: `case-2` carries the black-oil fluid and hashes its own PVT."""
    _run_dir, report = bo
    for job_id in ("bo_closed", "bo_depletion"):
        path = PATHS.resolve(report.artifacts[f"case.{job_id}"])
        case = CaseBundle.model_validate(json.loads(path.read_text(encoding="utf-8")))
        assert case.schema_version == "case-2", job_id
        assert isinstance(case.fluids, BlackOilFluidSpec)
        assert case.fluids.rv == 0.0
        assert case.fluids.academic_benchmark is True
        assert set(case.fluids.pvt_table_hashes) >= {"pvtw", "pvto", "pvdg", "relperm"}


# --------------------------------------------------------------------------------------
# 13.1 — the capability assertions, on the real published states
# --------------------------------------------------------------------------------------


def test_every_published_black_oil_state_is_a_physical_three_phase_state(
    results: dict[str, Any],
) -> None:
    """Plan 13.1's own assertion, applied to both fixtures at every published time.

    The sum is `1 + MINIMUM_COMPOSITIONAL_SATURATION` in this engine — JutulDarcy 0.3.11
    reconstructs the black-oil saturations as `1 - Sw + 1e-10` — so the drift is compared
    against the black-oil block's limit rather than against an exact one, and the
    saturations are never renormalised to make it vanish.
    """
    limit = load_blackoil_tolerances(PATHS.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH))[
        "blackoil_saturation_sum_abs_max"
    ]
    checked = 0
    for job_id, result in results.items():
        assert result.physics_class == "BO", job_id
        fields = _states(result)
        assert {"sg", "rs", "bg"} <= set(fields), job_id
        drift = float(np.abs(fields["sw"] + fields["so"] + fields["sg"] - 1.0).max())
        assert drift <= limit, (job_id, drift)
        assert (fields["sg"] >= -1e-8).all(), job_id
        assert (fields["rs"] >= 0.0).all(), job_id
        assert (fields["bo"] > 0.0).all() and (fields["bg"] > 0.0).all(), job_id
        assert (fields["bw"] > 0.0).all(), job_id
        assert (fields["pressure_pa"] > 0.0).all(), job_id
        checked += 1
    assert checked == 2, "both fixtures are checked; one successful constructor is not a pass"


def test_a_closed_black_oil_cell_with_no_source_does_not_move(results: dict[str, Any]) -> None:
    """The capability's equivalent of 9.5's closed cell: whatever moves is arithmetic."""
    tolerances = load_blackoil_tolerances(PATHS.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH))
    result = results["bo_closed"]
    fields = _states(result)
    for name in ("sw", "so", "sg"):
        drift = float(np.abs(fields[name][-1] - fields[name][0]).max())
        assert drift <= tolerances["blackoil_closed_saturation_drift_max"], (name, drift)
    assert float(fields["sg"].max()) <= 1e-10, "a closed cell above its bubble point has no gas"
    block = result.black_oil
    assert block is not None
    started = block.free_gas_m3_sc[0] + block.dissolved_gas_m3_sc[0]
    ended = block.free_gas_m3_sc[-1] + block.dissolved_gas_m3_sc[-1]
    assert started > 0.0, "an undersaturated oil holds dissolved gas"
    relative = abs(ended - started) / max(abs(started), 1e-6)
    assert relative <= tolerances["blackoil_closed_gas_inventory_relative_max"], relative


def test_free_gas_appears_only_once_the_pressure_is_below_the_bubble_point(
    bo: tuple[Path, SuiteReport], results: dict[str, Any]
) -> None:
    """13.1: gas appears at depletion below the bubble point — and not before it."""
    run_dir, _report = bo
    diagnostic = json.loads(
        (run_dir / "verification" / "blackoil.json").read_text(encoding="utf-8")
    )
    bubble = float(diagnostic["bubble_point_pa"])
    fields = _states(results["bo_depletion"])
    assert float(fields["sg"][0].max()) <= 1e-10, "the case starts undersaturated"
    assert float(fields["sg"][-1].max()) > 1e-3, "no free gas ever came out of solution"
    assert float(fields["pressure_pa"][-1].min()) < bubble
    # Cell by cell and time by time: a cell holding free gas is a cell at or below its own
    # bubble point, and one above it holds none. This is the assertion that would fail if
    # `Sg` were being reported for something other than a phase transition.
    gassy = fields["sg"] > 1e-8
    assert gassy.any()
    assert (fields["pressure_pa"][gassy] <= bubble * 1.001).all()
    assert (fields["sg"][fields["pressure_pa"] > bubble * 1.05] <= 1e-8).all()
    # Dissolved gas leaves the oil as it does so.
    assert float(fields["rs"][-1].min()) < float(fields["rs"][0].max())


def test_the_component_gas_balance_closes_with_the_dissolved_term(
    results: dict[str, Any],
) -> None:
    """13.1: `Rs*So/Bo` is inside the gas component, and the balance is closed on it.

    Two independent statements have to agree. The balance table is computed from the native
    `TotalMasses`, which knows nothing of the free/dissolved split; the split is recomputed
    here from the published saturations, ratio and formation volume factors. Their agreement
    is what makes the dissolved term a measured claim rather than a definition.
    """
    tolerances = load_blackoil_tolerances(PATHS.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH))
    for job_id, result in results.items():
        rows = pq.read_table(PATHS.resolve(str(result.balances_path))).to_pylist()
        components = {row["component"] for row in rows}
        assert components == set(BO_COMPONENTS), (job_id, sorted(components))
        for row in rows:
            if row["component"] != "gas":
                continue
            assert (
                abs(float(row["cumulative_relative"]))
                <= tolerances["blackoil_gas_balance_relative_max"]
            ), (job_id, row["balance"], row["cumulative_relative"])

        block = result.black_oil
        assert block is not None
        fields = _states(result)
        free = np.sum(fields["sg"] * fields["pore_volume_m3"] / fields["bg"], axis=1)
        dissolved = np.sum(
            fields["rs"] * fields["so"] * fields["pore_volume_m3"] / fields["bo"], axis=1
        )
        np.testing.assert_allclose(free, block.free_gas_m3_sc, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(dissolved, block.dissolved_gas_m3_sc, rtol=1e-12, atol=1e-9)

        # Against the RESERVOIR inventory: the split above is a sum over reservoir cells,
        # and the whole-model statement also holds the gas standing in the wellbores. That
        # difference is a real quantity — 0.31% of the total on the one-cell closed fixture —
        # so it is asserted to be positive rather than assumed away.
        gas = {str(row["balance"]): row for row in rows if row["component"] == "gas"}
        limit = tolerances["blackoil_gas_inventory_closure_relative_max"]
        native_initial = float(gas["reservoir_connections"]["initial_inventory_m3_sc"])
        native_final = float(gas["reservoir_connections"]["final_inventory_m3_sc"])

        def closes(split: float, native: float) -> float:
            return abs(split - native) / max(abs(native), 1e-6)

        # BOTH ends of the trajectory, and the last one is the load-bearing half. Both
        # fixtures start undersaturated, so `free[0]` is zero and the FIRST-state closure is
        # the identity `dissolved(t0) == native(t0)`: a real cross-check of the dissolved
        # term against the native `TotalMasses`, and silent about the free one.
        assert closes(float(free[0] + dissolved[0]), native_initial) <= limit, job_id
        assert closes(float(free[-1] + dissolved[-1]), native_final) <= limit, job_id
        assert float(gas["full_system_surface"]["initial_inventory_m3_sc"]) >= native_initial, (
            job_id
        )

        if job_id != "bo_depletion":
            continue
        # …and on the depletion the last-state closure really is weighing the free term: a
        # published run that dropped it entirely would still satisfy the first-state closure
        # exactly and would miss the last-state one by orders of magnitude. Asserted on the
        # real numbers, so this is a check that can fail rather than one that cannot.
        fraction = float(free[-1]) / float(free[-1] + dissolved[-1])
        assert fraction > 0.01, (
            "the last-state closure is measured where free gas is only "
            f"{fraction:.3g} of the inventory; it would be a statement about the dissolved "
            "term again"
        )
        assert closes(float(dissolved[-1]), native_final) > limit, (
            "dropping the free term entirely still satisfies the last-state closure; the "
            "gate cannot detect a free-gas error"
        )
        assert closes(float(dissolved[0]), native_initial) <= limit, (
            "the first-state closure is expected to be blind to the free term — it is the "
            "reason the last-state one was added"
        )


def test_the_published_capability_check_gates_both_ends_of_the_gas_closure(
    bo: tuple[Path, SuiteReport],
) -> None:
    """The session's own record carries the last-state closure, and it is a GATE there.

    The test above measures the closure from the published states. This one asserts that the
    session scored it too — a metric only a test computes is a metric the stage gate cannot
    use (plan 12.9) — and that it is paired with a threshold rather than merely reported.
    """
    _run_dir, report = bo
    check = next(c for c in report.checks if c.name == "black_oil")
    gated = {metric for metric, _threshold in BLACKOIL_GATES["black_oil"]}
    for metric in (
        "blackoil_gas_inventory_closure_relative",
        "blackoil_gas_inventory_closure_final_relative",
        "blackoil_final_free_gas_fraction_shortfall",
    ):
        assert metric in check.metrics, metric
        assert metric in gated, f"{metric} is measured and not gated"
    for metric, threshold in BLACKOIL_GATES["black_oil"]:
        assert threshold in check.thresholds, threshold
        assert check.metrics[metric] <= check.thresholds[threshold], (metric, threshold)
    # Reported beside them, so a reader can see the last-state closure has teeth.
    assert check.metrics["blackoil_final_free_gas_fraction"] > 0.01


def test_the_surface_gas_is_published_in_its_own_table_and_never_as_a_liquid(
    results: dict[str, Any],
) -> None:
    """13.4: gas units and source scales are not borrowed from the oil-water case."""
    result = results["bo_depletion"]
    block = result.black_oil
    assert block is not None
    gas = pq.read_table(PATHS.resolve(block.gas_monthly_path))
    assert set(gas.column_names) == {
        "well_id",
        "month_index",
        "start_s",
        "end_s",
        "gas_prod_m3_sc",
        "gas_inj_m3_sc",
    }
    assert Path(block.gas_monthly_path).name == GAS_MONTHLY_FILENAME
    produced = float(sum(gas.column("gas_prod_m3_sc").to_pylist()))
    assert produced > 0.0, "the depletion produced gas"
    assert produced == pytest.approx(block.surface_gas_m3_sc)

    monthly = pq.read_table(PATHS.resolve(str(result.monthly_path)))
    assert "gas_prod_m3_sc" not in monthly.column_names, (
        "a standard cubic metre of gas is not a standard cubic metre of liquid; it must not "
        "be a column of the oil-water monthly table"
    )
    liquid = np.asarray(monthly.column("liquid_prod_m3_sc").to_pylist(), dtype=np.float64)
    oil = np.asarray(monthly.column("oil_prod_m3_sc").to_pylist(), dtype=np.float64)
    water = np.asarray(monthly.column("water_prod_m3_sc").to_pylist(), dtype=np.float64)
    np.testing.assert_allclose(liquid, oil + water, rtol=0.0, atol=1e-12)
    assert float(oil.sum()) > 0.0
    # The produced gas is far larger than the produced liquid in standard volume — which is
    # exactly why the two are never summed.
    assert produced > float(liquid.sum())


# --------------------------------------------------------------------------------------
# 13.4 — the restart pair
# --------------------------------------------------------------------------------------


def test_the_continuation_is_the_same_run_on_every_black_oil_quantity(
    bo: tuple[Path, SuiteReport],
) -> None:
    """So/Sw/Sg/p/Rs, dissolved and free gas inventory, and surface Vo/Vw/Vg."""
    _run_dir, report = bo
    check = next(c for c in report.checks if c.name == "black_oil_restart")
    assert check.status == "PASS", check.reason
    tolerances = load_blackoil_tolerances(PATHS.resolve(DEFAULT_BLACKOIL_TOLERANCES_RELPATH))
    for metric, threshold in (
        ("blackoil_restart_saturation_abs", "blackoil_restart_saturation_abs_max"),
        ("blackoil_restart_pressure_relative", "blackoil_restart_pressure_relative_max"),
        ("blackoil_restart_rs_relative", "blackoil_restart_rs_relative_max"),
        ("blackoil_restart_free_gas_relative", "blackoil_restart_inventory_relative_max"),
        ("blackoil_restart_dissolved_gas_relative", "blackoil_restart_inventory_relative_max"),
        (
            "blackoil_restart_surface_volume_relative",
            "blackoil_restart_surface_volume_relative_max",
        ),
    ):
        assert metric in check.metrics, metric
        assert check.metrics[metric] <= tolerances[threshold], (metric, check.metrics[metric])
    # The comparison is not vacuous: the continuation really covered the whole horizon, and
    # the three surface volumes it is compared on are not all zero.
    assert check.metrics["blackoil_restart_completed_report_step"] >= 1.0
    for component in BO_COMPONENTS:
        assert check.metrics[f"blackoil_continuous_surface_{component}_m3_sc"] > 0.0, component


def test_the_two_sides_split_the_restart_after_the_same_report_step(
    bo: tuple[Path, SuiteReport],
) -> None:
    """13.4: the step Python cuts the prefix at is the step the diagnostic proves gas by.

    They are two constants in two languages with the same job. If they drifted, the
    checkpoint would move to before the phase transition, the continuation would stop
    exercising `BlackOilX`'s phase state, and every other test here would still pass —
    because each side asserts against its own copy. `_run_black_oil_restart` refuses the
    session outright when they disagree; this is the same comparison made against the
    published record, so a drift is visible in the artifacts as well as at run time.
    """
    run_dir, _report = bo
    diagnostic = json.loads(
        (run_dir / "verification" / "blackoil.json").read_text(encoding="utf-8")
    )
    assert int(diagnostic["restart_after_report_step"]) == BO_RESTART_AFTER_STEP


def test_the_checkpoint_the_continuation_read_carried_a_state_that_already_held_gas(
    bo: tuple[Path, SuiteReport],
) -> None:
    """13.4: the split is after the phase transition, so the phase state is what is restored.

    A black-oil primary unknown is a `BlackOilX` — a value AND the phase state it belongs to.
    A checkpoint taken before any gas appeared could be restored correctly by a continuation
    that carried saturations alone, and would prove nothing about the phase state; this
    asserts that the split really is on the other side of the transition.
    """
    run_dir, _report = bo
    diagnostic = json.loads(
        (run_dir / "verification" / "blackoil.json").read_text(encoding="utf-8")
    )
    measured = diagnostic["bo_depletion"]
    step = int(diagnostic["restart_after_report_step"])
    per_state = [float(v) for v in measured["max_sg_per_state"]]
    assert per_state[step] > 1e-3, (
        f"the restart is taken after report step {step}, where the largest gas saturation is "
        f"{per_state[step]}; the checkpoint has to carry a state that already holds free gas"
    )
    assert per_state[step - 1] <= 1e-10, "the transition must happen inside the prefix"
    assert float(measured["max_sg_at_restart_boundary"]) == pytest.approx(per_state[step])


def test_the_two_phase_controls_are_exactly_what_they_were_before_the_phase_dispatch_changed(
    bo: tuple[Path, SuiteReport],
) -> None:
    """13.3's regression: `native_control` at `n_phases = 2` builds the same control.

    The evidence is produced by the diagnostic, which builds each control three ways — the
    two-argument call the oil-water path has always made, the explicit two-phase one, and the
    three-phase one — and compares them by value. `InjectorControl` holds its mixture as an
    array, so structural equality would compare it by identity and always be false; the
    comparison is therefore on the native type, the target and the mixture's VALUE.
    """
    run_dir, _report = bo
    diagnostic = json.loads(
        (run_dir / "verification" / "blackoil.json").read_text(encoding="utf-8")
    )
    dispatch = diagnostic["phase_dispatch"]
    assert dispatch["water_injection_mixture_two_phase"] == [1.0, 0.0]
    assert dispatch["water_injection_mixture_from_count"]["2"] == [1.0, 0.0]
    assert dispatch["water_injection_mixture_from_count"]["3"] == [1.0, 0.0, 0.0]
    assert dispatch["controls"], "the regression compared no controls at all"
    for label, control in dispatch["controls"].items():
        assert control["default_equals_explicit_two_phase"], label
        assert control["target_unchanged_at_three_phases"], label
        if control["two_phase_mixture"]:
            assert control["two_phase_mixture"] == [1.0, 0.0], label
            assert control["three_phase_mixture"] == [1.0, 0.0, 0.0], label
