"""E01.0: measured resource profiles, the read-only probe and the session budget ledger.

Every boundary here is described to the policy as data, never produced by actually
consuming the resource: a machine with 16 GiB already resident, a full disk, a session
that has run past its wall budget and a job that has run past its timeout are all
`ResourceSnapshot` values handed to a pure function. Nothing in this file allocates
memory, fills a disk or sleeps, and nothing mocks the code under test — the injected
seam is the probe, which is the real boundary between the OS and the policy.

The one test that touches the live machine, `test_probe_reads_the_live_machine`, only
reads: it asserts the shape and the honesty of the measurement, not any particular value,
so it says the same thing on an 8 GiB laptop and on a 128 GiB workstation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest
from pydantic import ValidationError

from so_recon.config.load import ConfigError, load_project_config, resolved_config_dict
from so_recon.config.resources import (
    GIB,
    MIB,
    MIN_RESERVE_BYTES,
    P0_VERIFY_PROFILE,
    P1_E02_MATRIX_PROFILE,
    P1_E03_SMOKE_PROFILE,
    P1_LOOP_PROFILE,
    MissingResourceProfileError,
    ResourceProfile,
    UnapprovedResourceProfileError,
    require_resource_profile,
    resource_profile,
)
from so_recon.config.schema import ProjectConfig, SourcesConfig
from so_recon.environment.resources import (
    MemoryPressure,
    ResourceSnapshot,
    classify_pressure,
    probe_hardware,
    probe_resources,
    process_tree_usage,
    read_memory_pressure,
)
from so_recon.simulator.budget import (
    DISK_FAILURE_RESERVE_BYTES,
    BudgetLedger,
    BudgetStop,
    ResourceWatchdog,
    disk_headroom_stop,
    effective_caps,
    effective_hard_bytes,
    effective_soft_bytes,
)
from so_recon.simulator.contracts import (
    MAX_ATTEMPTS,
    TIME_CELL_AXES,
    ArrayRef,
    CostRecord,
    ForwardResult,
    ForwardStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CASE_SHA = "a" * 64
OTHER_CASE_SHA = "b" * 64
MODEL_HASH = "c" * 64
OTHER_MODEL_HASH = "d" * 64
ARRAY_SHA = "e" * 64


# --------------------------------------------------------------------------- fixtures


def pressure(
    status: str = "normal",
    raw: str | None = "1",
    method: str = "sysctlbyname:kern.memorystatus_vm_pressure_level",
) -> MemoryPressure:
    return MemoryPressure(status=status, raw=raw, method=method)  # type: ignore[arg-type]


def snapshot(**over: Any) -> ResourceSnapshot:
    """A healthy 24 GiB machine, overridden field by field to describe each boundary."""
    base: dict[str, Any] = {
        "total_bytes": 24 * GIB,
        "available_bytes": 16 * GIB,
        "process_rss_bytes": 1 * GIB,
        "process_cpu_s": 1.0,
        "process_measurement_method": "psutil process tree from pid 1: 1 processes read",
        "swap_used_bytes": 0,
        "pressure": pressure(),
        "disk_free_bytes": 100 * GIB,
        "monotonic_s": 0.0,
    }
    base.update(over)
    return ResourceSnapshot(**base)


class ScriptedProbe:
    """Hands out prepared snapshots in order; the last one repeats.

    This is the injected seam the plan asks for. It carries the clock too — the snapshot's
    `monotonic_s` is the only time source the policy reads — so a wall-budget or job-timeout
    boundary is reached by describing a later instant, never by sleeping through one.
    """

    def __init__(self, *snapshots: ResourceSnapshot) -> None:
        if not snapshots:
            raise ValueError("a scripted probe needs at least one snapshot")
        self.snapshots = list(snapshots)
        self.calls = 0

    def __call__(self) -> ResourceSnapshot:
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return self.snapshots[index]


def cost(**over: Any) -> CostRecord:
    base: dict[str, Any] = {
        "wall_s": 10.0,
        "cpu_s": 9.0,
        "peak_rss_bytes": 2 * GIB,
        "output_bytes": 4 * MIB,
        "accepted_steps": 12,
        "cut_steps": 0,
        "nonlinear_iterations": 40,
        "retry_count": 0,
        "measurement_method": "psutil process tree",
    }
    base.update(over)
    return CostRecord(**base)


def result(
    job_id: str,
    *,
    status: ForwardStatus = "COMPLETE",
    case_sha256: str = CASE_SHA,
    model_hash: str = MODEL_HASH,
    **cost_over: Any,
) -> ForwardResult:
    complete = status == "COMPLETE"
    states = {
        "sw": ArrayRef(
            path="artifacts/states.h5",
            dataset="sw",
            sha256=ARRAY_SHA,
            shape=(1, 8),
            dtype="float64",
            unit="1",
            axis_order=TIME_CELL_AXES,
        )
    }
    return ForwardResult(
        job_id=job_id,
        case_sha256=case_sha256,
        model_hash=model_hash,
        physics_class="OW-2p",
        status=status,
        reason=None if complete else f"injected {status}",
        completed_time_s=86400.0 if complete else 0.0,
        times_s=(86400.0,) if complete else (),
        states=states if complete else {},
        monthly_path="artifacts/monthly.parquet" if complete else None,
        connections_path="artifacts/connections.parquet" if complete else None,
        balances_path="artifacts/balances.parquet" if complete else None,
        solver_metadata={},
        cost=cost(**cost_over),
        parent_attempt_ids=(),
    )


def ledger(
    tmp_path: Path,
    probe: Callable[[], ResourceSnapshot] | None = None,
    profile: ResourceProfile = P0_VERIFY_PROFILE,
    session_id: str = "s1",
) -> BudgetLedger:
    return BudgetLedger.start(
        profile=profile,
        path=tmp_path / "ledger.json",
        session_id=session_id,
        probe=probe or ScriptedProbe(snapshot()),
    )


def reserve(book: BudgetLedger, job_id: str, **over: Any) -> None:
    params: dict[str, Any] = {
        "case_sha256": CASE_SHA,
        "model_hash": MODEL_HASH,
        "attempt": 1,
        "estimated_s": 10.0,
        "estimated_bytes": 4 * MIB,
    }
    params.update(over)
    book.reserve(job_id, params.pop("estimated_s"), params.pop("estimated_bytes"), **params)


# ------------------------------------------------------------------ the measured cap


def test_memory_limit_accounts_for_other_apps() -> None:
    """The plan's own kernel example: 8 GiB of the 24 is free, 4 is ours, 6 is reserved."""
    assert (
        effective_hard_bytes(
            configured=16 * GIB,
            total=24 * GIB,
            available=8 * GIB,
            project_rss=4 * GIB,
            reserve=6 * GIB,
        )
        == 6 * GIB
    )


def test_an_eight_gib_laptop_gets_a_non_negative_cap_that_forbids_starting() -> None:
    """total - reserve and rss + available - reserve are both negative here."""
    caps = effective_caps(
        P0_VERIFY_PROFILE,
        snapshot(total_bytes=8 * GIB, available_bytes=1 * GIB, process_rss_bytes=512 * MIB),
    )
    assert caps.hard_bytes == 0
    assert caps.soft_bytes == 0
    assert caps.allows_start is False


@pytest.mark.parametrize(
    ("total", "available", "project_rss"),
    [(8 * GIB, 1 * GIB, 0), (4 * GIB, 0, 0), (6 * GIB, 0, 0), (0, 0, 0)],
)
def test_the_cap_is_never_negative(total: int, available: int, project_rss: int) -> None:
    assert (
        effective_hard_bytes(
            configured=16 * GIB,
            total=total,
            available=available,
            project_rss=project_rss,
            reserve=6 * GIB,
        )
        == 0
    )


def test_the_configured_cap_wins_when_the_machine_is_large() -> None:
    assert (
        effective_hard_bytes(
            configured=16 * GIB,
            total=128 * GIB,
            available=100 * GIB,
            project_rss=2 * GIB,
            reserve=6 * GIB,
        )
        == 16 * GIB
    )


def test_the_soft_cap_follows_the_hard_cap_down() -> None:
    """A measured hard cap below the configured soft cap must pull the soft cap with it."""
    assert effective_soft_bytes(configured=12 * GIB, effective_hard=8 * GIB) == 6 * GIB
    assert effective_soft_bytes(configured=12 * GIB, effective_hard=16 * GIB) == 12 * GIB
    assert effective_soft_bytes(configured=12 * GIB, effective_hard=0) == 0


def test_a_zero_cap_refuses_the_reservation(tmp_path: Path) -> None:
    probe = ScriptedProbe(snapshot(total_bytes=8 * GIB, available_bytes=1 * GIB))
    book = ledger(tmp_path, probe)
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j1")
    assert excinfo.value.status == "RESOURCE_FAILURE"
    assert "cap" in excinfo.value.reason


# ---------------------------------------------------------------- profiles and config


def test_p0_profile_carries_the_normative_numbers() -> None:
    p = P0_VERIFY_PROFILE
    assert (p.profile, p.soft_bytes, p.hard_bytes) == ("P0_VERIFY", 12 * GIB, 16 * GIB)
    assert (p.reserve_bytes, p.disk_budget_bytes) == (6 * GIB, 5 * GIB)
    assert (p.wall_budget_s, p.job_timeout_s, p.startup_timeout_s) == (600, 300, 300)
    assert p.max_new_forward == 64
    assert (p.julia_workers, p.julia_threads, p.blas_threads) == (1, 4, 1)
    assert (p.poll_interval_s, p.max_swap_growth_bytes) == (0.25, 512 * MIB)


def test_p1_preset_changes_only_the_three_approved_limits() -> None:
    p0 = P0_VERIFY_PROFILE.model_dump()
    p1 = P1_LOOP_PROFILE.model_dump()
    assert {k for k in p0 if p0[k] != p1[k]} == {
        "profile",
        "wall_budget_s",
        "job_timeout_s",
        "max_new_forward",
    }
    assert (P1_LOOP_PROFILE.wall_budget_s, P1_LOOP_PROFILE.job_timeout_s) == (3600, 900)
    assert P1_LOOP_PROFILE.max_new_forward == 2000


def test_p1_e03_smoke_preset_moves_only_the_session_limits_of_p1_loop() -> None:
    """The E03 smoke cycle needs one complete N16 SMC method per session (plan E03 §12).

    Everything but the wall budget and the forward count stays P1_LOOP: a smoke session
    is the same machine policy with a longer single sitting, never a relabelled memory or
    disk allowance.
    """
    p1 = P1_LOOP_PROFILE.model_dump()
    smoke = P1_E03_SMOKE_PROFILE.model_dump()
    assert {k for k in p1 if p1[k] != smoke[k]} == {
        "profile",
        "wall_budget_s",
        "max_new_forward",
    }
    assert (P1_E03_SMOKE_PROFILE.wall_budget_s, P1_E03_SMOKE_PROFILE.max_new_forward) == (
        10800,
        512,
    )
    assert P1_E03_SMOKE_PROFILE.job_timeout_s == P1_LOOP_PROFILE.job_timeout_s
    assert resource_profile("P1_E03_SMOKE") == P1_E03_SMOKE_PROFILE
    assert require_resource_profile(P1_E03_SMOKE_PROFILE, command="forward") is (
        P1_E03_SMOKE_PROFILE
    )


def test_p1_e02_matrix_preset_moves_only_the_session_limits_of_p1_loop() -> None:
    """The remaining E02 native matrix needs one complete run per session (plan E03 §12).

    A measured N32 learned run took 510 forwards at ~20 s each; an N64 run is roughly
    double, and neither fits a P1_LOOP hour. Everything but the wall budget and the
    forward count stays P1_LOOP: a matrix session is the same machine policy with a
    longer single sitting, never a relabelled memory or disk allowance.
    """
    p1 = P1_LOOP_PROFILE.model_dump()
    matrix = P1_E02_MATRIX_PROFILE.model_dump()
    assert {k for k in p1 if p1[k] != matrix[k]} == {
        "profile",
        "wall_budget_s",
        "max_new_forward",
    }
    assert (P1_E02_MATRIX_PROFILE.wall_budget_s, P1_E02_MATRIX_PROFILE.max_new_forward) == (
        21600,
        1200,
    )
    assert P1_E02_MATRIX_PROFILE.job_timeout_s == P1_LOOP_PROFILE.job_timeout_s
    assert resource_profile("P1_E02_MATRIX") == P1_E02_MATRIX_PROFILE
    assert require_resource_profile(P1_E02_MATRIX_PROFILE, command="forward") is (
        P1_E02_MATRIX_PROFILE
    )


def test_presets_are_looked_up_by_name() -> None:
    assert resource_profile("P1_LOOP") == P1_LOOP_PROFILE
    with pytest.raises(ValueError, match="unknown resource profile"):
        resource_profile("P2_FIELD")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("soft_bytes", 20 * GIB),  # soft above hard
        ("reserve_bytes", MIN_RESERVE_BYTES - 1),  # below the normative OS reserve
        ("job_timeout_s", 601),  # a job that cannot fit inside the session
        ("startup_timeout_s", 601),  # cold startup is inside the wall budget
        ("julia_workers", 2),
        ("blas_threads", 2),
        ("max_new_forward", 0),
        ("poll_interval_s", 0.0),
    ],
)
def test_an_inconsistent_profile_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ResourceProfile(**{**P0_VERIFY_PROFILE.model_dump(), field: value})


def test_the_e01_config_carries_the_p0_profile() -> None:
    cfg = load_project_config(REPO_ROOT / "configs" / "e01.yml")
    assert cfg.resources == P0_VERIFY_PROFILE


def test_a_3_0_config_serialises_without_the_resource_field(tmp_project: Path) -> None:
    """Half one of the 3.0 rule: an omitted profile leaves the legacy bytes untouched."""
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    assert cfg.spec_version == "3.0"
    assert cfg.resources is None
    assert "resources" not in resolved_config_dict(cfg)


def test_a_3_0_config_that_sets_resources_is_rejected(tmp_project: Path) -> None:
    """Half two: the exclusion may only ever drop a None, so the hash cannot lie."""
    path = tmp_project / "configs" / "project.yml"
    path.write_text(
        path.read_text(encoding="utf-8") + "\nresources:\n  profile: P0_VERIFY\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="spec_version 3.0 has no 'resources' field"):
        load_project_config(path)


def test_a_3_0_project_config_with_a_profile_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError, match="spec_version 3.0 has no 'resources' field"):
        ProjectConfig(
            spec_version="3.0",
            config_version="t",
            sources=SourcesConfig(files=[]),
            resources=P0_VERIFY_PROFILE,
        )


def test_a_4_0_config_without_a_profile_is_still_valid() -> None:
    """Manifest and config-only checks need no budget; only a physical command does."""
    cfg = ProjectConfig(spec_version="4.0", config_version="t", sources=SourcesConfig(files=[]))
    assert cfg.resources is None
    assert resolved_config_dict(cfg)["resources"] is None


def test_a_physical_command_refuses_to_run_without_a_profile() -> None:
    with pytest.raises(MissingResourceProfileError, match="forward"):
        require_resource_profile(None, command="forward")
    assert require_resource_profile(P0_VERIFY_PROFILE, command="forward") is P0_VERIFY_PROFILE
    assert require_resource_profile(P1_LOOP_PROFILE, command="forward") is P1_LOOP_PROFILE


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Neither has a measured backstop the way the memory caps do: `total - reserve`
        # clamps those whatever the config claims, but a session length and a forward
        # count are spent exactly as written.
        ("wall_budget_s", 86400),
        ("max_new_forward", 100000),
        ("disk_budget_bytes", 500 * GIB),
    ],
)
def test_a_relabelled_profile_is_refused_by_a_physical_command(field: str, value: int) -> None:
    """A block may not claim an approved profile's name while carrying other numbers."""
    tampered = ResourceProfile(**{**P0_VERIFY_PROFILE.model_dump(), field: value})
    assert tampered.profile == "P0_VERIFY"  # internally consistent, so it validates
    with pytest.raises(UnapprovedResourceProfileError) as excinfo:
        require_resource_profile(tampered, command="forward")
    assert field in str(excinfo.value)
    assert str(value) in str(excinfo.value)


# ------------------------------------------------------------------------- the probe


def test_probe_reads_the_live_machine(tmp_path: Path) -> None:
    """Read-only and allocation-free: the shape of the measurement, not its value."""
    first = probe_resources(None, tmp_path)
    second = probe_resources(None, tmp_path)
    assert first.total_bytes > 0
    assert 0 <= first.available_bytes <= first.total_bytes
    # This process is readable by definition, so the tree really is measured here — the
    # nullable field carries a number, and the method says how many processes it covered.
    assert first.process_rss_bytes is not None and first.process_rss_bytes > 0
    assert first.process_cpu_s is not None and first.process_cpu_s >= 0.0
    assert "processes read" in first.process_measurement_method
    assert first.disk_free_bytes > 0
    assert second.monotonic_s >= first.monotonic_s
    assert first.pressure.status in {"normal", "warn", "critical", "unknown"}
    assert first.pressure.method


def test_the_live_pressure_probe_reports_a_status_and_its_method() -> None:
    measured = read_memory_pressure()
    assert measured.status in {"normal", "warn", "critical", "unknown"}
    assert measured.method
    if measured.status != "unknown":
        assert measured.raw is not None


@pytest.mark.parametrize(
    ("raw", "status", "kept"),
    [
        (1, "normal", "1"),
        (2, "warn", "2"),
        (4, "critical", "4"),
        # Read, but not a level this code knows: unknown, with the value kept. Rounding it
        # down to "normal" because it is not one of the alarming ones would be a guess.
        (3, "unknown", "3"),
        # Not read at all: unknown, and no fabricated number stands in for the absence.
        (None, "unknown", None),
    ],
)
def test_an_unknown_pressure_level_is_never_substituted(
    raw: int | None, status: str, kept: str | None
) -> None:
    measured = classify_pressure(raw, method="sysctlbyname:test")
    assert (measured.status, measured.raw) == (status, kept)
    assert "sysctlbyname:test" in measured.method


def _raising(error: Exception) -> Callable[[int], list[Any]]:
    """A tree reader that fails the way psutil fails when a tree cannot be listed."""

    def read(pid: int) -> list[Any]:
        raise error

    return read


class FakeProcess:
    """A process that answers, or raises the psutil error it was built with.

    Stands in for `psutil.Process` at the one place the probe touches the OS. The branch
    under test is this module's own handling of psutil's error types, which cannot be
    reached otherwise: a test run cannot arrange to be denied access to a process it owns.
    """

    def __init__(self, pid: int, rss: int = 0, cpu_s: float = 0.0, error: Exception | None = None):
        self.pid = pid
        self._rss = rss
        self._cpu_s = cpu_s
        self._error = error

    def oneshot(self) -> AbstractContextManager[None]:
        return nullcontext()

    def _check(self) -> None:
        if self._error is not None:
            raise self._error

    def memory_info(self) -> Any:
        self._check()
        return SimpleNamespace(rss=self._rss)

    def cpu_times(self) -> Any:
        self._check()
        return SimpleNamespace(user=self._cpu_s, system=0.0)


def test_a_readable_process_tree_is_summed_over_every_process() -> None:
    usage = process_tree_usage(
        11, tree=lambda pid: [FakeProcess(pid, rss=3 * GIB, cpu_s=2.0), FakeProcess(12, 1 * GIB)]
    )
    assert (usage.rss_bytes, usage.cpu_s, usage.sampled) == (4 * GIB, 2.0, 2)
    assert "2 processes read" in usage.method


def test_a_tree_whose_root_has_exited_measures_zero() -> None:
    """The documented race: something that has exited really is using nothing."""
    usage = process_tree_usage(11, tree=_raising(psutil.NoSuchProcess(11)))
    assert (usage.rss_bytes, usage.cpu_s) == (0, 0.0)
    assert "has exited" in usage.method


def test_a_child_that_exits_mid_scan_is_skipped_not_treated_as_unknown() -> None:
    usage = process_tree_usage(
        11,
        tree=lambda pid: [
            FakeProcess(pid, rss=2 * GIB),
            FakeProcess(12, error=psutil.NoSuchProcess(12)),
        ],
    )
    assert (usage.rss_bytes, usage.sampled) == (2 * GIB, 1)


@pytest.mark.parametrize("error", [psutil.AccessDenied(11), psutil.ZombieProcess(11)])
def test_an_unreadable_root_is_unknown_and_never_zero(error: Exception) -> None:
    usage = process_tree_usage(11, tree=_raising(error))
    assert usage.rss_bytes is None
    assert usage.cpu_s is None
    assert type(error).__name__ in usage.method


def test_an_unreadable_child_makes_the_whole_tree_unknown() -> None:
    """A partial sum would undercount, which is the unsafe direction for a memory guard."""
    usage = process_tree_usage(
        11,
        tree=lambda pid: [
            FakeProcess(pid, rss=2 * GIB),
            FakeProcess(12, error=psutil.AccessDenied(12)),
        ],
    )
    assert usage.rss_bytes is None
    assert "AccessDenied" in usage.method
    assert "pid 12" in usage.method


def test_an_unmeasurable_process_tree_refuses_to_start_work(tmp_path: Path) -> None:
    """The guard that cannot see the run must not be told the run occupies nothing."""
    blind = snapshot(
        process_rss_bytes=None,
        process_cpu_s=None,
        process_measurement_method="psutil process tree from pid 11: unreadable (AccessDenied)",
    )
    decision = ResourceWatchdog(P0_VERIFY_PROFILE, baseline=blind).observe(blind)
    assert decision.action == "drain"
    assert decision.status == "RESOURCE_FAILURE"
    assert any("could not be measured" in reason for reason in decision.reasons)
    book = ledger(tmp_path, ScriptedProbe(blind))
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j1")
    assert excinfo.value.status == "RESOURCE_FAILURE"
    assert "unknown measurement as zero" in excinfo.value.reason


def test_hardware_probe_records_the_host_without_installing_pytorch() -> None:
    hardware = probe_hardware(probe=lambda cmd: "julia version 1.12.7\n")
    assert hardware.arch and hardware.os
    assert hardware.total_ram_bytes > 0
    assert hardware.external_power in {"ac", "battery", "unknown"}
    assert isinstance(hardware.mps_capable, bool)
    assert isinstance(hardware.cuda_capable, bool)
    assert hardware.accelerator_method
    if hardware.julia_executable is not None:
        assert hardware.julia_version == "1.12.7"


# ---------------------------------------------------------------------- the watchdog


def watchdog(**baseline: Any) -> ResourceWatchdog:
    return ResourceWatchdog(P0_VERIFY_PROFILE, baseline=snapshot(**baseline))


def test_a_quiet_machine_keeps_running() -> None:
    decision = watchdog().observe(snapshot(process_rss_bytes=2 * GIB))
    assert decision.action == "continue"
    assert decision.accept_new_jobs is True
    assert decision.status is None


def test_a_soft_breach_warns_and_releases_completed_allocations() -> None:
    decision = watchdog().observe(snapshot(process_rss_bytes=13 * GIB))
    assert decision.action == "warn"
    assert decision.accept_new_jobs is True
    assert decision.release_completed is True
    assert decision.status is None


def test_a_hard_breach_stops_new_jobs_and_asks_to_finish_the_chunk() -> None:
    decision = watchdog().observe(snapshot(process_rss_bytes=17 * GIB))
    assert decision.action == "drain"
    assert decision.accept_new_jobs is False
    assert decision.terminate_process_group is False
    assert decision.status == "RESOURCE_FAILURE"


def test_continued_growth_after_a_hard_breach_terminates_the_process_group() -> None:
    guard = watchdog()
    assert guard.observe(snapshot(process_rss_bytes=17 * GIB)).action == "drain"
    decision = guard.observe(snapshot(process_rss_bytes=17 * GIB + 1))
    assert decision.action == "terminate"
    assert decision.terminate_process_group is True
    assert decision.status == "RESOURCE_FAILURE"
    # The decision is sticky: a later quiet snapshot does not un-kill the worker.
    assert guard.observe(snapshot(process_rss_bytes=1 * GIB)).action == "terminate"


def test_a_hard_breach_that_stops_growing_is_not_escalated() -> None:
    guard = watchdog()
    assert guard.observe(snapshot(process_rss_bytes=17 * GIB)).action == "drain"
    assert guard.observe(snapshot(process_rss_bytes=17 * GIB)).action == "drain"


def test_critical_os_pressure_stops_new_jobs_even_below_the_cap() -> None:
    decision = watchdog().observe(
        snapshot(process_rss_bytes=2 * GIB, pressure=pressure("critical", raw="4"))
    )
    assert decision.action == "drain"
    assert decision.status == "RESOURCE_FAILURE"
    assert any("critical" in reason for reason in decision.reasons)


def test_swap_growth_beyond_the_limit_stops_new_jobs() -> None:
    guard = watchdog(swap_used_bytes=1 * GIB)
    quiet = guard.observe(snapshot(swap_used_bytes=1 * GIB + 512 * MIB))
    assert quiet.action == "continue"
    decision = guard.observe(snapshot(swap_used_bytes=1 * GIB + 512 * MIB + 1))
    assert decision.action == "drain"
    assert decision.status == "RESOURCE_FAILURE"
    assert any("swap" in reason for reason in decision.reasons)


def test_unknown_pressure_keeps_the_available_and_swap_guards_and_says_so() -> None:
    """The unknown measurement is reported as unknown, not silently treated as normal."""
    blind = pressure("unknown", raw=None, method="unavailable: no probe for Linux")
    guard = watchdog(pressure=blind)
    quiet = guard.observe(snapshot(process_rss_bytes=2 * GIB, pressure=blind))
    assert quiet.action == "continue"
    assert any("pressure unavailable" in note for note in quiet.limitations)
    assert any("no probe for Linux" in note for note in quiet.limitations)
    starved = guard.observe(
        snapshot(process_rss_bytes=2 * GIB, total_bytes=8 * GIB, available_bytes=0, pressure=blind)
    )
    assert starved.action == "drain"
    assert starved.status == "RESOURCE_FAILURE"


def test_a_job_that_outruns_its_timeout_is_terminated() -> None:
    guard = watchdog()
    running = guard.observe(snapshot(monotonic_s=299.0), job_started_monotonic_s=0.0)
    assert running.action == "continue"
    decision = guard.observe(snapshot(monotonic_s=301.0), job_started_monotonic_s=0.0)
    assert decision.action == "terminate"
    assert decision.status == "TIMEOUT"


def test_a_disk_that_falls_into_the_record_reserve_terminates_the_running_job() -> None:
    """The reserve exists so the failure record can be written; eating it ends the job.

    Unlike a memory breach, this one is not escalated through a drain: finishing the chunk
    in hand writes more output, which is the opposite of the remedy. A run that filled the
    disk and then could not write down why is the silent stop SPEC 18.4 forbids.
    """
    guard = watchdog()
    assert guard.observe(snapshot(disk_free_bytes=1 * GIB)).action == "continue"
    decision = guard.observe(snapshot(disk_free_bytes=DISK_FAILURE_RESERVE_BYTES - 1))
    assert decision.action == "terminate"
    assert decision.terminate_process_group is True
    assert decision.status == "RESOURCE_FAILURE"
    assert any("failure record" in reason for reason in decision.reasons)
    # Sticky, like every other termination: a later roomy snapshot does not un-kill it.
    assert guard.observe(snapshot(disk_free_bytes=100 * GIB)).action == "terminate"


def test_a_committed_disk_budget_drains_without_killing_the_current_chunk() -> None:
    committed = P0_VERIFY_PROFILE.disk_budget_bytes - DISK_FAILURE_RESERVE_BYTES + 1
    decision = watchdog().observe(snapshot(), session_output_bytes=committed)
    assert decision.action == "drain"
    assert decision.accept_new_jobs is False
    assert decision.terminate_process_group is False
    assert decision.status == "INCOMPLETE_BUDGET"
    assert any("disk budget" in reason for reason in decision.reasons)


def test_the_mid_job_disk_guard_is_silent_while_there_is_room() -> None:
    assert (
        disk_headroom_stop(
            P0_VERIFY_PROFILE,
            session_output_bytes=P0_VERIFY_PROFILE.disk_budget_bytes - DISK_FAILURE_RESERVE_BYTES,
            free_bytes=DISK_FAILURE_RESERVE_BYTES,
        )
        is None
    )


def test_the_session_wall_budget_drains_without_killing_the_current_chunk() -> None:
    decision = watchdog().observe(snapshot(monotonic_s=601.0))
    assert decision.action == "drain"
    assert decision.accept_new_jobs is False
    assert decision.terminate_process_group is False
    assert decision.status == "INCOMPLETE_BUDGET"


# ------------------------------------------------------------------------- the ledger


def test_a_reservation_is_recorded_as_a_pending_attempt(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    assert book.pending_job_ids == ("j1",)
    assert book.attempted_job_ids == ("j1",)
    assert book.completed_job_ids == ()
    written = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert written["session_id"] == "s1"
    assert written["entries"][0]["case_sha256"] == CASE_SHA
    assert written["entries"][0]["state"] == "PENDING"


def test_completing_a_job_clears_its_reservation_and_records_the_cost(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    book.complete("j1", result("j1"))
    assert book.pending_job_ids == ()
    assert book.completed_job_ids == ("j1",)
    totals = book.cumulative_totals()
    assert (totals.forwards, totals.output_bytes) == (1, 4 * MIB)
    assert totals.wall_s == 10.0


def test_a_failed_attempt_still_counts_against_the_forward_budget(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    book.complete("j1", result("j1", status="NUMERICAL_FAILURE"))
    assert book.completed_job_ids == ()
    assert book.attempted_job_ids == ("j1",)
    assert book.cumulative_totals().forwards == 1


def test_the_attempt_limit_refuses_a_third_attempt_on_one_physical_model(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reserve(book, f"j{attempt}", attempt=attempt)
        book.complete(f"j{attempt}", result(f"j{attempt}", status="NUMERICAL_FAILURE"))
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j3", attempt=MAX_ATTEMPTS)
    assert excinfo.value.status == "PROTOCOL_FAILURE"
    assert "attempt" in excinfo.value.reason


def test_the_forward_limit_refuses_the_job_after_the_last_one(tmp_path: Path) -> None:
    tiny = ResourceProfile(**{**P0_VERIFY_PROFILE.model_dump(), "max_new_forward": 2})
    book = ledger(tmp_path, profile=tiny)
    for i in (1, 2):
        reserve(book, f"j{i}", model_hash=f"{i}" * 64)
        book.complete(f"j{i}", result(f"j{i}", model_hash=f"{i}" * 64))
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j3", model_hash=OTHER_MODEL_HASH)
    assert excinfo.value.status == "INCOMPLETE_BUDGET"
    assert "forward" in excinfo.value.reason


def test_the_wall_budget_refuses_a_job_that_cannot_finish_inside_it(tmp_path: Path) -> None:
    probe = ScriptedProbe(snapshot(monotonic_s=0.0), snapshot(monotonic_s=500.0))
    book = ledger(tmp_path, probe)
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j1", estimated_s=200.0)
    assert excinfo.value.status == "INCOMPLETE_BUDGET"
    assert "wall" in excinfo.value.reason


def test_a_job_longer_than_the_job_timeout_is_refused_before_it_starts(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    with pytest.raises(BudgetStop, match="job timeout"):
        reserve(book, "j1", estimated_s=301.0)


def test_disk_exhaustion_refuses_the_job_with_a_resource_failure(tmp_path: Path) -> None:
    probe = ScriptedProbe(snapshot(disk_free_bytes=DISK_FAILURE_RESERVE_BYTES + 1))
    book = ledger(tmp_path, probe)
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j1", estimated_bytes=2 * MIB)
    assert excinfo.value.status == "RESOURCE_FAILURE"
    assert "free disk" in excinfo.value.reason


def test_the_disk_guard_keeps_room_for_the_failure_and_ledger_records(tmp_path: Path) -> None:
    """Exactly the predicted output fits, but not the 64 MiB kept for the records."""
    free = 8 * MIB + DISK_FAILURE_RESERVE_BYTES
    book = ledger(tmp_path, ScriptedProbe(snapshot(disk_free_bytes=free)))
    reserve(book, "j0", estimated_bytes=8 * MIB)
    starved = ledger(
        tmp_path / "next",
        ScriptedProbe(snapshot(disk_free_bytes=free - 1)),
        session_id="s2",
    )
    with pytest.raises(BudgetStop, match="reserved for the failure"):
        reserve(starved, "j1", estimated_bytes=8 * MIB)


def test_the_session_disk_budget_counts_what_the_session_already_wrote(tmp_path: Path) -> None:
    small = ResourceProfile(**{**P0_VERIFY_PROFILE.model_dump(), "disk_budget_bytes": 100 * MIB})
    book = ledger(tmp_path, profile=small)
    reserve(book, "j0", estimated_bytes=32 * MIB)
    book.complete("j0", result("j0", output_bytes=32 * MIB))
    with pytest.raises(BudgetStop) as excinfo:
        reserve(book, "j1", estimated_bytes=32 * MIB, model_hash=OTHER_MODEL_HASH)
    assert excinfo.value.status == "INCOMPLETE_BUDGET"
    assert "disk budget" in excinfo.value.reason


def test_one_attempts_result_cannot_be_filed_against_another_attempt(tmp_path: Path) -> None:
    """Two attempts on one model share a case hash, so the case check alone is not enough.

    Filing j2's result on j1 would put the wrong cost and checkpoint on j1 and leave j2
    PENDING for the rest of the session, holding its reserved disk against the budget.
    """
    book = ledger(tmp_path)
    reserve(book, "j1", attempt=1)
    reserve(book, "j2", attempt=2)
    with pytest.raises(BudgetStop) as excinfo:
        book.complete("j1", result("j2"))
    assert excinfo.value.status == "PROTOCOL_FAILURE"
    assert "job_id" in excinfo.value.reason
    assert book.pending_job_ids == ("j1", "j2")


def test_a_result_for_a_different_physical_model_is_refused(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    with pytest.raises(BudgetStop) as excinfo:
        book.complete("j1", result("j1", model_hash=OTHER_MODEL_HASH))
    assert excinfo.value.status == "PROTOCOL_FAILURE"
    assert "model_hash" in excinfo.value.reason


def test_a_result_for_a_different_case_is_refused(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    with pytest.raises(BudgetStop) as excinfo:
        book.complete("j1", result("j1", case_sha256=OTHER_CASE_SHA))
    assert excinfo.value.status == "PROTOCOL_FAILURE"
    assert "case_sha256" in excinfo.value.reason


def test_an_outstanding_reservation_still_counts_against_the_disk_budget(
    tmp_path: Path,
) -> None:
    """Space promised to a job that has not finished is space the session no longer has."""
    small = ResourceProfile(**{**P0_VERIFY_PROFILE.model_dump(), "disk_budget_bytes": 100 * MIB})
    book = ledger(tmp_path, profile=small)
    reserve(book, "j0", estimated_bytes=32 * MIB)
    assert book.committed_output_bytes() == 32 * MIB
    with pytest.raises(BudgetStop, match="disk budget"):
        reserve(book, "j1", estimated_bytes=32 * MIB, model_hash=OTHER_MODEL_HASH)


def test_stopping_the_session_is_recorded_and_refuses_further_reservations(
    tmp_path: Path,
) -> None:
    book = ledger(tmp_path)
    reserve(book, "j1")
    book.stop("hard memory cap breached")
    written = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert written["stop_reason"] == "hard memory cap breached"
    # A system stop keeps what finished and what was still pending (SPEC 3.3).
    assert [entry["job_id"] for entry in written["entries"]] == ["j1"]
    with pytest.raises(BudgetStop, match="stopped"):
        reserve(book, "j2", model_hash=OTHER_MODEL_HASH)


def test_an_unwritable_ledger_surfaces_instead_of_passing_silently(tmp_path: Path) -> None:
    """A full disk must fail loudly: a budget nobody could record is not a budget."""
    (tmp_path / "ledger.json").mkdir()
    with pytest.raises(OSError):
        ledger(tmp_path)


# ---------------------------------------------------------------- resume accounting


def test_a_resumed_session_gets_a_fresh_wall_budget_and_keeps_the_parent_cost(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "ledger.json"
    parent = ledger(tmp_path, ScriptedProbe(snapshot(monotonic_s=0.0), snapshot(monotonic_s=590.0)))
    reserve(parent, "j1", estimated_s=5.0)
    parent.complete("j1", result("j1", wall_s=120.0, cpu_s=100.0))
    parent.stop("wall budget spent")

    child = BudgetLedger.resume(
        parent_path=parent_path,
        path=tmp_path / "resume" / "ledger.json",
        session_id="s2",
        probe=ScriptedProbe(snapshot(monotonic_s=10_000.0)),
    )
    # Its own wall budget: 590 + 250 would not fit, so the parent's clock did not follow.
    reserve(child, "j2", estimated_s=250.0, model_hash=OTHER_MODEL_HASH)
    assert child.record.parent_session_id == "s1"
    assert child.record.parent_ledger_sha256 is not None
    # The cumulative cost is not reset by the resume.
    totals = child.cumulative_totals()
    assert totals.wall_s == 120.0
    assert totals.forwards == 2


def test_a_completed_job_is_skipped_only_after_its_input_hash_verifies(tmp_path: Path) -> None:
    parent_path = tmp_path / "ledger.json"
    parent = ledger(tmp_path)
    reserve(parent, "j1")
    parent.complete("j1", result("j1"))
    child = BudgetLedger.resume(
        parent_path=parent_path,
        path=tmp_path / "resume" / "ledger.json",
        session_id="s2",
        probe=ScriptedProbe(snapshot()),
    )
    assert child.is_already_complete(model_hash=MODEL_HASH, case_sha256=CASE_SHA) is True
    # The same model, regenerated from a different case: the recorded work does not apply.
    assert child.is_already_complete(model_hash=MODEL_HASH, case_sha256=OTHER_CASE_SHA) is False
    assert child.is_already_complete(model_hash=OTHER_MODEL_HASH, case_sha256=CASE_SHA) is False


@pytest.mark.parametrize("finished", [False, True])
def test_an_incomplete_job_is_never_skipped(tmp_path: Path, finished: bool) -> None:
    parent_path = tmp_path / "ledger.json"
    parent = ledger(tmp_path)
    reserve(parent, "j1")
    if finished:
        parent.complete("j1", result("j1", status="NUMERICAL_FAILURE"))
    child = BudgetLedger.resume(
        parent_path=parent_path,
        path=tmp_path / "resume" / "ledger.json",
        session_id="s2",
        probe=ScriptedProbe(snapshot()),
    )
    assert child.is_already_complete(model_hash=MODEL_HASH, case_sha256=CASE_SHA) is False


def test_the_attempt_limit_survives_a_resume(tmp_path: Path) -> None:
    """Every attempt counts, in this session or an earlier one (SPEC 3.3)."""
    parent_path = tmp_path / "ledger.json"
    parent = ledger(tmp_path)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reserve(parent, f"j{attempt}", attempt=attempt)
        parent.complete(f"j{attempt}", result(f"j{attempt}", status="NUMERICAL_FAILURE"))
    child = BudgetLedger.resume(
        parent_path=parent_path,
        path=tmp_path / "resume" / "ledger.json",
        session_id="s2",
        probe=ScriptedProbe(snapshot()),
    )
    with pytest.raises(BudgetStop, match="attempt"):
        reserve(child, "j3", attempt=MAX_ATTEMPTS)


def test_a_corrupt_parent_ledger_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    parent_path = tmp_path / "ledger.json"
    parent_path.write_text('{"session_id": "s1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="parent ledger"):
        BudgetLedger.resume(
            parent_path=parent_path,
            path=tmp_path / "resume" / "ledger.json",
            session_id="s2",
            probe=ScriptedProbe(snapshot()),
        )
