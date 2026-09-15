"""Pure contracts for the registered T1/T2/T4 native inference matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so_recon.inference.contracts import DensitySchema, PriorContext
from so_recon.validation.physical_smc import (
    closed_preflight_case,
    convergence_screen,
    preflight_reproducibility_metrics,
    prior_context_payload,
    report_zone_matrix,
)


def _summary(
    particles: int,
    seed: int,
    medians: list[float],
    widths: list[float],
    family_one: float,
) -> dict[str, object]:
    return {
        "n_particles": particles,
        "seed": seed,
        "algorithm_status": "COMPLETE",
        "beta": 1.0,
        "zone_names": ["layer-0", "layer-1"],
        "zone_medians": medians,
        "zone_widths": widths,
        "family_probabilities": {"0": 1.0 - family_one, "1": family_one},
    }


def test_report_zones_include_layers_quadrants_and_t4_remote_support() -> None:
    common_names, common = report_zone_matrix("e02-t1-v1")
    assert common.shape == (10, 512)
    assert common_names[:2] == ("layer-0", "layer-1")
    np.testing.assert_array_equal(common[:2].sum(axis=1), [256.0, 256.0])

    remote_names, remote = report_zone_matrix("e02-t4-v1")
    assert remote.shape == (13, 512)
    assert remote_names[-3:] == ("remote-east", "remote-east-layer-0", "remote-east-layer-1")
    np.testing.assert_array_equal(remote[-3:].sum(axis=1), [64.0, 32.0, 32.0])


def test_prior_context_payload_round_trips_numpy_arrays_through_json_shape() -> None:
    schema = DensitySchema(
        schema_id="test",
        n_v=1,
        n_residual=0,
        families=(0,),
        basis_hash="a" * 64,
        transform_version="test-1",
    )
    context = PriorContext(
        density_schema=schema,
        n_geology=1,
        n_state_residual=0,
        mean=np.array([0.5]),
        chol=np.array([[2.0]]),
        rotation=np.array([[1.0]]),
        design={},
        g_hash="b" * 64,
        information_hash="c" * 64,
    )
    payload = prior_context_payload(context)
    assert payload["mean"] == [0.5]
    rebuilt = PriorContext.model_validate(payload)
    np.testing.assert_array_equal(rebuilt.chol, context.chol)


def test_convergence_screen_applies_paired_gaps_and_precision_limits() -> None:
    rows = [
        _summary(32, 11, [0.60, 0.50], [0.20, 0.10], 0.40),
        _summary(32, 12, [0.61, 0.51], [0.21, 0.11], 0.42),
        _summary(64, 11, [0.61, 0.51], [0.18, 0.09], 0.41),
        _summary(64, 12, [0.62, 0.52], [0.19, 0.10], 0.43),
    ]
    result = convergence_screen(rows)
    assert result["status"] == "PASS"
    assert result["checks"] == {
        "complete_beta_one": True,
        "zone_median_gap": True,
        "family_mass_gap": True,
        "zone_width_gap": True,
        "median_precision": True,
        "family_precision": True,
    }


def test_convergence_screen_does_not_widen_threshold_for_noisy_starts() -> None:
    rows = [
        _summary(32, 11, [0.40, 0.50], [0.20, 0.10], 0.20),
        _summary(32, 12, [0.60, 0.51], [0.21, 0.11], 0.60),
        _summary(64, 11, [0.41, 0.51], [0.18, 0.09], 0.21),
        _summary(64, 12, [0.61, 0.52], [0.19, 0.10], 0.61),
    ]
    result = convergence_screen(rows)
    assert result["status"] == "POSTERIOR_NOT_CONVERGED"
    assert result["checks"]["median_precision"] is False
    assert result["checks"]["family_precision"] is False


def test_near_zero_width_uses_absolute_not_relative_gap() -> None:
    rows = [
        _summary(32, 11, [0.6, 0.5], [0.01, 0.0], 0.4),
        _summary(32, 12, [0.6, 0.5], [0.01, 0.0], 0.4),
        _summary(64, 11, [0.6, 0.5], [0.019, 0.009], 0.4),
        _summary(64, 12, [0.6, 0.5], [0.019, 0.009], 0.4),
    ]
    assert convergence_screen(rows)["checks"]["zone_width_gap"] is True


def test_t4_closed_preflight_disables_surface_and_perforation_flow(tmp_path) -> None:
    from so_recon.inference.contracts import ThetaRecord
    from so_recon.paths import ProjectPaths
    from so_recon.registry.run import RunContext
    from so_recon.simulator.contracts import CaseBundle
    from so_recon.synthetic.inverse_worlds import make_inverse_world

    root = Path(tmp_path) / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-t4-preflight", argv=[], cfg=None, paths=paths)
    _context, _observations, truth = make_inverse_world("e02-t4-v1", 144, paths, ctx)
    payload = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))
    ThetaRecord.model_validate(payload["theta"])
    case = CaseBundle.model_validate(payload["case"])

    preflight = closed_preflight_case(case)

    assert len(preflight.report_edges_s) == 2
    assert preflight.report_edges_s == case.report_edges_s[:2]
    assert len(preflight.controls) == len(case.wells)
    assert all(
        control.role == "shut" and control.target == "disabled" for control in preflight.controls
    )
    assert all(not any(control.connection_open) for control in preflight.controls)
    assert preflight.model_hash != case.model_hash


def test_t4_preflight_reproducibility_is_strict_and_does_not_forbid_transient() -> None:
    first_pressure = np.array([[15.0, 15.1], [14.8, 15.3]])
    first_so = np.array([[0.8, 0.7], [0.79, 0.71]])
    metrics = preflight_reproducibility_metrics(
        first_pressure,
        first_so,
        first_pressure.copy(),
        first_so.copy(),
    )
    assert metrics["status"] == "PASS"
    assert metrics["pressure_transient_relative"] > 0.0
    assert metrics["so_transient_abs"] > 0.0

    failed = preflight_reproducibility_metrics(
        first_pressure,
        first_so,
        first_pressure * (1.0 + 2.0e-6),
        first_so + 2.0e-6,
    )
    assert failed["status"] == "FAIL"
