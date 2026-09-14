"""Diagnostic solver provenance must describe the actual native options."""

from so_recon.validation.fixtures import diagnostic_solver_evidence


def test_diagnostic_keeps_native_options_instead_of_inventing_worker_defaults() -> None:
    options = {"max_timestep": 432000.0, "tol_cnv": 1e-10, "tol_cnv_well": 1e-12}
    evidence = diagnostic_solver_evidence({"solver_options": options})
    assert evidence["solver_options"] == options
    assert evidence["options_available"] is True
    assert "max_timestep_days" not in evidence
    assert "max_nonlinear_iterations" not in evidence


def test_unrecorded_diagnostic_options_remain_unavailable() -> None:
    evidence = diagnostic_solver_evidence({})
    assert evidence["options_available"] is False
    assert evidence["solver_options"] is None
