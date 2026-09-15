"""Bounded native E02 reduced inversion, guarded by fresh accepted E01 evidence.

This test is intentionally fail-closed.  It is not part of the pure unit suite: an operator
must provide ``E02_E01_REPORT`` from the final physics tree and invoke it explicitly.  Once
started it records the truth/design outcome before asserting informativeness, and the
reference runner records an unresolved grid before the assertion can fail.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from so_recon.config.resources import P0_VERIFY_PROFILE
from so_recon.environment.resources import ResourceSnapshot, probe_resources
from so_recon.inference.contracts import FIXED_NOISE_THETA, HistoryRow, ObservationBundle
from so_recon.inference.target import require_complete_forward
from so_recon.observation.bins import rounding_grid
from so_recon.observation.noise import draw_history
from so_recon.observation.predict import predict_observations
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import write_json_artifact
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext
from so_recon.simulator.budget import BudgetLedger
from so_recon.simulator.contracts import OutputRequest
from so_recon.simulator.forward import SolverConfig, simulate
from so_recon.simulator.julia_bridge import JuliaNotFoundError, find_julia
from so_recon.simulator.worker import PersistentJuliaWorker
from so_recon.synthetic.reduced_inverse import ReducedDesign, build_reduced_case
from so_recon.validation.e01_dependency import require_e01_ow
from so_recon.validation.reference_inverse import run_reduced_reference

ROOT = Path(__file__).resolve().parents[2]


def _bounded_probe(session: Path) -> Callable[[], ResourceSnapshot]:
    def probe() -> ResourceSnapshot:
        return probe_resources(None, session).model_copy(
            update={
                "total_bytes": 64 * 1024**3,
                "available_bytes": 32 * 1024**3,
                "swap_used_bytes": 0,
            }
        )

    return probe


def _e01_report(paths: ProjectPaths) -> Path:
    raw = os.environ.get("E02_E01_REPORT")
    if raw is None:
        pytest.fail(
            "E02_E01_REPORT is required: regenerate E01 evidence on the final E02 physics "
            "tree before claiming a native reduced result"
        )
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else paths.root / candidate


def _julia() -> Path:
    try:
        return find_julia()
    except JuliaNotFoundError as exc:
        pytest.fail(f"native reduced gate requires Julia: {exc}")


def _blank_observations(design: ReducedDesign) -> ObservationBundle:
    grid = rounding_grid(0.01)
    rows = tuple(
        HistoryRow(
            well_id="P1",
            month_index=month,
            raw_value=0.0 if month not in {3, 9} else None,
            bin_index=0 if month not in {3, 9} else None,
            quality_group="watercut-0.01",
            observed_valid=month not in {3, 9},
            reset=month == 7,
        )
        for month in range(12)
    )
    return ObservationBundle(
        history=rows,
        logs=(),
        bin_edges_by_group={"watercut-0.01": tuple(float(v) for v in grid.edges)},
        cutoff_s=design.report_edges_s[-1],
        information_hash=sha256_json({"design": design.model_dump(mode="json"), "G": []}),
        observation_hash="0" * 64,
    )


@pytest.mark.julia
def test_reduced_physical_inverse_matches_an_independent_reference() -> None:
    paths = ProjectPaths.default(ROOT)
    paths.ensure_dirs()
    dependency = require_e01_ow(_e01_report(paths), paths)
    design = ReducedDesign()
    session = paths.artifacts / "e02-reduced-native"
    probe = _bounded_probe(session)
    ledger = BudgetLedger.start(
        profile=P0_VERIFY_PROFILE,
        path=session / "ledger.json",
        session_id="e02-reduced-native",
        probe=probe,
    )

    def run_factory(command: str, parent_run_ids: tuple[str, ...]) -> RunContext:
        return RunContext.start(
            command=command,
            argv=[],
            cfg=None,
            paths=paths,
            parent_run_ids=parent_run_ids,
            raw_input_hashes={"e01_dependency": sha256_json(dependency)},
        )

    truth_z = float(np.random.default_rng(701).standard_normal())
    blank = _blank_observations(design)
    with PersistentJuliaWorker(
        _julia(), ROOT / "julia", session, P0_VERIFY_PROFILE, paths=paths, probe=probe
    ) as worker:
        truth_ctx = run_factory("e02-reduced-truth", ())
        truth_case = build_reduced_case(truth_z, design, paths, truth_ctx)
        truth = require_complete_forward(
            simulate(
                truth_case,
                OutputRequest(
                    state_times_s=design.report_edges_s,
                    keep_native_restart=False,
                    chunk_months=1,
                ),
                worker=worker,
                ctx=truth_ctx,
                ledger=ledger,
                solver_config=SolverConfig(max_timestep_days=5.0, max_nonlinear_iterations=15),
            )
        )
        prediction = predict_observations(truth, blank, paths)
        observed_rows = draw_history(
            blank.history,
            prediction.fw,
            FIXED_NOISE_THETA,
            {"watercut-0.01": rounding_grid(0.01)},
            np.random.default_rng(8000),
        )
        observations = blank.model_copy(
            update={
                "history": observed_rows,
                "observation_hash": sha256_json(
                    {
                        "rows": [row.model_dump(mode="json") for row in observed_rows],
                        "truth_seed": 701,
                        "noise_seed": 8000,
                    }
                ),
            }
        )
        values = np.asarray(
            [value for value in prediction.fw.values() if value is not None], dtype=np.float64
        )
        informative = bool(values.size and np.ptp(values) >= 0.01 and values.max() >= 0.01)
        design_payload = {
            "schema_version": "e02-reduced-design-check-1",
            "status": "INFORMATIVE" if informative else "DESIGN_NOT_INFORMATIVE",
            "truth_z": truth_z,
            "model_hash": truth.model_hash,
            "watercut": values.tolist(),
            "watercut_range": float(np.ptp(values)) if values.size else None,
            "observations": observations.model_dump(mode="json"),
        }
        design_ref = write_json_artifact(
            truth_ctx.run_dir / "design_check.json",
            design_payload,
            paths,
            schema_version="e02-reduced-design-check-1",
            producer_run_id=truth_ctx.run_id,
            now=datetime.now(UTC),
        )
        truth_ctx.finish("PASS" if informative else "FAIL", outputs={"design_check": design_ref})
        assert informative, json.dumps(design_payload, sort_keys=True)

        reference_ref = run_reduced_reference(
            design,
            observations,
            worker,
            ledger,
            run_factory,
            parent_run_ids=(truth_ctx.run_id,),
        )

    reference = json.loads(paths.resolve(reference_ref.path).read_text(encoding="utf-8"))
    assert reference["status"] == "CONVERGED", reference["refinement"]
    assert len(reference["nodes"]) == 33
    assert ledger.session_totals().forwards == 34
