"""Fail-closed registration gate for the four pre-declared E02 physical parents."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.synthetic.inverse_worlds import make_inverse_world
from so_recon.validation.e01_dependency import require_e01_ow

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.julia
def test_every_declared_observability_parent_has_a_registered_generator() -> None:
    paths = ProjectPaths.default(ROOT)
    raw_evidence = os.environ.get("E02_E01_REPORT")
    if raw_evidence is None:
        pytest.fail("E02_E01_REPORT is required for the native observability gate")
    evidence = Path(raw_evidence)
    require_e01_ow(evidence if evidence.is_absolute() else ROOT / evidence, paths)
    manifest = json.loads((ROOT / "configs/e02_experiments.json").read_text(encoding="utf-8"))
    for experiment in manifest["experiments"]:
        ctx = RunContext.start(
            command="e02-observability-generator",
            argv=[experiment["experiment_id"]],
            cfg=None,
            paths=paths,
        )
        try:
            _context, _observations, truth = make_inverse_world(
                experiment["design_id"], experiment["truth_seed"], paths, ctx
            )
        except BaseException as exc:
            ctx.finish("FAIL", notes=[f"{type(exc).__name__}: {exc}"])
            raise
        ctx.finish("PASS", outputs={"truth": truth})
