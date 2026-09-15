"""E02 inverse inputs are allowlisted and experiment identities are frozen before scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.synthetic.inverse_worlds import inference_payload, make_inverse_world

ROOT = Path(__file__).resolve().parents[2]


def test_truth_cannot_enter_an_inverse_payload() -> None:
    with pytest.raises(ValueError, match="truth"):
        inference_payload({"context_ref": "context.json", "truth_ref": "truth/states.h5"})


@pytest.mark.parametrize(
    "payload",
    [
        {"context": {"theta": [1.0]}},
        {"context": {"initial_pressure_pa": [15.0e6]}},
        {"observations": {"path": "artifacts/world/truth/observations.parquet"}},
        {"seed": 141},
        {"full_permeability": [1.0]},
    ],
)
def test_hidden_generator_state_is_rejected_recursively(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        inference_payload(payload)


def test_allowlisted_input_is_stable_and_drops_nothing_silently() -> None:
    payload = {
        "context": {"design_id": "e02-t1-v1", "cutoff": "2003-01-01"},
        "G": {"values": [1.0, 2.0], "use": "condition_prior"},
        "U": {"controls_ref": "artifacts/public/controls.json"},
        "observations": {"history_ref": "artifacts/public/history.parquet"},
        "density_schema": {"schema_id": "p1-conditional-12"},
        "basis": {"basis_hash": "a" * 64},
    }
    assert inference_payload(payload) == payload
    with pytest.raises(ValueError, match="unexpected"):
        inference_payload({**payload, "extra": 1})


def test_t1_generator_conditions_on_numeric_g_and_keeps_truth_out_of_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)
    context, observations, truth = make_inverse_world("e02-t1-v1", 141, paths, ctx)
    assert context.mean.shape == (12,)
    assert np.linalg.norm(context.mean) > 0.0
    assert observations.information_hash == context.information_hash
    assert "/truth/" not in json.dumps(observations.model_dump(mode="json"))
    manifest = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))
    assert manifest["seed"] == 141
    assert manifest["design_id"] == "p1-two-layer-v2"


def test_experiment_manifest_freezes_four_parent_seeds_and_named_streams() -> None:
    payload = json.loads((ROOT / "configs/e02_experiments.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "e02-experiments-1"
    assert [(item["experiment_id"], item["truth_seed"]) for item in payload["experiments"]] == [
        ("e02-t1-v1-s141", 141),
        ("e02-t1-v1-s142", 142),
        ("e02-t2-v1-s143", 143),
        ("e02-t4-v1-s144", 144),
    ]
    assert payload["stream_names"] == ["truth_latent", "static_G", "history_noise", "schedule"]
