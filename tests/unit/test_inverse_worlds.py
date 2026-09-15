"""E02 inverse inputs are allowlisted and experiment identities are frozen before scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so_recon.geology.renderer import render_theta, swap_t2_layers, with_t4_remote_state
from so_recon.inference.contracts import ModelObservations, ThetaRecord
from so_recon.paths import ProjectPaths
from so_recon.registry.run import RunContext
from so_recon.synthetic.inverse_worlds import (
    generate_dynamic_history,
    inference_payload,
    make_inverse_world,
)

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


@pytest.mark.parametrize(
    ("design_id", "seed", "schema_id", "n_geology", "n_state_residual"),
    [
        ("e02-t2-v1", 143, "e02-t2-symmetric-12", 12, 0),
        ("e02-t4-v1", 144, "e02-t4-17d", 13, 1),
    ],
)
def test_ambiguous_generators_publish_real_cases_with_declared_latent_dimensions(
    tmp_path: Path,
    design_id: str,
    seed: int,
    schema_id: str,
    n_geology: int,
    n_state_residual: int,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)

    context, observations, truth = make_inverse_world(design_id, seed, paths, ctx)

    assert context.density_schema.schema_id == schema_id
    assert context.n_geology == n_geology
    assert context.n_state_residual == n_state_residual
    assert observations.information_hash == context.information_hash
    manifest = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))
    assert manifest["design_id"] == design_id
    assert manifest["seed"] == seed
    assert manifest["case"]["model_hash"] == manifest["model_hash"]
    assert manifest["renderer_checks"]["reproduces_truth_arrays"] is True


def test_t4_truth_contains_declared_remote_barrier_and_nonvirgin_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)
    context, _observations, truth = make_inverse_world("e02-t4-v1", 144, paths, ctx)
    manifest = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))

    assert context.density_schema.n_v == 11
    assert context.density_schema.n_residual == 6
    assert manifest["physical_design"]["remote_zone"] == {"x_index_min": 12, "y_index_min": 8}
    assert manifest["physical_design"]["initial_state_meaning"] == (
        "synthetic_nonvirgin_initial_state"
    )
    assert manifest["renderer_checks"]["remote_k_multiplier"] == 0.01
    assert manifest["renderer_checks"]["initial_sw_range"] > 0.0


def test_t2_pair_swaps_physical_layers_not_whitened_coordinate_labels(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)
    context, _observations, truth = make_inverse_world("e02-t2-v1", 143, paths, ctx)
    manifest = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))
    theta = ThetaRecord.model_validate(manifest["theta"])

    original = render_theta(theta, context)
    swapped_theta = swap_t2_layers(theta, context)
    swapped = render_theta(swapped_theta, context)
    cells = 16 * 16
    np.testing.assert_allclose(
        swapped.arrays["log_permeability_m2"][:cells],
        original.arrays["log_permeability_m2"][cells:],
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        swapped.arrays["log_permeability_m2"][cells:],
        original.arrays["log_permeability_m2"][:cells],
        atol=1.0e-12,
        rtol=0.0,
    )
    assert swapped_theta.v[8:] == theta.v[8:]


def test_t4_pair_changes_only_remote_initial_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)
    context, _observations, truth = make_inverse_world("e02-t4-v1", 144, paths, ctx)
    manifest = json.loads(paths.resolve(truth.path).read_text(encoding="utf-8"))
    theta = ThetaRecord.model_validate(manifest["theta"])

    low = render_theta(with_t4_remote_state(theta, context, -1.0), context)
    high = render_theta(with_t4_remote_state(theta, context, 1.0), context)
    np.testing.assert_array_equal(low.arrays["permeability_m2"], high.arrays["permeability_m2"])
    np.testing.assert_array_equal(low.arrays["porosity"], high.arrays["porosity"])
    sw_gap = high.arrays["sw"] - low.arrays["sw"]
    assert np.all(sw_gap[low.arrays["cell_centers_m"][:, 0] < 62.5] == 0.0)
    assert float(sw_gap.max()) > 0.15


def test_dynamic_history_is_drawn_after_forward_and_preserves_dry_masks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    paths = ProjectPaths.default(root)
    paths.ensure_dirs()
    ctx = RunContext.start(command="unit-inverse-world", argv=[], cfg=None, paths=paths)
    context, pending, _truth = make_inverse_world("e02-t1-v1", 141, paths, ctx)
    prediction = ModelObservations(
        fw={
            ("P1", 0): 0.0,
            ("P1", 1): 0.25,
            ("P1", 2): None,
            ("P2", 0): 1.0,
        },
        so_support={},
        model_hash="a" * 64,
    )

    first = generate_dynamic_history(context, prediction, seed=9101)
    second = generate_dynamic_history(context, prediction, seed=9101)

    assert first == second
    assert first.information_hash == pending.information_hash
    assert [(row.well_id, row.month_index) for row in first.history] == [
        ("P1", 0),
        ("P1", 1),
        ("P1", 2),
        ("P2", 0),
    ]
    assert first.history[2].observed_valid is False
    assert first.history[2].raw_value is None
    assert first.history[2].bin_index is None
    assert all(
        row.bin_index is not None and row.raw_value is not None
        for row in first.history
        if row.observed_valid
    )


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
