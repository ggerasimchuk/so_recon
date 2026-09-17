"""E03 Task 05 — the conditional NSF head and the frozen operational density (plan §9).

The subject is a NORMALISED operational law, so the tests mirror the §9 rows of the plan:

* the flow adapter is checked by 1D reference integration (tails accounted), by
  round-trip/logdet cancellation at central and tail points, and by autograd Jacobians —
  the logdet SIGN is thereby pinned against an independent derivation, not against the
  implementation's own bookkeeping;
* the categorical and residual factors are checked against `scipy.stats` and by exact
  identities, never against the helpers that build them;
* RNG isolation: one NumPy state draws one operational sample, and no global torch state
  is read or written on the way;
* the frozen checkpoint reloads to the same draws, the same log_prob and the same
  fingerprint, or it is refused;
* binding refuses an incompatible context spec, checkpoint, scaler or layout, and a theta
  from another basis is never scored.

No physics runs here: the context payload is the synthetic fixture of the Task 04 suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from so_recon.config.learning import EncoderParams, FlowParams
from so_recon.geology.density import Density
from so_recon.inference.contracts import DensitySchema, PriorContext, ThetaRecord
from so_recon.ml.checkpoint import (
    CheckpointHashes,
    FrozenCheckpoint,
    checkpoint_from_bytes,
    load_checkpoint,
    save_checkpoint,
    serialize_checkpoint,
)
from so_recon.ml.context import build_context_batch, permute_batch
from so_recon.ml.contracts import LayoutSupport, ProposalManifest
from so_recon.ml.flow import ConditionalNSF, condition_vector, standard_normal_log_prob
from so_recon.ml.normalization import FeatureScaler
from so_recon.ml.proposal import (
    LearnedProposal,
    bind_frozen_proposal,
    flow_config_payload,
    proposal_from_checkpoint,
)
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.hashing import sha256_json
from so_recon.simulator.schedule import month_edges_s
from so_recon.synthetic.inverse_corpus import WELL_TIME_FEATURES, default_context_spec

torch.manual_seed(0)

#: Registered central/tail points of the §9 round-trip row: inside the spline, at the
#: tail boundary, and beyond it in the linear tail.
ROUND_TRIP_POINTS = np.array(
    [
        [0.0, 0.0],
        [1.5, -2.25],
        [3.99, -3.99],
        [4.0, -4.0],
        [4.25, -7.5],
        [-4.0, 4.0],
        [8.5, -8.5],
    ],
    dtype=np.float64,
)


def _control(well: str, target: str, value: float) -> dict[str, Any]:
    return {
        "well_id": well,
        "start_s": 0.0,
        "end_s": 95_000_000.0,
        "role": "producer" if well.startswith("P") else "injector",
        "target": target,
        "value": value,
        "bhp_limit_pa": None,
        "connection_open": (True, True),
    }


def _payload() -> dict[str, Any]:
    wells = [("I1", 2, 2, 0), ("I2", 2, 13, 0), ("P1", 13, 2, 1), ("P2", 13, 13, 1)]
    design = {
        "design_id": "e02-t2-v2",
        "shape": [16, 16, 2],
        "n_months": 36,
        "start_date": "2000-01-01",
        "well_columns": [
            {"well_id": name, "column": [i, j], "role": "producer" if role else "injector"}
            for name, i, j, role in wells
        ],
    }
    months = 36
    edges = month_edges_s(date(2000, 1, 1), months)
    controls = []
    for name, _i, _j, role in wells:
        for month in range(months):
            controls.append(
                {
                    **_control(
                        name,
                        "water_rate" if role == 0 else "liquid_rate",
                        20.0,
                    ),
                    "start_s": edges[month],
                    "end_s": edges[month + 1],
                }
            )
    history = []
    for name, _i, _j, role in wells:
        if role == 0:
            continue
        for month in range(months):
            history.append(
                {
                    "well_id": name,
                    "month_index": month,
                    "raw_value": 0.31,
                    "bin_index": 31,
                    "quality_group": "watercut-0.01",
                    "observed_valid": True,
                    "reset": False,
                    "sigma_multiplier": 1.0,
                }
            )
    grid_edges = [round(value, 4) for value in np.linspace(0.0, 1.0, 101)]
    observations = {
        "history": history,
        "logs": [],
        "bin_edges_by_group": {"watercut-0.01": grid_edges},
        "cutoff_s": edges[-1],
        "information_hash": "a" * 64,
        "observation_hash": "b" * 64,
    }
    return {
        "schema_version": "e03-parent-context-1",
        "parent_id": "fixture-0000",
        "design_id": "e02-t2-v2",
        "split": "train",
        "cutoff_s": edges[-1],
        "well_ids": ["P1", "P2"],
        "inference_input": {
            "context": {
                "design": {"inverse_design": design},
                "density_schema": {
                    "schema_id": "e02-t2-symmetric-12",
                    "n_v": 11,
                    "n_residual": 4,
                    "families": [0],
                    "basis_hash": "c" * 64,
                    "transform_version": "e02-t2-v2-conditional-1",
                    "measure": "counting_x_latent_lebesgue",
                },
            },
            "G": [
                {
                    "observation_id": f"{name}-L0-log_permeability_m2",
                    "support_cell_ids": [index],
                    "value": -13.0 + 0.1 * index,
                    "sigma": 0.2,
                }
                for index, (name, _i, _j, _r) in enumerate(wells)
            ],
            "U": controls,
            "observations": observations,
        },
    }


def _prior_context(
    *,
    schema_id: str = "p1-conditional-12",
    transform_version: str = "p1-two-layer-v2-conditional-1",
    n_v: int = 11,
    n_residual: int = 4,
    families: tuple[int, ...] = (0, 1),
    n_geology: int = 12,
    n_state_residual: int = 0,
    seed: int = 3,
) -> PriorContext:
    """A minimal but well-formed conditioning context: whitening arrays need not come from
    the real G machinery to pin the proposal's contract."""
    rng = np.random.default_rng(seed)
    mean = 0.3 * rng.standard_normal(n_geology)
    rotation, _ = np.linalg.qr(rng.standard_normal((n_geology, n_geology)))
    # a sign flip per column keeps the basis orthogonal and the root's diagonal positive
    rotation *= np.sign(np.diag(rotation))
    root = rotation @ np.diag(np.linspace(0.6, 1.4, n_geology))
    schema = DensitySchema(
        schema_id=schema_id,
        n_v=n_v,
        n_residual=n_residual,
        families=families,
        basis_hash=sha256_json({"schema_id": schema_id, "seed": seed}),
        transform_version=transform_version,
    )
    return PriorContext(
        density_schema=schema,
        n_geology=n_geology,
        n_state_residual=n_state_residual,
        mean=mean,
        chol=root,
        rotation=rotation,
        design={},
        g_hash=sha256_json({"g": seed}),
        information_hash=sha256_json({"information": seed}),
    )


def _theta(context: PriorContext, coordinates: np.ndarray, s: int) -> ThetaRecord:
    schema = context.density_schema
    return ThetaRecord(
        schema_id=schema.schema_id,
        s=s,
        v=tuple(coordinates[: schema.n_v]),
        z_perp=tuple(coordinates[schema.n_v :]),
        basis_hash=schema.basis_hash,
    )


def _artifact_ref(name: str, sha: str, size_bytes: int) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=sha,
        path=f"artifacts/task05/{name}",
        sha256=sha,
        size_bytes=size_bytes,
        media_type="application/octet-stream",
        schema_version="e03-task05-test",
        producer_run_id="task05-test-run",
        parent_artifact_ids=[],
        created_at="2026-09-17T00:00:00",
    )


@dataclass(frozen=True)
class _Bundle:
    checkpoint: FrozenCheckpoint
    manifest: Any
    spec: Any
    batch: Any
    context: PriorContext
    n_head_families: int

    def bind(self) -> Any:
        return bind_frozen_proposal(
            checkpoint=self.checkpoint,
            manifest=self.manifest,
            spec=self.spec,
            batch=self.batch,
            schema=self.context.density_schema,
        )


def _build_bundle(
    root: Path,
    *,
    spec: Any,
    context: PriorContext,
    encoder_variant: str = "summary",
    families: int = 2,
    hidden_features: int = 16,
) -> _Bundle:
    torch.manual_seed(11)
    batch = build_context_batch(_payload(), spec=spec)
    encoder_params = EncoderParams(variant=encoder_variant, width=16, heads=2, context_dim=12)
    from so_recon.ml.encoders import build_encoder

    encoder = build_encoder(
        encoder_variant,
        encoder_params,
        len(WELL_TIME_FEATURES),
        len(spec.edge_features),
        len(spec.static_features),
    )
    flow = ConditionalNSF(
        FlowParams(),
        n_v=context.density_schema.n_v,
        context_dim=encoder_params.context_dim,
        n_families=families,
        hidden_features=hidden_features,
    )
    model = LearnedProposal(encoder, flow, n_families=families)
    scaler = FeatureScaler.fit([batch], spec)
    config = flow_config_payload(
        flow=FlowParams(),
        encoder_variant=encoder_variant,
        encoder_params=encoder_params,
        n_well_time_features=len(WELL_TIME_FEATURES),
        n_edge_features=len(spec.edge_features),
        n_static_features=len(spec.static_features),
        n_v=context.density_schema.n_v,
        context_dim=encoder_params.context_dim,
        hidden_features=hidden_features,
        n_families=families,
    )
    weights, config_bytes, scaler_bytes = serialize_checkpoint(
        model, flow_config=config, scaler_payload=scaler.payload()
    )
    hashes = save_checkpoint(
        root, module=model, flow_config=config, scaler_payload=scaler.payload()
    )
    checkpoint = load_checkpoint(root)
    schema = context.density_schema
    layout = LayoutSupport(
        schema_id=schema.schema_id,
        n_v=schema.n_v,
        n_residual=schema.n_residual,
        families=schema.families,
        basis_hash=schema.basis_hash,
    )
    manifest = ProposalManifest(
        weights_ref=_artifact_ref("weights.safetensors", hashes.weights_sha256, len(weights)),
        weights_sha256=hashes.weights_sha256,
        flow_config_ref=_artifact_ref(
            "flow_config.json", hashes.flow_config_sha256, len(config_bytes)
        ),
        flow_config_sha256=hashes.flow_config_sha256,
        scaler_ref=_artifact_ref("scaler.json", hashes.scaler_sha256, len(scaler_bytes)),
        scaler_sha256=hashes.scaler_sha256,
        supported_layouts=(layout,),
        operational_dtype="float64",
        operational_backend="cpu",
        architecture_version=FlowParams().architecture_version,
        context_builder_version=spec.builder_version,
    )
    return _Bundle(
        checkpoint=checkpoint,
        manifest=manifest,
        spec=spec,
        batch=batch,
        context=context,
        n_head_families=families,
    )


@pytest.fixture(scope="module")
def spec() -> Any:
    return default_context_spec(cutoff_s=9.5e7, max_wells=8, max_months=36)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory, spec: Any) -> _Bundle:
    root = tmp_path_factory.mktemp("bundle")
    return _build_bundle(root, spec=spec, context=_prior_context())


@pytest.fixture(scope="module")
def flow_1d() -> Any:
    torch.manual_seed(7)
    module = ConditionalNSF(
        FlowParams(n_transforms=1), n_v=1, context_dim=4, n_families=2, hidden_features=16
    ).to(torch.float64)
    module.eval()
    return module


@pytest.fixture(scope="module")
def flow_2d() -> Any:
    torch.manual_seed(5)
    module = ConditionalNSF(
        FlowParams(n_transforms=3), n_v=2, context_dim=4, n_families=2, hidden_features=16
    ).to(torch.float64)
    module.eval()
    return module


def _cond_1d(flow_1d: Any) -> torch.Tensor:
    rng = np.random.default_rng(9)
    return torch.tensor(rng.standard_normal((1, flow_1d.condition_dim)), dtype=torch.float64)


# --------------------------------------------------------------------------------------
# the §5.2 direction: sampling forward, scoring by the inverse
# --------------------------------------------------------------------------------------


def test_forward_is_the_sampling_direction_and_inverse_scores(flow_2d) -> None:
    eps = torch.tensor([[0.3, -1.2]], dtype=torch.float64)
    cond = torch.zeros((1, 6), dtype=torch.float64)
    v, logdet_fwd = flow_2d.forward(eps, cond)
    restored, logdet_inv = flow_2d.inverse(v, cond)
    np.testing.assert_allclose(restored.detach().numpy(), eps.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose((logdet_fwd + logdet_inv).detach().numpy(), 0.0, atol=1e-10)


def test_condition_vector_concatenates_a_one_hot_family() -> None:
    context = torch.zeros((2, 3), dtype=torch.float64)
    cond = condition_vector(context, torch.tensor([0, 1]), n_families=3)
    assert cond.shape == (2, 6)
    np.testing.assert_array_equal(cond.numpy(), np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]))


def test_standard_normal_log_prob_matches_scipy() -> None:
    eps = torch.tensor(np.random.default_rng(4).standard_normal((5, 4)), dtype=torch.float64)
    expected = np.sum(stats.norm.logpdf(eps.numpy()), axis=1)
    np.testing.assert_allclose(standard_normal_log_prob(eps).detach().numpy(), expected, atol=1e-12)


# --------------------------------------------------------------------------------------
# §9 small transform normalization (1D, tails accounted)
# --------------------------------------------------------------------------------------


def test_small_transform_normalizes_with_tails_accounted(flow_1d) -> None:
    cond = _cond_1d(flow_1d)

    def density_v(v: float) -> float:
        with torch.no_grad():
            log_q = flow_1d.log_prob(torch.tensor([[v]], dtype=torch.float64), cond)
        return float(torch.exp(log_q[0]))

    def pushforward_mass(eps_value: float) -> float:
        # substituting v = T(eps):  q(T(eps)) |det D T(eps)| must integrate to one, so a
        # wrong logdet SIGN in either direction cannot survive this integral
        tensor = torch.tensor([[eps_value]], dtype=torch.float64)
        with torch.no_grad():
            value, logabsdet = flow_1d.forward(tensor, cond)
            log_q = flow_1d.log_prob(value, cond)
        return float(torch.exp(log_q + logabsdet))

    with torch.no_grad():
        left, _ = flow_1d.forward(torch.tensor([[-18.0]], dtype=torch.float64), cond)
        right, _ = flow_1d.forward(torch.tensor([[18.0]], dtype=torch.float64), cond)

    mass_v, _ = integrate.quad(
        density_v, float(left[0, 0]), float(right[0, 0]), points=[-4.0, 4.0], limit=400
    )
    mass_eps, _ = integrate.quad(pushforward_mass, -18.0, 18.0, points=[-4.0, 4.0], limit=400)
    assert mass_v == pytest.approx(1.0, abs=1e-6)
    assert mass_eps == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# §9 round-trip / logdet
# --------------------------------------------------------------------------------------


def test_round_trip_is_exact_at_central_and_tail_points(flow_2d) -> None:
    cond = torch.zeros((1, 6), dtype=torch.float64)
    points = torch.tensor(ROUND_TRIP_POINTS, dtype=torch.float64)
    with torch.no_grad():
        v, _ = flow_2d.forward(points, cond)
        restored, _ = flow_2d.inverse(v, cond)
        _, logdet_fwd = flow_2d.forward(points, cond)
        _, logdet_inv = flow_2d.inverse(v, cond)
    np.testing.assert_allclose(restored.numpy(), points.numpy(), atol=1e-8)
    np.testing.assert_allclose((logdet_fwd + logdet_inv).numpy(), 0.0, atol=1e-8)


def test_log_prob_is_the_change_of_variables_formula(flow_2d) -> None:
    """The logdet-sign row: `log q(v) = log N(eps) + log|det D T^{-1}(v)|` against scipy."""
    rng = np.random.default_rng(21)
    cond = torch.tensor(rng.standard_normal((1, 6)), dtype=torch.float64)
    points = torch.tensor(rng.uniform(-6.0, 6.0, size=(6, 2)), dtype=torch.float64)
    with torch.no_grad():
        reported = flow_2d.log_prob(points, cond).numpy()
        eps, logabsdet = flow_2d.inverse(points, cond)
    expected = np.sum(stats.norm.logpdf(eps.numpy()), axis=1) + logabsdet.numpy()
    np.testing.assert_allclose(reported, expected, atol=1e-9)


def test_logdet_matches_the_autograd_jacobian(flow_2d) -> None:
    rng = np.random.default_rng(23)
    cond = torch.tensor(rng.standard_normal((1, 6)), dtype=torch.float64)
    v = torch.tensor(rng.uniform(-3.5, 3.5, size=(2,)), dtype=torch.float64)

    def inverse_map(point: torch.Tensor) -> torch.Tensor:
        out, _ = flow_2d.inverse(point.unsqueeze(0), cond)
        return out.squeeze(0)

    jacobian = torch.autograd.functional.jacobian(inverse_map, v)
    _, logabsdet_jac = torch.linalg.slogdet(jacobian)
    with torch.no_grad():
        _, logabsdet_reported = flow_2d.inverse(v.unsqueeze(0), cond)
    assert float(logabsdet_jac) == pytest.approx(float(logabsdet_reported[0]), abs=1e-6)


# --------------------------------------------------------------------------------------
# §9 categorical / residual law
# --------------------------------------------------------------------------------------


def test_frozen_law_implements_the_density_seam(bundle: _Bundle) -> None:
    assert isinstance(bundle.bind(), Density)


def test_allowed_categorical_mass_is_normalized(bundle: _Bundle) -> None:
    model = bundle.bind()
    context = bundle.context
    coordinates = np.random.default_rng(31).standard_normal(
        context.density_schema.n_v + context.density_schema.n_residual
    )
    masses = [
        np.exp(model.log_prob_parts(_theta(context, coordinates, s))[0])
        for s in context.density_schema.families
    ]
    assert float(np.sum(masses)) == pytest.approx(1.0, abs=1e-12)


def test_residual_block_is_exact_standard_normal_on_all_coordinates(bundle: _Bundle) -> None:
    model = bundle.bind()
    context = bundle.context
    rng = np.random.default_rng(37)
    coordinates = rng.standard_normal(
        context.density_schema.n_v + context.density_schema.n_residual
    )
    theta = _theta(context, coordinates, s=context.density_schema.families[0])
    log_s, log_v, log_z = model.log_prob_parts(theta)
    expected_z = float(np.sum(stats.norm.logpdf(np.asarray(theta.z_perp))))
    assert log_z == pytest.approx(expected_z, abs=1e-12)
    # doubling ONLY the residual block must move log_prob by exactly -3/2 ||z||^2
    doubled = ThetaRecord(
        schema_id=theta.schema_id,
        s=theta.s,
        v=theta.v,
        z_perp=tuple(2.0 * value for value in theta.z_perp),
        basis_hash=theta.basis_hash,
    )
    assert model.log_prob(doubled) - model.log_prob(theta) == pytest.approx(
        -1.5 * float(np.asarray(theta.z_perp) @ np.asarray(theta.z_perp)), abs=1e-9
    )
    # the flow factor is untouched by the residual block: same v, same s
    _, log_v2, _ = model.log_prob_parts(doubled)
    assert log_v2 == pytest.approx(log_v, abs=1e-12)


# --------------------------------------------------------------------------------------
# §9 RNG isolation
# --------------------------------------------------------------------------------------


def test_operational_sampling_consumes_only_the_passed_generator(bundle: _Bundle) -> None:
    model = bundle.bind()
    torch.manual_seed(1)
    before = torch.random.get_rng_state()
    first = model.sample(8, np.random.default_rng(101))
    assert torch.equal(torch.random.get_rng_state(), before)

    torch.manual_seed(987_654_321)
    mid = torch.random.get_rng_state()
    second = model.sample(8, np.random.default_rng(101))
    assert torch.equal(torch.random.get_rng_state(), mid)
    assert first == second

    third = model.sample(8, np.random.default_rng(102))
    assert third != first


def test_sample_refuses_a_nonpositive_count(bundle: _Bundle) -> None:
    with pytest.raises(ValueError):
        bundle.bind().sample(0, np.random.default_rng(5))


# --------------------------------------------------------------------------------------
# §9 frozen reload
# --------------------------------------------------------------------------------------


def test_frozen_reload_reproduces_draws_log_prob_and_fingerprint(tmp_path: Path, spec: Any) -> None:
    context = _prior_context()
    root = tmp_path / "ckpt"
    root.mkdir()
    bundle = _build_bundle(root, spec=spec, context=context)
    first = bundle.bind()

    reloaded = load_checkpoint(root)
    second = bind_frozen_proposal(
        checkpoint=reloaded,
        manifest=bundle.manifest,
        spec=spec,
        batch=bundle.batch,
        schema=context.density_schema,
    )
    draws_a = first.sample(16, np.random.default_rng(55))
    draws_b = second.sample(16, np.random.default_rng(55))
    assert draws_a == draws_b
    theta = draws_a[0]
    assert first.log_prob(theta) == second.log_prob(theta)
    assert first.fingerprint == second.fingerprint

    weights, config_bytes, scaler_bytes = serialize_checkpoint(
        proposal_from_checkpoint(reloaded),
        flow_config=reloaded.flow_config,
        scaler_payload=reloaded.scaler_payload,
    )
    from_bytes = checkpoint_from_bytes(weights, config_bytes, scaler_bytes)
    third = bind_frozen_proposal(
        checkpoint=from_bytes,
        manifest=bundle.manifest,
        spec=spec,
        batch=bundle.batch,
        schema=context.density_schema,
    )
    assert third.sample(16, np.random.default_rng(55)) == draws_a
    assert third.log_prob(theta) == first.log_prob(theta)


def test_checkpoint_reload_refuses_a_hash_mismatch(tmp_path: Path, spec: Any) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    bundle = _build_bundle(root, spec=spec, context=_prior_context())
    good = bundle.checkpoint.hashes
    tampered = CheckpointHashes(
        weights_sha256="0" * 64,
        flow_config_sha256=good.flow_config_sha256,
        scaler_sha256=good.scaler_sha256,
    )
    with pytest.raises(ValueError, match="weights"):
        load_checkpoint(root, expected=tampered)


# --------------------------------------------------------------------------------------
# §9 context permutation (density level)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_variant", ["summary", "graph"])
def test_context_permutation_leaves_log_prob_invariant(
    tmp_path: Path, spec: Any, encoder_variant: str
) -> None:
    bundle = _build_bundle(
        tmp_path / f"ckpt-{encoder_variant}",
        spec=spec,
        context=_prior_context(),
        encoder_variant=encoder_variant,
    )
    first = bundle.bind()
    schema = bundle.context.density_schema
    order = np.asarray([2, 0, 3, 1])
    permuted_batch = permute_batch(bundle.batch, order)
    second = bind_frozen_proposal(
        checkpoint=bundle.checkpoint,
        manifest=bundle.manifest,
        spec=bundle.spec,
        batch=permuted_batch,
        schema=bundle.context.density_schema,
    )
    rng = np.random.default_rng(77)
    deltas = []
    for _ in range(6):
        coordinates = rng.standard_normal(schema.n_v + schema.n_residual)
        theta = _theta(bundle.context, coordinates, s=int(rng.integers(0, len(schema.families))))
        deltas.append(abs(second.log_prob(theta) - first.log_prob(theta)))
    assert max(deltas) <= 1e-8


# --------------------------------------------------------------------------------------
# §9 residual law on the T4 layout: all six residual coordinates
# --------------------------------------------------------------------------------------


def test_t4_layout_declares_all_six_residual_coordinates(tmp_path: Path, spec: Any) -> None:
    from so_recon.synthetic.inverse_designs import inverse_design

    design = inverse_design("e02-t4-v1")
    assert design.n_geology == 13 and design.n_state_residual == 1
    geology_modes = design.n_geology - 8
    assert (geology_modes, design.n_state_residual) == (5, 1)

    context = _prior_context(
        schema_id="e02-t4-17d",
        transform_version="e02-t4-v1-conditional-1",
        n_residual=6,
        families=(0,),
        n_geology=13,
        n_state_residual=1,
        seed=13,
    )
    bundle = _build_bundle(tmp_path / "t4", spec=spec, context=context)
    model = bundle.bind()
    z = np.linspace(-1.2, 1.2, 6)
    theta = _theta(context, np.zeros(17), s=0)
    theta = ThetaRecord(
        schema_id=theta.schema_id,
        s=0,
        v=theta.v,
        z_perp=tuple(z),
        basis_hash=theta.basis_hash,
    )
    _log_s, _log_v, log_z = model.log_prob_parts(theta)
    assert log_z == pytest.approx(float(np.sum(stats.norm.logpdf(z))), abs=1e-12)
    # the law moves for every one of the six coordinates — none is silently dropped
    base = model.log_prob(theta)
    for index in range(6):
        shifted = list(z)
        shifted[index] += 0.5
        moved = ThetaRecord(
            schema_id=theta.schema_id,
            s=0,
            v=theta.v,
            z_perp=tuple(shifted),
            basis_hash=theta.basis_hash,
        )
        expected = stats.norm.logpdf(shifted[index]) - stats.norm.logpdf(z[index])
        assert model.log_prob(moved) - base == pytest.approx(float(expected), abs=1e-12)


# --------------------------------------------------------------------------------------
# binding refusals: context / checkpoint / scaler / basis-layout
# --------------------------------------------------------------------------------------


def test_binding_refuses_a_batch_from_another_context_spec(tmp_path: Path) -> None:
    spec_a = default_context_spec(cutoff_s=9.5e7, max_wells=8, max_months=36)
    context = _prior_context()
    bundle = _build_bundle(tmp_path / "ckpt", spec=spec_a, context=context)
    spec_b = default_context_spec(cutoff_s=1.0, max_wells=4, max_months=12)
    with pytest.raises(ValueError, match="context spec"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bundle.manifest,
            spec=spec_b,
            batch=bundle.batch,
            schema=context.density_schema,
        )


def test_binding_refuses_a_checkpoint_hash_mismatch(tmp_path: Path, spec: Any) -> None:
    context = _prior_context()
    bundle = _build_bundle(tmp_path / "ckpt", spec=spec, context=context)
    bad_weights = bundle.manifest.model_copy(update={"weights_sha256": "0" * 64})
    with pytest.raises(ValueError, match="weights"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bad_weights,
            spec=spec,
            batch=bundle.batch,
            schema=context.density_schema,
        )
    bad_scaler = bundle.manifest.model_copy(update={"scaler_sha256": "0" * 64})
    with pytest.raises(ValueError, match="scaler"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bad_scaler,
            spec=spec,
            batch=bundle.batch,
            schema=context.density_schema,
        )
    bad_config = bundle.manifest.model_copy(update={"flow_config_sha256": "0" * 64})
    with pytest.raises(ValueError, match="flow config"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bad_config,
            spec=spec,
            batch=bundle.batch,
            schema=context.density_schema,
        )


def test_binding_refuses_a_foreign_layout(tmp_path: Path, spec: Any) -> None:
    bundle = _build_bundle(tmp_path / "ckpt", spec=spec, context=_prior_context())
    foreign = _prior_context(
        schema_id="e03-t9-unknown-12",
        transform_version="e03-t9-unknown-1",
        seed=99,
    )
    with pytest.raises(ValueError, match="layout"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bundle.manifest,
            spec=spec,
            batch=bundle.batch,
            schema=foreign.density_schema,
        )
    wrong_dims = _prior_context(n_v=9, seed=98)
    with pytest.raises(ValueError, match="layout"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bundle.manifest,
            spec=spec,
            batch=bundle.batch,
            schema=wrong_dims.density_schema,
        )


def test_binding_refuses_a_same_shape_schema_of_another_basis(
    tmp_path: Path, spec: Any
) -> None:
    bundle = _build_bundle(tmp_path / "ckpt", spec=spec, context=_prior_context())
    other_basis = _prior_context(seed=42)
    trained = bundle.context.density_schema
    foreign = other_basis.density_schema
    assert foreign.schema_id == trained.schema_id
    assert (foreign.n_v, foreign.n_residual, foreign.families) == (
        trained.n_v,
        trained.n_residual,
        trained.families,
    )
    assert foreign.basis_hash != trained.basis_hash
    with pytest.raises(ValueError, match="basis"):
        bind_frozen_proposal(
            checkpoint=bundle.checkpoint,
            manifest=bundle.manifest,
            spec=spec,
            batch=bundle.batch,
            schema=foreign,
        )


def test_a_theta_from_another_basis_is_never_scored(bundle: _Bundle) -> None:
    model = bundle.bind()
    context = bundle.context
    foreign_basis = _prior_context(seed=42)
    theta = _theta(
        foreign_basis,
        np.zeros(context.density_schema.n_v + context.density_schema.n_residual),
        s=0,
    )
    with pytest.raises(ValueError, match="basis"):
        model.log_prob(theta)


def test_sampled_thetas_score_finite_and_families_follow_the_head(bundle: _Bundle) -> None:
    model = bundle.bind()
    draws = model.sample(2_000, np.random.default_rng(71))
    context = bundle.context
    schema = context.density_schema
    families = np.array([theta.s for theta in draws])
    assert set(families.tolist()) <= set(schema.families)
    _log_s, log_v, _log_z = model.log_prob_parts(draws[0])
    assert np.isfinite(model.log_prob(draws[0]))
    assert np.isfinite(log_v)
    for theta in draws:
        schema.validate_theta(theta)
