"""E03 Task 05 — the conditional proposal and its frozen operational law (plan §5.2, §7.5).

`LearnedProposal` is the trainable composite: an encoder trunk produces the context, a
linear head carries `q_phi(s | C)`, and `ConditionalNSF` carries `q_phi(v | s, C)`. The
proposal this module EXPORTS for inference is the residual-inclusive law of §5.2,

    q_phi(s, v, z | C) = q_phi(s | C) · q_phi(v | s, C) · p0(z | v, s, G),

where the last factor is the FULL N(0, I) of dimension `n_residual`: the residual block
is never optimised by the network, but it IS part of the operational log density, exactly
as in the conditional prior the flow replaces.

`FrozenConditionalNSF` binds that law once, to one world, on one CPU Float64 copy of the
model: checkpoint, scaler, context-builder, allowed context, prior/basis/layout and
dtype/backend are fixed at construction, and every incompatible pairing is refused there
instead of discovered in the numbers. All operational sampling consumes only the passed
NumPy Generator — categorical uniforms, normal base draws and residual draws — which are
then pushed through torch transforms as plain tensors. No library `Flow.sample()` with a
global torch RNG is ever called, and the same frozen copy both samples and scores.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from so_recon.config.learning import EncoderParams, FlowParams
from so_recon.inference.contracts import DensitySchema, ThetaRecord
from so_recon.ml.checkpoint import (
    FLOW_CONFIG_SCHEMA,
    FrozenCheckpoint,
)
from so_recon.ml.contracts import ContextBatch, ContextSpec, LayoutSupport, ProposalManifest
from so_recon.ml.flow import (
    FLOW_ARCHITECTURE_VERSION,
    ConditionalNSF,
    condition_vector,
)
from so_recon.ml.normalization import FeatureScaler
from so_recon.registry.hashing import sha256_bytes, sha256_json

_LOG_TWO_PI = math.log(2.0 * math.pi)


class CategoricalHead(nn.Module):
    """`q_phi(s | C)`: one logit per head class, from the pooled encoder context."""

    def __init__(self, context_dim: int, n_families: int) -> None:
        super().__init__()
        if n_families < 1:
            raise ValueError(f"a categorical head needs a class, got {n_families}")
        self.linear = nn.Linear(context_dim, n_families)

    @property
    def n_families(self) -> int:
        return int(self.linear.out_features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.linear(context)
        return logits


class LearnedProposal(nn.Module):
    """The trainable composite: encoder → context; head for `s`; conditional NSF for `v`."""

    def __init__(
        self,
        encoder: nn.Module,
        flow: ConditionalNSF,
        *,
        n_families: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.head = CategoricalHead(flow.context_dim, n_families)

    @property
    def n_families(self) -> int:
        return self.head.n_families

    def context_vector(self, batch: ContextBatch) -> torch.Tensor:
        """The pooled Float64-ready context row `(1, context_dim)` of one world."""
        context: torch.Tensor = self.encoder(batch)
        return context

    def condition_for(self, context: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """`[context, one-hot s]`, the conditioner input of the NSF."""
        return condition_vector(context, s, n_families=self.n_families)


def flow_config_payload(
    *,
    flow: FlowParams,
    encoder_variant: str,
    encoder_params: EncoderParams,
    n_well_time_features: int,
    n_edge_features: int,
    n_static_features: int,
    n_v: int,
    context_dim: int,
    hidden_features: int,
    n_families: int,
) -> dict[str, Any]:
    """The JSON-safe description a reload needs to rebuild the exact same modules."""
    return {
        "schema_version": FLOW_CONFIG_SCHEMA,
        "architecture_version": flow.architecture_version,
        "flow": flow.model_dump(),
        "encoder": {
            "variant": encoder_variant,
            "params": encoder_params.model_dump(),
            "n_well_time_features": n_well_time_features,
            "n_edge_features": n_edge_features,
            "n_static_features": n_static_features,
        },
        "n_v": n_v,
        "context_dim": context_dim,
        "hidden_features": hidden_features,
        "n_families": n_families,
    }


def proposal_from_checkpoint(checkpoint: FrozenCheckpoint) -> LearnedProposal:
    """Rebuild the composite from its exported config and Float64 tensors, strictly."""
    from so_recon.ml.encoders import build_encoder

    config = checkpoint.flow_config
    if config.get("schema_version") != FLOW_CONFIG_SCHEMA:
        raise ValueError(
            f"unknown flow config schema {config.get('schema_version')!r}: the checkpoint "
            "does not describe a proposal this code can rebuild"
        )
    if config.get("architecture_version") != FLOW_ARCHITECTURE_VERSION:
        raise ValueError(
            f"the checkpoint architecture {config.get('architecture_version')!r} is not "
            f"the {FLOW_ARCHITECTURE_VERSION!r} this adapter implements"
        )
    encoder_block = config["encoder"]
    encoder = build_encoder(
        str(encoder_block["variant"]),
        EncoderParams(**encoder_block["params"]),
        int(encoder_block["n_well_time_features"]),
        int(encoder_block["n_edge_features"]),
        int(encoder_block["n_static_features"]),
    )
    flow = ConditionalNSF(
        FlowParams(**config["flow"]),
        n_v=int(config["n_v"]),
        context_dim=int(config["context_dim"]),
        n_families=int(config["n_families"]),
        hidden_features=int(config["hidden_features"]),
    )
    model = LearnedProposal(encoder, flow, n_families=int(config["n_families"]))
    state = {
        str(name): torch.from_numpy(np.ascontiguousarray(tensor))
        for name, tensor in checkpoint.tensors.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(torch.float64).eval()


class FrozenConditionalNSF:
    """§7.5: the bound operational density `q_phi(s,v,z | C)` on CPU Float64.

    The conditioning context is recomputed ONCE, at construction, by the same Float64
    copy that will sample and score; nothing downstream re-encodes. The declared
    families of the bound schema are the categorical support — categories the layout
    does not declare are masked as impossible, and a category merely absent from the
    training corpus never is.
    """

    def __init__(
        self,
        *,
        model: LearnedProposal,
        scaler: FeatureScaler,
        batch: ContextBatch,
        schema: DensitySchema,
        architecture_version: str,
        context_builder_version: str,
        weights_sha256: str,
        flow_config_sha256: str,
        scaler_sha256: str,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._schema = schema
        self._families = tuple(schema.families)
        self._n_residual = schema.n_residual
        with torch.inference_mode():
            context = model.context_vector(scaler.transform(batch))
        if tuple(context.shape) != (1, model.flow.context_dim):
            raise ValueError(
                f"the encoder produced a {tuple(context.shape)} context, expected "
                f"(1, {model.flow.context_dim})"
            )
        self._context = context.detach().clone()
        with torch.inference_mode():
            logits = model.head(self._context)[0]
        allowed = logits[list(self._families)]
        log_probs = torch.log_softmax(allowed, dim=-1).numpy().astype(np.float64)
        self._family_log_probs = log_probs
        self._family_cumulative = np.cumsum(np.exp(log_probs))
        self._fingerprint = sha256_json(
            {
                "kind": "frozen_conditional_nsf",
                "architecture_version": architecture_version,
                "context_builder_version": context_builder_version,
                "weights_sha256": weights_sha256,
                "flow_config_sha256": flow_config_sha256,
                "scaler_sha256": scaler_sha256,
                "schema_id": schema.schema_id,
                "transform_version": schema.transform_version,
                "basis_hash": schema.basis_hash,
                "n_v": schema.n_v,
                "n_residual": schema.n_residual,
                "families": list(self._families),
                "measure": schema.measure,
                "context_sha256": sha256_bytes(
                    np.ascontiguousarray(self._context.numpy()).tobytes()
                ),
                "operational_dtype": "float64",
                "operational_backend": "cpu",
            }
        )

    @property
    def fingerprint(self) -> str:
        """Identifies the frozen law itself; sampling never moves it."""
        return self._fingerprint

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        """`n` draws consuming `rng` and no other source of randomness.

        The consumption order is fixed — categorical uniforms, then the normal base of
        the flow, then the residual normals — so one NumPy state means one sample.
        """
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got n={n}")
        schema = self._schema
        uniforms = rng.random(n)
        family_slots = np.searchsorted(self._family_cumulative, uniforms, side="right")
        family_slots = np.minimum(family_slots, len(self._families) - 1)
        chosen = np.asarray(self._families, dtype=np.int64)[family_slots]
        epsilon = rng.standard_normal((n, schema.n_v))
        residual = rng.standard_normal((n, self._n_residual))
        with torch.inference_mode():
            eps_tensor = torch.from_numpy(np.ascontiguousarray(epsilon))
            s_tensor = torch.from_numpy(chosen)
            condition = self._model.condition_for(self._context.expand(n, -1), s_tensor)
            v_tensor, _logabsdet = self._model.flow.forward(eps_tensor, condition)
        v_numpy = v_tensor.numpy()
        return tuple(
            ThetaRecord(
                schema_id=schema.schema_id,
                s=int(family),
                v=tuple(row.tolist()),
                z_perp=tuple(z_row.tolist()),
                basis_hash=schema.basis_hash,
            )
            for family, row, z_row in zip(chosen, v_numpy, residual, strict=True)
        )

    def log_prob_parts(self, theta: ThetaRecord) -> tuple[float, float, float]:
        """The `(log q(s|C), log q(v|s,C), log p0(z))` parts, as a diagnostic.

        `log_prob` is exactly their sum; the split exists so tests and the evaluator can
        check each factor against its own reference.
        """
        schema = self._schema
        schema.validate_theta(theta)
        slot = self._families.index(theta.s)
        log_s = float(self._family_log_probs[slot])
        with torch.inference_mode():
            v_tensor = torch.tensor([tuple(theta.v)], dtype=torch.float64)
            s_tensor = torch.tensor([theta.s], dtype=torch.int64)
            condition = self._model.condition_for(self._context, s_tensor)
            log_v = float(self._model.flow.log_prob(v_tensor, condition)[0])
        z = np.fromiter(theta.z_perp, dtype=np.float64, count=self._n_residual)
        log_z = float(-0.5 * (z @ z) - 0.5 * self._n_residual * _LOG_TWO_PI)
        return log_s, log_v, log_z

    def log_prob(self, theta: ThetaRecord) -> float:
        """The log density under the declared counting x Lebesgue latent measure."""
        log_s, log_v, log_z = self.log_prob_parts(theta)
        return log_s + log_v + log_z


def _matched_layout(supported: tuple[LayoutSupport, ...], schema: DensitySchema) -> LayoutSupport:
    candidates = [layout for layout in supported if layout.schema_id == schema.schema_id]
    if not candidates:
        raise ValueError(
            f"no supported layout names schema {schema.schema_id!r}: the proposal "
            f"declares {[layout.schema_id for layout in supported]}"
        )
    for layout in candidates:
        if (
            layout.n_v == schema.n_v
            and layout.n_residual == schema.n_residual
            and tuple(layout.families) == tuple(schema.families)
        ):
            return layout
    raise ValueError(
        f"schema {schema.schema_id!r} (n_v={schema.n_v}, n_residual={schema.n_residual}, "
        f"families={list(schema.families)}) matches no supported layout of the proposal: "
        "the latent layout this world declares was never trained for"
    )


def bind_frozen_proposal(
    *,
    checkpoint: FrozenCheckpoint,
    manifest: ProposalManifest,
    spec: ContextSpec,
    batch: ContextBatch,
    schema: DensitySchema,
) -> FrozenConditionalNSF:
    """Freeze the law against one world, refusing every incompatible pairing.

    Refused here, with a reason: a batch from another context spec or builder, a
    checkpoint whose streams fail the manifest digests, a scaler fit against another
    spec, and a latent layout the proposal does not support. The basis is fixed by the
    binding — the schema the wrapper validates every theta against — so a theta from
    another basis is refused at `log_prob` by the schema itself.
    """
    if manifest.operational_dtype != "float64" or manifest.operational_backend != "cpu":
        raise ValueError(
            "the frozen law is a CPU Float64 model; this manifest declares "
            f"{manifest.operational_dtype}/{manifest.operational_backend}"
        )
    config = checkpoint.flow_config
    if config.get("architecture_version") != manifest.architecture_version:
        raise ValueError(
            f"the checkpoint architecture {config.get('architecture_version')!r} is not "
            f"the manifest's {manifest.architecture_version!r}"
        )
    if manifest.context_builder_version != spec.builder_version:
        raise ValueError(
            f"the manifest freezes context builder {manifest.context_builder_version!r} "
            f"but the spec was built by {spec.builder_version!r}"
        )
    if batch.spec_hash != spec.spec_hash:
        raise ValueError(
            "the batch was built against another context spec: its tensors are not the "
            "features this proposal was trained on"
        )
    _matched_layout(manifest.supported_layouts, schema)
    if int(config["n_v"]) != schema.n_v:
        raise ValueError(
            f"the flow models n_v={config['n_v']} coordinates but the schema declares "
            f"{schema.n_v}: a theta of this layout cannot pass through it"
        )
    if max(schema.families) >= int(config["n_families"]):
        raise ValueError(
            f"the categorical head covers {config['n_families']} classes and cannot "
            f"serve family {max(schema.families)}"
        )
    if checkpoint.hashes.weights_sha256 != manifest.weights_sha256:
        raise ValueError(
            f"the checkpoint weights fail the manifest digest (got "
            f"{checkpoint.hashes.weights_sha256}, the manifest vouches for "
            f"{manifest.weights_sha256})"
        )
    if checkpoint.hashes.flow_config_sha256 != manifest.flow_config_sha256:
        raise ValueError(
            f"the checkpoint flow config fails the manifest digest (got "
            f"{checkpoint.hashes.flow_config_sha256}, the manifest vouches for "
            f"{manifest.flow_config_sha256})"
        )
    if checkpoint.hashes.scaler_sha256 != manifest.scaler_sha256:
        raise ValueError(
            f"the checkpoint scaler fails the manifest digest (got "
            f"{checkpoint.hashes.scaler_sha256}, the manifest vouches for "
            f"{manifest.scaler_sha256})"
        )
    scaler = FeatureScaler.from_payload(checkpoint.scaler_payload, spec)
    model = proposal_from_checkpoint(checkpoint)
    return FrozenConditionalNSF(
        model=model,
        scaler=scaler,
        batch=batch,
        schema=schema,
        architecture_version=manifest.architecture_version,
        context_builder_version=manifest.context_builder_version,
        weights_sha256=manifest.weights_sha256,
        flow_config_sha256=manifest.flow_config_sha256,
        scaler_sha256=manifest.scaler_sha256,
    )


__all__ = [
    "CategoricalHead",
    "FrozenConditionalNSF",
    "LearnedProposal",
    "bind_frozen_proposal",
    "flow_config_payload",
    "proposal_from_checkpoint",
]
