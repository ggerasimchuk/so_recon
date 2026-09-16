"""The three E03 encoders and one shared trunk (plan E03 §7.2).

All three produce the SAME context vector for the SAME NSF head, from the SAME dataset
contract; they differ only in how wells interact:

* `GraphEncoder` — the main path: causal temporal blocks per well, then edge-conditioned
  graph attention over the wells (relative position/distance only — never a conductance
  derived from the true K), then masked pooling;
* `TemporalSetEncoder` — the same temporal trunk without message passing, pooled
  permutation-invariantly (the ablation that isolates the graph's contribution);
* `SummaryEncoder` — declared per-well aggregates with set pooling, no temporal trunk.

Causality is structural: the temporal blocks convolve over the month axis with LEFT
padding only, so month `t` is a function of months `<= t` alone. Well permutation
invariance holds by construction (rows, edges and statics permute together) and is
pinned by test at Float64 to 1e-8.
"""

from __future__ import annotations

import torch
from torch import nn

from so_recon.config.learning import EncoderParams
from so_recon.ml.contracts import ContextBatch


class CausalTemporalBlock(nn.Module):
    """A residual causal conv block: month t sees months <= t only."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, T, F) -> channels-last conv over time per well
        shape = x.shape
        h = x.permute(0, 1, 3, 2).reshape(shape[0] * shape[1], shape[3], shape[2])
        h = self.act(self.conv1(h))[..., : shape[2]]
        h = torch.tanh(self.conv2(h))[..., : shape[2]]
        h = h.reshape(shape[0], shape[1], shape[3], shape[2]).permute(0, 1, 3, 2)
        return x + h


class EdgeGraphAttention(nn.Module):
    """Multi-head attention over wells with edge attributes as key/query modifiers."""

    def __init__(self, width: int, heads: int, edge_features: int) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.edge = nn.Linear(edge_features, heads)
        self.out = nn.Linear(width, width)
        self.scale = (width // heads) ** -0.5

    def forward(
        self,
        nodes: torch.Tensor,  # (B, W, F)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, Fe)
    ) -> torch.Tensor:
        b, w, f = nodes.shape
        heads = self.heads
        q = self.q(nodes).view(b, w, heads, f // heads).transpose(1, 2)  # (B,H,W,d)
        k = self.k(nodes).view(b, w, heads, f // heads).transpose(1, 2)
        v = self.v(nodes).view(b, w, heads, f // heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,W,W)
        bias = torch.zeros(w, w, device=nodes.device, dtype=nodes.dtype)
        if edge_index.numel():
            src = edge_index[0].to(nodes.device)
            dst = edge_index[1].to(nodes.device)
            bias.index_put_(
                (dst, src), self.edge(edge_attr).mean(dim=1).to(nodes.dtype), accumulate=False
            )
        scores = scores + bias.view(1, 1, w, w)
        attended = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attended, v)  # (B,H,W,d)
        mixed = mixed.transpose(1, 2).reshape(b, w, f)
        return nodes + self.out(mixed)


def _pad_to_width(features: torch.Tensor, width: int) -> torch.Tensor:
    if features.shape[-1] == width:
        return features
    if features.shape[-1] > width:
        return features[..., :width]
    padding = torch.full(
        (*features.shape[:-1], width - features.shape[-1]),
        0.0,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat([features, padding], dim=-1)


def _masked_pooling(nodes: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Mean and max pooling over wells, masked by each well's presence."""
    weights = presence.unsqueeze(-1).to(nodes.dtype)
    total = weights.sum().clamp(min=1.0)
    mean = (nodes * weights).sum(dim=1) / total
    neg = torch.finfo(nodes.dtype).min
    maxed = torch.where(
        presence.unsqueeze(-1), nodes, torch.full_like(nodes, neg)
    ).max(dim=1).values
    return torch.cat([mean, maxed], dim=-1)


def _to_tensors(batch: ContextBatch, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    well_time = torch.as_tensor(batch.well_time, dtype=dtype)
    mask = torch.as_tensor(batch.well_mask, dtype=dtype)
    static = torch.as_tensor(batch.static, dtype=dtype)
    edge_index = torch.as_tensor(batch.edge_index, dtype=torch.long)
    edge_attr = torch.as_tensor(batch.edge_attr, dtype=dtype)
    return well_time, mask, static, edge_index, edge_attr


class GraphEncoder(nn.Module):
    """The main path: causal temporal trunk + edge-conditioned graph attention."""

    def __init__(self, params: EncoderParams, n_well_time_features: int, n_edge_features: int) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.blocks = nn.ModuleList(CausalTemporalBlock(params.width) for _ in range(2))
        self.graph1 = EdgeGraphAttention(params.width, params.heads, n_edge_features)
        self.graph2 = EdgeGraphAttention(params.width, params.heads, n_edge_features)
        self.static = nn.Linear(1, params.width) if params.width else None
        self.head = nn.Sequential(
            nn.Linear(4 * params.width, params.context_dim),
            nn.GELU(),
            nn.Linear(params.context_dim, params.context_dim),
        )

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, static, edge_index, edge_attr = _to_tensors(
            batch, torch.get_default_dtype()
        )
        nodes = well_time.unsqueeze(0)  # (1, W, T, F)
        h = self.input(nodes)
        for block in self.blocks:
            h = block(h) * mask.unsqueeze(-1).unsqueeze(0)
        presence = mask.any(dim=1)  # (W,)
        temporal = (h * mask.unsqueeze(-1).unsqueeze(0)).sum(dim=2) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0).unsqueeze(0)
        nodes_flat = (temporal + self.static(static.mean(dim=-1, keepdim=True)).unsqueeze(0)).squeeze(0)
        nodes_flat = self.graph1(nodes_flat.unsqueeze(0), edge_index, edge_attr)
        nodes_flat = self.graph2(nodes_flat, edge_index, edge_attr)
        pooled = _masked_pooling(nodes_flat, presence.unsqueeze(0))
        return self.head(pooled)


class TemporalSetEncoder(nn.Module):
    """The ablation: the same temporal trunk, no message passing between wells."""

    def __init__(self, params: EncoderParams, n_well_time_features: int) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.blocks = nn.ModuleList(CausalTemporalBlock(params.width) for _ in range(2))
        self.head = nn.Sequential(
            nn.Linear(2 * params.width, params.context_dim),
            nn.GELU(),
            nn.Linear(params.context_dim, params.context_dim),
        )

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, _static, _edge_index, _edge_attr = _to_tensors(
            batch, torch.get_default_dtype()
        )
        nodes = well_time.unsqueeze(0)
        h = self.input(nodes)
        for block in self.blocks:
            h = block(h) * mask.unsqueeze(-1).unsqueeze(0)
        presence = mask.any(dim=1)
        temporal = (h * mask.unsqueeze(-1).unsqueeze(0)).sum(dim=2) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0).unsqueeze(0)
        pooled = _masked_pooling(temporal.squeeze(0), presence.unsqueeze(0))
        return self.head(pooled)


class SummaryEncoder(nn.Module):
    """Declared per-well aggregates with set pooling: no temporal trunk at all."""

    def __init__(self, params: EncoderParams, n_well_time_features: int) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.project = nn.Sequential(
            nn.Linear(params.width, params.width),
            nn.GELU(),
            nn.Linear(params.width, params.width),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * params.width + 1, params.context_dim),
            nn.GELU(),
            nn.Linear(params.context_dim, params.context_dim),
        )

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, static, _edge_index, _edge_attr = _to_tensors(
            batch, torch.get_default_dtype()
        )
        projected = self.input(well_time)
        presence = mask.any(dim=1)
        # declared aggregates over the observed months only
        observed = mask.unsqueeze(-1)
        mean_feature = (projected * observed).sum(dim=1) / observed.sum(dim=1).clamp(min=1.0)
        neg = torch.finfo(projected.dtype).min
        max_feature = torch.where(
            observed, projected, torch.full_like(projected, neg)
        ).max(dim=1).values
        summary = torch.cat(
            [mean_feature, max_feature, static.mean(dim=1, keepdim=True)], dim=-1
        )
        pooled = _masked_pooling(self.project(summary), presence.unsqueeze(0))
        return self.head(pooled)


def build_encoder(
    variant: str, params: EncoderParams, n_well_time_features: int, n_edge_features: int
) -> nn.Module:
    if variant == "graph":
        return GraphEncoder(params, n_well_time_features, n_edge_features)
    if variant == "temporal_set":
        return TemporalSetEncoder(params, n_well_time_features)
    if variant == "summary":
        return SummaryEncoder(params, n_well_time_features)
    raise ValueError(f"unknown encoder variant {variant!r}")


__all__ = [
    "CausalTemporalBlock",
    "EdgeGraphAttention",
    "GraphEncoder",
    "SummaryEncoder",
    "TemporalSetEncoder",
    "build_encoder",
]
