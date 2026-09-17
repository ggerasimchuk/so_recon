"""The three E03 encoders and one shared trunk (plan E03 §7.2).

All three produce the SAME context vector for the SAME NSF head, from the SAME dataset
contract; they differ only in how wells interact:

* `GraphEncoder` — the main path: causal temporal blocks per well, then edge-conditioned
  graph attention over the wells (relative position/distance only — never a conductance
  derived from the true K), then masked pooling;
* `TemporalSetEncoder` — the same temporal trunk and statics without message passing,
  pooled permutation-invariantly (the ablation that isolates the graph's contribution);
* `SummaryEncoder` — declared per-well aggregates with set pooling, no temporal trunk.

Each encoder ends the same way: per-well node vectors of width `width`, masked mean+max
pooling over wells to `2 * width`, and one shared head shape to `context_dim`.

Causality is structural: the temporal blocks convolve over the month axis with LEFT
padding only, so month `t` is a function of months `<= t` alone. Well permutation
invariance holds by construction (rows, edges and statics permute together) and is
pinned by test at Float64 to 1e-8.

No global torch state is read or written: the compute dtype is whatever dtype the
module's parameters carry, so the Float64 operational pass casts the module (`.double()`)
and training keeps Float32 without anyone touching `torch.set_default_dtype`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from so_recon.config.learning import EncoderParams
from so_recon.ml.contracts import ContextBatch

#: Static feature count of the default context spec; the real caller passes the spec's
#: own count, the default only keeps direct construction honest.
DEFAULT_N_STATIC_FEATURES = 5


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
    """Multi-head attention over wells with edge attributes as score biases."""

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


def _to_tensors(
    batch: ContextBatch, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    # np.array copies the read-only contract arrays, so no tensor ever aliases them.
    # The mask stays boolean: it selects (`torch.where`) as often as it scales.
    well_time = torch.from_numpy(np.array(batch.well_time)).to(dtype=dtype, device=device)
    mask = torch.from_numpy(np.array(batch.well_mask)).to(device=device)
    static = torch.from_numpy(np.array(batch.static)).to(dtype=dtype, device=device)
    edge_index = torch.from_numpy(np.array(batch.edge_index)).to(device=device)
    edge_attr = torch.from_numpy(np.array(batch.edge_attr)).to(dtype=dtype, device=device)
    return well_time, mask, static, edge_index, edge_attr


def _masked_pooling(nodes: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Mean and max pooling over wells, masked by each well's presence."""
    weights = presence.unsqueeze(-1).to(nodes.dtype)
    total = weights.sum().clamp(min=1.0)
    mean = (nodes * weights).sum(dim=1) / total
    neg = torch.finfo(nodes.dtype).min
    maxed = (
        torch.where(presence.unsqueeze(-1), nodes, torch.full_like(nodes, neg)).max(dim=1).values
    )
    return torch.cat([mean, maxed], dim=-1)


def _temporal_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over each well's valid months; masked months contribute nothing.

    `h` is (1, W, T, width), `mask` is (W, T): the mean runs over the time axis and
    divides by each well's own count of valid months.
    """
    weights = mask.unsqueeze(-1)  # (W, T, 1)
    summed = (h * weights).sum(dim=2)  # (1, W, width)
    count = weights.sum(dim=1)  # (W, 1)
    return summed / count.clamp(min=1.0)


def _context_head(width: int, context_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * width, context_dim),
        nn.GELU(),
        nn.Linear(context_dim, context_dim),
    )


class GraphEncoder(nn.Module):
    """The main path: causal temporal trunk + statics + edge-conditioned attention."""

    def __init__(
        self,
        params: EncoderParams,
        n_well_time_features: int,
        n_edge_features: int,
        n_static_features: int = DEFAULT_N_STATIC_FEATURES,
    ) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.blocks = nn.ModuleList(CausalTemporalBlock(params.width) for _ in range(2))
        self.static = nn.Linear(n_static_features, params.width)
        self.graph1 = EdgeGraphAttention(params.width, params.heads, n_edge_features)
        self.graph2 = EdgeGraphAttention(params.width, params.heads, n_edge_features)
        self.head = _context_head(params.width, params.context_dim)

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, static, edge_index, edge_attr = _to_tensors(
            batch, self.input.weight.dtype, self.input.weight.device
        )
        h = self.input(well_time.unsqueeze(0))  # (1, W, T, width)
        for block in self.blocks:
            h = block(h) * mask.unsqueeze(-1).unsqueeze(0)
        presence = mask.any(dim=1)  # (W,)
        nodes = _temporal_pool(h, mask) + self.static(static)  # (1, W, width)
        nodes = self.graph1(nodes, edge_index, edge_attr)
        nodes = self.graph2(nodes, edge_index, edge_attr)
        pooled = _masked_pooling(nodes, presence.unsqueeze(0))  # (1, 2*width)
        return self.head(pooled)  # (1, context_dim)


class TemporalSetEncoder(nn.Module):
    """The ablation: the same trunk and statics, no message passing between wells."""

    def __init__(
        self,
        params: EncoderParams,
        n_well_time_features: int,
        n_static_features: int = DEFAULT_N_STATIC_FEATURES,
    ) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.blocks = nn.ModuleList(CausalTemporalBlock(params.width) for _ in range(2))
        self.static = nn.Linear(n_static_features, params.width)
        self.head = _context_head(params.width, params.context_dim)

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, static, _edge_index, _edge_attr = _to_tensors(
            batch, self.input.weight.dtype, self.input.weight.device
        )
        h = self.input(well_time.unsqueeze(0))
        for block in self.blocks:
            h = block(h) * mask.unsqueeze(-1).unsqueeze(0)
        presence = mask.any(dim=1)
        nodes = _temporal_pool(h, mask) + self.static(static)
        pooled = _masked_pooling(nodes, presence.unsqueeze(0))
        return self.head(pooled)


class SummaryEncoder(nn.Module):
    """Declared per-well aggregates with set pooling: no temporal trunk at all."""

    def __init__(
        self,
        params: EncoderParams,
        n_well_time_features: int,
        n_static_features: int = DEFAULT_N_STATIC_FEATURES,
    ) -> None:
        super().__init__()
        self.params = params
        self.input = nn.Linear(n_well_time_features, params.width)
        self.static = nn.Linear(n_static_features, params.width)
        self.combine = nn.Sequential(
            nn.Linear(3 * params.width, params.width),
            nn.GELU(),
        )
        self.head = _context_head(params.width, params.context_dim)

    def forward(self, batch: ContextBatch) -> torch.Tensor:
        well_time, mask, static, _edge_index, _edge_attr = _to_tensors(
            batch, self.input.weight.dtype, self.input.weight.device
        )
        projected = self.input(well_time)  # (W, T, width)
        presence = mask.any(dim=1)
        # declared aggregates over the observed months only
        observed = mask.unsqueeze(-1)
        mean_feature = (projected * observed).sum(dim=1) / observed.sum(dim=1).clamp(min=1.0)
        neg = torch.finfo(projected.dtype).min
        max_feature = (
            torch.where(observed, projected, torch.full_like(projected, neg)).max(dim=1).values
        )
        nodes = self.combine(torch.cat([mean_feature, max_feature, self.static(static)], dim=-1))
        pooled = _masked_pooling(nodes, presence.unsqueeze(0))
        return self.head(pooled)


def build_encoder(
    variant: str,
    params: EncoderParams,
    n_well_time_features: int,
    n_edge_features: int,
    n_static_features: int = DEFAULT_N_STATIC_FEATURES,
) -> nn.Module:
    if variant == "graph":
        return GraphEncoder(params, n_well_time_features, n_edge_features, n_static_features)
    if variant == "temporal_set":
        return TemporalSetEncoder(params, n_well_time_features, n_static_features)
    if variant == "summary":
        return SummaryEncoder(params, n_well_time_features, n_static_features)
    raise ValueError(f"unknown encoder variant {variant!r}")


__all__ = [
    "CausalTemporalBlock",
    "EdgeGraphAttention",
    "GraphEncoder",
    "SummaryEncoder",
    "TemporalSetEncoder",
    "build_encoder",
]
