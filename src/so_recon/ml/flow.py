"""E03 Task 05 — the conditional NSF head: a thin typed adapter over nflows (plan §7.3).

The adapter owns exactly one thing nflows leaves open: the DIRECTION of the transform.
Here `forward` is the SAMPLING direction `v = T_phi(eps; s, C)` with `eps ~ N(0, I)`, and
`inverse` returns `(eps, log|det D T_phi^{-1}(v)|)` so that, per plan §5.2,

    log q_phi(v | s, C) = log N(T_phi^{-1}(v); 0, I) + log|det D T_phi^{-1}(v)|.

nflows' own `Flow.log_prob` assumes the opposite convention (its transform's forward maps
data to noise), which is why this module never calls it: sampling and scoring both go
through the pair declared here, and the two can no longer use opposite logdet signs.

Everything else stays nflows': `PiecewiseRationalQuadraticCouplingTransform` stacks with
alternating masks and reverse permutations between them, linear tails beyond
`tail_bound`, dropout 0 and no batch-dependent normalisation — the settings that make the
operational density unambiguous (§7.3). The conditioner is an nflows `ResidualNet` fed
the concatenated encoder context and one-hot `s`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from so_recon.config.learning import FlowParams

#: Version of the flow architecture this adapter implements; must equal
#: `FlowParams.architecture_version` of the config the head was built from.
FLOW_ARCHITECTURE_VERSION = "e03-conditional-nsf-1"

#: Default width of the coupling conditioners (`ResidualNet` hidden features). A project
#: choice of the adapter, recorded in the exported flow config for reload determinism.
DEFAULT_CONDITIONER_HIDDEN_FEATURES = 64

_LOG_TWO_PI = math.log(2.0 * math.pi)


def standard_normal_log_prob(epsilon: torch.Tensor) -> torch.Tensor:
    """`log N(eps; 0, I)` row-wise, computed by the same formula for every caller."""
    return -0.5 * (epsilon * epsilon).sum(dim=-1) - 0.5 * epsilon.shape[-1] * _LOG_TWO_PI


def one_hot_families(s: torch.Tensor, n_families: int, dtype: torch.dtype) -> torch.Tensor:
    """The one-hot block appended to the context so the conditioner receives `s`."""
    rows = torch.zeros((int(s.shape[0]), n_families), dtype=dtype, device=s.device)
    rows[torch.arange(int(s.shape[0]), device=s.device), s.to(dtype=torch.long)] = 1.0
    return rows


def condition_vector(context: torch.Tensor, s: torch.Tensor, *, n_families: int) -> torch.Tensor:
    """The conditioner input: `[encoder context, one-hot s]` (plan §7.3)."""
    return torch.cat([context, one_hot_families(s, n_families, context.dtype)], dim=-1)


class ConditionalNSF(nn.Module):
    """`q_phi(v | s, C)` as a stack of rational-quadratic conditional couplings.

    The chain alternates masks (`transform t` transforms the features of parity `t`) and
    reverses the feature order between couplings, so every coordinate is both conditioned
    and conditioning across the stack. With `n_v == 1` the single feature is conditioned
    on the context alone, which is the 1D instance the normalisation test integrates.
    """

    def __init__(
        self,
        params: FlowParams,
        *,
        n_v: int,
        context_dim: int,
        n_families: int,
        hidden_features: int = DEFAULT_CONDITIONER_HIDDEN_FEATURES,
    ) -> None:
        from nflows.transforms import (
            CompositeTransform,
            PiecewiseRationalQuadraticCouplingTransform,
            ReversePermutation,
        )

        super().__init__()
        if n_v < 1:
            raise ValueError(f"the flow needs at least one coordinate, got n_v={n_v}")
        self.params = params
        self._n_v = n_v
        self._context_dim = context_dim
        self._n_families = n_families
        self._hidden_features = hidden_features
        conditioner_dim = context_dim + n_families
        chain: list[nn.Module] = []
        for index in range(params.n_transforms):
            mask = [float((i + index) % 2 == 0) for i in range(n_v)]

            def make_conditioner(
                num_input_features: int,
                num_output_features: int,
                _hidden: int = hidden_features,
                _context: int = conditioner_dim,
            ) -> nn.Module:
                from nflows.nn.nets import ResidualNet

                net: nn.Module = ResidualNet(
                    in_features=num_input_features,
                    out_features=num_output_features,
                    hidden_features=_hidden,
                    context_features=_context,
                    num_blocks=2,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                )
                return net

            chain.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=make_conditioner,
                    num_bins=params.n_bins,
                    tails=params.tails,
                    tail_bound=params.tail_bound,
                    min_bin_width=params.min_bin_width,
                    min_bin_height=params.min_bin_height,
                    min_derivative=params.min_derivative,
                )
            )
            if index + 1 < params.n_transforms:
                chain.append(ReversePermutation(n_v))
        self.chain = CompositeTransform(chain)

    @property
    def n_v(self) -> int:
        return self._n_v

    @property
    def context_dim(self) -> int:
        """The dimension of the ENCODER context half of the conditioner input."""
        return self._context_dim

    @property
    def n_families(self) -> int:
        """The width of the one-hot `s` block appended to the context."""
        return self._n_families

    @property
    def condition_dim(self) -> int:
        """The full conditioner input width: encoder context plus one-hot `s`."""
        return self._context_dim + self._n_families

    @property
    def hidden_features(self) -> int:
        return self._hidden_features

    def _batch_condition(self, condition: torch.Tensor, batch: int) -> torch.Tensor:
        """A condition row per input row: a single frozen context broadcasts over a batch."""
        if condition.shape[0] == batch:
            return condition
        if condition.shape[0] == 1:
            return condition.expand(batch, -1)
        raise ValueError(f"{batch} inputs cannot share {condition.shape[0]} condition rows")

    def forward(
        self, epsilon: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The sampling direction: `(v, log|det D T(eps)|)` with `v = T(eps; s, C)`."""
        if epsilon.shape[-1] != self._n_v:
            raise ValueError(
                f"epsilon carries {epsilon.shape[-1]} coordinates, the flow declares {self._n_v}"
            )
        pushed: tuple[torch.Tensor, torch.Tensor] = self.chain(
            epsilon, context=self._batch_condition(condition, epsilon.shape[0])
        )
        return pushed

    def inverse(
        self, v: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The scoring direction: `(eps, log|det D T^{-1}(v)|)` (plan §5.2)."""
        if v.shape[-1] != self._n_v:
            raise ValueError(f"v carries {v.shape[-1]} coordinates, the flow declares {self._n_v}")
        pulled: tuple[torch.Tensor, torch.Tensor] = self.chain.inverse(
            v, context=self._batch_condition(condition, v.shape[0])
        )
        return pulled

    def log_prob(self, v: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """`log q_phi(v | s, C)`: the base density plus the inverse-direction logdet."""
        epsilon, logabsdet = self.inverse(v, condition)
        return standard_normal_log_prob(epsilon) + logabsdet


__all__ = [
    "DEFAULT_CONDITIONER_HIDDEN_FEATURES",
    "FLOW_ARCHITECTURE_VERSION",
    "ConditionalNSF",
    "condition_vector",
    "one_hot_families",
    "standard_normal_log_prob",
]
