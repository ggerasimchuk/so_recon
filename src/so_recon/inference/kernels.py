"""Fixed gradient-free proposal kernels and their complete MH correction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from so_recon.geology.density import Density
from so_recon.inference.contracts import F64, DensitySchema, TargetEvaluation, ThetaRecord
from so_recon.inference.weights import bridge_log_target

BASE_KERNEL_PROBABILITIES = {
    "global": 0.25,
    "rw": 0.35,
    "pcn": 0.25,
    "family": 0.15,
}


@dataclass(frozen=True)
class Move:
    """One proposed state and ``log q(old|new) - log q(new|old)``."""

    proposed: ThetaRecord
    log_reverse_minus_forward: float
    kernel: str

    def __post_init__(self) -> None:
        if not self.kernel:
            raise ValueError("a move names its kernel")
        ratio = self.log_reverse_minus_forward
        if math.isnan(ratio) or ratio == math.inf:
            raise ValueError(f"proposal log ratio must be finite or -inf, got {ratio!r}")


def _scale(value: float, *, upper: float | None = None) -> float:
    if not math.isfinite(value) or value <= 0.0 or (upper is not None and value >= upper):
        interval = "(0,1)" if upper == 1.0 else "(0,infinity)"
        raise ValueError(f"proposal scale must be in {interval}, got {value!r}")
    return value


def pcn_reverse_minus_forward(old: F64, new: F64) -> float:
    """The pCN reverse/forward ratio under its standard-normal reference measure."""
    old_array = np.asarray(old, dtype=np.float64)
    new_array = np.asarray(new, dtype=np.float64)
    if old_array.ndim != 1 or old_array.shape != new_array.shape:
        raise ValueError(
            f"pCN vectors must have one equal shape, got {old_array.shape}/{new_array.shape}"
        )
    if not np.isfinite(old_array).all() or not np.isfinite(new_array).all():
        raise ValueError("pCN vectors must be finite")
    return float(0.5 * (new_array @ new_array - old_array @ old_array))


def propose_global(theta: ThetaRecord, r: Density, rng: np.random.Generator) -> Move:
    """Draw the whole state from r and account for both directions of that draw."""
    proposed = r.sample(1, rng)[0]
    return Move(
        proposed=proposed,
        log_reverse_minus_forward=r.log_prob(theta) - r.log_prob(proposed),
        kernel="global",
    )


def propose_rw(theta: ThetaRecord, scale: float, rng: np.random.Generator) -> Move:
    """Symmetric Gaussian random walk on the full ``v`` block only."""
    step = _scale(scale) * rng.standard_normal(len(theta.v))
    proposed = theta.model_copy(
        update={"v": tuple((np.asarray(theta.v, dtype=np.float64) + step).tolist())}
    )
    return Move(proposed=proposed, log_reverse_minus_forward=0.0, kernel="rw")


def propose_pcn(theta: ThetaRecord, scale: float, rng: np.random.Generator) -> Move:
    """pCN move on the complete residual block with its non-symmetric proposal ratio."""
    h = _scale(scale, upper=1.0)
    old = np.asarray(theta.z_perp, dtype=np.float64)
    if old.size == 0:
        raise ValueError("pCN requires a non-empty z_perp block")
    new = math.sqrt(1.0 - h * h) * old + h * rng.standard_normal(old.size)
    proposed = theta.model_copy(update={"z_perp": tuple(new.tolist())})
    return Move(
        proposed=proposed,
        log_reverse_minus_forward=pcn_reverse_minus_forward(old, new),
        kernel="pcn",
    )


def propose_family(theta: ThetaRecord, schema: DensitySchema, rng: np.random.Generator) -> Move:
    """Uniform switch to another supported family, retaining all continuous coordinates."""
    schema.validate_theta(theta)
    alternatives = tuple(family for family in schema.families if family != theta.s)
    if not alternatives:
        raise ValueError("a family switch requires at least two supported families")
    selected = alternatives[int(rng.integers(0, len(alternatives)))]
    return Move(
        proposed=theta.model_copy(update={"s": selected}),
        log_reverse_minus_forward=0.0,
        kernel="family",
    )


def kernel_probabilities(schema: DensitySchema) -> dict[str, float]:
    """Return the fixed pre-run law after removing blocks absent from this schema."""
    enabled = {"global", "rw"}
    if schema.n_residual:
        enabled.add("pcn")
    if len(schema.families) > 1:
        enabled.add("family")
    total = math.fsum(
        probability for name, probability in BASE_KERNEL_PROBABILITIES.items() if name in enabled
    )
    return {
        name: probability / total
        for name, probability in BASE_KERNEL_PROBABILITIES.items()
        if name in enabled
    }


def mh_log_accept(
    old: TargetEvaluation,
    new: TargetEvaluation,
    beta: float,
    move: Move,
) -> float:
    """Full log acceptance, including the target bridge and proposal asymmetry."""
    if move.proposed != new.theta:
        raise ValueError("the scored new theta is not the move's proposed theta")
    old_target = bridge_log_target(beta, old.log_p0, old.log_l, old.log_r)
    new_target = bridge_log_target(beta, new.log_p0, new.log_l, new.log_r)
    if old_target == -math.inf:
        raise ValueError("the current state has zero mass under the tempered target")
    if new_target == -math.inf or move.log_reverse_minus_forward == -math.inf:
        return -math.inf
    ratio = new_target - old_target + move.log_reverse_minus_forward
    if not math.isfinite(ratio):
        raise ValueError(f"Metropolis-Hastings log ratio is nonfinite: {ratio!r}")
    return float(min(0.0, ratio))


__all__ = [
    "BASE_KERNEL_PROBABILITIES",
    "Move",
    "kernel_probabilities",
    "mh_log_accept",
    "pcn_reverse_minus_forward",
    "propose_family",
    "propose_global",
    "propose_pcn",
    "propose_rw",
]
