"""E02.2 — the density interface of the stage, and the conditional prior that implements it.

`Density` is the seam SMC talks to. It is deliberately three members wide — draw, score,
identify — because E03 replaces the conditional prior with a trained flow and must do so
without touching the algorithm around it.

`GaussianConditionalPrior` is the law of plan §3: a fair coin over the two `kz/kx`
hypotheses times independent standard normals over the whitened coordinates. Its
normalisation is exact and written down, not estimated.

Two properties are worth stating because they are easy to lose:

**The measure is `counting x latent Lebesgue`.** `log_prob` is the density with respect to
the counting measure on `s` and Lebesgue measure on `(v, z_perp)`. The renderer's Jacobian
belongs to a density over the PHYSICAL coefficients and is not folded in here; adding it
would make the numbers below stop integrating to one.

**Nothing is truncated.** A latent point whose geology the E01 kernel refuses to render is
still a point of this prior with a finite log density. It fails in the physical evaluation,
where the failure is visible, instead of being quietly assigned zero probability — a
finite physical support would be a different prior version owing its own normaliser.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from so_recon.inference.contracts import PriorContext, ThetaRecord
from so_recon.registry.hashing import sha256_json


@runtime_checkable
class Density(Protocol):
    """A normalised latent distribution: draw from it, score a point, and name it."""

    @property
    def fingerprint(self) -> str:
        """Identifies the distribution itself, so two of them cannot be mixed unnoticed."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        """`n` independent draws, consuming `rng` and no other source of randomness."""
        ...

    def log_prob(self, theta: ThetaRecord) -> float:
        """The log density of `theta` under the declared measure."""
        ...


class GaussianConditionalPrior:
    """`p0(theta | G)`: a uniform choice of family and standard normals in the whitened basis.

    The conditioning lives in the context — `mean`, `chol` and `rotation` already carry `G`
    — so the latent coordinates themselves are standard normal. That is the whole point of
    whitening: this is NOT an unconditional `N(0, I)` over physical coefficients, and the
    renderer is what puts `G` back into the rock.
    """

    def __init__(self, context: PriorContext) -> None:
        schema = context.density_schema
        self._context = context
        self._dimension = schema.n_v + schema.n_residual
        self._n_families = len(schema.families)
        self._log_normaliser = -0.5 * self._dimension * math.log(2.0 * math.pi) - math.log(
            self._n_families
        )
        self._fingerprint = sha256_json(
            {
                "kind": "gaussian_conditional_prior",
                "schema_id": schema.schema_id,
                "n_v": schema.n_v,
                "n_residual": schema.n_residual,
                "families": list(schema.families),
                "measure": schema.measure,
                "transform_version": schema.transform_version,
                "basis_hash": schema.basis_hash,
                "n_geology": context.n_geology,
                "n_state_residual": context.n_state_residual,
                "g_hash": context.g_hash,
                "information_hash": context.information_hash,
            }
        )

    @property
    def context(self) -> PriorContext:
        return self._context

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        """Draw the family uniformly, then the whitened coordinates, in that order."""
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got n={n}")
        schema = self._context.density_schema
        families = np.asarray(schema.families, dtype=np.int64)
        chosen = families[rng.integers(0, families.size, size=n)]
        coordinates = rng.standard_normal((n, self._dimension))
        return tuple(
            ThetaRecord(
                schema_id=schema.schema_id,
                s=int(family),
                v=tuple(row[: schema.n_v].tolist()),
                z_perp=tuple(row[schema.n_v :].tolist()),
                basis_hash=schema.basis_hash,
            )
            for family, row in zip(chosen, coordinates, strict=True)
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        """The normalised log density. A theta of another schema or basis is refused."""
        self._context.density_schema.validate_theta(theta)
        coordinates = np.fromiter(
            (*theta.v, *theta.z_perp), dtype=np.float64, count=self._dimension
        )
        return float(self._log_normaliser - 0.5 * float(coordinates @ coordinates))


__all__ = ["Density", "GaussianConditionalPrior"]
