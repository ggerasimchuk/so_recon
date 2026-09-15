"""Latent schema and normalized prior for the registered reduced-v2 experiment."""

from __future__ import annotations

import math

import numpy as np

from so_recon.inference.contracts import DensitySchema, ThetaRecord
from so_recon.registry.hashing import sha256_json
from so_recon.synthetic.reduced_inverse import REDUCED_RENDERER_VERSION, ReducedDesign


def reduced_density_schema(design: ReducedDesign) -> DensitySchema:
    """Identity of the single standard-normal coordinate used by reduced-v2."""
    return DensitySchema(
        schema_id="e02-reduced-z-1",
        n_v=1,
        n_residual=0,
        families=(0,),
        basis_hash=sha256_json(
            {
                "renderer": REDUCED_RENDERER_VERSION,
                "design": design.model_dump(mode="json"),
                "coordinate": "z-standard-normal",
            }
        ),
        transform_version="oil-corey-exponent-probit-1",
    )


class ReducedGaussianPrior:
    """The normalized ``z ~ N(0, 1)`` law of the reduced physical experiment."""

    def __init__(self, design: ReducedDesign) -> None:
        self.schema = reduced_density_schema(design)
        self._fingerprint = sha256_json(
            {
                "kind": "e02-reduced-standard-normal-1",
                "schema": self.schema.model_dump(mode="json"),
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def sample(self, n: int, rng: np.random.Generator) -> tuple[ThetaRecord, ...]:
        if n < 1:
            raise ValueError(f"a sample holds at least one point, got n={n}")
        return tuple(
            ThetaRecord(
                schema_id=self.schema.schema_id,
                s=0,
                v=(float(z),),
                z_perp=(),
                basis_hash=self.schema.basis_hash,
            )
            for z in rng.standard_normal(n)
        )

    def log_prob(self, theta: ThetaRecord) -> float:
        self.schema.validate_theta(theta)
        z = theta.v[0]
        return float(-0.5 * math.log(2.0 * math.pi) - 0.5 * z * z)


__all__ = ["ReducedGaussianPrior", "reduced_density_schema"]
