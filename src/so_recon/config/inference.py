"""The SMC settings a 4.0 configuration may declare, as validated data.

This module is pure configuration, like `so_recon.config.resources` beside it: it states
the numbers SPEC §13.2 fixes for the sampler and nothing about how the sampler works. It
exists so that the engine of Task 9 can be written and tested against declared settings
instead of guessing a target ESS.

Every number here is a SETTING, adapted nowhere. SPEC §13.2 fixes the starting CESS target
at 0.8N and the resampling threshold at ESS < 0.5N; `max_beta_steps`, `moves_per_level`,
`rw_scale` and `pcn_scale` are this stage's declared kernel budget. A run that wants other
values changes the configuration and says so in its record — it does not discover them
while it runs, because a silently adapted proposal is a different algorithm from the one
the run record names.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    """One declared SMC budget.

    Frozen and `extra='forbid'` like every other typed record in the project. It spells its
    `model_config` out instead of inheriting `StrictModel`, because `ProjectConfig` in
    `config.schema` holds a field of this type and importing the base class from there
    would be circular — the same reason `ResourceProfile` does it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: SPEC §13.2 and COMPUTE §10: a small session. Two particles is the smallest set a
    #: resampling step can be defined on at all, so it is the floor rather than the aim.
    n_particles: int = Field(default=32, ge=2)
    #: «Стартовая цель — 0,8N» for CESS (SPEC §13.2).
    cess_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    #: «по умолчанию ESS < 0,5N» for resampling (SPEC §13.2).
    resample_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    max_beta_steps: int = Field(default=24, ge=1)
    moves_per_level: int = Field(default=2, ge=1)
    #: Declared, never drawn from entropy: the same seed must reproduce the same sequence.
    seed: int = Field(ge=0)
    rw_scale: float = Field(default=0.25, gt=0.0)
    #: A pCN step mixes with `sqrt(1 - pcn_scale**2)`, so the scale is strictly inside (0,1).
    pcn_scale: float = Field(default=0.2, gt=0.0, lt=1.0)
