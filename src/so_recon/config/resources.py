"""Resource profiles: the budget a physical E01 command is allowed to spend.

This module is pure configuration. It knows nothing about the operating system and
nothing about policy: it only states, as validated data, the numbers COMPUTE §§2, 5, 7
and 10 fix for E01, so that the probe (`so_recon.environment.resources`) and the policy
(`so_recon.simulator.budget`) can be written and tested against them without ever
guessing a limit.

Memory is GiB = 2^30 bytes throughout. `soft_bytes`/`hard_bytes`/`reserve_bytes` and the
disk budget come straight from COMPUTE §§5 and 7; `wall_budget_s` and `max_new_forward`
from the per-profile session limits of COMPUTE §§2 and 10. `job_timeout_s`,
`startup_timeout_s` and `max_swap_growth_bytes` are conservative E01 decisions taken on
top of those normative session limits, not normative values themselves — a single job
that wants more than a fraction of the session, or a run that has pushed half a gigabyte
into swap, is a run that has already gone wrong. Cold startup is inside the wall budget,
which is why `startup_timeout_s` may not exceed it.

The historical smoke timeout (`JuliaConfig.timeout_s`, 1800 s) is a different thing
entirely and is deliberately left alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Memory is stated in GiB = 2^30 bytes (COMPUTE §5), never in decimal gigabytes.
GIB = 1024**3
MIB = 1024**2

ResourceProfileId = Literal["P0_VERIFY", "P1_LOOP"]

#: COMPUTE §5: «резервом не менее 6 GiB для ОС и прочих процессов». A profile that
#: reserves less than this is refused rather than quietly starving the machine.
MIN_RESERVE_BYTES = 6 * GIB


class MissingResourceProfileError(RuntimeError):
    """A physical command was asked to run without a measured resource budget."""


class UnapprovedResourceProfileError(RuntimeError):
    """A profile block claims an approved profile name but does not carry its numbers."""


class ResourceProfile(BaseModel):
    """One approved budget, named by its profile id.

    Frozen and `extra='forbid'` like every other typed record in the project. It spells
    its `model_config` out instead of inheriting `StrictModel`, because `ProjectConfig`
    in `config.schema` holds a field of this type and importing the base class from there
    would be circular.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: ResourceProfileId
    soft_bytes: int = Field(gt=0)
    hard_bytes: int = Field(gt=0)
    reserve_bytes: int = Field(gt=0)
    disk_budget_bytes: int = Field(gt=0)
    wall_budget_s: int = Field(gt=0)
    job_timeout_s: int = Field(gt=0)
    startup_timeout_s: int = Field(gt=0)
    max_new_forward: int = Field(gt=0)
    julia_workers: int = Field(default=1, gt=0)
    julia_threads: int = Field(default=4, gt=0)
    blas_threads: int = Field(default=1, gt=0)
    poll_interval_s: float = Field(default=0.25, gt=0.0)
    max_swap_growth_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _limits_are_consistent(self) -> ResourceProfile:
        if self.soft_bytes > self.hard_bytes:
            raise ValueError(
                f"soft_bytes {self.soft_bytes} exceeds hard_bytes {self.hard_bytes}: "
                "the warning threshold must sit below the limit it warns about"
            )
        if self.reserve_bytes < MIN_RESERVE_BYTES:
            raise ValueError(
                f"reserve_bytes {self.reserve_bytes} is below the {MIN_RESERVE_BYTES} bytes "
                "COMPUTE §5 reserves for the OS and other processes"
            )
        if self.job_timeout_s > self.wall_budget_s:
            raise ValueError(
                f"job_timeout_s {self.job_timeout_s} exceeds wall_budget_s "
                f"{self.wall_budget_s}: a single job that cannot fit inside the session "
                "budget would always be refused"
            )
        if self.startup_timeout_s > self.wall_budget_s:
            raise ValueError(
                f"startup_timeout_s {self.startup_timeout_s} exceeds wall_budget_s "
                f"{self.wall_budget_s}: cold startup is spent inside the wall budget"
            )
        # COMPUTE §5 fixes both of these for E01. They are fields rather than constants so
        # that a profile states the whole environment it runs in, but a value other than 1
        # is a specification change, not a configuration choice.
        if self.julia_workers != 1:
            raise ValueError(
                f"julia_workers must be 1 in E01 (COMPUTE §5), got {self.julia_workers}"
            )
        if self.blas_threads != 1:
            raise ValueError(f"blas_threads must be 1 in E01 (COMPUTE §5), got {self.blas_threads}")
        return self


#: P0_VERIFY — COMPUTE §§2, 10: ten minutes and 64 new forwards for a verification session.
P0_VERIFY_PROFILE = ResourceProfile(
    profile="P0_VERIFY",
    soft_bytes=12 * GIB,
    hard_bytes=16 * GIB,
    reserve_bytes=6 * GIB,
    disk_budget_bytes=5 * GIB,
    wall_budget_s=600,
    job_timeout_s=300,
    startup_timeout_s=300,
    max_new_forward=64,
    julia_workers=1,
    julia_threads=4,
    blas_threads=1,
    poll_interval_s=0.25,
    max_swap_growth_bytes=512 * MIB,
)

#: P1_LOOP — COMPUTE §§2, 10: one hour and 2000 new forwards. Memory, disk, threads, the
#: poll interval and the swap limit are unchanged; only the session's own length, its
#: forward count and the per-job timeout that follows from them move.
P1_LOOP_PROFILE = ResourceProfile(
    profile="P1_LOOP",
    soft_bytes=12 * GIB,
    hard_bytes=16 * GIB,
    reserve_bytes=6 * GIB,
    disk_budget_bytes=5 * GIB,
    wall_budget_s=3600,
    job_timeout_s=900,
    startup_timeout_s=300,
    max_new_forward=2000,
    julia_workers=1,
    julia_threads=4,
    blas_threads=1,
    poll_interval_s=0.25,
    max_swap_growth_bytes=512 * MIB,
)

RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    P0_VERIFY_PROFILE.profile: P0_VERIFY_PROFILE,
    P1_LOOP_PROFILE.profile: P1_LOOP_PROFILE,
}


def resource_profile(name: str) -> ResourceProfile:
    """Look an approved preset up by name; an explicit CLI override goes through here."""
    try:
        return RESOURCE_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown resource profile {name!r}; approved profiles are {sorted(RESOURCE_PROFILES)}"
        ) from None


def require_resource_profile(profile: ResourceProfile | None, *, command: str) -> ResourceProfile:
    """Gate a physical command on having an approved, measured budget.

    A missing profile is legal: a 3.0 configuration has no such field at all, and a 4.0
    manifest or config-only check spends nothing that needs bounding. Anything that
    actually runs physics does, and refuses here rather than discovering the absence
    halfway through a simulation — that is what makes «resource failure не даёт
    физический нулевой likelihood» checkable (SPEC 18.4).

    The numbers are checked here too, not only their internal consistency. The field
    validators above cannot tell that a block labelled `P0_VERIFY` carries P0_VERIFY's
    limits, and `wall_budget_s` and `max_new_forward` have no measured backstop the way
    the memory caps do — `total - reserve` clamps those whatever the config claims, while
    a session length or a forward count is spent exactly as written. Since COMPUTE §§2
    and 10 fix those per profile, a block that names an approved profile must be that
    profile. A different budget needs a new approved preset, not a relabelled one.
    """
    if profile is None:
        raise MissingResourceProfileError(
            f"command {command!r} runs physics and needs a resource profile, but the "
            "configuration declares none; add a `resources:` block (spec_version 4.0) or "
            "select an approved preset explicitly"
        )
    approved = RESOURCE_PROFILES.get(profile.profile)
    if approved is None:
        raise UnapprovedResourceProfileError(
            f"command {command!r}: profile {profile.profile!r} is not an approved preset; "
            f"approved profiles are {sorted(RESOURCE_PROFILES)}"
        )
    if profile != approved:
        differences = {
            name: (value, getattr(approved, name))
            for name, value in profile.model_dump().items()
            if value != getattr(approved, name)
        }
        raise UnapprovedResourceProfileError(
            f"command {command!r}: the resource block names {profile.profile!r} but does "
            f"not carry its approved limits (COMPUTE §§2, 5, 7, 10). Differences as "
            f"{{field: (configured, approved)}}: {differences}"
        )
    return profile
