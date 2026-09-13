"""Typed configuration schema (SPEC 19.1: versioned YAML validated by typed schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from so_recon import SpecVersion
from so_recon.config.resources import ResourceProfile
from so_recon.paths import validate_relative_path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictModel):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts"
    reports: str = "reports"
    configs: str = "configs"
    julia: str = "julia"

    @field_validator("raw", "interim", "processed", "artifacts", "reports", "configs", "julia")
    @classmethod
    def _relative(cls, v: str) -> str:
        return validate_relative_path(v)


class SourceFileSpec(StrictModel):
    """A raw source file. E00 opens it read-only: it is never moved or rewritten."""

    name: str
    path: str
    encoding: Literal["utf-8", "utf-8-sig", "cp1251"]
    delimiter: Literal[",", ";"]
    decimal: Literal[".", ","]
    header_lines: int = Field(default=1, ge=0)
    required: bool = True

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        return validate_relative_path(v)


class SourcesConfig(StrictModel):
    files: list[SourceFileSpec]

    @field_validator("files")
    @classmethod
    def _unique_names(cls, v: list[SourceFileSpec]) -> list[SourceFileSpec]:
        names = [f.name for f in v]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate source names: {duplicates}")
        return v


class SmokeFixtureConfig(StrictModel):
    seed: int = 20260913
    n_wells: int = Field(default=4, ge=2)
    n_months: int = Field(default=12, ge=1)
    nx: int = Field(default=20, ge=3)
    n_steps: int = Field(default=12, ge=1)
    rel_tol: float = Field(default=1e-6, gt=0)


class JuliaConfig(StrictModel):
    project: str = "julia"
    smoke_script: str = "julia/smoke/smoke_case.jl"
    timeout_s: int = Field(default=1800, ge=1)

    @field_validator("project", "smoke_script")
    @classmethod
    def _relative(cls, v: str) -> str:
        return validate_relative_path(v)


class ProjectConfig(StrictModel):
    # A 3.0 config keeps its original meaning: it is read, hashed and stamped as 3.0.
    # Nothing here upgrades a legacy config, and nothing marks a 4.0 config as 3.0.
    spec_version: SpecVersion
    config_version: str
    project_name: str = "SO-RECON"
    paths: PathsConfig = PathsConfig()
    sources: SourcesConfig
    smoke: SmokeFixtureConfig = SmokeFixtureConfig()
    julia: JuliaConfig = JuliaConfig()
    # 4.0 only. Absence is legal — a manifest or config-only check spends nothing that
    # needs bounding — but a physical E01 command refuses without one; see
    # `so_recon.config.resources.require_resource_profile`.
    resources: ResourceProfile | None = None

    @field_validator("resources", mode="before")
    @classmethod
    def _resources_are_a_4_0_field(cls, v: object, info: ValidationInfo) -> object:
        """A 3.0 config may not carry a budget it cannot describe.

        `resolved_config_dict` drops `resources` from a 3.0 dump so that every historical
        E00 `resolved_config_hash` still resolves. If a 3.0 config were allowed to SET the
        field, that exclusion would silently drop a value that really was in force, and
        the hash would stop describing the configuration actually in use. Refusing here
        means the exclusion only ever removes a None.

        A `before` validator so that this fires ahead of the profile's own field checks:
        an operator who put a resource block in a 3.0 file is told exactly that, instead
        of being handed a list of missing profile fields. `spec_version` is declared first
        in this model, so it is already validated and available in `info.data` here.
        """
        if v is not None and info.data.get("spec_version") == "3.0":
            raise ValueError(
                "spec_version 3.0 has no 'resources' field: a resource profile is a 4.0 "
                "addition, and a 3.0 config that set one would be hashed without it. "
                "Declare spec_version 4.0, or remove the resources block"
            )
        return v
