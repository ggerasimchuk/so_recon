"""Typed configuration schema (SPEC 19.1: versioned YAML validated by typed schemas)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    spec_version: Literal["3.0"]
    config_version: str
    project_name: str = "SO-RECON"
    paths: PathsConfig = PathsConfig()
    sources: SourcesConfig
    smoke: SmokeFixtureConfig = SmokeFixtureConfig()
    julia: JuliaConfig = JuliaConfig()
