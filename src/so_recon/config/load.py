"""Loading and hashing of the project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from so_recon.config.schema import ProjectConfig
from so_recon.registry.hashing import sha256_json


class ConfigError(ValueError):
    """Configuration file is missing, malformed or violates the schema."""


def load_project_config(path: Path) -> ProjectConfig:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


def resolved_config_dict(cfg: ProjectConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def config_hash(cfg: ProjectConfig) -> str:
    return sha256_json(resolved_config_dict(cfg))
