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


#: Configuration fields that exist only from spec 4.0 on. A 3.0 config must serialise
#: exactly the way E00 serialised it — the `resolved_config_hash` of every historical run
#: record depends on it — so a 4.0-only field is dropped from a 3.0 dump by name. A global
#: `exclude_defaults` would do this too, but it would also drop the pre-existing defaults
#: and change every legacy hash. Empty until E01 adds its first optional field.
SPEC_4_0_ONLY_FIELDS: frozenset[str] = frozenset()


def resolved_config_dict(cfg: ProjectConfig) -> dict[str, Any]:
    data: dict[str, Any] = cfg.model_dump(mode="json")
    if cfg.spec_version == "3.0":
        return {key: value for key, value in data.items() if key not in SPEC_4_0_ONLY_FIELDS}
    return data


def config_hash(cfg: ProjectConfig) -> str:
    return sha256_json(resolved_config_dict(cfg))
