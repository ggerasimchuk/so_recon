"""E01.0: the spec version travels from the validated config into every lineage record.

E00 stamped a module-level constant, so a 4.0 configuration would still have produced
3.0 run records and 3.0 source manifests. The version now comes from the validated
`cfg.spec_version`, and a run opened without a configuration says `unavailable` instead
of inventing one (SPEC 0, 17.4.1). Historical 3.0 records keep their original meaning:
reading one must not rewrite it, and a 3.0 config must still serialise byte-for-byte the
way E00 serialised it.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

import so_recon
from so_recon.cli import main
from so_recon.config.load import (
    SPEC_4_0_ONLY_FIELDS,
    ConfigError,
    load_project_config,
    resolved_config_dict,
)
from so_recon.config.schema import JuliaConfig, PathsConfig
from so_recon.paths import ProjectPaths
from so_recon.registry.hashing import sha256_json
from so_recon.registry.run import RunContext, RunRecord, RunStatus
from so_recon.registry.source_manifest import (
    build_source_manifest,
    load_source_manifest,
    manifest_bytes,
)
from so_recon.runner import execute_run

REPO_ROOT = Path(__file__).resolve().parents[2]

# Recorded from the E00 code before E01 touched the version plumbing. They are drift
# detectors, not decoration: a 3.0 config must serialise exactly as it did then, or the
# `resolved_config_hash` in every historical E00 run record stops resolving.
LEGACY_REPO_CONFIG_SHA256 = "cc154ecfd84b0f88dff989f5db579b5eb6a85dd07b6e7d98b955b165b5cdcf16"
LEGACY_FIXTURE_CONFIG_SHA256 = "a120a7f0b35dd45c385bffe8d4e4c98f1bae26895205466d9406b23a568a61e0"
LEGACY_FIXTURE_MANIFEST_SHA256 = "38b0a4a66a056aa030d1c0c213180cd8d92cd74beb5e625e86b3d81bdd48129a"

# An E00-era run.json, spelled out in full so that reading it back proves no field is
# dropped, renamed or re-stamped on the way through the current model.
E00_RUN_RECORD: dict[str, Any] = {
    "run_id": "20260913T120000Z-manifest-0123abcd",
    "command": "manifest",
    "argv": ["so-recon", "manifest"],
    "created_at": "2026-09-13T12:00:00+00:00",
    "finished_at": "2026-09-13T12:00:02+00:00",
    "git_commit": "0" * 40,
    "git_dirty": False,
    "spec_version": "3.0",
    "config_version": "E00.2",
    "resolved_config_hash": "a" * 64,
    "environment_lock_hash": "b" * 64,
    "python_version": "3.13.5",
    "julia_version": None,
    "jutul_version": None,
    "jutuldarcy_version": None,
    "model_checkpoint_hash": None,
    "schema_versions": {"source_manifest": "2", "run_record": "1"},
    "raw_input_hashes": {"coords": "c" * 64},
    "parent_run_ids": [],
    "status": "PASS",
    "outputs": {
        "source_manifest": {
            "artifact_id": "d" * 64,
            "path": "reports/manifests/source_manifest.json",
            "sha256": "d" * 64,
            "size_bytes": 1234,
            "media_type": "application/json",
            "schema_version": "2",
            "producer_run_id": "20260913T120000Z-manifest-0123abcd",
            "parent_artifact_ids": [],
            "created_at": "2026-09-13T12:00:01+00:00",
        }
    },
    "notes": [],
}


def _retarget(config: Path, version: str) -> None:
    """Rewrite the fixture config so that it declares `version` instead of 3.0."""
    config.write_text(
        config.read_text(encoding="utf-8").replace('"3.0"', f'"{version}"'), encoding="utf-8"
    )


@pytest.mark.parametrize("version", ["3.0", "4.0"])
def test_lineage_uses_validated_config(tmp_project: Path, version: str) -> None:
    path = tmp_project / "configs" / "project.yml"
    _retarget(path, version)
    cfg = load_project_config(path)
    paths = ProjectPaths.from_config(tmp_project, cfg.paths)
    ctx = RunContext.start(command="probe", argv=[], cfg=cfg, paths=paths)
    manifest = build_source_manifest(
        cfg.sources,
        paths,
        config_version=cfg.config_version,
        spec_version=cfg.spec_version,
    )
    assert ctx.record.spec_version == manifest.spec_version == version
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["spec_version"] == version


def test_package_declares_both_supported_versions() -> None:
    assert so_recon.SUPPORTED_SPEC_VERSIONS == ("3.0", "4.0")
    assert so_recon.LATEST_SPEC_VERSION == "4.0"
    # The E00 stamp survives as a legacy alias; it must name a version still supported.
    assert so_recon.SPEC_VERSION in so_recon.SUPPORTED_SPEC_VERSIONS


@pytest.mark.parametrize("version", ["5.0", "4", "4.0.0", "2.9"])
def test_unsupported_spec_version_is_rejected(tmp_project: Path, version: str) -> None:
    path = tmp_project / "configs" / "project.yml"
    _retarget(path, version)
    with pytest.raises(ConfigError):
        load_project_config(path)


@pytest.mark.parametrize(
    "body",
    [
        'spec_version: "4.0"\nsources: [this is not a mapping\n',  # unparsable YAML
        "- 4.0\n",  # valid YAML, but the root is a sequence
        "",  # empty file: no mapping at all
    ],
)
def test_malformed_config_is_rejected(tmp_project: Path, body: str) -> None:
    path = tmp_project / "configs" / "malformed.yml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(path)


def test_start_without_config_has_no_invented_spec(tmp_project: Path) -> None:
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    paths = ProjectPaths.from_config(tmp_project, cfg.paths)
    ctx = RunContext.start(command="invalid", argv=[], cfg=None, paths=paths)
    ctx.finish("FAIL", notes=["config unavailable"])
    assert ctx.record.spec_version == "unavailable"
    assert ctx.record.status == "FAIL"


def test_runner_records_unavailable_spec_for_a_config_failure(tmp_project: Path) -> None:
    """The same absence, through the runner every command actually uses (invariant I6)."""
    paths = ProjectPaths.default(tmp_project)

    def body(ctx: RunContext, log: logging.Logger) -> tuple[RunStatus, list[str]]:
        return "FAIL", ["config error: injected"]

    ctx = execute_run(
        command="manifest", argv=["so-recon", "manifest"], cfg=None, paths=paths, body=body
    )
    record = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["spec_version"] == "unavailable"
    assert record["config_version"] == "unavailable"
    assert record["status"] == "FAIL"


def test_historical_3_0_run_record_reads_back_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(E00_RUN_RECORD), encoding="utf-8")
    record = RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert record.spec_version == "3.0"
    assert record.model_dump(mode="json") == E00_RUN_RECORD


def test_published_e00_manifest_still_reads_as_3_0() -> None:
    """The committed E00 manifest is frozen: reading it must reproduce its exact bytes."""
    published = REPO_ROOT / "reports" / "manifests" / "source_manifest.json"
    manifest = load_source_manifest(published)
    assert manifest.spec_version == "3.0"
    assert manifest_bytes(manifest) == published.read_bytes()


def test_legacy_3_0_config_serialisation_has_not_drifted() -> None:
    cfg = load_project_config(REPO_ROOT / "configs" / "project.yml")
    assert cfg.spec_version == "3.0"
    assert sha256_json(resolved_config_dict(cfg)) == LEGACY_REPO_CONFIG_SHA256


def test_legacy_3_0_manifest_bytes_have_not_drifted(tmp_project: Path) -> None:
    """Also pins the default: an unversioned call still builds a 3.0 manifest."""
    cfg = load_project_config(tmp_project / "configs" / "project.yml")
    paths = ProjectPaths.from_config(tmp_project, cfg.paths)
    assert sha256_json(resolved_config_dict(cfg)) == LEGACY_FIXTURE_CONFIG_SHA256
    manifest = build_source_manifest(cfg.sources, paths, config_version=cfg.config_version)
    assert manifest.spec_version == "3.0"
    assert hashlib.sha256(manifest_bytes(manifest)).hexdigest() == LEGACY_FIXTURE_MANIFEST_SHA256


def test_4_0_only_fields_are_absent_from_a_3_0_resolved_config(tmp_project: Path) -> None:
    """The same file under both specs differs exactly by the version and the 4.0 fields.

    `resources` was the first such field and E02's `inference` is the second. The 3.0 dump
    does not carry either key at all, which is what keeps every historical
    `resolved_config_hash` resolving; the 4.0 dump carries both as null because this
    fixture declares neither block. A 3.0 config that actually SET one is refused by the
    schema, so the exclusion below can only ever be dropping a null.
    """
    path = tmp_project / "configs" / "project.yml"
    legacy = resolved_config_dict(load_project_config(path))
    _retarget(path, "4.0")
    current = resolved_config_dict(load_project_config(path))
    assert (legacy["spec_version"], current["spec_version"]) == ("3.0", "4.0")
    assert set(SPEC_4_0_ONLY_FIELDS) == {"resources", "inference"}
    assert not SPEC_4_0_ONLY_FIELDS & legacy.keys()
    assert current["resources"] is None
    assert current["inference"] is None
    shared = {"spec_version", *SPEC_4_0_ONLY_FIELDS}
    assert {k: v for k, v in current.items() if k not in shared} == {
        k: v for k, v in legacy.items() if k != "spec_version"
    }


@pytest.mark.parametrize(
    ("version", "published", "absent"),
    [
        ("3.0", "source_manifest.json", "source_manifest-4.0.json"),
        ("4.0", "source_manifest-4.0.json", "source_manifest.json"),
    ],
)
def test_manifest_command_publishes_one_file_per_spec_version(
    tmp_project: Path, version: str, published: str, absent: str
) -> None:
    """A 4.0 run must not overwrite the frozen 3.0 manifest that E00 published."""
    _retarget(tmp_project / "configs" / "project.yml", version)
    assert main(["--root", str(tmp_project), "manifest"]) == 0
    manifests = tmp_project / "reports" / "manifests"
    payload = json.loads((manifests / published).read_text(encoding="utf-8"))
    assert payload["spec_version"] == version
    assert payload["manifest_version"] == "2"  # the schema is unchanged; only the value moved
    assert not (manifests / absent).exists()
    runs = sorted((tmp_project / "artifacts" / "runs").iterdir())
    record = json.loads((runs[0] / "run.json").read_text(encoding="utf-8"))
    assert record["spec_version"] == version


def test_e01_config_is_valid_and_declares_4_0() -> None:
    cfg = load_project_config(REPO_ROOT / "configs" / "e01.yml")
    assert cfg.spec_version == so_recon.LATEST_SPEC_VERSION
    assert cfg.config_version == "E01.1"
    assert cfg.project_name == "SO-RECON"
    assert cfg.sources.files == []
    # The existing paths/julia defaults, plus the 4.0-only resource profile.
    assert cfg.paths == PathsConfig()
    assert cfg.julia == JuliaConfig()
    assert cfg.resources is not None and cfg.resources.profile == "P0_VERIFY"
    # Smoke stays on the legacy config, which keeps its own version.
    assert load_project_config(REPO_ROOT / "configs" / "project.yml").spec_version == "3.0"
