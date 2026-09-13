"""Run identity and lineage record (SPEC 19.12, 19.13, 20.9)."""

from __future__ import annotations

import platform
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from so_recon import SPEC_VERSION
from so_recon.config.load import config_hash, resolved_config_dict
from so_recon.config.schema import ProjectConfig, StrictModel
from so_recon.paths import ProjectPaths
from so_recon.registry.artifact import ArtifactRef
from so_recon.registry.atomic import write_json_atomic
from so_recon.registry.gitinfo import git_commit, git_is_dirty
from so_recon.registry.hashing import sha256_bytes, sha256_file, sha256_json

RunStatus = Literal["RUNNING", "PASS", "FAIL"]

#: Marker used when a run has to be recorded before the configuration could be loaded.
UNAVAILABLE = "unavailable"

LOCK_FILES = ("uv.lock", "julia/Manifest.toml", "julia/.julia-version")


def make_run_id(command: str, config_hash: str, git_commit: str | None, now: datetime) -> str:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = sha256_bytes(f"{command}|{config_hash}|{git_commit or 'nogit'}".encode())[:8]
    return f"{stamp}-{command}-{short}"


def environment_lock_hash(paths: ProjectPaths) -> str:
    parts: dict[str, str] = {}
    for rel in LOCK_FILES:
        p = paths.root / rel
        parts[rel] = sha256_file(p) if p.is_file() else "missing"
    return sha256_json(parts)


def _claim_run_dir(runs_root: Path, base_id: str, max_suffix: int = 99) -> tuple[str, Path]:
    """Create a unique run directory; same-second reruns get -01, -02, ... suffixes."""
    runs_root.mkdir(parents=True, exist_ok=True)
    for i in range(max_suffix + 1):
        run_id = base_id if i == 0 else f"{base_id}-{i:02d}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise FileExistsError(f"more than {max_suffix} runs with id {base_id} in one second")


class RunRecord(StrictModel):
    run_id: str
    command: str
    argv: list[str]
    created_at: str
    finished_at: str | None = None
    git_commit: str | None
    git_dirty: bool | None
    spec_version: str
    config_version: str
    resolved_config_hash: str
    environment_lock_hash: str
    python_version: str
    julia_version: str | None = None
    jutul_version: str | None = None
    jutuldarcy_version: str | None = None
    model_checkpoint_hash: str | None = None
    schema_versions: dict[str, str]
    raw_input_hashes: dict[str, str]
    parent_run_ids: list[str]
    status: RunStatus
    outputs: dict[str, ArtifactRef]
    notes: list[str]


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    record: RunRecord

    @classmethod
    def start(
        cls,
        *,
        command: str,
        argv: Sequence[str],
        cfg: ProjectConfig | None,
        paths: ProjectPaths,
        parent_run_ids: Sequence[str] = (),
        raw_input_hashes: dict[str, str] | None = None,
        schema_versions: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> RunContext:
        """Open a run. cfg may be None so that a configuration failure can still be
        recorded as a FAIL run (invariant I6)."""
        now = now or datetime.now(UTC)
        cfg_hash = config_hash(cfg) if cfg is not None else UNAVAILABLE
        commit = git_commit(paths.root)
        dirty = git_is_dirty(paths.root)
        notes: list[str] = []
        if commit is None:
            # A silent null commit is indistinguishable from "no git installed", "not a
            # repository" and "git failed" — and SPEC 19.12 exists precisely to make a run
            # traceable. Say so in the record rather than leaving a bare null.
            notes.append("git commit unavailable: not a git repository, or git could not be run")
        if dirty:
            notes.append("working tree was dirty at run start")
        base_id = make_run_id(command, cfg_hash, commit, now)
        run_id, run_dir = _claim_run_dir(paths.runs, base_id)
        record = RunRecord(
            run_id=run_id,
            command=command,
            argv=list(argv),
            created_at=now.isoformat(),
            git_commit=commit,
            git_dirty=dirty,
            spec_version=SPEC_VERSION,
            config_version=cfg.config_version if cfg is not None else UNAVAILABLE,
            resolved_config_hash=cfg_hash,
            environment_lock_hash=environment_lock_hash(paths),
            python_version=platform.python_version(),
            schema_versions=dict(schema_versions or {}),
            raw_input_hashes=dict(raw_input_hashes or {}),
            parent_run_ids=list(parent_run_ids),
            status="RUNNING",
            outputs={},
            notes=notes,
        )
        ctx = cls(run_id=run_id, run_dir=run_dir, record=record)
        try:
            ctx.write()
        except BaseException:
            # A claimed directory holding no run.json is a run that happened but cannot be
            # recorded — exactly what invariant I6 forbids. Release the claim instead.
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        # run.json now exists, so a failure below is recorded by the caller's FAIL path
        # rather than vanishing.
        if cfg is not None:
            write_json_atomic(run_dir / "resolved_config.json", resolved_config_dict(cfg))
        return ctx

    def write(self) -> None:
        write_json_atomic(self.run_dir / "run.json", self.record.model_dump(mode="json"))

    def update(self, **fields: Any) -> None:
        """Merge fields into the record, re-validating the result.

        model_copy(update=...) skips validation, and pydantic then serialises a bad value in
        warn mode rather than raising — so a wrong status or a malformed output would be
        written into run.json silently. Static typing does not close this: values reaching
        here can arrive through Any-typed boundaries such as parsed JSON or subprocess
        output. Lineage is the one thing that must not be quietly corrupt.
        """
        self.record = RunRecord.model_validate({**self.record.model_dump(), **fields})
        self.write()

    def add_output(self, key: str, ref: ArtifactRef) -> None:
        self.update(outputs={**self.record.outputs, key: ref})

    def finish(
        self,
        status: RunStatus,
        outputs: dict[str, ArtifactRef] | None = None,
        notes: Sequence[str] = (),
    ) -> None:
        self.update(
            status=status,
            outputs={**self.record.outputs, **(outputs or {})},
            notes=[*self.record.notes, *notes],
            finished_at=datetime.now(UTC).isoformat(),
        )
