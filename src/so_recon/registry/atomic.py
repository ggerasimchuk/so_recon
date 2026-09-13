"""Atomic writes: a reader never observes a half-written artifact (invariant I3).

Every scientific artifact goes through here. The temporary file is created in the
destination directory so that os.replace stays within one filesystem and is atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def fsync_dir(directory: Path) -> None:
    """Persist a rename into `directory`, so a crash cannot lose the entry itself."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_path(path: Path) -> Path:
    """Reserve a unique, empty temporary neighbour of `path`.

    For writers that insist on owning their own file handle — HDF5 is one — so they
    cannot be handed the descriptor `write_bytes_atomic` opens. The neighbour lives in
    the destination directory, which keeps the later rename on one filesystem and
    therefore atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(name)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # Path.replace, not os.replace: the ruff PTH ruleset this project selects forbids
        # os.replace (PTH105), and Path.replace delegates to it anyway, looking the attribute
        # up on the os module at call time — so monkeypatching os.replace still intercepts it.
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path, obj: object, *, indent: int = 2) -> bytes:
    """Deterministic pretty JSON. Serialisation happens before any file is touched,
    so an unserialisable object leaves the previous content intact."""
    payload = (
        json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload)
    return payload
