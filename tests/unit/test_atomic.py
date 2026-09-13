import json
from pathlib import Path

import pytest

from so_recon.registry import atomic as atomic_module
from so_recon.registry.atomic import write_bytes_atomic, write_json_atomic, write_text_atomic


def test_write_bytes_atomic_creates_parents(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b" / "f.bin"
    write_bytes_atomic(p, b"payload")
    assert p.read_bytes() == b"payload"


def test_write_bytes_atomic_replaces_existing(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    write_bytes_atomic(p, b"one")
    write_bytes_atomic(p, b"two")
    assert p.read_bytes() == b"two"


def test_write_atomic_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_text_atomic(tmp_path / "f.txt", "hello")
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.txt"]


def test_serialisation_failure_keeps_old_content(tmp_path: Path) -> None:
    """An unserialisable object must fail before the filesystem is touched at all.

    This does NOT exercise the rollback branch — see the replace-failure test for that.
    """
    p = tmp_path / "f.json"
    write_json_atomic(p, {"ok": 1})
    with pytest.raises(ValueError):
        write_json_atomic(p, {"bad": float("inf")})
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": 1}
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.json"]


def test_cleanup_removes_the_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the rollback branch directly.

    A serialisation failure raises before any file is touched, so it never reaches the
    cleanup path. Without this test a broken `tmp.unlink` would ship green.
    """
    target = tmp_path / "f.bin"
    write_bytes_atomic(target, b"original")

    def boom(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_module.os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        write_bytes_atomic(target, b"replacement")
    assert target.read_bytes() == b"original", "a failed write must not damage the old artifact"
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.bin"], "temp file was left behind"


def test_write_json_atomic_is_deterministic_and_returns_bytes(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    first = write_json_atomic(p, {"b": 1, "a": "ё"})
    second = write_json_atomic(tmp_path / "g.json", {"a": "ё", "b": 1})
    assert first == second
    assert first == p.read_bytes()
    assert first.decode("utf-8") == '{\n  "a": "ё",\n  "b": 1\n}\n'
