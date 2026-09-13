import json
from pathlib import Path

import pytest

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


def test_failed_write_leaves_no_debris_and_keeps_old_content(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    write_json_atomic(p, {"ok": 1})
    with pytest.raises(ValueError):
        write_json_atomic(p, {"bad": float("inf")})
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": 1}
    assert sorted(q.name for q in tmp_path.iterdir()) == ["f.json"]


def test_write_json_atomic_is_deterministic_and_returns_bytes(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    first = write_json_atomic(p, {"b": 1, "a": "ё"})
    second = write_json_atomic(tmp_path / "g.json", {"a": "ё", "b": 1})
    assert first == second
    assert first == p.read_bytes()
    assert first.decode("utf-8") == '{\n  "a": "ё",\n  "b": 1\n}\n'
