import hashlib
from pathlib import Path

import pytest

from so_recon.registry.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json


def test_sha256_bytes_matches_hashlib() -> None:
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_sha256_file_streams_large_file(tmp_path: Path) -> None:
    p = tmp_path / "big.bin"
    payload = b"x" * (3 * 1024 * 1024 + 17)
    p.write_bytes(payload)
    assert sha256_file(p, chunk_size=1024) == hashlib.sha256(payload).hexdigest()


def test_canonical_json_is_key_order_independent() -> None:
    a = canonical_json({"b": 1, "a": [1, 2, {"z": None, "y": "ё"}]})
    b = canonical_json({"a": [1, 2, {"y": "ё", "z": None}], "b": 1})
    assert a == b
    assert a == '{"a":[1,2,{"y":"ё","z":null}],"b":1}'


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_json_is_stable() -> None:
    assert sha256_json({"k": 1}) == sha256_bytes(b'{"k":1}')
