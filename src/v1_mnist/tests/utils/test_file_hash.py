"""Tests for the shared file fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.v1_mnist.component.utils.file_hash import sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The streamed digest equals hashlib's digest of the whole file."""
    data = b"phase 3" * 300_000
    path = tmp_path / "data.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
    assert sha256_file(str(path)) == sha256_file(path)
