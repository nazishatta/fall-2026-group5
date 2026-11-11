"""SHA-256 fingerprints of files, used for provenance checks."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 hex digest for a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
