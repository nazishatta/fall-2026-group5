"""Resolve paths from configs safely inside a repository root."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_inside(root: Path, value: str | Path) -> Path:
    """Return ``root / value`` as an absolute path, refusing paths that leave ``root``.

    The check works on the path as written: ``..`` parts are collapsed, but symbolic
    links are not followed, so a data file linked into the repository (such as the
    SVHN file on EC2) is still accepted.

    Args:
        root: Repository root the path must stay inside.
        value: Path written in a config, relative to ``root``.

    Returns:
        The absolute, normalized path.

    Raises:
        ValueError: If the path points outside ``root``.
    """
    base = Path(os.path.normpath(Path(root).absolute()))
    candidate = Path(os.path.normpath(base / value))
    if not candidate.is_relative_to(base):
        raise ValueError(f"Path outside repository: {value}")
    return candidate
