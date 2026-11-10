"""Resolve paths from configs safely inside a repository root."""

from __future__ import annotations

from pathlib import Path


def resolve_inside(root: Path, value: str | Path) -> Path:
    """Return ``root / value`` resolved, refusing paths that leave ``root``.

    Args:
        root: Repository root the path must stay inside.
        value: Path written in a config, relative to ``root``.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path points outside ``root``.
    """
    base = Path(root).resolve()
    candidate = (base / value).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"Path outside repository: {value}")
    return candidate
