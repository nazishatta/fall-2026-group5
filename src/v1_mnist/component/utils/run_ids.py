"""Check run IDs before a step creates its output folder."""

from __future__ import annotations

from typing import Any


def check_run_id(run_id: Any) -> str:
    """Return ``run_id`` if it is a safe folder name, else raise ``ValueError``.

    A safe run ID is a non-empty string of letters, digits, ``_`` and ``-``, so it
    can never point outside the step's output folder.

    Args:
        run_id: The ``run_id`` value read from a config.

    Returns:
        The same run ID as a string.

    Raises:
        ValueError: If the run ID is empty or has any other character.
    """
    if (
        not isinstance(run_id, str)
        or not run_id
        or not all(char.isalnum() or char in "_-" for char in run_id)
    ):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    return run_id
