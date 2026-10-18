"""Run-ID preflight: a finished run is never overwritten."""

from __future__ import annotations

from pathlib import Path


def ensure_not_done(results_path: Path) -> None:
    """Stop if this run's results already exist; a new run needs a new revision in the config."""
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists. Results are never overwritten: set a new run.revision "
            "in the config to redo the check."
        )
