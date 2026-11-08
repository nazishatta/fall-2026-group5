"""Run-ID preflight and SHA-256 provenance helpers shared by the Phase 4 steps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.v1_mnist.component.utils.som_io import sha256_file


def ensure_not_done(results_path: Path) -> None:
    """Stop if this run's results already exist; a new run needs a new revision in the config."""
    if results_path.exists():
        raise FileExistsError(
            f"{results_path} already exists. Results are never overwritten: set a new run.revision "
            "in the config to redo the step."
        )


def hash_files(files: Mapping[str, Path]) -> dict[str, str]:
    """SHA-256 of each named file."""
    return {name: sha256_file(path) for name, path in files.items()}


def write_json(path: Path, content: Mapping[str, Any]) -> None:
    """Write a small JSON record (sorted keys, so reruns diff cleanly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
