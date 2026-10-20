"""Tests for the configured historical SVHN inspection runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection.inspect_svhn_scores import (
    inspect,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = (
    REPO_ROOT / "src" / "v1_mnist" / "component"
    / "pipeline" / "p3_novelty_detection" / "inspect_svhn_scores.py"
)
CONFIG = (
    REPO_ROOT / "src" / "v1_mnist" / "component"
    / "configs" / "historical_svhn_inspection.yaml"
)


def test_historical_inspection_requires_config() -> None:
    result = subprocess.run(
        [sys.executable, "-m",
         "src.v1_mnist.component.pipeline.p3_novelty_detection.inspect_svhn_scores"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "--config" in result.stderr


def test_historical_inspection_reads_configured_folder(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert "historical_run_directory" in config

    # Using an empty directory proves the loader uses its argument,
    # rather than a hard-coded historical run location.
    import pytest

    with pytest.raises(FileNotFoundError):
        inspect(tmp_path)
