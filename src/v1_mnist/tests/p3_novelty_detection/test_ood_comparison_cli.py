"""Regression tests for Phase 3 OOD comparison configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = (
    REPO_ROOT
    / "src"
    / "v1_mnist"
    / "component"
    / "pipeline"
    / "p3_novelty_detection"
    / "run_ood_comparison.py"
)


def test_ood_comparison_requires_explicit_config() -> None:
    """Refuse an incomplete CLI invocation before loading any data."""
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "--config" in result.stderr
