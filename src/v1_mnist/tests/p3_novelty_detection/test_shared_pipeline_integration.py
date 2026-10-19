"""Verify shared Phase 2 compatibility and standalone Phase 3 dry-runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"
PHASE3 = (
    ROOT
    / "src/v1_mnist/component/pipeline/p3_novelty_detection/"
    "run_phase3_stage.py"
)
CONFIG_DIR = ROOT / "src/v1_mnist/component/configs"


def test_shared_phase2_pipeline_unchanged() -> None:
    """The original shared pipeline must still dry-run Phase 2."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ALL),
            "--from",
            "error_pattern_analysis",
            "--to",
            "error_pattern_analysis",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "error_pattern_analysis" in output
    assert "Dry run finished." in output


@pytest.mark.parametrize(
    ("stage", "config"),
    [
        ("qe_novelty", "qe_novelty.yaml"),
        ("qe_ood", "qe_ood_evaluation.yaml"),
        ("svhn_benchmark", "novelty_benchmark.yaml"),
    ],
)
def test_standalone_phase3_dry_run(stage: str, config: str) -> None:
    """Phase 3 adapter builds commands without running experiments."""
    result = subprocess.run(
        [
            sys.executable,
            str(PHASE3),
            "--stage",
            stage,
            "--config",
            str(CONFIG_DIR / config),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert f"Phase 3 stage: {stage}" in output
    assert "Dry run: no experiment executed" in output
