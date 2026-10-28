"""Regression tests for standalone Phase 3 and unchanged shared pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"
PHASE3 = (
    ROOT
    / "src/v1_mnist/component/pipeline/p3_novelty_detection/run_phase3_stage.py"
)


def invoke(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Invoke a pipeline CLI in a subprocess."""
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_phase3_dry_run() -> None:
    """Standalone Phase 3 previews the requested stage and config."""
    result = invoke(
        PHASE3,
        "--stage", "fair_budget_baselines",
        "--config", "src/v1_mnist/component/configs/fair_budget_baselines.yaml",
        "--dry-run",
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Phase 3 stage: fair_budget_baselines" in output
    assert "fair_budget_baselines.yaml" in output
    assert "Dry run: no experiment executed" in output


def test_default_pipeline_stays_phase2() -> None:
    """Shared default pipeline must remain Phase 1/2 only."""
    result = invoke(RUN_ALL, "--dry-run")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "som_nnsom_training" in output
    assert "error_pattern_analysis" in output
    assert "interactive report" in output
    assert "phase3_novelty" not in output


def test_invalid_phase3_stage_rejected() -> None:
    """Standalone Phase 3 rejects unknown stages."""
    result = invoke(
        PHASE3,
        "--stage", "not_a_real_stage",
        "--config", "src/v1_mnist/component/configs/fair_budget_baselines.yaml",
        "--dry-run",
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "invalid choice" in output
