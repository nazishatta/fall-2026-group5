"""Regression tests for Phase 3 registration in shared run_all."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"


def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run only the shared pipeline command-line interface."""
    return subprocess.run(
        [sys.executable, str(RUN_ALL), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_phase3_dry_run() -> None:
    """Explicit Phase 3 routes the requested stage and config."""
    result = invoke(
        "--from", "phase3_novelty",
        "--to", "phase3_novelty",
        "--phase3-stage", "fair_budget_baselines",
        "--phase3-config",
        "src/v1_mnist/component/configs/fair_budget_baselines.yaml",
        "--no-report",
        "--dry-run",
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "run_all: phase3_novelty" in output
    assert "Phase 3 stage: fair_budget_baselines" in output
    assert "fair_budget_baselines.yaml" in output
    assert "Dry run: no experiment executed" in output
    assert "interactive report" not in output


def test_default_pipeline_stays_phase2() -> None:
    """Default pipeline must not trigger Phase 3."""
    result = invoke("--dry-run")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "som_nnsom_training" in output
    assert "error_pattern_analysis" in output
    assert "interactive report" in output
    assert "phase3_novelty" not in output


def test_invalid_phase3_stage_rejected() -> None:
    """Unknown Phase 3 stages cannot execute."""
    result = invoke(
        "--from", "phase3_novelty",
        "--to", "phase3_novelty",
        "--phase3-stage", "not_a_real_stage",
        "--dry-run",
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "invalid choice" in output
