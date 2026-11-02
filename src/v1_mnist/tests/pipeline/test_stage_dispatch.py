"""Regression tests for shared pipeline dispatch compatibility."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"


def invoke(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Invoke a pipeline CLI in a subprocess."""
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


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


def test_phase3_has_independent_pipeline_entry() -> None:
    """Phase 3 has its own entry point without modifying the shared pipeline."""
    runner = (
        ROOT / "src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/"
        "run_novelty_detection.py"
    )
    assert runner.is_file()
    assert runner.name == "run_novelty_detection.py"


def test_phase3_pipeline_entry_dry_run() -> None:
    """Dedicated Phase 3 entry point delegates without executing experiments."""
    runner = (
        ROOT / "src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/"
        "run_novelty_detection.py"
    )
    result = invoke(
        runner,
        "--stage",
        "qe_novelty",
        "--config",
        "src/v1_mnist/component/configs/qe_novelty.yaml",
        "--dry-run",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Phase 3 stage: qe_novelty" in output
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
