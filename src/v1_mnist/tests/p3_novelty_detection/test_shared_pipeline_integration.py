"""Regression tests for Phase 3 integration with the shared pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        (
            "error_pattern_analysis",
            "error_pattern_analysis",
            ["error_pattern_analysis", "interactive report"],
        ),
        (
            "qe_novelty",
            "svhn_benchmark",
            ["qe_novelty", "qe_ood", "svhn_benchmark"],
        ),
    ],
)
def test_shared_pipeline_dry_run(
    first: str,
    last: str,
    expected: list[str],
) -> None:
    """Existing Phase 2 behavior and Phase 3 ordering must remain intact."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ALL),
            "--from",
            first,
            "--to",
            last,
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    output = result.stdout + result.stderr
    positions = [output.index(f"] {name}\n") for name in expected]
    assert positions == sorted(positions)
    assert "Dry run finished." in output or "no experiment executed" in output
