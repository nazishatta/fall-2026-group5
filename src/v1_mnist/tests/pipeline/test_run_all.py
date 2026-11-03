"""run_all.py runs every step runner in proposal order (checked with --dry-run)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from src.v1_mnist.component.pipeline.common import STEP_RUNNERS

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "run_all.py"


def run_all(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(RUN_ALL), *args, "--dry-run"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def test_dry_run_lists_every_step_in_order() -> None:
    result = run_all("--from", "data_pipeline")
    assert result.returncode == 0, result.stdout + result.stderr
    starts = [result.stdout.index(f"$ python {path.relative_to(REPO_ROOT).as_posix()}")
              for path in STEP_RUNNERS.values()]
    assert starts == sorted(starts)
    assert "run_interactive_report.py" in result.stdout


def test_from_and_to_limit_the_steps() -> None:
    result = run_all("--from", "cluster_analysis", "--to", "cluster_analysis")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_cluster_analysis.py" in result.stdout
    assert "run_error_pattern_analysis.py" not in result.stdout
    assert "run_interactive_report.py" not in result.stdout


def test_from_after_to_is_refused() -> None:
    result = run_all("--from", "error_pattern_analysis", "--to", "cluster_analysis")
    assert result.returncode != 0
    assert "--from must not come after --to" in result.stderr
