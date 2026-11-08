"""run_all.py runs Phase 3 in the order its steps depend on each other (checked with --dry-run)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_plan import (
    REPRODUCTION_STEPS, RUNNER_STEPS, groups, phase3_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "run_all.py"
RUNNERS = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "p3_novelty_detection"


def run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script), *args, "--dry-run"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)


def flat(include_reproductions: bool) -> list[str]:
    return [name for _, names in groups(include_reproductions) for name in names]


def test_every_phase3_step_is_planned_exactly_once() -> None:
    planned = flat(include_reproductions=True)
    every_step = [step.name for steps in RUNNER_STEPS.values() for step in steps]
    assert sorted(planned) == sorted(every_step)
    assert len(planned) == len(set(planned))


def test_reproduction_steps_are_left_out_by_default() -> None:
    assert not REPRODUCTION_STEPS & set(flat(include_reproductions=False))
    assert REPRODUCTION_STEPS <= set(flat(include_reproductions=True))


def test_steps_come_after_the_steps_whose_results_they_read() -> None:
    names = flat(include_reproductions=False)
    first = names.index("qe_novelty")
    benchmark = names.index("svhn_benchmark")
    assert first < benchmark
    for later in ("qe_threshold_sweep", "qe_umatrix_validation_figure", "proposal_gap_analysis", "svhn_som_overlay"):
        assert benchmark < names.index(later), later


def test_planned_steps_keep_each_runners_own_order() -> None:
    """Splitting a runner into groups must not reorder its steps."""
    for runner, steps in RUNNER_STEPS.items():
        planned = [name for r, names in groups(include_reproductions=True) if r == runner for name in names]
        assert planned == [step.name for step in steps], runner


def test_plan_builds_one_step_per_group_with_the_steps_option() -> None:
    plan = phase3_steps(dry_run=True)
    assert len(plan) == len(groups(include_reproductions=False))
    assert all("--steps" in step.args and "--dry-run" in step.args for step in plan)


def test_default_run_all_leaves_phase3_out() -> None:
    result = run(RUN_ALL)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "p3_novelty_detection" not in result.stdout


def test_from_novelty_detection_runs_only_phase3_in_order() -> None:
    result = run(RUN_ALL, "--from", "novelty_detection")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_som_nnsom_training.py" not in result.stdout
    assert "run_interactive_report.py" not in result.stdout
    starts = [result.stdout.index(f"$ python src/v1_mnist/component/pipeline/p3_novelty_detection/{runner}")
              for runner, _ in groups(include_reproductions=False)[:2]]
    assert starts == sorted(starts)
    assert "qe_threshold_sweep" in result.stdout and "svhn_som_overlay" in result.stdout
    assert "--steps qe_ood" not in result.stdout


def test_with_reproductions_adds_the_three_historical_steps() -> None:
    result = run(RUN_ALL, "--from", "novelty_detection", "--with-reproductions")
    assert result.returncode == 0, result.stdout + result.stderr
    for name in REPRODUCTION_STEPS:
        assert name in result.stdout


def test_to_novelty_detection_builds_the_report_before_phase3() -> None:
    result = run(RUN_ALL, "--from", "error_pattern_analysis", "--to", "novelty_detection")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("run_interactive_report.py") < result.stdout.index("p3_novelty_detection")


def test_runner_refuses_only_and_steps_together() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNERS / "run_qe_novelty_detection.py"), "--only", "qe_novelty", "--steps", "qe_ood"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode != 0
    assert "use --only or --steps, not both" in result.stderr
