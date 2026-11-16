"""The step runners in pipeline/ list their steps in order and never overwrite results.

- run_all.py runs every runner in proposal order, and Phase 3 in the order its steps
  depend on each other (checked with --dry-run).
- The three Phase 3 runners list every step with an existing script and config, run one
  step with --only, and never overwrite a run folder that already exists.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.v1_mnist.component.pipeline.common import (
    CONFIG_DIR, P3, STEP_RUNNERS, P3Step, results_folder,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_novelty_benchmark, run_qe_novelty_detection, run_u_matrix_analysis,
)
from src.v1_mnist.component.pipeline.run_all import (
    REPRODUCTION_STEPS, RUNNER_STEPS, groups, phase3_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ALL = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "run_all.py"
PIPELINE = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "p3_novelty_detection"
RUNNERS: dict[str, list[P3Step]] = {
    "run_qe_novelty_detection.py": run_qe_novelty_detection.STEPS,
    "run_u_matrix_analysis.py": run_u_matrix_analysis.STEPS,
    "run_novelty_benchmark.py": run_novelty_benchmark.STEPS,
}


def run_all(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(RUN_ALL), *args, "--dry-run"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def run_runner(runner: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PIPELINE / runner), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)


def flat(include_reproductions: bool) -> list[str]:
    return [name for _, names in groups(include_reproductions) for name in names]


# ---- run_all.py: Phases 1 and 2 ----------------------------------------------


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


# ---- run_all.py: Phase 3 order -----------------------------------------------


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
    result = run_all()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "p3_novelty_detection" not in result.stdout


def test_from_novelty_detection_runs_only_phase3_in_order() -> None:
    result = run_all("--from", "novelty_detection")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_som_nnsom_training.py" not in result.stdout
    assert "run_interactive_report.py" not in result.stdout
    starts = [result.stdout.index(f"$ python src/v1_mnist/component/pipeline/p3_novelty_detection/{runner}")
              for runner, _ in groups(include_reproductions=False)[:2]]
    assert starts == sorted(starts)
    assert "qe_threshold_sweep" in result.stdout and "svhn_som_overlay" in result.stdout
    assert "--steps qe_ood" not in result.stdout


def test_with_reproductions_adds_the_three_historical_steps() -> None:
    result = run_all("--from", "novelty_detection", "--with-reproductions")
    assert result.returncode == 0, result.stdout + result.stderr
    for name in REPRODUCTION_STEPS:
        assert name in result.stdout


def test_to_novelty_detection_builds_the_report_before_phase3() -> None:
    result = run_all("--from", "error_pattern_analysis", "--to", "novelty_detection")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.index("run_interactive_report.py") < result.stdout.index("p3_novelty_detection")


def test_runner_refuses_only_and_steps_together() -> None:
    result = subprocess.run(
        [sys.executable, str(PIPELINE / "run_qe_novelty_detection.py"), "--only", "qe_novelty", "--steps", "qe_ood"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode != 0
    assert "use --only or --steps, not both" in result.stderr


# ---- the Phase 3 runners ------------------------------------------------------


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_step_has_a_script_and_a_config(runner: str) -> None:
    for step in RUNNERS[runner]:
        assert (P3 / step.script).is_file(), step.script
        assert (REPO_ROOT / CONFIG_DIR / step.config).is_file(), step.config


@pytest.mark.parametrize("runner", RUNNERS)
def test_step_names_are_unique(runner: str) -> None:
    names = [step.name for step in RUNNERS[runner]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_dry_run_lists_every_step_in_order(runner: str) -> None:
    result = run_runner(runner, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    starts = [result.stdout.index(f"] {step.label}") for step in RUNNERS[runner]]
    assert starts == sorted(starts)
    assert "Dry run finished." in result.stdout


@pytest.mark.parametrize("runner", RUNNERS)
def test_only_runs_one_step(runner: str) -> None:
    steps = RUNNERS[runner]
    result = run_runner(runner, "--only", steps[0].name, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert steps[0].label in result.stdout
    assert all(step.label not in result.stdout for step in steps[1:])


@pytest.mark.parametrize("runner", RUNNERS)
def test_unknown_step_is_refused(runner: str) -> None:
    result = run_runner(runner, "--only", "not_a_step")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_config_needs_only() -> None:
    result = run_runner("run_u_matrix_analysis.py", "--config", "other.yaml")
    assert result.returncode != 0
    assert "--config needs --only" in result.stderr


def test_results_folder_is_output_folder_plus_run_id(tmp_path: Path) -> None:
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump({"run_id": "run_a", "output_root": str(tmp_path / "out")}))
    nested = tmp_path / "nested.yaml"
    nested.write_text(yaml.safe_dump({"ood_datasets": [{"run_id": "run_b"}], "output_dir": str(tmp_path / "out")}))
    unnamed = tmp_path / "unnamed.yaml"
    unnamed.write_text(yaml.safe_dump({"seed": 42}))
    assert results_folder(plain) == tmp_path / "out" / "run_a"
    assert results_folder(nested) == tmp_path / "out" / "run_b"
    assert results_folder(unnamed) is None


def test_existing_results_are_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "out" / "run_a").mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"run_id": "run_a", "output_root": str(tmp_path / "out")}))
    first = RUNNERS["run_qe_novelty_detection.py"][0]
    result = run_runner("run_qe_novelty_detection.py", "--only", first.name, "--config", str(config))
    assert result.returncode != 0
    assert "never overwritten" in result.stdout + result.stderr


# Configs of the other phases (1, 2 and 4); only Phase 3 configs must belong to a Phase 3 runner.
OTHER_PHASE_CONFIGS = {
    "base.yaml", "cnn_baseline.yaml", "som.yaml", "som_10x10_robustness.yaml", "som_20x20_robustness.yaml",
    "error_pattern_baselines.yaml", "retraining.yaml", "som_reanalysis.yaml",
}
# Not a step: the shared input fingerprints.
SUPPORTING_CONFIGS = {"phase3_shared_provenance.yaml"}


def test_every_phase3_config_is_a_step_or_listed_as_supporting() -> None:
    """A new Phase 3 config must be added to a runner (or listed here), so no analysis is left out."""
    step_configs = {step.config for steps in RUNNERS.values() for step in steps}
    on_disk = {path.name for path in (REPO_ROOT / CONFIG_DIR).glob("*.yaml")}
    missing = on_disk - OTHER_PHASE_CONFIGS - SUPPORTING_CONFIGS - step_configs
    assert not missing, f"Configs that no Phase 3 runner uses: {sorted(missing)}"
    assert not step_configs - on_disk, "A step names a config that does not exist"


def test_a_step_never_comes_before_the_step_whose_results_it_reads() -> None:
    """svhn_score_export writes the arrays that qe_energy_knn_paired reads."""
    names = [step.name for step in run_novelty_benchmark.STEPS]
    assert names.index("svhn_score_export") < names.index("qe_energy_knn_paired")
    assert names.index("unified_novelty_benchmark") < names.index("autoencoder_scaling_diagnostic")
    for source in ("svhn_benchmark", "required_baselines_knn", "qe_msp_paired_comparison"):
        assert names.index(source) < names.index("consolidate_evidence")
