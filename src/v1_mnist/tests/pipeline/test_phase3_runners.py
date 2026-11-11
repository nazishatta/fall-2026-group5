"""The Phase 3 step runners list their steps in order and never overwrite results (checked with --dry-run)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_novelty_benchmark, run_qe_novelty_detection, run_u_matrix_analysis,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_steps import (
    CONFIG_DIR, P3, P3Step, results_folder,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE = REPO_ROOT / "src" / "v1_mnist" / "component" / "pipeline" / "p3_novelty_detection"

RUNNERS: dict[str, list[P3Step]] = {
    "run_qe_novelty_detection.py": run_qe_novelty_detection.STEPS,
    "run_u_matrix_analysis.py": run_u_matrix_analysis.STEPS,
    "run_novelty_benchmark.py": run_novelty_benchmark.STEPS,
}


def run_runner(runner: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PIPELINE / runner), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)


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
def test_dry_run_lists_every_step_in_order(runner: str) -> None:
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


# Configs of other phases, and the one Phase 3 file that is read by steps but is not a step itself.
OTHER_PHASE_CONFIGS = {
    "base.yaml", "cnn_baseline.yaml", "som.yaml", "som_10x10_robustness.yaml", "som_20x20_robustness.yaml",
}
# Not steps: shared input fingerprints, and the report-figure list read by copy_report_figures.py.
SUPPORTING_CONFIGS = {"phase3_shared_provenance.yaml", "phase3_report_figures.yaml"}


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
