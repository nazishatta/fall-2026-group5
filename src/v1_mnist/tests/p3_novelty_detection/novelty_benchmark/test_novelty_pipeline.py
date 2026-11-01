"""Validate registered Phase 3 stage commands without execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.run_phase3_stage import (
    BENCHMARK_CONFIG,
    PIPELINE_DIR,
    REPO_ROOT,
    STAGES,
    build_command,
)

CONFIG_DIR = Path("src/v1_mnist/component/configs")

CASES = [
    ("qe_novelty", "qe_novelty.yaml"),
    ("qe_ood", "qe_ood_evaluation.yaml"),
    ("svhn_benchmark", "novelty_benchmark.yaml"),
    ("historical_baseline_verification", "historical_baseline_verification.yaml"),
    ("qe_energy_knn_paired", "qe_energy_knn_paired.yaml"),
    ("qe_random_control", "qe_random_control.yaml"),
    ("qe_u_matrix", "qe_u_matrix_reproduction.yaml"),
    ("qe_umatrix_random_ood", "qe_u_matrix_random_ood.yaml"),
    ("required_baselines", "required_ood_baselines.yaml"),
    ("standardized_autoencoder", "autoencoder_standardized_diagnostic.yaml"),
    ("qe_umatrix_validation_figure", "qe_umatrix_validation_figure.yaml"),
    ("u_matrix_grid_figure", "u_matrix_grid_figure.yaml"),
    ("missed_svhn_figure", "missed_svhn_figure.yaml"),
    ("fair_budget_baselines", "fair_budget_baselines.yaml"),
]


@pytest.mark.parametrize(("stage", "filename"), CASES)
def test_phase3_registered_stage_command(
    stage: str, filename: str
) -> None:
    """Commands use the correct scripts and CLI arguments."""
    config = CONFIG_DIR / filename
    assert (REPO_ROOT / config).is_file()

    command = build_command(stage, config)
    script, extra = STAGES[stage]

    assert command[0] == sys.executable
    if stage in {
            "qe_novelty",
            "qe_ood",
            "qe_random_control",
            "historical_baseline_verification",
            "qe_energy_knn_paired",
            "required_baselines",
            "fair_budget_baselines",
            "standardized_autoencoder",
            "missed_svhn_figure",
            "svhn_benchmark",
            "qe_u_matrix",
            "qe_umatrix_random_ood",
            "qe_umatrix_validation_figure",
            "u_matrix_grid_figure",
            "proposal_gap_analysis",
        }:
        assert command[1] == "-m"
        assert command[2] == (
            "src.v1_mnist.component.p3_novelty_detection."
            f"{'novelty_benchmark' if stage in {'historical_baseline_verification', 'qe_energy_knn_paired', 'required_baselines', 'fair_budget_baselines', 'standardized_autoencoder', 'missed_svhn_figure', 'svhn_benchmark', 'qe_umatrix_random_ood'} else 'qe_novelty_detection' if stage in {'qe_novelty', 'qe_ood', 'qe_random_control'} else 'u_matrix_analysis'}.{Path(script).stem}"
        )
    else:
        assert command[1] == str(PIPELINE_DIR / script)
    if stage == "svhn_benchmark" or stage in {"historical_baseline_verification", "qe_energy_knn_paired", "required_baselines", "fair_budget_baselines", "standardized_autoencoder", "missed_svhn_figure"}:
        expected_file = (
            PIPELINE_DIR.parent.parent
            / "p3_novelty_detection/novelty_benchmark"
            / Path(script).name
        )
    elif stage == "qe_random_control":
        expected_file = (
            PIPELINE_DIR.parent.parent
            / "p3_novelty_detection/qe_novelty_detection"
            / Path(script).name
        )
    else:
        expected_file = PIPELINE_DIR / script

    assert expected_file.is_file()

    if stage == "svhn_benchmark":
        assert (REPO_ROOT / config).resolve() == BENCHMARK_CONFIG.resolve()
        assert command == [
            sys.executable,
            "-m",
            ("src.v1_mnist.component.p3_novelty_detection."
            "novelty_benchmark.run_svhn_benchmark"),
            "--config",
            str((REPO_ROOT / config).resolve()),
        ]
    else:
        if stage in {
            "qe_novelty",
            "qe_ood",
            "qe_random_control",
            "historical_baseline_verification",
            "qe_energy_knn_paired",
            "required_baselines",
            "fair_budget_baselines",
            "standardized_autoencoder",
            "missed_svhn_figure",
            "qe_u_matrix",
            "qe_umatrix_random_ood",
            "qe_umatrix_validation_figure",
            "u_matrix_grid_figure",
            "proposal_gap_analysis",
        }:
            expected_command = [
                sys.executable,
                "-m",
                (
                    "src.v1_mnist.component.p3_novelty_detection."
                    f"{'novelty_benchmark' if stage in {'historical_baseline_verification', 'qe_energy_knn_paired', 'required_baselines', 'fair_budget_baselines', 'standardized_autoencoder', 'missed_svhn_figure', 'qe_umatrix_random_ood'} else 'qe_novelty_detection' if stage in {'qe_novelty', 'qe_ood', 'qe_random_control'} else 'u_matrix_analysis'}.{Path(script).stem}"
                ),
                "--config",
                str((REPO_ROOT / config).resolve()),
                *extra,
            ]
        else:
            expected_command = [
                sys.executable,
                str(PIPELINE_DIR / script),
                "--config",
                str((REPO_ROOT / config).resolve()),
                *extra,
            ]
        assert command == expected_command


def test_required_stages_registered() -> None:
    """All compatible Phase 3 runners have stage entries."""
    assert {stage for stage, _ in CASES}.issubset(STAGES)
    assert "ood_comparison" in STAGES
    assert "proposal_gap_analysis" in STAGES


def test_benchmark_forwards_selected_config() -> None:
    """SVHN benchmark must receive the explicitly selected configuration."""
    config = CONFIG_DIR / "novelty_benchmark.yaml"
    command = build_command("svhn_benchmark", config)

    assert command[-2:] == ["--config", str(config.resolve())]
    assert command[1] == "-m"
    assert command[2].endswith(".run_svhn_benchmark")


def test_benchmark_forwards_alternative_config() -> None:
    """Command construction must not silently substitute the historical YAML."""
    config = Path("alternative_svhn_config.yaml")
    command = build_command("svhn_benchmark", config)

    assert command[-2:] == [
        "--config",
        str((REPO_ROOT / config).resolve()),
    ]


@pytest.mark.parametrize(
    ("stage", "config_name"),
    [
        ("qe_novelty", "qe_novelty.yaml"),
        ("qe_ood", "qe_ood_evaluation.yaml"),
        ("svhn_benchmark", "novelty_benchmark.yaml"),
    ],
)
def test_no_duplicate_src_path(stage: str, config_name: str) -> None:
    """Generated commands must not contain duplicated source directories."""
    command = build_command(stage, CONFIG_DIR / config_name)
    assert "src/src/" not in " ".join(command)


ROOT = REPO_ROOT
RUN_ALL = ROOT / "src/v1_mnist/component/pipeline/run_all.py"
PHASE3 = ROOT / (
    "src/v1_mnist/component/p3_novelty_detection/"
    "novelty_benchmark/run_phase3_stage.py"
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


def test_shared_phase3_dry_run() -> None:
    """Verify Phase 3 dispatch without running an experiment."""
    runner = REPO_ROOT / "src/v1_mnist/component/pipeline/run_all.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--phase3-stage", "qe_novelty",
            "--phase3-config",
            "src/v1_mnist/component/configs/qe_novelty.yaml",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "run_phase3_stage.py" in output
    assert "dry run: nothing is executed" in output.lower()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--phase3-stage", "qe_novelty", "--dry-run"],
        [
            "--phase3-config",
            "src/v1_mnist/component/configs/qe_novelty.yaml",
            "--dry-run",
        ],
        [
            "--phase3-stage", "invalid_stage",
            "--phase3-config",
            "src/v1_mnist/component/configs/qe_novelty.yaml",
            "--dry-run",
        ],
        [
            "--phase3-stage", "ood_comparison",
            "--phase3-config",
            "src/v1_mnist/component/configs/qe_novelty.yaml",
            "--dry-run",
        ],
    ],
)
def test_shared_phase3_rejects_unsupported_arguments(
    arguments: list[str],
) -> None:
    """Reject invalid requests before any scientific execution."""
    runner = REPO_ROOT / "src/v1_mnist/component/pipeline/run_all.py"
    result = subprocess.run(
        [sys.executable, str(runner), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr.lower()


@pytest.mark.parametrize(
    "conflicting_args",
    [
        ["--from", "som_nnsom_training"],
        ["--to", "error_pattern_analysis"],
        ["--skip-training"],
        ["--no-report"],
        ["--cnn-config", "other.yaml"],
        ["--som-config=other.yaml"],
    ],
)
def test_shared_phase3_rejects_phase12_options(
    conflicting_args: list[str],
) -> None:
    """Phase 3 must not silently ignore Phase 1/2 arguments."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ALL),
            "--phase3-stage",
            "svhn_benchmark",
            "--phase3-config",
            str(CONFIG_DIR / "novelty_benchmark.yaml"),
            "--dry-run",
            *conflicting_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Phase 1/2 options cannot be combined" in result.stderr


def test_shared_phase3_dispatch_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify dispatch with the execution function mocked."""
    from src.v1_mnist.component.pipeline import run_all

    recorded = []

    def mock_run_steps(title: str, steps: list, dry_run: bool) -> None:
        recorded.append((title, steps, dry_run))

    monkeypatch.setattr(run_all, "run_steps", mock_run_steps)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all.py",
            "--phase3-stage", "qe_novelty",
            "--phase3-config",
            "src/v1_mnist/component/configs/qe_novelty.yaml",
        ],
    )

    run_all.main()

    assert len(recorded) == 1
    title, steps, dry_run = recorded[0]
    assert title == "Phase 3 novelty"
    assert dry_run is False
    assert steps[0].script.name == "run_phase3_stage.py"
    assert "--dry-run" not in steps[0].args
