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
def test_phase3_registered_stage_command(stage: str, filename: str) -> None:
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
    if stage == "svhn_benchmark" or stage in {
        "historical_baseline_verification",
        "qe_energy_knn_paired",
        "required_baselines",
        "fair_budget_baselines",
        "standardized_autoencoder",
        "missed_svhn_figure",
    }:
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
            (
                "src.v1_mnist.component.p3_novelty_detection."
                "novelty_benchmark.run_svhn_benchmark"
            ),
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
    "src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/run_phase3_stage.py"
)


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


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stage", "invalid_stage", "--dry-run"],
        ["--stage", "qe_novelty", "--unknown-option", "--dry-run"],
    ],
)
def test_standalone_phase3_rejects_invalid_arguments(
    arguments: list[str],
) -> None:
    """Reject invalid standalone Phase 3 requests without running experiments."""
    result = subprocess.run(
        [sys.executable, str(PHASE3), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
