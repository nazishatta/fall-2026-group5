"""Validate registered Phase 3 stage commands without execution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_phase3_stage import (
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
    assert command[1] == str(PIPELINE_DIR / script)
    assert (PIPELINE_DIR / script).is_file()

    if stage == "svhn_benchmark":
        assert (REPO_ROOT / config).resolve() == BENCHMARK_CONFIG.resolve()
        assert command == [sys.executable, str(PIPELINE_DIR / script)]
    else:
        assert command == [
            sys.executable,
            str(PIPELINE_DIR / script),
            "--config",
            str((REPO_ROOT / config).resolve()),
            *extra,
        ]


def test_registered_stage_count() -> None:
    """All compatible Phase 3 runners have stage entries."""
    assert len(STAGES) == 14
