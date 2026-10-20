"""Regression tests for the independent Phase 3 pipeline adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_phase3_stage import (
    BENCHMARK_CONFIG,
    PIPELINE_DIR,
    REPO_ROOT,
    build_command,
)


@pytest.mark.parametrize(
    ("stage", "config_name", "script_name"),
    [
        ("qe_novelty", "qe_novelty.yaml", "run_qe_novelty.py"),
        ("qe_ood", "qe_ood_evaluation.yaml", "run_qe_ood.py"),
        ("svhn_benchmark", "novelty_benchmark.yaml", "run_svhn_benchmark.py"),
    ],
)
def test_adapter_uses_correct_paths(
    stage: str, config_name: str, script_name: str
) -> None:
    """Every stage must point to the intended script and config."""
    config = Path("src/v1_mnist/component/configs") / config_name
    command = build_command(stage, config)

    assert command[0] == sys.executable
    assert command[1] == str(PIPELINE_DIR / script_name)
    assert (REPO_ROOT / config).is_file()
    assert "src/src/" not in " ".join(command)

    if stage == "svhn_benchmark":
        assert str(BENCHMARK_CONFIG) not in command
    else:
        assert command[2:4] == ["--config", str((REPO_ROOT / config).resolve())]


def test_benchmark_rejects_other_config() -> None:
    """Historical benchmark must not silently accept another config."""
    with pytest.raises(ValueError):
        build_command("svhn_benchmark", Path("other.yaml"))
