"""Regression checks for the separate MNIST-SVHN kNN benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/svhn_knn_baselines.yaml"


def test_svhn_knn_configuration() -> None:
    """Keep a new run ID and a fixed kNN setting."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["run_id"] == "mnist_svhn_required_baselines_knn_s42_v1"
    assert config["evaluation"]["knn_neighbors"] == 5
    assert config["evaluation"]["target_tpr"] == 0.95
    assert config["evaluation"]["bootstrap_iterations"] == 2000
    assert config["inputs"]["svhn_file"].endswith("test_32x32.mat")


def test_knn_baseline_is_importable() -> None:
    """The benchmark uses the existing kNN novelty implementation."""
    assert callable(knn_novelty)
