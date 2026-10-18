"""Ensure Phase 3 OOD settings inherit the shared base configuration."""

from __future__ import annotations

from pathlib import Path

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    read_settings,
)
from src.v1_mnist.component.utils.config import load_config

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/v1_mnist/component/configs/qe_ood_evaluation.yaml"


def test_ood_settings_inherit_base_config() -> None:
    """Phase 3 must use the same image geometry and encoder as Phase 1."""
    shared = load_config(CONFIG)
    effective = read_settings(CONFIG)

    assert effective["image_size"] == [shared.dataset.image_size] * 2
    assert effective["feature_layer"] == shared.model.feature_layer
    assert effective["image_size"] == [28, 28]
    assert effective["feature_layer"] == "fc2"


def test_ood_settings_preserve_run_ids_and_evaluation_rules() -> None:
    """Refactoring configuration must not change the frozen evaluation plan."""
    effective = read_settings(CONFIG)

    assert effective["id_split"] == "val"
    assert {entry["name"] for entry in effective["ood_datasets"]} == {
        "svhn",
        "fashion_mnist",
    }
    assert [entry["run_id"] for entry in effective["ood_datasets"]] == [
        "mnist_svhn_qe_dev_initial",
        "mnist_fashion_mnist_qe_dev_initial",
    ]
    assert effective["threshold_percentile"] == 95.0
    assert effective["bootstrap"]["iterations"] == 2000
