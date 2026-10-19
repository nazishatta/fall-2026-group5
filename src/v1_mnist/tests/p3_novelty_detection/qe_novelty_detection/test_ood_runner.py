"""Guardrails for the independent QE OOD evaluation runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    read_settings,
    resolved,
)


def _config() -> dict[str, object]:
    return {
        "id_dataset": "mnist",
        "id_split": "val",
        "ood_datasets": [
            {"name": "svhn", "split": "test", "run_id": "svhn_new"},
            {"name": "fashion_mnist", "split": "test", "run_id": "fashion_new"},
        ],
        "dataset": {"image_size": 28},
        "model": {"feature_layer": "fc2"},
        "bootstrap": {
            "iterations": 2000,
            "seed": 2026,
            "confidence": 0.95,
        },
        "som_config": "src/v1_mnist/component/configs/som.yaml",
        "qe_config": "src/v1_mnist/component/configs/qe_novelty.yaml",
        "cnn_checkpoint": "checkpoints/checkpoint.pth",
        "output_dir": "outputs/new_qe",
        "threshold_percentile": 95.0,
        "batch_size": 256,
        "data_root": "data/ood",
        "training_distance_path": "outputs/train.npy",
        "validation_distance_path": "outputs/val.npy",
    }


def test_dev_config_accepts_two_pairs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    assert len(read_settings(config_path)["ood_datasets"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"id_split": "test"},
        {"id_dataset": "fashion_mnist"},
        {"model": {"feature_layer": "fc1"}},
        {"threshold_percentile": 0},
    ],
)
def test_rejects_unsafe_or_invalid_config(
    tmp_path: Path, change: dict[str, object]
) -> None:
    config = _config()
    config.update(change)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        read_settings(path)


def test_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    config = _config()
    config["ood_datasets"][1]["run_id"] = "svhn_new"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_settings(path)


def test_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="escapes repository"):
        resolved("../outside")


def test_numeric_score_validation_without_model() -> None:
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_scoring import (
        score_embeddings,
    )

    with pytest.raises(ValueError):
        score_embeddings(object(), np.array([[np.nan]]))
