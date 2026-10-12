"""Dataset and model facts come from the config, not from the code (utils/dataset_info.py)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.v1_mnist.component.utils.dataset_info import dataset_info
from src.v1_mnist.component.utils.run_settings import load_dataset_info
from src.v1_mnist.component.utils.som_io import load_split

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_mnist_values_come_from_the_som_configs() -> None:
    for cfg in ("som.yaml", "som_10x10_robustness.yaml", "som_20x20_robustness.yaml"):
        info = load_dataset_info(f"src/v1_mnist/component/configs/{cfg}", REPO_ROOT)
        assert (info.num_classes, info.feature_dim) == (10, 84)
        assert (info.display_name, info.class_label) == ("MNIST", "digit")
        assert (info.model_display_name, info.feature_layer) == ("LeNet", "fc2")


def test_another_dataset_needs_only_a_different_config() -> None:
    config = SimpleNamespace(
        dataset=SimpleNamespace(name="cifar10", num_classes=10),
        model=SimpleNamespace(architecture="ResNet18", feature_layer="avgpool", feature_dim=512),
    )
    info = dataset_info(config)
    assert info.display_name == "CIFAR10" and info.class_label == "class"
    assert info.model_display_name == "ResNet18"
    assert (info.feature_layer, info.feature_dim) == ("avgpool", 512)


def test_missing_values_are_reported() -> None:
    config = SimpleNamespace(dataset=SimpleNamespace(name="x"), model=SimpleNamespace(feature_layer="f", feature_dim=3))
    with pytest.raises(KeyError, match="num_classes"):
        dataset_info(config)


def _write_split(folder: Path, n: int, dim: int, num_classes: int) -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, num_classes, n)
    preds = labels.copy()
    preds[:3] = (labels[:3] + 1) % num_classes
    arrays = {
        "features": rng.normal(size=(n, dim)).astype(np.float32),
        "labels": labels, "preds": preds, "correct": preds == labels,
        "sample_ids": np.arange(n), "confidence": rng.random(n),
    }
    for name, arr in arrays.items():
        np.save(folder / f"val_{name}.npy", arr)


def test_load_split_uses_the_given_embedding_size(tmp_path: Path) -> None:
    _write_split(tmp_path, n=50, dim=512, num_classes=15)
    arrays = load_split("val", embeddings_dir=tmp_path, feature_dim=512)
    assert arrays["features"].shape == (50, 512)
    with pytest.raises(ValueError, match="expected"):
        load_split("val", embeddings_dir=tmp_path, feature_dim=84)
