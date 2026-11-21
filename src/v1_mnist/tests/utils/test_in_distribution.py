"""The in-distribution dataset of a config, and the names built from it.

The key promise: for MNIST every name and text is exactly what the code wrote before
``in_distribution`` existed, so MNIST results are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.in_distribution import InDistribution, in_distribution

ROOT = Path(__file__).resolve().parents[4]
MNIST_CONFIGS = ROOT / "src/v1_mnist/component/configs"
FASHION_CONFIGS = ROOT / "src/v1_fashion_mnist/component/configs"


def test_mnist_config_folder_gives_mnist() -> None:
    dataset = in_distribution(MNIST_CONFIGS / "qe_ood_evaluation.yaml")
    assert dataset == InDistribution()
    assert dataset.is_mnist


def test_mnist_names_are_the_historical_ones() -> None:
    dataset = InDistribution()
    assert dataset.val_tag == "mnist_val"
    assert dataset.validation_key == "mnist_validation"
    assert dataset.train_key == "mnist_train"
    assert dataset.key("training_budget") == "mnist_training_budget"
    assert dataset.expected_count({"mnist_train_count": 49000}, "train") == 49000


def test_fashion_config_folder_gives_fashion_names() -> None:
    dataset = in_distribution(FASHION_CONFIGS / "som.yaml")
    assert dataset.name == "fashion_mnist"
    assert dataset.display_name == "Fashion-MNIST"
    assert dataset.class_label == "class"
    assert not dataset.is_mnist
    assert dataset.val_tag == "fashion_mnist_val"
    assert dataset.validation_key == "fashion_mnist_validation"


def test_folder_without_base_yaml_is_mnist(tmp_path: Path) -> None:
    config = tmp_path / "some_step.yaml"
    config.write_text("run_id: x\n", encoding="utf-8")
    assert in_distribution(config) == InDistribution()


def test_other_dataset_read_from_base_yaml(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "dataset:\n  name: cifar10\n  display_name: CIFAR-10\n  class_label: class\n", encoding="utf-8"
    )
    dataset = in_distribution(tmp_path / "step.yaml")
    assert (dataset.name, dataset.display_name, dataset.val_tag) == ("cifar10", "CIFAR-10", "cifar10_val")


def test_expected_count_names_the_missing_key() -> None:
    with pytest.raises(KeyError, match="fashion_mnist_train_count"):
        in_distribution(FASHION_CONFIGS / "som.yaml").expected_count({"mnist_train_count": 1}, "train")


@pytest.mark.parametrize("text", [
    "Frozen CNN. Mahalanobis fitted on MNIST training embeddings only.",
    "as a typical MNIST digit in that cell",
    "Fashion-MNIST and other OOD sets are not included.",
    "src/v1_mnist/component",
])
def test_mnist_wording_is_unchanged(text: str) -> None:
    assert InDistribution().wording(text) == text


def test_wording_for_another_dataset() -> None:
    fashion = InDistribution("fashion_mnist", "Fashion-MNIST", "class")
    assert fashion.wording("MNIST validation and SVHN test") == "Fashion-MNIST validation and SVHN test"
    assert fashion.wording("a typical MNIST digit") == "a typical Fashion-MNIST class"
    assert fashion.wording("how many MNIST digits") == "how many Fashion-MNIST class images"
    # Names that end in MNIST and lower-case names are left alone; "MNIST-like" is not.
    assert fashion.wording("Fashion-MNIST, v1_mnist") == "Fashion-MNIST, v1_mnist"
    assert fashion.wording("MNIST-like norms, MNIST-validation QE") == (
        "Fashion-MNIST-like norms, Fashion-MNIST-validation QE"
    )
