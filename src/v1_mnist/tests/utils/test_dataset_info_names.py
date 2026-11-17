"""Class names for figure titles come from dataset.class_names in the config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.v1_mnist.component.utils.dataset_info import dataset_info


def _config(**dataset: object) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(name="x", num_classes=2, **dataset),
        model=SimpleNamespace(architecture="LeNet5", feature_layer="fc2", feature_dim=84),
    )


def test_names_are_used_when_given() -> None:
    info = dataset_info(_config(class_names=["Coat", "Bag"]))
    assert (info.class_name(0), info.class_name(1)) == ("Coat", "Bag")


def test_numbers_are_used_without_names() -> None:
    assert dataset_info(_config()).class_name(1) == "1"


def test_a_wrong_number_of_names_is_refused() -> None:
    with pytest.raises(ValueError, match="class_names"):
        dataset_info(_config(class_names=["Coat"]))
