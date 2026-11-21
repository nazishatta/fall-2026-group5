"""The dataset registry and the conversion of stored images to network input."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.v1_mnist.component.utils.image_datasets import (
    SOURCES,
    DatasetSpec,
    dataset_source,
    dataset_spec,
    to_model_input,
)


def _config(**dataset: object) -> SimpleNamespace:
    return SimpleNamespace(dataset=SimpleNamespace(**dataset))


def test_grey_28x28_is_only_scaled_exactly_as_before() -> None:
    """MNIST, Fashion-MNIST and KMNIST keep the original Phase 1 arrays (uint8 / 255, N x 28 x 28)."""
    raw = np.random.default_rng(0).integers(0, 256, size=(5, 28, 28)).astype(np.uint8)
    expected = raw.astype(np.float32)
    expected /= 255.0
    out = to_model_input(raw, channels=1, image_size=28)
    assert out.dtype == np.float32 and out.shape == (5, 28, 28)
    assert np.array_equal(out, expected)


def test_colour_images_are_made_grey_and_resized() -> None:
    raw = np.full((3, 32, 32, 3), 255, dtype=np.uint8)          # N x H x W x C, as CIFAR-10 stores it
    out = to_model_input(raw, channels=1, image_size=28)
    assert out.shape == (3, 28, 28)
    assert np.allclose(out, 1.0, atol=1e-5)                       # white stays white


def test_grey_images_can_be_given_three_channels() -> None:
    raw = np.zeros((2, 28, 28), dtype=np.uint8)
    out = to_model_input(raw, channels=3, image_size=32)
    assert out.shape == (2, 3, 32, 32)


def test_channels_other_than_one_or_three_are_refused() -> None:
    with pytest.raises(ValueError, match="channels"):
        to_model_input(np.zeros((1, 28, 28), dtype=np.uint8), channels=2, image_size=28)


def test_spec_comes_from_the_config() -> None:
    spec = dataset_spec(_config(name="fashion_mnist", channels=1, image_size=28))
    assert spec == DatasetSpec("fashion_mnist", 1, 28)


def test_spec_falls_back_to_the_stored_format() -> None:
    assert dataset_spec(_config(name="cifar10")) == DatasetSpec("cifar10", 3, 32)


def test_default_spec_is_mnist() -> None:
    assert DatasetSpec() == DatasetSpec("mnist", 1, 28)


def test_unknown_dataset_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="fashion_mnist"):
        dataset_source("no_such_set")


def test_fashion_mnist_uses_the_torchvision_folder_and_idx_files() -> None:
    source = SOURCES["fashion_mnist"]
    assert (source.torchvision_class, source.folder, source.idx_files) == ("FashionMNIST", "FashionMNIST", True)
