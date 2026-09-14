from types import SimpleNamespace

import numpy as np
import pytest

from src.component.data.mnist_dataset import get_dataloaders


def make_config():
    return SimpleNamespace(
        paths=SimpleNamespace(data_root="../../All_Data"),
        dataset=SimpleNamespace(
            name="mnist",
            total_samples=70_000,
            num_classes=10,
            channels=1,
            image_size=28,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        ),
        dataloader=SimpleNamespace(batch_size=256, num_workers=0, pin_memory=False),
        training=SimpleNamespace(seed=42),
    )


@pytest.fixture(scope="session")
def loaders():
    return get_dataloaders(
        make_config(),
        train_shuffle=False,
        return_sample_ids=False,
    )


def test_split_sizes(loaders):
    train, val, test, *_ = loaders
    assert [len(train.dataset), len(val.dataset), len(test.dataset)] == [
        49_000, 10_500, 10_500
    ]


def test_image_properties(loaders):
    train, *_ = loaders
    images, _ = next(iter(train))
    assert images.shape[1:] == (1, 28, 28)
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


def test_all_classes_present(loaders):
    *_, train_labels, _, _ = loaders
    assert np.array_equal(np.unique(train_labels), np.arange(10))
