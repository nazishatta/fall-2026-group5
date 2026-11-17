"""frozen_split reads the dataset named by its DatasetSpec (MNIST by default)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.utils.frozen_split import (
    IMAGE_MAGIC,
    LABEL_MAGIC,
    load_train_val_images,
)
from src.v1_mnist.component.utils.image_datasets import DatasetSpec


def _write_idx(path: Path, magic: int, array: np.ndarray) -> None:
    header = magic.to_bytes(4, "big") + len(array).to_bytes(4, "big")
    if magic == IMAGE_MAGIC:
        header += array.shape[1].to_bytes(4, "big") + array.shape[2].to_bytes(4, "big")
    path.write_bytes(header + array.astype(np.uint8).tobytes())


def _dataset(root: Path, folder: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    raw = root / folder / "raw"
    raw.mkdir(parents=True)
    images = np.random.default_rng(seed).integers(0, 256, size=(10, 28, 28))
    labels = np.arange(10)
    _write_idx(raw / "train-images-idx3-ubyte", IMAGE_MAGIC, images[:6])
    _write_idx(raw / "train-labels-idx1-ubyte", LABEL_MAGIC, labels[:6])
    _write_idx(raw / "t10k-images-idx3-ubyte", IMAGE_MAGIC, images[6:])
    _write_idx(raw / "t10k-labels-idx1-ubyte", LABEL_MAGIC, labels[6:])
    return images, labels


@pytest.fixture()
def two_datasets(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    data = tmp_path / "data"
    mnist, labels = _dataset(data, "MNIST", seed=0)
    fashion, _ = _dataset(data, "FashionMNIST", seed=1)
    emb = tmp_path / "emb"
    emb.mkdir()
    for split, ids in {"train": [7, 0, 3, 9], "val": [1, 5]}.items():
        np.save(emb / f"{split}_sample_ids.npy", np.array(ids))
        np.save(emb / f"{split}_labels.npy", labels[ids])
    return data, emb, mnist, fashion


def test_default_reads_mnist(two_datasets: tuple[Path, Path, np.ndarray, np.ndarray]) -> None:
    data, emb, mnist, _ = two_datasets
    train, _ = load_train_val_images(data, emb)
    assert np.allclose(train.images, mnist[[7, 0, 3, 9]] / 255.0)


def test_fashion_mnist_is_read_from_its_own_folder(two_datasets: tuple[Path, Path, np.ndarray, np.ndarray]) -> None:
    data, emb, _, fashion = two_datasets
    train, val = load_train_val_images(data, emb, DatasetSpec("fashion_mnist", 1, 28))
    assert np.allclose(train.images, fashion[[7, 0, 3, 9]] / 255.0)
    assert np.allclose(val.images, fashion[[1, 5]] / 255.0)


def test_a_missing_dataset_folder_says_which_file(tmp_path: Path) -> None:
    emb = tmp_path / "emb"
    emb.mkdir()
    with pytest.raises(FileNotFoundError, match="KMNIST"):
        load_train_val_images(tmp_path, emb, DatasetSpec("kmnist", 1, 28))
