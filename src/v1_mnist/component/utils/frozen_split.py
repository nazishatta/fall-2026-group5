"""Rebuild the project's frozen data split from the saved sample IDs.

The split is not re-drawn with a random seed. Phase 1 saved the sample IDs of every split next to
the embeddings (an ID is a position in the 70,000 images: MNIST train 0..59999, then MNIST test
60000..69999), so the images are looked up by those IDs. That keeps the split identical whatever
training seed a Phase 4 run uses, and every look-up is checked against the saved labels.

The test split has its own function (``load_test_images``). Training code never calls it.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


@dataclass(frozen=True)
class SplitImages:
    """Images (scaled to 0..1, shape N x 28 x 28), labels and sample IDs of one split."""

    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray


def _read_bytes(folder: Path, name: str) -> bytes:
    plain, packed = folder / name, folder / f"{name}.gz"
    if plain.is_file():
        return plain.read_bytes()
    if packed.is_file():
        return gzip.decompress(packed.read_bytes())
    raise FileNotFoundError(f"MNIST file not found: {plain} (or .gz). Check paths.data_root.")


def read_idx(folder: Path, name: str, magic: int) -> np.ndarray:
    """Read one MNIST idx file (images or labels)."""
    raw = _read_bytes(folder, name)
    if int.from_bytes(raw[:4], "big") != magic:
        raise ValueError(f"{name}: unexpected file header.")
    count = int.from_bytes(raw[4:8], "big")
    if magic == IMAGE_MAGIC:
        rows, cols = int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols)
    return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(count)


def load_combined_mnist(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """All 70,000 MNIST images (float32, 0..1) and labels, in sample-ID order."""
    folder = Path(data_root) / "MNIST" / "raw"
    images = np.concatenate([
        read_idx(folder, MNIST_FILES["train_images"], IMAGE_MAGIC),
        read_idx(folder, MNIST_FILES["test_images"], IMAGE_MAGIC),
    ]).astype(np.float32) / 255.0
    labels = np.concatenate([
        read_idx(folder, MNIST_FILES["train_labels"], LABEL_MAGIC),
        read_idx(folder, MNIST_FILES["test_labels"], LABEL_MAGIC),
    ]).astype(np.int64)
    return images, labels


def _split_from_ids(
    split: str, embeddings_dir: Path, images: np.ndarray, labels: np.ndarray
) -> SplitImages:
    ids = np.load(Path(embeddings_dir) / f"{split}_sample_ids.npy").astype(np.int64)
    saved_labels = np.load(Path(embeddings_dir) / f"{split}_labels.npy").astype(np.int64)
    if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(labels):
        raise ValueError(f"{split}: sample IDs must be unique and inside 0..{len(labels) - 1}.")
    if not np.array_equal(labels[ids], saved_labels):
        raise ValueError(
            f"{split}: MNIST labels at the saved sample IDs differ from the saved labels. "
            "paths.data_root points at different data than Phase 1 used."
        )
    return SplitImages(images=images[ids], labels=saved_labels, sample_ids=ids)


def load_train_val_images(
    data_root: Path, embeddings_dir: Path
) -> tuple[SplitImages, SplitImages]:
    """Training and validation images in the saved order. The test split is not touched."""
    images, labels = load_combined_mnist(data_root)
    train = _split_from_ids("train", embeddings_dir, images, labels)
    val = _split_from_ids("val", embeddings_dir, images, labels)
    if set(train.sample_ids.tolist()) & set(val.sample_ids.tolist()):
        raise ValueError("Training and validation sample IDs overlap.")
    return train, val


def load_test_images(data_root: Path, embeddings_dir: Path) -> SplitImages:
    """Test images. Only the final test stage may call this (protocol must be frozen)."""
    images, labels = load_combined_mnist(data_root)
    return _split_from_ids("test", embeddings_dir, images, labels)
