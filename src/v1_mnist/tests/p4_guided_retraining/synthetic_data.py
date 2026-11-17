"""A tiny made-up RetrainingData for the Phase 4 tests (no MNIST files, no PyTorch)."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.utils.frozen_split import SplitImages
from src.v1_mnist.component.utils.retraining_data import RetrainingData

NUM_CLASSES = 3


def make_data(per_class: int = 40) -> RetrainingData:
    """120 training images (3 digits), 12 in the SOM pool, 4 in a hotspot; 30 validation images."""
    rng = np.random.default_rng(0)
    n = NUM_CLASSES * per_class
    labels = np.repeat(np.arange(NUM_CLASSES), per_class)
    images = rng.random((n, 28, 28)).astype(np.float32)
    som = np.zeros(n, dtype=bool)
    som[[0, 1, 2, 3, 40, 41, 42, 43, 80, 81, 82, 83]] = True
    hotspot = np.zeros(n, dtype=bool)
    hotspot[[0, 1, 40, 80]] = True
    msp = np.zeros(n, dtype=bool)
    msp[[5, 6, 7, 8, 45, 46, 47, 48, 85, 86, 87, 88]] = True
    val_labels = np.repeat(np.arange(NUM_CLASSES), 10)
    return RetrainingData(
        train=SplitImages(images, labels, np.arange(n) + 1000),
        val=SplitImages(rng.random((30, 28, 28)).astype(np.float32), val_labels, np.arange(30) + 5000),
        val_in_hotspot=np.arange(30) < 6,
        hotspot_neurons=np.array([1, 2]),
        train_in_hotspot=hotspot,
        som_pool=som,
        msp_pool=msp,
        rules_satisfied=som.astype(np.int64) + hotspot.astype(np.int64),
        bmu_distance=rng.random(n),
        confidence=rng.random(n),
        sha256={},
    )
