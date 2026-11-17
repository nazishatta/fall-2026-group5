"""Tests for the kNN novelty baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)


def test_distant_points_are_more_novel() -> None:
    training = np.array([[0.0], [1.0], [2.0]])
    evaluation = np.array([[1.1], [20.0]])

    scores = knn_novelty(training, evaluation, neighbors=2)

    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        knn_novelty(
            np.zeros((3, 2)),
            np.zeros((2, 3)),
            neighbors=2,
        )


def test_rejects_invalid_neighbor_count() -> None:
    with pytest.raises(ValueError, match="neighbors"):
        knn_novelty(
            np.zeros((3, 2)),
            np.zeros((2, 2)),
            neighbors=4,
        )


def test_rejects_nonfinite_embeddings() -> None:
    with pytest.raises(ValueError, match="finite"):
        knn_novelty(
            np.array([[0.0], [np.nan]]),
            np.array([[1.0]]),
        )
