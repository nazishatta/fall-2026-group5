"""k-nearest-neighbor distance baseline for novelty detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def knn_novelty(
    training_features: NDArray[np.floating],
    evaluation_features: NDArray[np.floating],
    *,
    neighbors: int = 5,
) -> NDArray[np.float64]:
    """Return mean neighbor distance; larger values indicate novelty."""
    training = np.asarray(training_features, dtype=np.float64)
    evaluation = np.asarray(evaluation_features, dtype=np.float64)

    if training.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("Both embedding arrays must be 2D")
    if training.shape[1] != evaluation.shape[1]:
        raise ValueError("Embedding dimensions must match")
    if len(training) == 0 or len(evaluation) == 0:
        raise ValueError("Embedding arrays must be nonempty")
    if not np.isfinite(training).all() or not np.isfinite(evaluation).all():
        raise ValueError("Embeddings must be finite")
    if neighbors < 1 or neighbors > len(training):
        raise ValueError("Invalid number of neighbors")

    model = NearestNeighbors(
        n_neighbors=neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=1,
    )
    model.fit(training)

    distances, _ = model.kneighbors(evaluation)
    return distances.mean(axis=1)
