"""Score frozen CNN embeddings using the upstream SOM BMU-distance convention."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def score_embeddings(som: Any, features: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return one verified distance per sample using NNSOM.cluster_data."""
    values = np.asarray(features)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite two-dimensional embeddings")
    from src.v1_mnist.component.utils.som_io import reconstruct_assignments

    clusters, distances, _, _ = som.cluster_data(values)
    _, per_sample = reconstruct_assignments(clusters, distances, values.shape[0])
    if np.any(per_sample < 0):
        raise RuntimeError("Negative SOM distances")
    return per_sample
