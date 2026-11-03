"""U-Matrix scores using the existing SOM neighbor rule."""

import numpy as np


def neuron_u_matrix(weights: np.ndarray, topology: np.ndarray) -> np.ndarray:
    """Calculate mean distance to adjacent SOM neurons."""
    weights = np.asarray(weights, dtype=float)
    topology = np.asarray(topology, dtype=float)

    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("Invalid SOM weights")

    n = len(weights)
    if topology.shape != (n, n) or not np.isfinite(topology).all():
        raise ValueError("Invalid SOM topology")

    neighbors = (topology <= 1.001) & ~np.eye(n, dtype=bool)
    counts = neighbors.sum(axis=1)

    if not counts.all():
        raise ValueError("SOM neuron has no neighbors")

    distances = np.linalg.norm(
        weights[:, None, :] - weights[None, :, :], axis=2
    )
    return (distances * neighbors).sum(axis=1) / counts


def sample_u_matrix(
    neuron_scores: np.ndarray, bmu_indices: np.ndarray
) -> np.ndarray:
    """Assign each sample the U-Matrix score of its BMU."""
    scores = np.asarray(neuron_scores, dtype=float)
    indices = np.asarray(bmu_indices)

    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("Invalid BMU indices")
    if (indices < 0).any() or (indices >= len(scores)).any():
        raise ValueError("BMU index outside SOM")

    return scores[indices]
