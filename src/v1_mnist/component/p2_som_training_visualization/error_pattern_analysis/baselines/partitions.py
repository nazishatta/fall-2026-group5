"""The simple alternatives to the SOM: k-means cells and kNN impurity (max-softmax: utils/scores.py).

None of them uses a validation image's own label. k-means and kNN are fitted on the training
embeddings only, after scaling with the same scaler rule as the SOM (fitted on training data).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def scale_features(
    train: np.ndarray, val: np.ndarray, feature_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max scale both splits with a scaler fitted on the training split only."""
    scaler = MinMaxScaler(feature_range=feature_range).fit(train)
    return scaler.transform(train), scaler.transform(val)


def fit_kmeans_cells(
    train_scaled: np.ndarray, val_scaled: np.ndarray, num_clusters: int, seed: int, n_init: int
) -> tuple[np.ndarray, float]:
    """Fit k-means on training embeddings; each validation image takes its nearest centroid.

    Returns the cell index of every validation image and the training inertia.
    """
    model = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=seed).fit(train_scaled)
    return model.predict(val_scaled).astype(int), float(model.inertia_)


def knn_impurity(
    train_scaled: np.ndarray,
    train_labels: np.ndarray,
    val_scaled: np.ndarray,
    neighbors: int,
    num_classes: int,
) -> np.ndarray:
    """1 - largest class share among the nearest training embeddings of each validation image."""
    _, nearest = NearestNeighbors(n_neighbors=neighbors).fit(train_scaled).kneighbors(val_scaled)
    neighbor_labels = train_labels[nearest]
    counts = np.stack([(neighbor_labels == c).sum(axis=1) for c in range(num_classes)], axis=1)
    return 1.0 - counts.max(axis=1) / neighbors

