"""Required novelty baselines on frozen CNN outputs and embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf


def max_softmax_novelty(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Higher score means lower CNN classification confidence."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Expected nonempty 2D class logits")
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite")

    return np.asarray(1.0 - softmax(values, axis=1).max(axis=1))


def energy_novelty(
    logits: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Energy score: higher values indicate greater novelty."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Expected nonempty 2D class logits")
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")

    return np.asarray(
        -temperature * logsumexp(values / temperature, axis=1)
    )


def mahalanobis_novelty(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.int64],
    evaluation_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Minimum class-conditional squared Mahalanobis distance.

    Class means and a shared shrinkage covariance are fitted on
    training embeddings only. Evaluation samples are never fitted.
    """
    train = np.asarray(train_features, dtype=np.float64)
    labels = np.asarray(train_labels)
    evaluation = np.asarray(evaluation_features, dtype=np.float64)

    if train.ndim != 2 or evaluation.ndim != 2:
        raise ValueError("Feature arrays must be 2D")
    if train.shape[0] < 3 or train.shape[1] == 0:
        raise ValueError("Insufficient training features")
    if evaluation.shape[0] == 0 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("Evaluation feature dimensions must match")
    if labels.shape != (len(train),) or labels.dtype.kind not in "iu":
        raise ValueError("Training labels must be aligned integers")
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Features must be finite")

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two training classes required")

    means = np.stack([train[labels == label].mean(axis=0) for label in classes])
    centered = train - means[np.searchsorted(classes, labels)]

    precision = LedoitWolf().fit(centered).precision_

    # Chunk samples to avoid large evaluation-by-class-by-feature arrays.
    result = np.empty(len(evaluation), dtype=np.float64)
    for start in range(0, len(evaluation), 1024):
        stop = min(start + 1024, len(evaluation))
        differences = evaluation[start:stop, None, :] - means[None, :, :]
        squared = np.einsum(
            "ncd,df,ncf->nc",
            differences,
            precision,
            differences,
            optimize=True,
        )
        result[start:stop] = np.maximum(squared.min(axis=1), 0.0)

    return result
