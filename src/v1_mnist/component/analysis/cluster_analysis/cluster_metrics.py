"""Week 4 SOM cluster-structure metrics for MNIST.

This module is downstream analysis only. It does not train or modify the
baseline CNN, embeddings, scaler, or SOM.

The functions operate on already-computed sample-to-BMU assignments and
provide class/cluster statistics needed for representation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NUM_CLASSES = 10


@dataclass(frozen=True)
class ClusterMetrics:
    """Per-neuron class-structure statistics."""

    neuron_id: int
    support: int
    dominant_class: int | None
    dominant_count: int
    purity: float
    normalized_entropy: float
    class_counts: tuple[int, ...]


def normalized_entropy(counts: np.ndarray) -> float:
    """Return class entropy normalized to [0, 1].

    Empty clusters return NaN.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional.")

    total = float(counts.sum())

    if total == 0:
        return float("nan")

    probabilities = counts[counts > 0] / total
    entropy = -np.sum(probabilities * np.log(probabilities))

    if len(counts) <= 1:
        return 0.0

    return float(entropy / np.log(len(counts)))


def compute_cluster_metrics(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int = NUM_CLASSES,
) -> list[ClusterMetrics]:
    """Compute class composition, purity, and entropy for every SOM neuron."""

    labels = np.asarray(labels)
    bmu = np.asarray(bmu)

    if labels.ndim != 1 or bmu.ndim != 1:
        raise ValueError("labels and bmu must be one-dimensional.")

    if len(labels) != len(bmu):
        raise ValueError("labels and bmu must contain the same number of samples.")

    if num_neurons <= 0:
        raise ValueError("num_neurons must be positive.")

    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")

    if len(bmu) and (np.any(bmu < 0) or np.any(bmu >= num_neurons)):
        raise ValueError("BMU index outside valid neuron range.")

    if len(labels) and (
        np.any(labels < 0) or np.any(labels >= num_classes)
    ):
        raise ValueError("Class label outside valid class range.")

    results: list[ClusterMetrics] = []

    for neuron_id in range(num_neurons):
        idx = np.flatnonzero(bmu == neuron_id)
        support = int(idx.size)

        counts = np.bincount(
            labels[idx].astype(np.int64),
            minlength=num_classes,
        ).astype(np.int64)

        if support:
            dominant_class = int(np.argmax(counts))
            dominant_count = int(counts[dominant_class])
            purity = float(dominant_count / support)
            entropy = normalized_entropy(counts)
        else:
            dominant_class = None
            dominant_count = 0
            purity = float("nan")
            entropy = float("nan")

        results.append(
            ClusterMetrics(
                neuron_id=neuron_id,
                support=support,
                dominant_class=dominant_class,
                dominant_count=dominant_count,
                purity=purity,
                normalized_entropy=entropy,
                class_counts=tuple(int(x) for x in counts),
            )
        )

    return results


def class_neuron_intersection(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Return class-by-neuron sample counts.

    Matrix element [c, n] is the number of samples from class c assigned
    to SOM neuron n.
    """

    labels = np.asarray(labels)
    bmu = np.asarray(bmu)

    if labels.ndim != 1 or bmu.ndim != 1:
        raise ValueError("labels and bmu must be one-dimensional.")

    if len(labels) != len(bmu):
        raise ValueError("labels and bmu must contain the same number of samples.")

    matrix = np.zeros((num_classes, num_neurons), dtype=np.int64)

    if len(labels):
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError("Class label outside valid class range.")

        if np.any(bmu < 0) or np.any(bmu >= num_neurons):
            raise ValueError("BMU index outside valid neuron range.")

        np.add.at(
            matrix,
            (labels.astype(np.int64), bmu.astype(np.int64)),
            1,
        )

    return matrix


def weighted_cluster_purity(metrics: list[ClusterMetrics]) -> float:
    """Return sample-weighted cluster purity over occupied neurons."""

    occupied = [m for m in metrics if m.support > 0]

    if not occupied:
        return float("nan")

    total = sum(m.support for m in occupied)
    correct_majority = sum(m.dominant_count for m in occupied)

    return float(correct_majority / total)
