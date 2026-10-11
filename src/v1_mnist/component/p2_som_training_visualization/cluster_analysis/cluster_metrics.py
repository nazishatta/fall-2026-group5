"""Week 4 SOM cluster-structure metrics (Cluster Analysis).

Downstream analysis only: it does not train or modify the CNN, the
embeddings, the scaler or the SOM. The functions work on sample-to-BMU
assignments that were already computed.

The class-by-neuron counts come from NNSOM's own post-training tool,
``cal_class_cluster_intersect``, as the project proposal asks ("compute
cluster purity and class-cluster intersection metrics (available in
NNSOM)"). Entropy comes from utils/stats.py, the project's only copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from NNSOM.utils import cal_class_cluster_intersect

from src.v1_mnist.component.utils.stats import normalized_entropy



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


def _check_inputs(labels: np.ndarray, bmu: np.ndarray, num_neurons: int, num_classes: int) -> None:
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


def class_neuron_intersection(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int,
) -> np.ndarray:
    """Return class-by-neuron sample counts, using NNSOM's cal_class_cluster_intersect.

    Matrix element [c, n] is the number of samples from class c assigned
    to SOM neuron n.
    """

    labels = np.asarray(labels)
    bmu = np.asarray(bmu)
    _check_inputs(labels, bmu, num_neurons, num_classes)

    # NNSOM takes one index array per neuron and one per class, and returns
    # a (neurons x classes) matrix of float counts.
    clusters = [np.flatnonzero(bmu == neuron) for neuron in range(num_neurons)]
    class_indices = [np.flatnonzero(labels == c) for c in range(num_classes)]
    per_neuron = np.asarray(
        cal_class_cluster_intersect(clusters, *class_indices),
        dtype=np.float64,
    ).reshape(num_neurons, num_classes)

    return np.rint(per_neuron).astype(np.int64).T


def compute_cluster_metrics(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int,
) -> list[ClusterMetrics]:
    """Compute class composition, purity, and entropy for every SOM neuron."""

    counts_matrix = class_neuron_intersection(labels, bmu, num_neurons, num_classes)

    results: list[ClusterMetrics] = []

    for neuron_id in range(num_neurons):
        counts = counts_matrix[:, neuron_id]
        support = int(counts.sum())

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


def weighted_cluster_purity(metrics: list[ClusterMetrics]) -> float:
    """Return sample-weighted cluster purity over occupied neurons."""

    occupied = [m for m in metrics if m.support > 0]

    if not occupied:
        return float("nan")

    total = sum(m.support for m in occupied)
    correct_majority = sum(m.dominant_count for m in occupied)

    return float(correct_majority / total)


def weighted_mean_entropy(metrics: list[ClusterMetrics]) -> float:
    """Return sample-weighted mean normalized class entropy over occupied neurons."""

    occupied = [m for m in metrics if m.support > 0]

    if not occupied:
        return float("nan")

    support = np.asarray([m.support for m in occupied], dtype=np.float64)
    entropy = np.asarray([m.normalized_entropy for m in occupied], dtype=np.float64)

    return float(np.average(entropy, weights=support))


def weighted_purity_from_labels(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int,
) -> float:
    """Sample-weighted purity: share of images whose class is their neuron's majority class."""
    counts = np.bincount(
        np.asarray(bmu, dtype=np.int64) * num_classes + np.asarray(labels, dtype=np.int64),
        minlength=num_neurons * num_classes,
    ).reshape(num_neurons, num_classes)
    return float(counts.max(axis=1).sum() / len(labels))


def purity_label_shuffle(
    labels: np.ndarray,
    bmu: np.ndarray,
    num_neurons: int,
    num_classes: int,
    *,
    shuffles: int,
    seed: int,
) -> dict:
    """Compare the weighted purity with purities after shuffling the class labels.

    Each image keeps its neuron (BMU); only the labels are shuffled, so the null
    keeps the map's occupancy and the class sizes but removes any link between a
    neuron and the classes it holds. p-value: (1 + shuffles with purity >= the
    observed purity) / (1 + shuffles).
    """
    labels = np.asarray(labels, dtype=np.int64)
    bmu = np.asarray(bmu, dtype=np.int64)
    observed = weighted_purity_from_labels(labels, bmu, num_neurons, num_classes)
    rng = np.random.default_rng(seed)
    null = np.array([
        weighted_purity_from_labels(rng.permutation(labels), bmu, num_neurons, num_classes)
        for _ in range(shuffles)
    ])
    return {
        "method": "class labels shuffled across images; each image keeps its BMU",
        "shuffles": int(shuffles),
        "seed": int(seed),
        "observed_purity": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if shuffles > 1 else float("nan"),
        "null_95_range": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "null_max": float(null.max()),
        "p_value": float((1 + np.count_nonzero(null >= observed)) / (1 + shuffles)),
    }
