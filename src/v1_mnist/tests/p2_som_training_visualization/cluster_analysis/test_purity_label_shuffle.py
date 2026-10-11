"""Week 4: label-shuffle comparison for the sample-weighted cluster purity."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.cluster_analysis.cluster_metrics import (
    compute_cluster_metrics,
    purity_label_shuffle,
    weighted_cluster_purity,
    weighted_purity_from_labels,
)


def make_map(rng: np.random.Generator, pure: bool) -> tuple[np.ndarray, np.ndarray]:
    """20 neurons, 3 classes, 600 images; 'pure' maps put each class on its own neurons."""
    labels = rng.integers(0, 3, size=600)
    bmu = (labels * 6 + rng.integers(0, 6, size=600)) if pure else rng.integers(0, 20, size=600)
    return labels, bmu


def test_fast_purity_matches_nnsom_based_purity() -> None:
    labels, bmu = make_map(np.random.default_rng(1), pure=False)
    metrics = compute_cluster_metrics(labels, bmu, num_neurons=20, num_classes=3)
    assert np.isclose(weighted_purity_from_labels(labels, bmu, 20, 3), weighted_cluster_purity(metrics))


def test_structured_map_beats_the_shuffles() -> None:
    labels, bmu = make_map(np.random.default_rng(2), pure=True)
    result = purity_label_shuffle(labels, bmu, 20, 3, shuffles=200, seed=7)
    assert result["observed_purity"] == 1.0
    assert result["null_95_range"][1] < 0.6
    assert result["p_value"] == 1 / 201          # smallest possible with 200 shuffles


def test_random_map_does_not_beat_the_shuffles() -> None:
    labels, bmu = make_map(np.random.default_rng(3), pure=False)
    result = purity_label_shuffle(labels, bmu, 20, 3, shuffles=200, seed=7)
    assert result["p_value"] > 0.05


def test_same_seed_same_result() -> None:
    labels, bmu = make_map(np.random.default_rng(4), pure=False)
    a = purity_label_shuffle(labels, bmu, 20, 3, shuffles=50, seed=11)
    b = purity_label_shuffle(labels, bmu, 20, 3, shuffles=50, seed=11)
    assert a == b
