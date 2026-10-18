from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.partitions import (
    fit_kmeans_cells,
    knn_impurity,
    scale_features,
)


def test_scaler_is_fitted_on_training_only() -> None:
    train = np.array([[0.0], [10.0]])
    val = np.array([[20.0]])
    train_scaled, val_scaled = scale_features(train, val, (-1.0, 1.0))
    assert train_scaled.min() == -1.0 and train_scaled.max() == 1.0
    assert val_scaled[0, 0] == 3.0  # outside the training range: the scaler did not see it


def test_knn_impurity_is_zero_for_pure_neighbours_and_positive_for_mixed() -> None:
    train = np.array([[0.0], [0.1], [0.2], [4.9], [5.1], [10.0]])
    labels = np.array([0, 0, 0, 0, 1, 1])
    val = np.array([[0.05], [5.0]])
    impurity = knn_impurity(train, labels, val, neighbors=3, num_classes=2)
    assert impurity[0] == 0.0
    assert np.isclose(impurity[1], 1 / 3)  # neighbours: 4.9 (0), 5.1 (1), 0.2 (0)

from src.v1_mnist.component.utils.scores import max_softmax_score


def test_max_softmax_score_is_one_minus_confidence() -> None:
    assert np.allclose(max_softmax_score(np.array([1.0, 0.25])), [0.0, 0.75])


def test_kmeans_cells_are_in_range_and_repeatable() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 4))
    val = rng.normal(size=(50, 4))
    first, inertia = fit_kmeans_cells(train, val, num_clusters=6, seed=3, n_init=2)
    second, _ = fit_kmeans_cells(train, val, num_clusters=6, seed=3, n_init=2)
    assert first.shape == (50,) and first.min() >= 0 and first.max() < 6
    assert np.array_equal(first, second)
    assert inertia > 0
