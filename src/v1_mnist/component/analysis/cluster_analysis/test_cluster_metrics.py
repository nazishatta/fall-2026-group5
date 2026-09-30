import numpy as np

from src.v1_mnist.component.cluster_analysis.cluster_metrics import (
    class_neuron_intersection,
    compute_cluster_metrics,
    normalized_entropy,
    weighted_cluster_purity,
)


def test_normalized_entropy_pure_cluster():
    counts = np.array([10, 0, 0, 0])
    assert np.isclose(normalized_entropy(counts), 0.0)


def test_normalized_entropy_uniform_cluster():
    counts = np.array([5, 5, 5, 5])
    assert np.isclose(normalized_entropy(counts), 1.0)


def test_compute_cluster_metrics():
    labels = np.array([0, 0, 1, 1, 1, 2])
    bmu = np.array([0, 0, 0, 1, 1, 2])

    metrics = compute_cluster_metrics(
        labels,
        bmu,
        num_neurons=4,
        num_classes=3,
    )

    assert len(metrics) == 4

    # Neuron 0: labels [0, 0, 1]
    assert metrics[0].support == 3
    assert metrics[0].dominant_class == 0
    assert metrics[0].dominant_count == 2
    assert np.isclose(metrics[0].purity, 2 / 3)
    assert metrics[0].class_counts == (2, 1, 0)

    # Neuron 1: labels [1, 1]
    assert metrics[1].support == 2
    assert metrics[1].dominant_class == 1
    assert np.isclose(metrics[1].purity, 1.0)

    # Neuron 2: label [2]
    assert metrics[2].support == 1
    assert metrics[2].dominant_class == 2
    assert np.isclose(metrics[2].purity, 1.0)

    # Neuron 3: empty
    assert metrics[3].support == 0
    assert metrics[3].dominant_class is None
    assert np.isnan(metrics[3].purity)
    assert np.isnan(metrics[3].normalized_entropy)


def test_class_neuron_intersection():
    labels = np.array([0, 0, 1, 1, 2])
    bmu = np.array([0, 1, 1, 2, 2])

    matrix = class_neuron_intersection(
        labels,
        bmu,
        num_neurons=3,
        num_classes=3,
    )

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )

    np.testing.assert_array_equal(matrix, expected)


def test_weighted_cluster_purity():
    labels = np.array([0, 0, 1, 1, 1, 2])
    bmu = np.array([0, 0, 0, 1, 1, 2])

    metrics = compute_cluster_metrics(
        labels,
        bmu,
        num_neurons=3,
        num_classes=3,
    )

    # Majority counts = 2 + 2 + 1 = 5 of 6 samples.
    assert np.isclose(weighted_cluster_purity(metrics), 5 / 6)


def test_invalid_bmu_rejected():
    labels = np.array([0, 1])
    bmu = np.array([0, 3])

    try:
        compute_cluster_metrics(
            labels,
            bmu,
            num_neurons=3,
            num_classes=2,
        )
    except ValueError as exc:
        assert "BMU index" in str(exc)
    else:
        raise AssertionError("Expected invalid BMU to raise ValueError.")
