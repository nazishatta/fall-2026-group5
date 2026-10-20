"""Tests for U-Matrix novelty calculations."""

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
    neuron_u_matrix,
    sample_u_matrix,
)


def test_u_matrix() -> None:
    weights = np.array([[0.0], [2.0], [5.0]])
    topology = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ])
    result = neuron_u_matrix(weights, topology)
    np.testing.assert_allclose(result, [2.0, 2.5, 3.0])


def test_sample_scores() -> None:
    scores = np.array([2.0, 2.5, 3.0])
    result = sample_u_matrix(scores, np.array([2, 0, 1]))
    np.testing.assert_allclose(result, [3.0, 2.0, 2.5])


def test_invalid_bmu() -> None:
    with pytest.raises(ValueError):
        sample_u_matrix(np.array([1.0, 2.0]), np.array([0, 2]))
