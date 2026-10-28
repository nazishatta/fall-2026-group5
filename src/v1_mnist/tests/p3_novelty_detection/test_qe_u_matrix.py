"""Tests for Phase 3 validation-only U-Matrix reproduction."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_u_matrix import (
    evaluate_validation,
)


def sample_som() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small SOM with known weights, topology and sample assignments."""
    weights = np.array([[0.0], [1.0], [3.0], [6.0]])
    topology = np.abs(
        np.arange(4)[:, None] - np.arange(4)[None, :]
    ).astype(float)
    bmu = np.array([0, 1, 2, 3], dtype=np.int64)
    qe = np.array([0.1, 0.3, 0.6, 0.9])
    return weights, topology, bmu, qe


def test_u_matrix_validation_statistics() -> None:
    """Use real U-Matrix formulas rather than hard-coded reported metrics."""
    weights, topology, bmu, qe = sample_som()

    result = evaluate_validation(weights, topology, bmu, qe)

    # Neighbor distances: [1, (1+2)/2, (2+3)/2, 3].
    assert result["som_neurons"] == 4
    assert result["validation_samples"] == 4
    assert result["qe_mean"] == pytest.approx(0.475)
    assert result["u_matrix_mean"] == pytest.approx(2.0)
    assert result["spearman_correlation"] == pytest.approx(1.0)
    assert result["ood_evaluation"] == "not_performed"


def test_u_matrix_rejects_misaligned_samples() -> None:
    weights, topology, _, qe = sample_som()

    with pytest.raises(ValueError, match="aligned"):
        evaluate_validation(
            weights,
            topology,
            np.array([0, 1]),
            qe,
        )


def test_u_matrix_rejects_invalid_bmu() -> None:
    weights, topology, bmu, qe = sample_som()
    bmu[2] = 4

    with pytest.raises(ValueError, match="outside SOM"):
        evaluate_validation(weights, topology, bmu, qe)


def test_u_matrix_rejects_invalid_qe() -> None:
    weights, topology, bmu, qe = sample_som()
    qe[1] = np.nan

    with pytest.raises(ValueError, match="invalid distances"):
        evaluate_validation(weights, topology, bmu, qe)


def test_u_matrix_rejects_constant_qe() -> None:
    weights, topology, bmu, _ = sample_som()
    qe = np.ones(4)

    with pytest.raises(ValueError, match="undefined"):
        evaluate_validation(weights, topology, bmu, qe)
