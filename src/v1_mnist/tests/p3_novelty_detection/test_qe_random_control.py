"""Deterministic tests for the Phase 3 QE random-selection control."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_random_control import (
    compute_control,
)

SEEDS = [11, 23, 37, 47, 59]


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Return aligned validation QE and CNN-correctness arrays."""
    distances = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3])
    correct = np.array([True, False, True, True, False, False])
    return distances, correct


def test_control_recomputes_counts() -> None:
    distances, correct = sample_data()

    result = compute_control(distances, correct, 0.5, SEEDS)

    assert result["validation_samples"] == 6
    assert result["qe_flagged_samples"] == 3
    assert result["cnn_errors"] == 3
    assert result["qe_captured_errors"] == 2
    assert result["qe_error_capture_rate"] == pytest.approx(2 / 3)
    assert result["random_expected_errors"] == pytest.approx(1.5)
    assert result["leakage_control"]["final_test_used"] is False


def test_control_reproducible_for_same_seeds() -> None:
    distances, correct = sample_data()

    first = compute_control(distances, correct, 0.5, SEEDS)
    second = compute_control(distances, correct, 0.5, SEEDS)

    assert first["random_seeded_control"] == second["random_seeded_control"]
    assert len(first["random_seeded_control"]["captured_errors"]) == 5


def test_control_rejects_misaligned_correctness() -> None:
    distances, _ = sample_data()

    with pytest.raises(ValueError, match="aligned boolean"):
        compute_control(
            distances,
            np.array([True, False]),
            0.5,
            SEEDS,
        )


def test_control_requires_five_unique_seeds() -> None:
    distances, correct = sample_data()

    with pytest.raises(ValueError, match="five unique"):
        compute_control(
            distances,
            correct,
            0.5,
            [11, 11, 23, 37, 47],
        )
