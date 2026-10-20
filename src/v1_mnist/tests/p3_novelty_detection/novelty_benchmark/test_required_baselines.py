"""Tests for the three required frozen-CNN novelty baselines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
    max_softmax_novelty,
)


def test_max_softmax_matches_probability_definition() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 0.0]])

    actual = max_softmax_novelty(logits)
    expected = 1.0 - softmax(logits, axis=1).max(axis=1)

    np.testing.assert_allclose(actual, expected)
    assert actual[0] < actual[1]


def test_energy_matches_definition() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 0.0]])

    actual = energy_novelty(logits, temperature=1.0)
    expected = -logsumexp(logits, axis=1)

    np.testing.assert_allclose(actual, expected)
    assert actual[0] < actual[1]


def test_energy_temperature_is_explicit() -> None:
    logits = np.array([[2.0, 1.0]])

    actual = energy_novelty(logits, temperature=2.0)
    expected = -2.0 * logsumexp(logits / 2.0, axis=1)

    np.testing.assert_allclose(actual, expected)


def test_logits_reject_invalid_values() -> None:
    invalid = np.array([[np.nan, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        max_softmax_novelty(invalid)

    with pytest.raises(ValueError, match="finite"):
        energy_novelty(invalid, temperature=1.0)

    with pytest.raises(ValueError, match="Temperature"):
        energy_novelty(np.array([[1.0, 2.0]]), temperature=0.0)


def test_mahalanobis_scores_far_samples_higher() -> None:
    train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [3.0, 3.0],
        [3.1, 3.0],
        [3.0, 3.1],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])

    evaluation = np.array([
        [0.03, 0.03],
        [20.0, 20.0],
    ])

    scores = mahalanobis_novelty(train, labels, evaluation)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    assert scores[0] < scores[1]


def test_mahalanobis_rejects_invalid_labels() -> None:
    train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [3.0, 3.0],
        [3.1, 3.0],
    ])
    evaluation = np.array([[1.0, 1.0]])

    with pytest.raises(ValueError, match="aligned integers"):
        mahalanobis_novelty(
            train,
            np.array([0, 1]),
            evaluation,
        )


def test_mahalanobis_rejects_nonfinite_features() -> None:
    train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [3.0, 3.0],
        [3.1, 3.0],
    ])
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="finite"):
        mahalanobis_novelty(
            train,
            labels,
            np.array([[np.nan, 1.0]]),
        )
