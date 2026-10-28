"""Synthetic tests for the SVHN embedding-norm diagnostic."""

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.svhn_inverted_norm_diagnostic import (
    compare_norms,
    norm_summary,
)


def test_norm_summary() -> None:
    """Known Euclidean norms are measured correctly."""
    result = norm_summary(np.array([[3.0, 4.0], [0.0, 5.0]]))
    assert result["count"] == 2
    assert result["mean"] == pytest.approx(5.0)


def test_paired_comparison() -> None:
    """Inverted samples preserve pairwise order."""
    normal = np.array([[3.0, 4.0], [0.0, 5.0]])
    inverted = normal * 2
    result = compare_norms(normal, inverted)
    assert result["paired_delta_mean"] == pytest.approx(5.0)
    assert result["inverted_greater_fraction"] == 1.0


def test_reject_nonfinite() -> None:
    """NaNs must never appear in reported statistics."""
    with pytest.raises(ValueError):
        norm_summary(np.array([[float("nan"), 1.0]]))


def test_reject_unaligned() -> None:
    """Different sample counts are not paired samples."""
    with pytest.raises(ValueError):
        compare_norms(np.ones((2, 3)), np.ones((3, 3)))
