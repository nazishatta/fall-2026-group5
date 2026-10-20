"""Focused tests for additive Phase 3 scaler isolation."""
import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.run_standardized_autoencoder import (
    scaled_embeddings,
)


def test_training_only_statistics() -> None:
    train = np.array([[0., 10.], [2., 12.], [4., 14.]], dtype=np.float32)
    val = np.array([[100., 200.]], dtype=np.float32)
    ood = np.array([[-100., -200.]], dtype=np.float32)
    fitted, evaluated = scaled_embeddings(train, val, ood)
    np.testing.assert_allclose(fitted.mean(axis=0), [0, 0], atol=1e-6)
    assert evaluated[0, 0] > 10  # validation value did not enter scaler fit
    assert evaluated[1, 0] < -10


def test_incompatible_dimensions_refused() -> None:
    with pytest.raises(ValueError, match="dimension"):
        scaled_embeddings(np.ones((3, 2)), np.ones((2, 3)), np.ones((4, 2)))


def test_nonfinite_refused() -> None:
    with pytest.raises(ValueError, match="Nonfinite"):
        scaled_embeddings(np.array([[np.nan, 0.]]), np.ones((2, 2)), np.ones((4, 2)))
