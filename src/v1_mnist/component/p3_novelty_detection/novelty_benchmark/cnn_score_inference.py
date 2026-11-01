"""Frozen CNN score reconstruction and validation for Phase 3."""

from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_logits(
    model: Any, features: np.ndarray
) -> np.ndarray:
    """Apply frozen LeNet-5 ReLU and classifier to saved fc2 activations."""
    import torch

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 2D fc2 features")

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy(values[start : start + 4096])
            logits = model.fc3(torch.relu(batch))
            batches.append(logits.cpu().numpy())

    if not batches:
        raise ValueError("Empty feature array")
    return np.concatenate(batches, axis=0)


def verify_predictions(
    logits: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    tolerance: float,
) -> None:
    """Confirm reconstructed logits agree with original CNN outputs."""
    from scipy.special import softmax

    predicted = np.argmax(logits, axis=1)
    probabilities = softmax(logits.astype(np.float64), axis=1)
    recovered_confidence = probabilities.max(axis=1)

    if predicted.shape != predictions.shape or not np.array_equal(
        predicted, predictions
    ):
        raise RuntimeError("CNN predictions disagree with saved outputs")

    if confidence.shape != recovered_confidence.shape:
        raise RuntimeError("CNN confidence array length mismatch")
    if not np.allclose(
        recovered_confidence, confidence, rtol=0, atol=tolerance
    ):
        raise RuntimeError("CNN confidence reconstruction failed")
