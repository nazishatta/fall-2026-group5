"""Synthetic-only checks; no actual test dataset or upstream artifact access."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)


def test_perfect_separation() -> None:
    result = ood_metrics(np.array([0.1, 0.2]), np.array([0.8, 0.9]), threshold=0.5, target_tpr=0.95)
    assert result["auroc"] == 1.0
    assert result["aupr_ood"] == 1.0
    assert result["fpr_at_target_tpr"] == 0.0
    assert result["ood_detection_rate"] == 1.0
    assert result["id_false_positive_rate"] == 0.0


def test_threshold_strictness() -> None:
    result = ood_metrics(np.array([0.5, 0.4]), np.array([0.5, 0.8]), threshold=0.5, target_tpr=0.95)
    assert result["id_false_positive_rate"] == 0.0
    assert result["ood_detection_rate"] == 0.5


def test_bootstrap_reproducibility() -> None:
    id_scores = np.array([0.1, 0.2, 0.4, 0.5])
    ood_scores = np.array([0.3, 0.6, 0.7, 0.8])
    first = stratified_auroc_interval(id_scores, ood_scores, iterations=30, seed=42, confidence=0.95)
    assert first == stratified_auroc_interval(
        id_scores, ood_scores, iterations=30, seed=42, confidence=0.95
    )
    assert 0 <= first[0] <= first[1] <= 1


@pytest.mark.parametrize(
    "values", [np.array([]), np.array([[1.0]]), np.array([np.nan])]
)
def test_reject_bad_scores(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ood_metrics(values, np.array([1.0]), threshold=0.5, target_tpr=0.95)
