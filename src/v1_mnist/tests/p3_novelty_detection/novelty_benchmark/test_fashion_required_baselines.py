"""Lightweight regression tests for evidence checks (no image download)."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.verify_fashion_required_baselines import (
    calculate_metrics,
    verify_metrics,
)


def test_score_direction_and_bootstrap_repeatability() -> None:
    """OOD points with larger scores should have an AUROC of 1."""
    ids = np.array([0.1, 0.2, 0.3])
    oods = np.array([0.7, 0.8, 0.9])
    first = calculate_metrics(ids, oods, samples=30, seed=2026, tpr=0.95)
    second = calculate_metrics(ids, oods, samples=30, seed=2026, tpr=0.95)
    assert first == second
    assert first["auroc"] == 1.0


def test_reproduction_rejects_changed_metric() -> None:
    metrics = {
        "auroc": 0.9, "auroc_ci95": [0.8, 0.95],
        "aupr_ood": 0.85, "fpr_at_95_tpr": 0.2,
    }
    verify_metrics(metrics, dict(metrics))
    with pytest.raises(RuntimeError, match="Reproduction mismatch"):
        verify_metrics(metrics, {**metrics, "auroc": 0.5})


def test_nonfinite_scores_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_metrics(
            np.array([np.nan]), np.array([0.4]), samples=5, seed=1, tpr=0.95
        )
