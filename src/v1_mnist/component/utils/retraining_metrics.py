"""Metrics of one retrained CNN (plain NumPy, so they work for any dataset with N classes).

Hotspot images are the images whose neuron in the ORIGINAL SOM is one of the error hotspots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.v1_mnist.component.utils.stats import wilson_interval


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix: rows are true digits, columns are predicted digits."""
    flat = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _rate(errors: np.ndarray) -> dict[str, Any]:
    n = int(errors.size)
    k = int(errors.sum())
    low, high = wilson_interval(n - k, n) if n else (float("nan"), float("nan"))
    return {
        "images": n,
        "errors": k,
        "error_rate": (k / n) if n else float("nan"),
        # Wilson interval of the ERROR rate: 1 - (interval of the accuracy).
        "error_wilson_95": [1.0 - high, 1.0 - low],
    }


def evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, in_hotspot: np.ndarray, num_classes: int
) -> dict[str, Any]:
    """Overall, hotspot, non-hotspot and per-digit error of one model on one split."""
    wrong = y_true != y_pred
    per_class = [
        float(wrong[y_true == c].mean()) if (y_true == c).any() else float("nan")
        for c in range(num_classes)
    ]
    return {
        "overall": _rate(wrong),
        "hotspot": _rate(wrong[in_hotspot]),
        "non_hotspot": _rate(wrong[~in_hotspot]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))),
        "per_class_error": per_class,
    }
