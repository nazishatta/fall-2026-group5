"""Dataset-independent OOD metrics; higher scores always indicate novelty."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _scores(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    """Validate a nonempty vector of finite novelty scores."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError(f"{name}: expected nonempty finite 1D scores")
    return data



def minimum_fpr_at_target_tpr(
    fpr: NDArray[np.floating],
    tpr: NDArray[np.floating],
    target_tpr: float,
) -> float:
    """Minimum observed FPR where the ROC reaches the target TPR."""
    false_rates = np.asarray(fpr, dtype=np.float64)
    true_rates = np.asarray(tpr, dtype=np.float64)
    if (
        false_rates.ndim != 1
        or true_rates.ndim != 1
        or false_rates.shape != true_rates.shape
        or not false_rates.size
        or not np.isfinite(false_rates).all()
        or not np.isfinite(true_rates).all()
        or not 0 < target_tpr <= 1
    ):
        raise ValueError("Invalid ROC curve or target TPR")
    qualified = false_rates[true_rates >= target_tpr]
    if not qualified.size:
        raise ValueError("ROC curve does not reach target TPR")
    return float(np.min(qualified))

def ood_metrics(
    id_scores: NDArray[np.floating],
    ood_scores: NDArray[np.floating],
    threshold: float,
    target_tpr: float = 0.95,
) -> dict[str, float | int]:
    """Compute AUROC, OOD-positive AUPR, FPR@TPR, and threshold rates."""
    id_values = _scores(id_scores, "ID")
    ood_values = _scores(ood_scores, "OOD")
    if not np.isfinite(threshold) or not 0 < target_tpr <= 1:
        raise ValueError("Invalid threshold or target TPR")
    labels = np.r_[
        np.zeros(id_values.size, dtype=np.int8), np.ones(ood_values.size, dtype=np.int8)
    ]
    scores = np.r_[id_values, ood_values]
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "id_count": int(id_values.size),
        "ood_count": int(ood_values.size),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_ood": float(average_precision_score(labels, scores)),
        "fpr_at_target_tpr": minimum_fpr_at_target_tpr(fpr, tpr, target_tpr),
        "target_tpr": float(target_tpr),
        "ood_detection_rate": float(np.mean(ood_values > threshold)),
        "id_false_positive_rate": float(np.mean(id_values > threshold)),
    }


def stratified_auroc_interval(
    id_scores: NDArray[np.floating],
    ood_scores: NDArray[np.floating],
    *,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a reproducible stratified percentile bootstrap AUROC interval."""
    ids = _scores(id_scores, "ID")
    oods = _scores(ood_scores, "OOD")
    if iterations < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    sample_labels = np.r_[
        np.zeros(ids.size, dtype=np.int8), np.ones(oods.size, dtype=np.int8)
    ]
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled_ids = ids[rng.integers(ids.size, size=ids.size)]
        sampled_oods = oods[rng.integers(oods.size, size=oods.size)]
        values[index] = roc_auc_score(sample_labels, np.r_[sampled_ids, sampled_oods])
    tail = 100.0 * (1.0 - confidence) / 2.0
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    return float(lower), float(upper)
