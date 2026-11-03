"""Evaluation and uncertainty for the exploratory unified novelty benchmark."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METHODS = (
    "qe", "u_matrix", "isolation_forest", "one_class_svm", "autoencoder",
    "max_softmax", "energy", "mahalanobis", "knn",
)


def subset_indices(size: int, budget: int, seed: int) -> np.ndarray:
    """Select the same training records for every newly fitted detector."""
    if not 0 < budget <= size:
        raise ValueError("Invalid training budget")
    return np.random.default_rng(seed).choice(size, budget, replace=False)


def subset_fingerprint(indices: np.ndarray) -> str:
    """Record exact chosen training rows without persisting embeddings."""
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def evaluate(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_target: float) -> dict[str, float]:
    """Evaluate OOD-positive scores with a consistent ROC operating point."""
    id_values = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_values = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if not (len(id_values) and len(ood_values)) or not 0 < tpr_target <= 1:
        raise ValueError("Invalid evaluation inputs")
    if not np.isfinite(id_values).all() or not np.isfinite(ood_values).all():
        raise ValueError("Nonfinite novelty scores")
    labels = np.r_[np.zeros(len(id_values), dtype=int), np.ones(len(ood_values), dtype=int)]
    scores = np.r_[id_values, ood_values]
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr[tpr >= tpr_target]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr_at_target_tpr": float(eligible.min()),
    }


def stratified_bootstrap(
    id_scores: np.ndarray, ood_scores: np.ndarray, target_tpr: float,
    repeats: int, seed: int,
) -> dict[str, list[float]]:
    """Bootstrap fixed-model uncertainty without altering training or tuning."""
    if repeats < 1:
        raise ValueError("At least one bootstrap resample is required")
    rng = np.random.default_rng(seed)
    id_values = np.asarray(id_scores)
    ood_values = np.asarray(ood_scores)
    keys = ("auroc", "aupr", "fpr_at_target_tpr")
    samples = {key: [] for key in keys}
    for _ in range(repeats):
        sample = evaluate(
            id_values[rng.integers(len(id_values), size=len(id_values))],
            ood_values[rng.integers(len(ood_values), size=len(ood_values))],
            target_tpr,
        )
        for key in keys:
            samples[key].append(sample[key])
    return {key: np.quantile(samples[key], [0.025, 0.975]).tolist() for key in keys}


def seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize variability of newly fitted detectors across independent seeds."""
    if len(rows) < 5:
        raise ValueError("At least five seeds are required")
    methods = rows[0]["metrics"]
    summary: dict[str, Any] = {}
    for name in methods:
        summary[name] = {}
        for key in ("auroc", "aupr", "fpr_at_target_tpr"):
            values = np.array([row["metrics"][name][key] for row in rows], dtype=float)
            summary[name][key] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "per_seed": values.tolist(),
            }
    return summary
