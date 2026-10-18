"""Held-out OOD comparison; never fit detectors or calibrate on evaluation samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    minimum_fpr_at_target_tpr,
)

LOGGER = logging.getLogger(__name__)


def evaluate_scores(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    bootstrap_repeats: int,
    seed: int,
    target_tpr: float,
) -> dict[str, object]:
    """Evaluate higher-is-more-novel scores on held-out ID and OOD examples.

    The 95% TPR cutoff is evaluated on the combined labeled evaluation data;
    it is not a deployable training-calibrated novelty threshold.
    """
    normal = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    novel = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if not len(normal) or not len(novel):
        raise ValueError("Both ID and OOD samples are required for OOD metrics")
    if not np.isfinite(normal).all() or not np.isfinite(novel).all():
        raise ValueError("Score arrays must contain only finite values")
    if target_tpr != 0.95:
        raise ValueError("Historical fpr_at_95_tpr schema requires target_tpr=0.95")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")

    labels = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(novel), dtype=int)]
    scores = np.r_[normal, novel]
    auc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = minimum_fpr_at_target_tpr(fpr, tpr, target_tpr)
    rng = np.random.default_rng(seed)
    estimates: list[list[float]] = []
    for _ in range(bootstrap_repeats):
        b_normal = normal[rng.integers(len(normal), size=len(normal))]
        b_novel = novel[rng.integers(len(novel), size=len(novel))]
        b_labels = np.r_[np.zeros(len(b_normal), dtype=int), np.ones(len(b_novel), dtype=int)]
        b_scores = np.r_[b_normal, b_novel]
        b_fpr, b_tpr, _ = roc_curve(b_labels, b_scores)
        estimates.append([
            float(roc_auc_score(b_labels, b_scores)),
            float(average_precision_score(b_labels, b_scores)),
            minimum_fpr_at_target_tpr(b_fpr, b_tpr, target_tpr),
        ])
    intervals = (
        np.quantile(np.asarray(estimates), [0.025, 0.975], axis=0)
        if estimates else None
    )
    result: dict[str, object] = {
        "n_id": len(normal), "n_ood": len(novel), "seed": seed,
        "bootstrap_repeats": bootstrap_repeats,
        "target_tpr": target_tpr,
        "auroc": auc, "aupr": aupr, "fpr_at_95_tpr": fpr95,
        "confidence_intervals_95": (
            {name: [float(intervals[0, index]), float(intervals[1, index])]
             for index, name in enumerate(("auroc", "aupr", "fpr_at_95_tpr"))}
            if intervals is not None else None
        ),
        "metric_definition": "Positive class is OOD; higher scores indicate more novelty",
    }
    LOGGER.info("Evaluated %d ID and %d OOD samples", len(normal), len(novel))
    return result


def analyze_misses(
    ood_scores: np.ndarray,
    *,
    threshold: float,
    sample_ids: np.ndarray | None = None,
    max_examples: int = 20,
) -> dict[str, object]:
    """Report genuine OOD samples missed by a previously frozen threshold."""
    values = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    identifiers = np.arange(len(values)) if sample_ids is None else np.asarray(sample_ids)
    if len(identifiers) != len(values):
        raise ValueError("sample_ids length must equal score length")
    missed = np.flatnonzero(values <= threshold)
    ordered = missed[np.argsort(values[missed])][:max_examples]
    return {
        "n_ood": len(values), "threshold": float(threshold),
        "detected": int(np.sum(values > threshold)),
        "missed": len(missed),
        "detection_rate": float(np.mean(values > threshold)) if len(values) else None,
        "lowest_scoring_misses": [
            {"sample_id": str(identifiers[index]), "score": float(values[index])}
            for index in ordered
        ],
        "limitation": "Explains which samples were missed; causal reasons require review of those samples",
    }


def save_new_json(path: Path, payload: dict[str, object]) -> None:
    """Write a new small metrics JSON without overwriting any existing run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    LOGGER.info("Created %s", path)
