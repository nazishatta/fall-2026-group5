"""Metrics of one split from its score arrays (protocol sections 5 and 6). NumPy and scikit-learn only.

S1 to S4 and D1 come from the novelty-after-retraining scoring functions (which call Phase 3's functions);
S5 (nearest-neighbour distance) and the label-shuffle control are added here with Phase 3's functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import knn_novelty
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.qe_threshold_sweep import (
    permutation_auroc_null,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.scoring import (
    TARGET_TPR,
    cell_metrics,
    competitive,
    score_split,
)

KNN = "knn"
SCORES = ("qe", "msp", "energy", "mahalanobis", KNN)   # S1 to S5


def all_scores(features: dict[str, np.ndarray], logits: dict[str, np.ndarray], qe: dict[str, np.ndarray],
               train_labels: np.ndarray, temperature: float, neighbors: int) -> dict[str, dict[str, np.ndarray]]:
    """S1 to S5 and D1 for the 'normal' and 'novel' images, and S1 to S4 and D1 for 'train' (for thresholds)."""
    train_x = features["train"]
    out = {k: score_split(features[k], logits[k], qe[k], train_x, train_labels, temperature)
           for k in ("train", "normal", "novel")}
    for k in ("normal", "novel"):
        out[k][KNN] = knn_novelty(train_x, features[k], neighbors=neighbors)
    return out


def split_metrics(scores: dict[str, dict[str, np.ndarray]], percentile: float, resamples: int, seed: int,
                  confidence: float, margin: float, null_iterations: int, null_seed: int) -> dict[str, Any]:
    """Every per-split number of the protocol: AUROC and intervals, QE minus each score, control, length."""
    out = cell_metrics(scores["train"], scores["normal"], scores["novel"], percentile, resamples, seed, confidence)
    normal, novel = scores["normal"], scores["novel"]
    # S5 has no training-score threshold (a training image is its own nearest neighbour), so the
    # threshold-based rates are not reported for it.
    knn = ood_metrics(normal[KNN], novel[KNN], float(np.max(normal[KNN])) + 1.0, TARGET_TPR)
    low, high = stratified_auroc_interval(normal[KNN], novel[KNN], iterations=resamples, seed=seed,
                                          confidence=confidence)
    knn.pop("ood_detection_rate", None)
    knn.pop("id_false_positive_rate", None)
    out["scores"][KNN] = {**knn, "auroc_interval": [low, high]}
    paired = compare_auroc_paired(normal["qe"], novel["qe"], normal[KNN], novel[KNN],
                                  iterations=resamples, seed=seed)
    out["qe_minus"][KNN] = {"difference": paired["qe_minus_msp_auroc"], "interval": paired["difference_ci_95"]}
    out["competitive_rq4"] = competitive(out["qe_minus"], out["scores"], margin)
    out["qe_label_shuffle_control"] = permutation_auroc_null(normal["qe"], novel["qe"], iterations=null_iterations,
                                                             seed=null_seed)
    return out


def per_digit_qe_auroc(normal_qe: np.ndarray, novel_qe: np.ndarray, novel_digits: np.ndarray) -> dict[str, float]:
    """QE AUROC of each unknown digit against all normal images (descriptive: which new digits are easy)."""
    out = {}
    for d in np.unique(novel_digits):
        nov = novel_qe[novel_digits == d]
        labels = np.r_[np.zeros(len(normal_qe)), np.ones(len(nov))]
        out[str(int(d))] = float(roc_auc_score(labels, np.r_[normal_qe, nov]))
    return out
