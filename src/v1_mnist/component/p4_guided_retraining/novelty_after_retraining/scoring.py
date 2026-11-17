"""The five novelty scores of one cell and their metrics (protocol sections 3 and 4). NumPy only.

Higher scores mean "more novel". Every fitted quantity comes from the cell's training embeddings.
S2 to S4 are Phase 3's functions, unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
    max_softmax_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)

DETECTORS = ("qe", "msp", "energy", "mahalanobis")   # S1 to S4
DIAGNOSTIC = "length"                                  # D1
TARGET_TPR = 0.95


def relu_logits(features: np.ndarray, fc3_weight: np.ndarray, fc3_bias: np.ndarray) -> np.ndarray:
    """Logits from fc2 activations: the frozen LeNet-5 applies ReLU, then fc3 (as in Phase 3)."""
    return np.maximum(features, 0.0).astype(np.float64) @ fc3_weight.T.astype(np.float64) + fc3_bias


def score_split(features: np.ndarray, logits: np.ndarray, qe: np.ndarray, train_features: np.ndarray,
                train_labels: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    """S1 to S4 and D1 for one set of images."""
    if not (len(features) == len(logits) == len(qe)):
        raise ValueError("Features, logits and QE must describe the same images.")
    return {
        "qe": np.asarray(qe, dtype=np.float64),
        "msp": max_softmax_novelty(logits),
        "energy": energy_novelty(logits, temperature),
        "mahalanobis": mahalanobis_novelty(train_features, train_labels, features),
        DIAGNOSTIC: -np.linalg.norm(np.asarray(features, dtype=np.float64), axis=1),
    }


def cell_metrics(train: dict[str, np.ndarray], normal: dict[str, np.ndarray], novel: dict[str, np.ndarray],
                 percentile: float, resamples: int, seed: int, confidence: float) -> dict[str, Any]:
    """Per-score metrics, AUROC intervals, the paired QE comparisons and the length check."""
    out: dict[str, Any] = {"scores": {}, "qe_minus": {}}
    for name in (*DETECTORS, DIAGNOSTIC):
        threshold = float(np.percentile(train[name], percentile))   # training images only
        metrics = ood_metrics(normal[name], novel[name], threshold, TARGET_TPR)
        low, high = stratified_auroc_interval(normal[name], novel[name], iterations=resamples, seed=seed,
                                              confidence=confidence)
        out["scores"][name] = {**metrics, "threshold_from_training": threshold, "auroc_interval": [low, high]}
    for other in DETECTORS[1:]:
        paired = compare_auroc_paired(normal["qe"], novel["qe"], normal[other], novel[other],
                                      iterations=resamples, seed=seed)
        out["qe_minus"][other] = {"difference": paired["qe_minus_msp_auroc"],
                                  "interval": paired["difference_ci_95"]}
    both_qe = np.r_[normal["qe"], novel["qe"]]
    both_len = -np.r_[normal[DIAGNOSTIC], novel[DIAGNOSTIC]]
    out["qe_length_spearman"] = float(spearmanr(both_qe, both_len).statistic)
    return out


def competitive(qe_minus: dict[str, dict[str, Any]], scores: dict[str, dict[str, Any]], margin: float) -> bool:
    """Proposal RQ4 rule: QE minus the best of S2 to S4 has its interval's lower end above -margin."""
    best = max(DETECTORS[1:], key=lambda n: scores[n]["auroc"])
    return bool(qe_minus[best]["interval"][0] > -margin)
