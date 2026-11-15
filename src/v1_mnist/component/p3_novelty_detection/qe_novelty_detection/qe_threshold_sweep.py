"""Threshold sweep for the QE novelty score: operating points and a threshold-free AUROC.

AUROC does not depend on the cutoff. It ranks every image by its QE, so it is the
same at the 95th and the 99th percentile. The cutoff only decides the *operating
point*: how many MNIST digits raise a false alarm and how many SVHN images are
caught. This module reports both, with Wilson intervals for the rates, a
bootstrap interval for AUROC and a label-permutation null.

Pure numpy/scikit-learn, so it runs without PyTorch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.quantization_novelty import (
    calculate_threshold,
)
from src.v1_mnist.component.utils.stats import wilson_interval


def check_percentiles(percentiles: list[float]) -> list[float]:
    """Return the percentiles as floats; they must be unique and strictly between 0 and 100."""
    values = [float(value) for value in percentiles]
    if not values or len(set(values)) != len(values):
        raise ValueError("Percentiles must be a nonempty list of unique values")
    if any(not 0 < value < 100 for value in values):
        raise ValueError("Percentiles must be strictly between 0 and 100")
    return values


def check_scores(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    """Validate a nonempty vector of finite, nonnegative QE values."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"{name}: expected nonempty finite nonnegative 1D QE values")
    return data


def threshold_sweep(
    train_qe: NDArray[np.floating],
    id_qe: NDArray[np.floating],
    ood_qe: NDArray[np.floating],
    percentiles: list[float],
) -> list[dict[str, float | int]]:
    """One row per percentile. Cutoffs come from the training QE only.

    An image is flagged as novel when its QE is strictly above the cutoff.
    """
    train = check_scores(train_qe, "training")
    ids = check_scores(id_qe, "ID")
    oods = check_scores(ood_qe, "OOD")
    rows: list[dict[str, float | int]] = []
    for percentile in check_percentiles(percentiles):
        cutoff = calculate_threshold(train, percentile)
        id_flagged = int((ids > cutoff).sum())
        ood_flagged = int((oods > cutoff).sum())
        id_low, id_high = wilson_interval(id_flagged, ids.size)
        ood_low, ood_high = wilson_interval(ood_flagged, oods.size)
        rows.append(
            {
                "percentile": percentile,
                "threshold": cutoff,
                "training_flag_rate": float((train > cutoff).mean()),
                "id_false_alarm_rate": id_flagged / ids.size,
                "id_false_alarm_low": id_low,
                "id_false_alarm_high": id_high,
                "ood_detection_rate": ood_flagged / oods.size,
                "ood_detection_low": ood_low,
                "ood_detection_high": ood_high,
                "ood_missed": int(oods.size - ood_flagged),
                "detection_minus_false_alarm": ood_flagged / oods.size - id_flagged / ids.size,
            }
        )
    return rows


def permutation_auroc_null(
    id_qe: NDArray[np.floating],
    ood_qe: NDArray[np.floating],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    """AUROC when the ID/OOD labels are shuffled (the control: it should sit near 0.5).

    p_value is the share of shuffles, counting the real labelling once, whose AUROC is
    at least as large as the observed one. The smallest possible value is 1 / (iterations + 1).
    """
    ids = check_scores(id_qe, "ID")
    oods = check_scores(ood_qe, "OOD")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    pooled = np.r_[ids, oods]
    labels = np.r_[np.zeros(ids.size, dtype=np.int8), np.ones(oods.size, dtype=np.int8)]
    observed = float(roc_auc_score(labels, pooled))
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        values[index] = roc_auc_score(rng.permutation(labels), pooled)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "observed_auroc": observed,
        "shuffled_mean": float(values.mean()),
        "shuffled_low": float(low),
        "shuffled_high": float(high),
        "shuffled_max": float(values.max()),
        "iterations": iterations,
        "p_value": float((1 + (values >= observed).sum()) / (iterations + 1)),
    }
