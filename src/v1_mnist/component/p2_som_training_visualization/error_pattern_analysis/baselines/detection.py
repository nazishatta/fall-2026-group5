"""How well does each score tell the CNN's mistakes from its correct predictions? (AUROC, average precision)

Cell scores (SOM cell, k-means cell) are an error rate learned on one random half of the validation
images and used on the other half, in both directions. Scores that need no fitting (kNN impurity,
max-softmax) are scored on the same halves, so all methods see the same images. Uncertainty:
the spread over the repeated splits, and a bootstrap interval for the scores that need no fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.error_pattern_robustness import (
    stratified_halves,
)

LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5


@dataclass(frozen=True)
class MethodScores:
    """Per-repeat AUROC and average precision of one method (each averaged over both directions)."""

    auroc: np.ndarray
    average_precision: np.ndarray
    auroc_per_partition: np.ndarray


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean and the 2.5 to 97.5 percentile range of the values."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "lower": float(np.percentile(values, LOWER_PERCENTILE)),
        "upper": float(np.percentile(values, UPPER_PERCENTILE)),
    }


def cell_error_scores(
    cells: np.ndarray, errors: np.ndarray, fit_idx: np.ndarray, test_idx: np.ndarray, num_cells: int
) -> np.ndarray:
    """Error rate of each test image's cell, learned on the fit images (overall rate for unseen cells)."""
    n = np.bincount(cells[fit_idx], minlength=num_cells)
    e = np.bincount(cells[fit_idx], weights=errors[fit_idx], minlength=num_cells)
    rate = np.where(n > 0, e / np.maximum(n, 1), errors[fit_idx].mean())
    return rate[cells[test_idx]]


def half_split_detection(
    partitions: Mapping[str, Sequence[np.ndarray]],
    fixed_scores: Mapping[str, np.ndarray],
    errors: np.ndarray,
    num_cells: int,
    repeats: int,
    split_seed_base: int,
) -> dict[str, MethodScores]:
    """Score every method on the same random stratified half-splits.

    ``partitions`` maps a method to one or more cell assignments (several k-means seeds are averaged);
    ``fixed_scores`` maps a method to a score per validation image that needs no fitting.
    """
    err = errors.astype(float)
    names = list(partitions) + list(fixed_scores)
    auc = {name: [] for name in names}
    ap = {name: [] for name in names}
    per_part = {name: [] for name in partitions}

    for rep in range(repeats):
        half_a, half_b = stratified_halves(errors, rep, split_seed_base)
        direction_auc = {name: [] for name in names}
        direction_ap = {name: [] for name in names}
        direction_part = {name: [] for name in partitions}
        for fit, test in ((half_a, half_b), (half_b, half_a)):
            truth = err[test]
            for name, cell_sets in partitions.items():
                part_auc = []
                part_ap = []
                for cells in cell_sets:
                    score = cell_error_scores(cells, err, fit, test, num_cells)
                    part_auc.append(roc_auc_score(truth, score))
                    part_ap.append(average_precision_score(truth, score))
                direction_auc[name].append(np.mean(part_auc))
                direction_ap[name].append(np.mean(part_ap))
                direction_part[name].append(part_auc)
            for name, score in fixed_scores.items():
                direction_auc[name].append(roc_auc_score(truth, score[test]))
                direction_ap[name].append(average_precision_score(truth, score[test]))
        for name in names:
            auc[name].append(np.mean(direction_auc[name]))
            ap[name].append(np.mean(direction_ap[name]))
        for name in partitions:
            per_part[name].append(np.mean(direction_part[name], axis=0))

    results: dict[str, MethodScores] = {}
    for name in names:
        per_partition = np.asarray(per_part[name]).T if name in per_part else np.asarray([auc[name]])
        results[name] = MethodScores(np.asarray(auc[name]), np.asarray(ap[name]), per_partition)
    return results


def bootstrap_auroc(
    errors: np.ndarray, scores: Mapping[str, np.ndarray], resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    """AUROC on all validation images with a bootstrap interval (same resamples for every score)."""
    err = errors.astype(float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in scores}
    for _ in range(resamples):
        idx = rng.integers(0, err.size, err.size)
        if err[idx].sum() == 0:
            continue
        for name, score in scores.items():
            draws[name].append(roc_auc_score(err[idx], score[idx]))
    out: dict[str, dict[str, float]] = {}
    for name, score in scores.items():
        interval = summarize(np.asarray(draws[name]))
        out[name] = {"point": float(roc_auc_score(err, score)), "lower": interval["lower"], "upper": interval["upper"]}
    return out
