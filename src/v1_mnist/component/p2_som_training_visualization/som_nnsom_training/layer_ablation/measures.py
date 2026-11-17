"""Measures A1 to A3 of one map, and the reported extras (protocol section 3). NumPy and scikit-learn only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.detection import (
    half_split_detection,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_measures import (
    hotspot_measures,
    neuron_majority,
    weighted_purity,
)


def error_cell_auroc(val_bmu: np.ndarray, errors: np.ndarray, num_neurons: int, repeats: int,
                     split_seed_base: int) -> dict[str, float]:
    """A1: the Phase 2 half-split error-cell AUROC (mean over the repeats, with its 2.5-97.5% range)."""
    scores = half_split_detection({"som_cell": [np.asarray(val_bmu)]}, {}, np.asarray(errors, dtype=bool),
                                  num_neurons, repeats, split_seed_base)["som_cell"].auroc
    return {"mean": float(scores.mean()), "low": float(np.percentile(scores, 2.5)),
            "high": float(np.percentile(scores, 97.5))}


def qe_novelty_auroc(normal_qe: np.ndarray, novel_qe: np.ndarray) -> float:
    """A3: AUROC of the QE with the novel images as the positive class."""
    y = np.r_[np.zeros(len(normal_qe)), np.ones(len(novel_qe))]
    return float(roc_auc_score(y, np.r_[normal_qe, novel_qe]))


def map_measures(val_bmu: np.ndarray, val_labels: np.ndarray, val_preds: np.ndarray, val_qe: np.ndarray,
                 svhn_qe: np.ndarray, num_neurons: int, num_classes: int, hotspot_fraction: float,
                 min_hotspots: int, repeats: int, split_seed_base: int) -> dict[str, Any]:
    """A1, A2, A3 and the reported extras for one map, on validation images (A3 also on SVHN)."""
    errors = (val_preds != val_labels)
    counts = neuron_majority(val_bmu, val_labels, num_neurons, num_classes)
    hot = hotspot_measures(val_bmu, errors.astype(float), num_neurons, hotspot_fraction, min_hotspots)
    return {
        "A1_error_cell_auroc": error_cell_auroc(val_bmu, errors, num_neurons, repeats, split_seed_base),
        "A2_weighted_purity": weighted_purity(counts),
        "A3_qe_novelty_auroc": qe_novelty_auroc(val_qe, svhn_qe),
        "hotspot_error_share": hot["hotspot_error_share"],
        "hotspot_error_rate": hot["hotspot_error_rate"],
        "hotspots": hot["hotspots"],
        "occupied_neurons": int((counts.sum(axis=1) > 0).sum()),
        "val_qe_mean": float(np.mean(val_qe)),
        "svhn_qe_mean": float(np.mean(svhn_qe)),
        "validation_errors": int(errors.sum()),
    }
