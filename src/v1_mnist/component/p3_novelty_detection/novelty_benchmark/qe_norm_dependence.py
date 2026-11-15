"""How much of the QE novelty signal is an embedding-norm effect?

Pure NumPy and SciPy functions; ``run_qe_norm_dependence.py`` loads the data.
The norm-only score itself is the length-only control already saved by the
autoencoder scaling diagnostic; this module adds what that run does not answer:
how closely QE follows the norm, and whether QE still separates SVHN from MNIST
among images whose norm is ordinary. Every reference value comes from MNIST
training embeddings only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)


def embedding_norms(features: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return the L2 norm of every row of a finite, non-empty 2-D embedding matrix."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected a finite, non-empty 2-D embedding matrix")
    return np.linalg.norm(values, axis=1)


def norm_summary(norms: NDArray[np.floating]) -> dict[str, float | int]:
    """Count, mean, median, sample SD and 5th/95th percentiles of a norm vector."""
    values = np.asarray(norms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def spearman(first: NDArray[np.floating], second: NDArray[np.floating]) -> float:
    """Spearman rank correlation of two aligned score vectors."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError("Spearman needs two aligned vectors of at least three values")
    return float(spearmanr(a, b).statistic)


def norm_matched_qe(
    train_norms: NDArray[np.floating],
    id_norms: NDArray[np.floating],
    ood_norms: NDArray[np.floating],
    id_qe: NDArray[np.floating],
    ood_qe: NDArray[np.floating],
    *,
    band: Sequence[float],
    qe_threshold: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """QE metrics using only images whose norm lies inside a training-norm band.

    If QE still separates SVHN from MNIST when both sets have ordinary norms, its
    signal is not only an embedding-length effect.
    """
    low, high = (
        float(v) for v in np.percentile(np.asarray(train_norms, dtype=np.float64), band)
    )
    id_keep = (id_norms >= low) & (id_norms <= high)
    ood_keep = (ood_norms >= low) & (ood_norms <= high)
    result: dict[str, Any] = {
        "band_percentiles": [float(band[0]), float(band[1])],
        "band_norms": [low, high],
        "id_count": int(id_keep.sum()),
        "ood_count": int(ood_keep.sum()),
    }
    if result["id_count"] < 2 or result["ood_count"] < 2:
        result["qe"] = None
        result["note"] = "Too few images inside the band to compute an AUROC."
        return result
    metrics: dict[str, Any] = dict(
        ood_metrics(
            id_qe[id_keep],
            ood_qe[ood_keep],
            qe_threshold,
            float(settings["target_tpr"]),
        )
    )
    metrics["auroc_ci"] = list(
        stratified_auroc_interval(
            id_qe[id_keep],
            ood_qe[ood_keep],
            iterations=int(settings["bootstrap_iterations"]),
            seed=int(settings["seed"]),
            confidence=float(settings["confidence"]),
        )
    )
    result["qe"] = metrics
    return result


def analyze(
    train: NDArray[np.floating],
    validation: NDArray[np.floating],
    svhn: NDArray[np.floating],
    validation_qe: NDArray[np.floating],
    svhn_qe: NDArray[np.floating],
    *,
    qe_threshold: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Norm summaries, the QE reference, QE-norm correlations and the norm-band check."""
    train_norms = embedding_norms(train)
    id_norms = embedding_norms(validation)
    ood_norms = embedding_norms(svhn)
    id_qe = np.asarray(validation_qe, dtype=np.float64)
    ood_qe = np.asarray(svhn_qe, dtype=np.float64)
    if id_qe.shape != id_norms.shape or ood_qe.shape != ood_norms.shape:
        raise ValueError("QE scores and embeddings are not aligned")
    qe_reference: dict[str, Any] = dict(
        ood_metrics(id_qe, ood_qe, qe_threshold, float(settings["target_tpr"]))
    )
    qe_reference["auroc_ci"] = list(
        stratified_auroc_interval(
            id_qe,
            ood_qe,
            iterations=int(settings["bootstrap_iterations"]),
            seed=int(settings["seed"]),
            confidence=float(settings["confidence"]),
        )
    )
    qe_reference["threshold"] = float(qe_threshold)
    qe_reference["threshold_source"] = "frozen MNIST training-QE 95th percentile"
    return {
        "norms": {
            "mnist_train": norm_summary(train_norms),
            "mnist_validation": norm_summary(id_norms),
            "svhn_test": norm_summary(ood_norms),
        },
        "qe_reference": qe_reference,
        "qe_norm_spearman": {
            "mnist_validation": spearman(id_qe, id_norms),
            "svhn_test": spearman(ood_qe, ood_norms),
            "pooled": spearman(np.r_[id_qe, ood_qe], np.r_[id_norms, ood_norms]),
        },
        "norm_matched_qe": norm_matched_qe(
            train_norms,
            id_norms,
            ood_norms,
            id_qe,
            ood_qe,
            band=settings["norm_matched_band"],
            qe_threshold=qe_threshold,
            settings=settings,
        ),
    }
