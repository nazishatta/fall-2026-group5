"""Paired stratified bootstrap for comparing two OOD AUROCs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


def compare_auroc_paired(
    qe_id: NDArray[np.floating],
    qe_ood: NDArray[np.floating],
    msp_id: NDArray[np.floating],
    msp_ood: NDArray[np.floating],
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Estimate QE-minus-MSP AUROC with aligned stratified resampling."""
    arrays = tuple(
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (qe_id, qe_ood, msp_id, msp_ood)
    )

    q_id, q_ood, m_id, m_ood = arrays
    if not all(len(a) for a in arrays):
        raise ValueError("Every score array must be nonempty")
    if len(q_id) != len(m_id) or len(q_ood) != len(m_ood):
        raise ValueError("QE and MSP score arrays must be aligned")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("All scores must be finite")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")

    labels = np.r_[
        np.zeros(len(q_id), dtype=int),
        np.ones(len(q_ood), dtype=int),
    ]

    def auc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        """Calculate OOD-positive AUROC."""
        return float(
            roc_auc_score(labels, np.r_[id_scores, ood_scores])
        )

    qe_auc = auc(q_id, q_ood)
    msp_auc = auc(m_id, m_ood)

    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)

    for index in range(iterations):
        id_indices = rng.integers(len(q_id), size=len(q_id))
        ood_indices = rng.integers(len(q_ood), size=len(q_ood))

        qe_sample = auc(q_id[id_indices], q_ood[ood_indices])
        msp_sample = auc(m_id[id_indices], m_ood[ood_indices])
        differences[index] = qe_sample - msp_sample

    low, high = np.quantile(differences, [0.025, 0.975])

    return {
        "qe_auroc": qe_auc,
        "msp_auroc": msp_auc,
        "qe_minus_msp_auroc": qe_auc - msp_auc,
        "difference_ci_95": [float(low), float(high)],
        "bootstrap_iterations": iterations,
        "seed": seed,
        "n_id": len(q_id),
        "n_ood": len(q_ood),
        "positive_class": "OOD",
        "limitation": (
            "Paired validity requires identically ordered QE and MSP "
            "scores for each evaluation split."
        ),
    }
