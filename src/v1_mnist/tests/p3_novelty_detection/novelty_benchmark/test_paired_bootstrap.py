"""Tests for paired QE-versus-MSP bootstrap comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)


def test_identical_methods_have_zero_difference() -> None:
    qe_id = np.array([0.1, 0.3, 0.4, 0.2])
    qe_ood = np.array([0.5, 0.9, 0.4, 0.8])

    result = compare_auroc_paired(
        qe_id, qe_ood, qe_id, qe_ood, iterations=100, seed=42
    )

    assert result["qe_minus_msp_auroc"] == 0.0
    assert result["difference_ci_95"] == [0.0, 0.0]


def test_paired_comparison_is_deterministic() -> None:
    qe_id = np.array([0.1, 0.5, 0.3, 0.4])
    qe_ood = np.array([0.6, 0.2, 0.8, 0.9])
    msp_id = np.array([0.2, 0.3, 0.1, 0.5])
    msp_ood = np.array([0.7, 0.4, 0.6, 0.8])

    first = compare_auroc_paired(
        qe_id, qe_ood, msp_id, msp_ood,
        iterations=100, seed=2026,
    )
    second = compare_auroc_paired(
        qe_id, qe_ood, msp_id, msp_ood,
        iterations=100, seed=2026,
    )

    assert first == second


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="aligned"):
        compare_auroc_paired(
            np.array([0.1, 0.2]),
            np.array([0.8]),
            np.array([0.1]),
            np.array([0.8]),
            iterations=10,
            seed=42,
        )


def test_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        compare_auroc_paired(
            np.array([np.nan]),
            np.array([0.8]),
            np.array([0.1]),
            np.array([0.8]),
            iterations=10,
            seed=42,
        )


def test_rejects_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        compare_auroc_paired(
            np.array([0.1]),
            np.array([0.8]),
            np.array([0.1]),
            np.array([0.8]),
            iterations=0,
            seed=42,
        )
