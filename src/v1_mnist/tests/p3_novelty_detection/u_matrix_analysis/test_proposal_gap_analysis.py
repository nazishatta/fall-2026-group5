"""Tests for Phase 3 descriptive work; no Phase 1/2 files modified."""

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.novelty_overlay import (
    aggregate_bmu_scores,
    save_novelty_overlay,
    summarize_qe_misses,
)


def test_aggregate_preserves_empty_neurons() -> None:
    means, counts = aggregate_bmu_scores(np.array([0, 0, 2]), np.array([1.0, 3.0, 7.0]), 4)
    assert np.allclose(means[[0, 2]], [2.0, 7.0])
    assert np.isnan(means[[1, 3]]).all()
    assert counts.tolist() == [2, 0, 1, 0]


def test_aggregate_rejects_bad_indices() -> None:
    with pytest.raises(ValueError):
        aggregate_bmu_scores(np.array([-1]), np.array([1.0]), 4)


def test_misses_use_strict_cutoff() -> None:
    result = summarize_qe_misses(np.array([0.5, 1.0, 1.5]), 1.0)
    assert result["missed"] == 2
    assert result["detected"] == 1
    assert result["missed_qe_median"] == 0.75


def test_pdf_and_no_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "pdf" / "overlay.pdf"
    result = save_novelty_overlay(np.array([0, 1]), np.array([0.5, 1.5]), 2, 2, destination)
    assert result["empty_neurons"] == 2
    assert destination.is_file()
    with pytest.raises(FileExistsError):
        save_novelty_overlay(np.array([0]), np.array([0.5]), 2, 2, destination)
