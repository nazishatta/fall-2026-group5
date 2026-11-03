"""Pure QE tests that do not need DVC artifacts or final test data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.quantization_novelty import (
    calculate_threshold,
    error_association,
    summarize,
)


def test_train_only_calibration() -> None:
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    val = np.array([0.5, 2.5, 3.8, 3.9, 5.0])
    assert calculate_threshold(train, 95) == pytest.approx(3.8)
    summary = summarize(train, val, 95)
    assert summary["validation_flagged"] == 2
    assert summary["threshold_source"] == "train_only"
    assert summary["final_test_used"] is False


@pytest.mark.parametrize(
    "values",
    [
        np.array([]),
        np.array([[1.0, 2.0]]),
        np.array([-1.0, 2.0]),
        np.array([1.0, np.nan]),
    ],
)
def test_bad_training_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        calculate_threshold(values, 95)


def test_error_association() -> None:
    distances = np.array([1.0, 2.0, 3.0, 4.0])
    correct = np.array([True, False, True, False])
    result = error_association(distances, correct, 2.5)
    assert result["cnn_errors"] == 2
    assert result["flagged_and_error"] == 1
    assert result["flagged_samples"] == 2


def test_nonboolean_correctness_rejected() -> None:
    with pytest.raises(ValueError):
        error_association(np.array([1.0]), np.array([1]), 0.5)


def test_pdf_only_figure(tmp_path: Path) -> None:
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.novelty_figures import (
        save_qe_distribution,
    )

    train = np.linspace(0.1, 2, 20)
    val = np.linspace(0.2, 2.5, 20)
    pdf = tmp_path / "qe_score_distribution.pdf"
    save_qe_distribution(train, val, 1.8, pdf)
    actual_pdf = tmp_path / "pdf" / "qe_score_distribution.pdf"
    assert actual_pdf.read_bytes().startswith(b"%PDF-")
    with pytest.raises(ValueError, match="PDF"):
        save_qe_distribution(train, val, 1.8, tmp_path / "plot.svg")
