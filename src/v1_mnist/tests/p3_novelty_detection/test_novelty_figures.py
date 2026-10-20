"""Verify Phase 3 figure conventions without writing to experiment folders."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.novelty_figures import (
    save_qe_distribution,
)


def test_qe_figure_uses_shared_pdf_layout(tmp_path: Path) -> None:
    """A result figure must be a PDF in the shared pdf/ subdirectory."""
    requested = tmp_path / "qe_score_distribution.pdf"

    save_qe_distribution(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.2, 0.4, 0.5]),
        0.35,
        requested,
    )

    actual = tmp_path / "pdf" / "qe_score_distribution.pdf"
    assert actual.is_file()
    assert actual.read_bytes().startswith(b"%PDF-")
    assert not requested.exists()


def test_qe_figure_refuses_overwrite(tmp_path: Path) -> None:
    """Existing figures must not be replaced by another save."""
    requested = tmp_path / "qe_score_distribution.pdf"

    save_qe_distribution(
        np.array([0.1, 0.2]),
        np.array([0.3, 0.4]),
        0.25,
        requested,
    )

    original = (tmp_path / "pdf" / "qe_score_distribution.pdf").read_bytes()

    with pytest.raises(FileExistsError):
        save_qe_distribution(
            np.array([0.5, 0.6]),
            np.array([0.7, 0.8]),
            0.65,
            requested,
        )

    assert (
        tmp_path / "pdf" / "qe_score_distribution.pdf"
    ).read_bytes() == original
