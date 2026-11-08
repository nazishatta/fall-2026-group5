"""Small PDF-only figures for training-calibrated QE validation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.figures import (
    pdf_figure_path,
    save_pdf,
)


def save_qe_distribution(
    train: np.ndarray, validation: np.ndarray, threshold: float, destination: Path
) -> None:
    """Save an explanatory distribution plot; never use final-test data."""
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Results figures must use PDF format")
    output = pdf_figure_path(destination)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {output}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        train, bins=60, alpha=0.55, density=True, label="Train QE", rasterized=True
    )
    ax.hist(
        validation,
        bins=60,
        alpha=0.55,
        density=True,
        label="Validation QE",
        rasterized=True,
    )
    ax.axvline(threshold, linestyle="--", label="Training-calibrated threshold")
    ax.set(
        xlabel="Distance to SOM BMU",
        ylabel="Density",
        title="QE score distributions",
    )
    ax.legend()
    save_pdf(fig, destination)
