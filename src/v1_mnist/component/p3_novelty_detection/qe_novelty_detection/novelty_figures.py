"""Small PDF-only figures for training-calibrated QE validation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_qe_distribution(
    train: np.ndarray, validation: np.ndarray, threshold: float, destination: Path
) -> None:
    """Save an explanatory distribution plot; never use final-test data."""
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Results figures must use PDF format")
    with plt.rc_context({"pdf.fonttype": 42}):
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
        fig.savefig(destination, bbox_inches="tight")
        plt.close(fig)
