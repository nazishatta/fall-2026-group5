"""Confusion-matrix figures: CF1 (before), CF2 (after) and their difference. PDF only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.figures import save_pdf


def _heat(ax: plt.Axes, matrix: np.ndarray, title: str, cmap: str, centre_zero: bool) -> None:
    limit = float(np.abs(matrix).max()) or 1.0
    image = ax.imshow(matrix, cmap=cmap, vmin=-limit if centre_zero else 0.0, vmax=limit)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(matrix.shape[0]))
    ax.set_yticks(range(matrix.shape[0]))
    for (row, col), value in np.ndenumerate(matrix):
        if value and (row != col or centre_zero):
            ax.text(col, row, f"{value:.1f}" if centre_zero else f"{value:.0f}", ha="center",
                    va="center", fontsize=6)
    plt.colorbar(image, ax=ax, fraction=0.046)


def plot_confusion_comparison(
    before: np.ndarray, after: np.ndarray, before_name: str, after_name: str, output_path: Path
) -> Path:
    """CF1, CF2 and CF2 - CF1 (test images, averaged over seeds) in one PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _heat(axes[0], before, f"CF1: {before_name}", "Blues", centre_zero=False)
    _heat(axes[1], after, f"CF2: {after_name}", "Blues", centre_zero=False)
    _heat(axes[2], after - before, "CF2 minus CF1 (blue: fewer, red: more)", "RdBu_r", centre_zero=True)
    fig.tight_layout()
    return save_pdf(fig, output_path)
