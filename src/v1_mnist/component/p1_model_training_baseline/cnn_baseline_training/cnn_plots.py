"""Week 2 (Train Baseline CNN) figures: training curves and the test confusion matrix.

Split from visualization/plots.py; the Week 1 data figures are in
p1_model_training_baseline/data_pipeline/data_plots.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None

from src.v1_mnist.component.utils.figures import save_pdf


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    classes: Sequence[Any],
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot confusion matrix heatmap and save it as a PDF (pdf/ folder)."""

    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
    else:
        im = ax.imshow(conf_matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_yticklabels([str(c) for c in classes])
        max_val = np.max(conf_matrix) if conf_matrix.size > 0 else 1
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, str(conf_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > max_val / 2 else "black",
                )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    return save_pdf(fig, output_path)


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot training and validation loss/accuracy curves and save them as a PDF (pdf/ folder)."""

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss", fontsize=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy", fontsize=14)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()
    return save_pdf(fig, output_path)
