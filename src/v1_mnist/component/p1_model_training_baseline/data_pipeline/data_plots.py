"""Week 1 (Environment Setup & Data Pipeline) figures: class distributions and sample images.

Split from visualization/plots.py; the Week 2 CNN figures are in
p1_model_training_baseline/cnn_baseline_training/cnn_plots.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None

from src.v1_mnist.component.utils.figures import save_pdf

def plot_class_distribution(
    counts_dict: Mapping[Any, Mapping[str, Any]],
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot class frequency distribution and save it as a PDF (pdf/ folder)."""

    classes = list(counts_dict.keys())
    counts = [d["count"] for d in counts_dict.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.barplot(x=classes, y=counts, palette="viridis", ax=ax)
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(classes))))
        ax.bar(classes, counts, color=colors)
        ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticks(classes)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            format(height, ".0f"),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    fig.tight_layout()
    return save_pdf(fig, output_path)


def plot_sample_images(
    dataset: Any,
    num_samples_per_class: int,
    output_path: str | Path,
) -> Path:
    """Plot representative sample images per class and save them as a PDF (pdf/ folder)."""

    num_classes = 10
    fig, axes = plt.subplots(
        num_classes,
        num_samples_per_class,
        figsize=(num_samples_per_class * 1.5, num_classes * 1.5),
    )

    class_counts = {i: 0 for i in range(num_classes)}

    for img, label in dataset:
        lbl = label.item() if hasattr(label, "item") else int(label)
        if class_counts[lbl] < num_samples_per_class:
            ax = axes[lbl, class_counts[lbl]]
            # Convert from [1, 28, 28] tensor to [28, 28] numpy array
            img_np = img.squeeze().numpy() if hasattr(img, "numpy") else np.squeeze(img)
            ax.imshow(img_np, cmap="gray")
            ax.axis("off")
            if class_counts[lbl] == 0:
                ax.set_title(f"Class {lbl}", loc="left")
            class_counts[lbl] += 1

        if all(count == num_samples_per_class for count in class_counts.values()):
            break

    fig.tight_layout()
    return save_pdf(fig, output_path)
