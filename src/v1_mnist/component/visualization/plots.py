from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None

RECOGNIZED_CATEGORIES = {
    "core",
    "class_maps",
    "feature_maps",
    "component_planes",
    "analysis_maps",
    "extended",
    "convergence",
}


def _route_dual_paths(output_path: str | Path) -> tuple[Path, Path]:
    """Route an output path to twin SVG and PDF paths under dedicated svg/ and pdf/ directories."""
    path = Path(output_path)
    parts = list(path.parts)
    dir_parts = parts[:-1]

    if "svg" in dir_parts:
        idx = dir_parts.index("svg")
        pdf_parts = list(parts)
        pdf_parts[idx] = "pdf"
        svg_path = Path(*parts).with_suffix(".svg")
        pdf_path = Path(*pdf_parts).with_suffix(".pdf")
        return svg_path, pdf_path

    if "pdf" in dir_parts:
        idx = dir_parts.index("pdf")
        svg_parts = list(parts)
        svg_parts[idx] = "svg"
        svg_path = Path(*svg_parts).with_suffix(".svg")
        pdf_path = Path(*parts).with_suffix(".pdf")
        return svg_path, pdf_path

    parent_name = path.parent.name
    if parent_name in RECOGNIZED_CATEGORIES:
        root = path.parent.parent
        stem = path.stem
        svg_path = root / "svg" / parent_name / f"{stem}.svg"
        pdf_path = root / "pdf" / parent_name / f"{stem}.pdf"
        return svg_path, pdf_path

    stem = path.stem
    svg_path = path.parent / "svg" / f"{stem}.svg"
    pdf_path = path.parent / "pdf" / f"{stem}.pdf"
    return svg_path, pdf_path


def plot_class_distribution(
    counts_dict: Mapping[Any, Mapping[str, Any]],
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot class frequency distribution and save as vector SVG and PDF."""
    svg_path, pdf_path = _route_dual_paths(output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    classes: Sequence[Any],
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot confusion matrix heatmap and save as vector SVG and PDF."""
    svg_path, pdf_path = _route_dual_paths(output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Plot training and validation loss/accuracy curves and save as vector SVG and PDF."""
    svg_path, pdf_path = _route_dual_paths(output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_sample_images(
    dataset: Any,
    num_samples_per_class: int,
    output_path: str | Path,
) -> Path:
    """Plot representative sample images per class and save as vector SVG and PDF."""
    svg_path, pdf_path = _route_dual_paths(output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return svg_path
