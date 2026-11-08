"""Figures of the SOM re-analysis. PDF only; the dense layers are rasterized, text stays vector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    cell_name,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_measures import (
    neuron_majority,
)
from src.v1_mnist.component.utils.figures import save_pdf

PANELS = (
    ("weighted_purity", "Weighted purity (higher is better)"),
    ("low_purity_share", "Share of images in low-purity neurons"),
    ("class_scatter_index", "Class scatter index"),
    ("topographic_error_pct", "Topographic error (%)"),
    ("hotspot_error_rate", "Hotspot error rate"),
    ("hotspot_error_share", "Share of errors in hotspots"),
)


def plot_measures(table: dict[str, dict[str, Any]], settings: ReanalysisSettings, path: Path) -> Path:
    """One panel per measure: a dot per seed for each arm, lines join the same seed, dashed = Phase 2 map."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (measure, title) in zip(axes.ravel(), PANELS):
        values = np.array([[table[cell_name(a, s)][measure] for a in settings.arms] for s in settings.seeds])
        for row in values:
            ax.plot(range(len(settings.arms)), row, color="0.7", lw=0.8, zorder=1)
        ax.scatter(np.tile(range(len(settings.arms)), len(settings.seeds)), values.ravel(), s=18, zorder=2)
        ax.scatter(range(len(settings.arms)), values.mean(axis=0), marker="_", s=400, color="k", zorder=3)
        ax.axhline(table[settings.reference][measure], ls="--", color="tab:red", lw=1)
        ax.set_xticks(range(len(settings.arms)))
        ax.set_xticklabels(settings.arms, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Validation images; dots = CNN seeds, bar = mean, dashed = Phase 2 map", fontsize=9)
    fig.tight_layout()
    return save_pdf(fig, path)


def plot_maps(settings: ReanalysisSettings, table: dict[str, dict[str, Any]], path: Path) -> Path:
    """Purity map of the Phase 2 SOM and of the example seed's maps; hotspot neurons are outlined."""
    cells = [settings.reference, *[cell_name(a, settings.example_seed) for a in settings.arms]]
    labels = np.load(settings.retraining.analysis.embeddings_dir / "val_labels.npy").astype(np.int64)
    num_classes = settings.retraining.train.num_classes
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.8))
    image = None
    for ax, cell in zip(np.atleast_1d(axes), cells):
        npz = np.load(settings.folder("assignments") / f"{cell}.npz")
        counts = neuron_majority(npz["val_bmu"], labels, npz["neuron_dist"].shape[0], num_classes)
        size = counts.sum(axis=1)
        purity = np.where(size > 0, counts.max(axis=1) / np.maximum(size, 1), np.nan)
        x, y = npz["pos"]
        image = ax.scatter(x, y, c=purity, marker="s", s=70, vmin=0.5, vmax=1.0, cmap="viridis",
                           rasterized=True)
        spots = table[cell]["hotspots"]
        ax.scatter(x[spots], y[spots], marker="s", s=70, facecolors="none", edgecolors="red", lw=1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(cell, fontsize=9)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="neuron purity (validation)")
    return save_pdf(fig, path, rasterize_dense=True)


def plot_cka(cka: dict[str, Any], path: Path) -> Path:
    """Linear CKA between the validation embeddings of all cells."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mat = np.array(cka["matrix"])
    image = ax.imshow(mat, vmin=min(0.0, float(mat.min())), vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(cka["cells"])))
    ax.set_yticks(range(len(cka["cells"])))
    ax.set_xticklabels(cka["cells"], rotation=90, fontsize=6)
    ax.set_yticklabels(cka["cells"], fontsize=6)
    fig.colorbar(image, label="linear CKA")
    return save_pdf(fig, path, rasterize_dense=True)


def make_figures(settings: ReanalysisSettings, results: dict[str, Any]) -> list[Path]:
    folder = settings.folder("figures")
    return [
        plot_measures(results["table"], settings, folder / "map_measures_by_arm.pdf"),
        plot_maps(settings, results["table"], folder / "purity_maps.pdf"),
        plot_cka(results["cka"], folder / "embedding_similarity_cka.pdf"),
    ]


