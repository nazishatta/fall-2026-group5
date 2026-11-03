"""SVHN on the SOM grid: where novel images land and how ordinary they look to the SOM.

Pure numpy/matplotlib helpers (no model, no data loading), so they can be tested
without PyTorch. The runner ``run_svhn_som_overlay.py`` supplies the assignments.

Everything here is descriptive. It says where the missed SVHN images sit on the
map and how close they are to the SOM's prototypes. It does not prove why a
particular image was missed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from numpy.typing import NDArray

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.run_u_matrix_grid_figure import (
    hexagon_vertices,
)
from src.v1_mnist.component.utils.figures import pdf_figure_path, save_pdf

NO_CLASS = -1


def check_assignments(
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    neuron_count: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate aligned BMU indices and QE values; return them as int64 and float64."""
    if neuron_count < 1:
        raise ValueError("neuron_count must be positive")
    index = np.asarray(bmu)
    values = np.asarray(qe, dtype=np.float64)
    if index.ndim != 1 or index.size == 0 or index.dtype.kind not in "iu":
        raise ValueError("BMUs must be a nonempty 1D integer array")
    if values.shape != index.shape or not np.isfinite(values).all():
        raise ValueError("QE values must be finite and aligned with the BMUs")
    if index.min() < 0 or index.max() >= neuron_count:
        raise ValueError("BMU index outside the SOM")
    return index.astype(np.int64), values


def check_threshold(threshold: float) -> float:
    """Return the frozen QE cutoff as a float, rejecting invalid values."""
    value = float(threshold)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Invalid training-derived QE cutoff")
    return value


def neuron_dominant_class(
    class_counts: NDArray[np.number],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Dominant MNIST class and its share for every neuron.

    Args:
        class_counts: array of shape (classes, neurons) with the number of
            training samples of each class whose BMU is that neuron.

    Returns:
        (dominant class, purity). Neurons that no sample reached get
        ``NO_CLASS`` and a NaN purity.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.size == 0:
        raise ValueError("Class counts must be a nonempty (classes, neurons) array")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Class counts must be finite and non-negative")
    totals = counts.sum(axis=0)
    dominant = np.where(totals > 0, counts.argmax(axis=0), NO_CLASS).astype(np.int64)
    purity = np.full(counts.shape[1], np.nan, dtype=np.float64)
    np.divide(counts.max(axis=0), totals, out=purity, where=totals > 0)
    return dominant, purity


def svhn_neuron_summary(
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    threshold: float,
    neuron_count: int,
) -> dict[str, NDArray[np.generic]]:
    """Per-neuron SVHN hit count, missed count and missed share.

    An image is missed when its QE is at or below the frozen cutoff, the same
    rule the earlier Phase 3 runs use. Neurons with no SVHN image get a NaN share.
    """
    index, values = check_assignments(bmu, qe, neuron_count)
    cutoff = check_threshold(threshold)
    hits = np.bincount(index, minlength=neuron_count).astype(np.int64)
    missed = np.bincount(index[values <= cutoff], minlength=neuron_count).astype(np.int64)
    share = np.full(neuron_count, np.nan, dtype=np.float64)
    np.divide(missed, hits, out=share, where=hits > 0)
    return {"hits": hits, "missed": missed, "miss_rate": share}


def class_landing_profile(
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    threshold: float,
    dominant_class: NDArray[np.integer],
    class_count: int,
) -> list[dict[str, float | int | str | None]]:
    """Group SVHN images by the MNIST class that dominates the neuron they land in.

    One row per class (plus a row for neurons no training sample reached, if any
    SVHN image lands there). The rows add up to all SVHN images and all misses.
    """
    classes = np.asarray(dominant_class)
    index, values = check_assignments(bmu, qe, len(classes))
    cutoff = check_threshold(threshold)
    landing = classes[index]
    missed = values <= cutoff
    total_missed = int(missed.sum())
    rows: list[dict[str, float | int | str | None]] = []
    for label in (*range(class_count), NO_CLASS):
        here = landing == label
        images = int(here.sum())
        if label == NO_CLASS and images == 0:
            continue
        misses = int((here & missed).sum())
        rows.append(
            {
                "dominant_mnist_class": "none" if label == NO_CLASS else int(label),
                "svhn_images": images,
                "missed": misses,
                "miss_rate": misses / images if images else None,
                "share_of_all_missed": misses / total_missed if total_missed else None,
            }
        )
    return rows


def miss_typicality(
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    threshold: float,
    validation_bmu: NDArray[np.integer],
    validation_qe: NDArray[np.floating],
    neuron_count: int,
    min_validation_hits: int,
) -> dict[str, float | int | str | None]:
    """How ordinary the missed SVHN images look to the SOM.

    Each missed image's QE is divided by the median QE of the MNIST validation
    digits in the same neuron. A ratio near or below 1 means the SOM represents
    the image as closely as a typical MNIST digit in that cell, which is why a
    QE cutoff cannot separate it. Neurons with fewer than ``min_validation_hits``
    validation digits are left out of the comparison.
    """
    index, values = check_assignments(bmu, qe, neuron_count)
    ref_index, ref_values = check_assignments(validation_bmu, validation_qe, neuron_count)
    cutoff = check_threshold(threshold)
    if min_validation_hits < 1:
        raise ValueError("min_validation_hits must be positive")
    median = np.full(neuron_count, np.nan, dtype=np.float64)
    for neuron in np.unique(ref_index):
        members = ref_values[ref_index == neuron]
        if members.size >= min_validation_hits:
            median[neuron] = float(np.median(members))
    missed = values <= cutoff
    reference = median[index[missed]]
    usable = np.isfinite(reference) & (reference > 0)
    ratio = values[missed][usable] / reference[usable]
    return {
        "missed_images": int(missed.sum()),
        "compared_images": int(usable.sum()),
        "median_qe_ratio": float(np.median(ratio)) if ratio.size else None,
        "share_at_or_below_cell_median": float(np.mean(ratio <= 1.0)) if ratio.size else None,
        "interpretation": (
            "Ratio = missed image QE / median MNIST-validation QE in the same neuron. "
            "Values near or below 1 mean the SOM represents the image about as well "
            "as a typical MNIST digit in that cell. This describes the score; it is "
            "not proof of a visual or causal reason."
        ),
    }


def top_missed_neurons(
    summary: dict[str, NDArray[np.generic]],
    dominant_class: NDArray[np.integer],
    purity: NDArray[np.floating],
    count: int,
) -> list[dict[str, float | int | None]]:
    """The neurons that hold the most missed SVHN images (ties keep neuron order)."""
    if count < 1:
        raise ValueError("count must be positive")
    missed = np.asarray(summary["missed"])
    rows: list[dict[str, float | int | None]] = []
    for neuron in np.argsort(-missed, kind="stable")[:count]:
        if missed[neuron] == 0:
            break
        share = float(summary["miss_rate"][neuron])
        quality = float(purity[neuron])
        rows.append(
            {
                "neuron": int(neuron),
                "missed": int(missed[neuron]),
                "svhn_images": int(summary["hits"][neuron]),
                "miss_rate": None if np.isnan(share) else share,
                "dominant_mnist_class": (
                    None if dominant_class[neuron] == NO_CLASS else int(dominant_class[neuron])
                ),
                "class_purity": None if np.isnan(quality) else quality,
            }
        )
    return rows


def save_svhn_overlay(
    centers: NDArray[np.floating],
    summary: dict[str, NDArray[np.generic]],
    dominant_class: NDArray[np.integer],
    class_count: int,
    threshold: float,
    radius: float,
    destination: Path,
) -> Path:
    """Draw three hexagonal SOM maps and save them as one PDF (never overwrites).

    Panels: SVHN images per neuron, share of those images missed by the QE
    cutoff, and the MNIST class that dominates each neuron.
    """
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Only PDF result figures are allowed")
    output = pdf_figure_path(destination)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    neuron_count = len(dominant_class)
    if np.asarray(centers).shape != (neuron_count, 2):
        raise ValueError("Expected one (x, y) position per neuron")
    if any(len(summary[key]) != neuron_count for key in ("hits", "missed", "miss_rate")):
        raise ValueError("Summary arrays must have one value per neuron")
    if not 1 <= class_count <= 20:
        raise ValueError("class_count must be between 1 and 20")
    vertices = hexagon_vertices(np.asarray(centers, dtype=np.float64), radius)
    hits = np.where(summary["hits"] > 0, summary["hits"], np.nan).astype(np.float64)
    classes = np.where(dominant_class >= 0, dominant_class, np.nan).astype(np.float64)
    class_map = "tab10" if class_count <= 10 else "tab20"
    panels = (
        ("SVHN images per neuron", hits, "viridis", "SVHN images (BMU hits)", None),
        ("Share of SVHN images missed", summary["miss_rate"], "magma", "Missed share (QE at or below cutoff)", None),
        ("Dominant MNIST class per neuron", classes, plt.get_cmap(class_map, class_count), "MNIST class", (-0.5, class_count - 0.5)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, (title, values, cmap, label, limits) in zip(axes, panels):
        colors = plt.get_cmap(cmap).copy()
        colors.set_bad("lightgray")
        layer = PolyCollection(
            vertices, array=np.ma.masked_invalid(values), cmap=colors,
            edgecolors="white", linewidths=0.25,
        )
        if limits is not None:
            layer.set_clim(*limits)
        ax.add_collection(layer)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="SOM horizontal coordinate", ylabel="SOM vertical coordinate")
        bar = fig.colorbar(layer, ax=ax, label=label, shrink=0.8)
        if limits is not None:
            bar.set_ticks(range(class_count))
    fig.suptitle(f"SVHN on the frozen SOM (training QE cutoff {check_threshold(threshold):.3f}); gray = no samples")
    fig.tight_layout()
    return save_pdf(fig, destination)
