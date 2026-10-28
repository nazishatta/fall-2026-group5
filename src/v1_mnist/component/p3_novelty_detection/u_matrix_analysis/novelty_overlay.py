"""Phase 3 validation-only SOM overlay and descriptive miss analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.v1_mnist.component.utils.figures import pdf_figure_path, save_pdf


def aggregate_bmu_scores(
    bmu: NDArray[np.integer],
    scores: NDArray[np.floating],
    neuron_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute per-neuron mean novelty scores and hit counts (no empty-cell imputation)."""
    indices = np.asarray(bmu)
    values = np.asarray(scores, dtype=np.float64)
    if neuron_count < 1:
        raise ValueError("neuron_count must be positive")
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("BMUs must be a 1D integer array")
    if values.ndim != 1 or values.shape != indices.shape or values.size == 0:
        raise ValueError("Scores must be nonempty and aligned with BMUs")
    if not np.isfinite(values).all() or (indices >= neuron_count).any():
        raise ValueError("Invalid scores or out-of-range BMU")
    if (indices < 0).any():
        raise ValueError("Negative BMU index")
    counts = np.bincount(indices.astype(np.int64), minlength=neuron_count)
    sums = np.bincount(indices.astype(np.int64), weights=values, minlength=neuron_count)
    mean = np.full(neuron_count, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean, counts.astype(np.int64)


def save_novelty_overlay(
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    grid_height: int,
    grid_width: int,
    destination: Path,
) -> dict[str, int]:
    """Map MNIST-validation mean QE to BMU locations; no SVHN claims."""
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Only PDF result figures are allowed")
    output = pdf_figure_path(destination)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    means, counts = aggregate_bmu_scores(bmu, qe, grid_height * grid_width)
    image = np.ma.masked_invalid(means.reshape(grid_height, grid_width))
    fig, ax = plt.subplots(figsize=(7, 5))
    heat = ax.imshow(image, interpolation="nearest", cmap="viridis")
    ax.set(title="Mean MNIST validation QE by SOM BMU", xlabel="SOM column", ylabel="SOM row")
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))
    ax.tick_params(labelsize=7)
    fig.colorbar(heat, ax=ax, label="Mean distance to BMU (QE)")
    fig.text(0.5, 0.01, "Empty neurons are blank; no SVHN images are shown", ha="center", fontsize=9)
    save_pdf(fig, destination)
    return {"samples": int(counts.sum()), "occupied_neurons": int(np.count_nonzero(counts)), "empty_neurons": int(np.count_nonzero(counts == 0))}


def summarize_qe_misses(
    svhn_qe: NDArray[np.floating], threshold: float
) -> dict[str, float | int | str]:
    """Describe missed OOD scores without assigning unsupported causal explanations."""
    scores = np.asarray(svhn_qe, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("SVHN QE scores must be finite and nonempty")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid training-derived cutoff")
    missed = scores[scores <= threshold]
    return {
        "svhn_samples": int(scores.size),
        "detected": int(np.count_nonzero(scores > threshold)),
        "missed": int(missed.size),
        "miss_rate": float(missed.size / scores.size),
        "missed_qe_median": float(np.median(missed)) if missed.size else None,
        "missed_qe_min": float(missed.min()) if missed.size else None,
        "missed_qe_max": float(missed.max()) if missed.size else None,
        "interpretation": "Misses fall at or below the frozen QE cutoff. This is a score-based description, not proof of visual or causal mechanisms.",
    }
