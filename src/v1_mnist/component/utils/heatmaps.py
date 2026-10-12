from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.figures import save_pdf


def matrix_from_neuron_values(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Package native NNSOM neuron coordinates with one value per neuron.

    Preserves actual NNSOM coordinates and supports arbitrary grid sizes
    (e.g., 10x10, 15x15, 20x20) and staggered topologies.

    Args:
        coordinates: Shape (2, num_neurons) array of 2D neuron positions.
        values: Shape (num_neurons,) array of scalar values.

    Returns:
        Tuple of (coordinates, values) as float64 arrays.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[0] != 2:
        raise RuntimeError(
            f"Unexpected SOM coordinate shape: {coords.shape}; expected (2, num_neurons)."
        )

    num_neurons = coords.shape[1]

    if vals.ndim != 1 or vals.shape[0] != num_neurons:
        raise RuntimeError(
            f"Unexpected neuron-value shape: {vals.shape}; expected ({num_neurons},)."
        )

    unique_positions = np.unique(coords.T, axis=0)
    if unique_positions.shape[0] != num_neurons:
        raise RuntimeError(
            f"SOM neuron positions are not unique: {unique_positions.shape[0]} "
            f"unique positions for {num_neurons} neurons."
        )

    return coords, vals


def class_colormap_name(num_classes: int) -> str:
    """A qualitative colour map with enough distinct colours for the classes."""
    if num_classes <= 10:
        return "tab10"
    if num_classes <= 20:
        return "tab20"
    return "turbo"


def save_heatmap(
    path_base: Path,
    matrix: tuple[np.ndarray, np.ndarray],
    title: str,
    colorbar_label: str,
    *,
    categorical: bool = False,
    num_classes: int | None = None,
) -> list[Path]:
    """Save a SOM neuron map using the model's native topology positions as a PDF.

    Args:
        path_base: Output file path without suffix.
        matrix: Tuple of (coordinates, values).
        title: Plot title string.
        colorbar_label: Label for the colorbar.
        categorical: Whether the values represent discrete class categories.
        num_classes: Number of classes (required when categorical).

    Returns:
        List containing the path of the saved PDF (pdf/ folder).
    """
    coordinates, values = matrix
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if categorical and not num_classes:
        raise ValueError("num_classes is required for a categorical (class) map.")

    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError(
            f"No finite neuron values available for figure: {title}"
        )

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_axes([0.11, 0.11, 0.72, 0.80])

    # Draw every neuron location first so empty / undefined neurons
    # remain visible as part of the SOM topology.
    ax.scatter(
        coordinates[0],
        coordinates[1],
        s=115,
        marker="o",
        facecolors="none",
        edgecolors="0.80",
        linewidths=0.45,
    )

    if categorical:
        image = ax.scatter(
            coordinates[0, finite],
            coordinates[1, finite],
            c=values[finite],
            s=100,
            marker="o",
            cmap=plt.get_cmap(class_colormap_name(num_classes), num_classes),
            vmin=-0.5,
            vmax=num_classes - 0.5,
            linewidths=0.0,
        )
    else:
        image = ax.scatter(
            coordinates[0, finite],
            coordinates[1, finite],
            c=values[finite],
            s=100,
            marker="o",
            linewidths=0.0,
        )

    ax.set_title(title)
    ax.set_xlabel("NNSOM topology coordinate 0")
    ax.set_ylabel("NNSOM topology coordinate 1")
    ax.set_aspect("equal", adjustable="box")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(colorbar_label)

    if categorical:
        colorbar.set_ticks(list(range(num_classes)))

    return [save_pdf(fig, path_base)]


def read_neuron_table(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Read a per-neuron table (one row per neuron, in neuron order) with pos_0 / pos_1 columns.

    Returns:
        Tuple of (coordinates of shape (2, num_neurons), rows as dictionaries).
    """
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if [int(r["neuron_id"]) for r in rows] != list(range(len(rows))):
        raise RuntimeError(f"{path}: rows are not one per neuron in neuron order.")

    coordinates = np.asarray(
        [[float(r["pos_0"]) for r in rows], [float(r["pos_1"]) for r in rows]],
        dtype=np.float64,
    )
    return coordinates, rows


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    """One numeric column of a neuron table; empty cells and 'None' become NaN."""
    return np.asarray(
        [float("nan") if r[name] in ("", "None", "nan") else float(r[name]) for r in rows],
        dtype=np.float64,
    )
