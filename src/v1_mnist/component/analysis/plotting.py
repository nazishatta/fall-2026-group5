from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_COUNT = 10


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


def save_heatmap(
    path_base: Path,
    matrix: tuple[np.ndarray, np.ndarray],
    title: str,
    colorbar_label: str,
    *,
    categorical: bool = False,
) -> list[Path]:
    """Save a SOM neuron map using the model's native topology positions as vector SVG and PDF.

    Args:
        path_base: Output file path without suffix.
        matrix: Tuple of (coordinates, values).
        title: Plot title string.
        colorbar_label: Label for the colorbar.
        categorical: Whether the values represent discrete class categories.

    Returns:
        List containing paths to saved SVG and PDF files.
    """
    coordinates, values = matrix
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError(
            f"No finite neuron values available for figure: {title}"
        )

    svg_path, pdf_path = _route_dual_paths(path_base)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
            cmap=plt.get_cmap("tab10", CLASS_COUNT),
            vmin=-0.5,
            vmax=9.5,
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
        colorbar.set_ticks(list(range(CLASS_COUNT)))

    # Save SVG vector graphic
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    # Save PDF vector graphic
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)
    return [svg_path, pdf_path]
