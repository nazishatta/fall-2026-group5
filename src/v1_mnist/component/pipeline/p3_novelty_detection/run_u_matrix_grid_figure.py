"""Plot the frozen SOM's U-Matrix on its actual hexagonal topology."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import PolyCollection

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
    neuron_u_matrix,
)
from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.som_io import (
    load_selected_som,
    sha256_file,
    som_coordinates,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]


def repository_path(value: str) -> Path:
    """Resolve an artifact path inside the repository."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path outside repository")
    return path


def hexagon_vertices(
    centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Build point-up hexagons centered on actual SOM positions."""
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Expected neuron positions as (n, 2)")
    if not np.isfinite(centers).all() or not 0 < radius < 1:
        raise ValueError("Invalid hexagon geometry")

    angles = np.arange(6) * (np.pi / 3) + np.pi / 6
    offsets = radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    return centers[:, None, :] + offsets[None, :, :]


def run(config_path: Path) -> Path:
    """Save a new provenance-verified PDF without modifying SOM inputs."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_id = str(config["run_id"])
    if not run_id or not all(
        char.isalnum() or char in "_-" for char in run_id
    ):
        raise ValueError("Invalid run ID")

    destination = repository_path(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"STOP: Existing run: {destination}")

    model_path = repository_path(config["model_path"])
    model_sha256 = sha256_file(model_path)
    if model_sha256 != config["model_sha256"]:
        raise RuntimeError("STOP: Frozen SOM SHA-256 mismatch")

    height = int(config["grid_height"])
    width = int(config["grid_width"])
    feature_dim = int(config["feature_dim"])

    som = load_selected_som(model_path, height, width, feature_dim)
    centers = som_coordinates(som).T
    scores = neuron_u_matrix(
        np.asarray(som.w),
        np.asarray(som.neuron_dist),
    )

    count = height * width
    if centers.shape != (count, 2) or scores.shape != (count,):
        raise RuntimeError("Unexpected SOM geometry")
    if not np.isfinite(scores).all() or not np.isfinite(centers).all():
        raise RuntimeError("Invalid U-Matrix data")
    if len(np.unique(centers, axis=0)) != count:
        raise RuntimeError("Duplicate SOM coordinates")

    vertices = hexagon_vertices(centers, float(config["hex_radius"]))
    fig, ax = plt.subplots(figsize=(9, 7))

    collection = PolyCollection(
        vertices,
        array=scores,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.25,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("SOM horizontal coordinate")
    ax.set_ylabel("SOM vertical coordinate")
    ax.set_title(f"Frozen {height}×{width} SOM: U-Matrix")
    fig.colorbar(
        collection,
        ax=ax,
        label="Mean distance to neighboring neurons",
        shrink=0.82,
    )
    fig.tight_layout()

    figure_name = str(config["figure_name"])
    if Path(figure_name).name != figure_name or not figure_name.endswith(".pdf"):
        raise ValueError("Invalid PDF filename")

    destination.mkdir(parents=True, exist_ok=False)
    figure_path = save_pdf(fig, destination / figure_name)

    metadata = {
        "run_id": run_id,
        "figure": figure_name,
        "som_model": model_path.name,
        "model_sha256": model_sha256,
        "config_sha256": sha256_file(config_path),
        "figure_sha256": sha256_file(figure_path),
        "grid_height": height,
        "grid_width": width,
        "feature_dim": feature_dim,
        "neuron_count": count,
        "u_matrix_min": float(scores.min()),
        "u_matrix_max": float(scores.max()),
        "u_matrix_mean": float(scores.mean()),
        "leakage_control": (
            "Frozen trained SOM only; no fitting, threshold tuning, "
            "or access to the MNIST final test set."
        ),
        "reproduction_note": (
            "Reconstructed from original SOM neuron weights, topology "
            "and coordinates; visual styling may differ from historical PDF."
        ),
    }

    with (destination / "figure_metrics.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write(chr(10))

    LOGGER.info("Saved U-Matrix PDF: %s", figure_path)
    return figure_path


def main() -> None:
    """Run configured SOM-grid PDF generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    run(args.config)


if __name__ == "__main__":
    main()
