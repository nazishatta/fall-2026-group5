from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from NNSOM.plots import SOMPlots

GRID_HEIGHT = 15
GRID_WIDTH = 15
NUM_NEURONS = GRID_HEIGHT * GRID_WIDTH
FEATURE_DIM = 84

SPLIT_FILES = {
    "features",
    "labels",
    "preds",
    "sample_ids",
    "confidence",
    "correct",
}


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 hex digest for a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path_obj = Path(path)
    digest = hashlib.sha256()

    with path_obj.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_float(value: float) -> float | None:
    """Convert non-finite float values to None (JSON null).

    Args:
        value: Float value to convert.

    Returns:
        float if finite, otherwise None.
    """
    val = float(value)
    if not math.isfinite(val):
        return None
    return val


def load_split(
    split: str,
    embeddings_dir: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Load and validate the frozen feature/label files for one split.

    Args:
        split: Split name ('train' or 'val').
        embeddings_dir: Optional path to embeddings directory.

    Returns:
        Dictionary mapping file suffix to loaded NumPy array.
    """
    if split not in {"train", "val"}:
        raise ValueError("Only train and val are permitted in this analysis.")

    if embeddings_dir is None:
        repo_root = Path(__file__).resolve().parents[4]
        emb_dir = repo_root / "outputs/v1_mnist/cnn_baseline/embeddings/mnist"
    else:
        emb_dir = Path(embeddings_dir)

    arrays: dict[str, np.ndarray] = {}

    for suffix in sorted(SPLIT_FILES):
        path = emb_dir / f"{split}_{suffix}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required embedding split file: {path}")
        arrays[suffix] = np.load(path)

    n = arrays["features"].shape[0]

    expected_shapes = {
        "features": (n, FEATURE_DIM),
        "labels": (n,),
        "preds": (n,),
        "sample_ids": (n,),
        "confidence": (n,),
        "correct": (n,),
    }

    for name, expected in expected_shapes.items():
        observed = arrays[name].shape
        if observed != expected:
            raise ValueError(
                f"{split}_{name}: shape {observed}; expected {expected}"
            )

    if not np.isfinite(arrays["features"]).all():
        raise ValueError(f"{split}: non-finite features detected.")

    if not np.isfinite(arrays["confidence"]).all():
        raise ValueError(f"{split}: non-finite confidence detected.")

    if not np.array_equal(
        arrays["correct"],
        arrays["preds"] == arrays["labels"],
    ):
        raise ValueError(f"{split}: correctness metadata mismatch.")

    if len(np.unique(arrays["sample_ids"])) != n:
        raise ValueError(f"{split}: sample IDs are not unique.")

    if np.any((arrays["confidence"] < 0.0) | (arrays["confidence"] > 1.0)):
        raise ValueError(f"{split}: confidence outside [0, 1].")

    return arrays


def load_selected_som(
    model_path: Path | str | None = None,
    grid_height: int = GRID_HEIGHT,
    grid_width: int = GRID_WIDTH,
) -> Any:
    """Load and validate the trained SOM model.

    Args:
        model_path: Optional path to pickled SOM model file.
        grid_height: Height of the SOM grid.
        grid_width: Width of the SOM grid.

    Returns:
        Loaded SOMPlots model object.
    """
    if model_path is None:
        repo_root = Path(__file__).resolve().parents[4]
        m_path = (
            repo_root
            / "outputs/v1_mnist/som/som_models"
            / "som_15x15_seed42_final"
        )
    else:
        m_path = Path(model_path)

    if not m_path.is_file():
        raise FileNotFoundError(f"SOM model file not found: {m_path}")

    som = SOMPlots(dimensions=(grid_height, grid_width))
    loaded = som.load_pickle(m_path.name, str(m_path.parent) + os.sep)

    if loaded is not None and hasattr(loaded, "cluster_data"):
        som = loaded

    expected_neurons = grid_height * grid_width
    weights = np.asarray(som.w)

    if weights.shape != (expected_neurons, FEATURE_DIM):
        raise RuntimeError(
            f"Loaded SOM weight shape mismatch: {weights.shape}; "
            f"expected ({expected_neurons}, {FEATURE_DIM})"
        )

    if int(som.numNeurons) != expected_neurons:
        raise RuntimeError(
            f"Loaded SOM neuron-count mismatch: {som.numNeurons}; "
            f"expected {expected_neurons}"
        )

    if bool(som.sim_flag):
        raise RuntimeError("Loaded SOM is marked untrained.")

    return som


def som_coordinates(som: Any) -> np.ndarray:
    """Return SOM positions as shape (2, num_neurons).

    Args:
        som: Trained SOM model instance.

    Returns:
        2D coordinate array of shape (2, num_neurons).
    """
    positions = np.asarray(som.pos)
    num_neurons = getattr(som, "numNeurons", None)

    if positions.ndim != 2:
        raise RuntimeError(f"Unexpected SOM position dimensionality: {positions.ndim}")

    if num_neurons is not None:
        n = int(num_neurons)
        if positions.shape == (2, n):
            return positions
        if positions.shape == (n, 2):
            return positions.T

    if positions.shape[0] == 2:
        return positions
    if positions.shape[1] == 2:
        return positions.T

    raise RuntimeError(f"Unexpected SOM position shape: {positions.shape}")


def reconstruct_assignments(
    clusters: Sequence[Sequence[int] | np.ndarray],
    cluster_distances: Sequence[Sequence[float] | np.ndarray],
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert NNSOM cluster lists into per-sample BMU arrays.

    Supports arbitrary numbers of neurons.

    Args:
        clusters: List of sample index arrays, one per neuron.
        cluster_distances: List of sample distance arrays, one per neuron.
        n_samples: Total number of samples expected.

    Returns:
        Tuple of (bmu_array, distance_array).
    """
    num_neurons = len(clusters)
    if len(cluster_distances) != num_neurons:
        raise RuntimeError(
            f"Cluster distance count ({len(cluster_distances)}) does not match "
            f"neuron cluster count ({num_neurons})."
        )

    bmu = np.full(n_samples, -1, dtype=np.int64)
    distance = np.full(n_samples, np.nan, dtype=np.float64)
    assigned = np.zeros(n_samples, dtype=bool)

    for neuron_id, (sample_indices, distances) in enumerate(
        zip(clusters, cluster_distances)
    ):
        s_indices = np.asarray(sample_indices, dtype=np.int64)
        s_distances = np.asarray(distances, dtype=np.float64)

        if s_indices.shape != s_distances.shape:
            raise RuntimeError(
                f"Neuron {neuron_id}: index/distance shape mismatch."
            )

        if np.any((s_indices < 0) | (s_indices >= n_samples)):
            raise RuntimeError(f"Neuron {neuron_id}: invalid sample index.")

        if np.any(assigned[s_indices]):
            raise RuntimeError("A sample was assigned to multiple BMUs.")

        bmu[s_indices] = neuron_id
        distance[s_indices] = s_distances
        assigned[s_indices] = True

    if not assigned.all():
        missing = int((~assigned).sum())
        raise RuntimeError(f"{missing} samples have no BMU assignment.")

    if np.any(bmu < 0):
        raise RuntimeError("Negative BMU index after assignment.")

    if not np.isfinite(distance).all():
        raise RuntimeError("Non-finite BMU distance detected.")

    return bmu, distance


def write_sample_assignments(
    path: Path | str,
    split: str,
    arrays: dict[str, np.ndarray],
    bmu: np.ndarray,
    distance: np.ndarray,
    coordinates: np.ndarray,
) -> None:
    """Write traceable per-sample BMU assignments to CSV.

    Args:
        path: Output CSV path.
        split: Split name string.
        arrays: Dictionary of embedding arrays.
        bmu: Per-sample BMU index array.
        distance: Per-sample BMU distance array.
        coordinates: Shape (2, num_neurons) coordinate array.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "split_index",
        "sample_id",
        "label",
        "prediction",
        "correct",
        "confidence",
        "bmu_index",
        "bmu_pos_0",
        "bmu_pos_1",
        "bmu_distance",
    ]

    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(bmu)):
            neuron_id = int(bmu[i])
            writer.writerow(
                {
                    "split": split,
                    "split_index": i,
                    "sample_id": int(arrays["sample_ids"][i]),
                    "label": int(arrays["labels"][i]),
                    "prediction": int(arrays["preds"][i]),
                    "correct": int(arrays["correct"][i]),
                    "confidence": float(arrays["confidence"][i]),
                    "bmu_index": neuron_id,
                    "bmu_pos_0": float(coordinates[0, neuron_id]),
                    "bmu_pos_1": float(coordinates[1, neuron_id]),
                    "bmu_distance": float(distance[i]),
                }
            )


def write_rows(
    path: Path | str,
    rows: list[dict[str, Any]],
) -> None:
    """Write list of row dictionaries to CSV.

    Args:
        path: Target CSV file path.
        rows: Non-empty list of dictionary rows.
    """
    if not rows:
        raise ValueError("Cannot write empty table.")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
