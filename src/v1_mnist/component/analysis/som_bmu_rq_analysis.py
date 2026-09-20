"""BMU, representation-structure, and error-geography analysis.

Uses only the frozen training and validation embeddings.
The final test split is intentionally excluded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from NNSOM.plots import SOMPlots
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "som_20x20_seed42_rq1_rq2_v1"

EMBEDDINGS_DIR = (
    REPO_ROOT
    / "outputs/week_2/embeddings/mnist"
)

MODEL_PATH = (
    REPO_ROOT
    / "outputs/week_3/som_models/"
    "som_20x20_seed42_gridstudy"
)

FROZEN_METRICS_PATH = (
    REPO_ROOT
    / "outputs/week_3/logs/"
    "som_20x20_seed42_gridstudy_metrics.json"
)

OUTPUT_BASE = (
    REPO_ROOT
    / "outputs/week_3/analysis"
)

OUTPUT_DIR = OUTPUT_BASE / RUN_ID
TEMP_DIR = OUTPUT_BASE / f".{RUN_ID}.tmp"

EXPECTED_MODEL_SHA256 = (
    "528bb5e11cac077c155ba42855ed1c423"
    "aba7d0e70d86f5f42a0c7dee813c576"
)

GRID_HEIGHT = 20
GRID_WIDTH = 20
NUM_NEURONS = GRID_HEIGHT * GRID_WIDTH
FEATURE_DIM = 84

CLASS_COUNT = 10

SPLIT_FILES = {
    "features",
    "labels",
    "preds",
    "sample_ids",
    "confidence",
    "correct",
}


def sha256_file(path: Path) -> str:
    """Return SHA-256 for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalized_entropy(counts: np.ndarray) -> float:
    """Normalized Shannon entropy over the ten MNIST classes."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())

    if total <= 0:
        return float("nan")

    probabilities = counts[counts > 0] / total
    entropy = -float(
        np.sum(probabilities * np.log(probabilities))
    )

    return entropy / math.log(CLASS_COUNT)


def wilson_lower_bound(
    errors: int,
    total: int,
    z: float = 1.959963984540054,
) -> float:
    """95% Wilson lower bound for a binomial proportion."""
    if total <= 0:
        return float("nan")

    p_hat = errors / total
    z2 = z * z

    denominator = 1.0 + z2 / total

    center = (
        p_hat
        + z2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                p_hat * (1.0 - p_hat) / total
                + z2 / (4.0 * total * total)
            )
        )
        / denominator
    )

    return max(0.0, center - half_width)


def safe_mean(values: np.ndarray) -> float:
    """Mean or NaN for an empty array."""
    values = np.asarray(values)

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def safe_median(values: np.ndarray) -> float:
    """Median or NaN for an empty array."""
    values = np.asarray(values)

    if values.size == 0:
        return float("nan")

    return float(np.median(values))


def json_float(value: float):
    """Convert non-finite values to JSON null."""
    value = float(value)

    if not math.isfinite(value):
        return None

    return value


def load_split(split: str) -> dict[str, np.ndarray]:
    """Load only the frozen files required for one development split."""
    if split not in {"train", "val"}:
        raise ValueError(
            "Only train and val are permitted in this analysis."
        )

    arrays = {}

    for suffix in sorted(SPLIT_FILES):
        path = EMBEDDINGS_DIR / f"{split}_{suffix}.npy"

        if not path.is_file():
            raise FileNotFoundError(path)

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
                f"{split}_{name}: shape {observed}; "
                f"expected {expected}"
            )

    if not np.isfinite(arrays["features"]).all():
        raise ValueError(
            f"{split}: non-finite features"
        )

    if not np.isfinite(arrays["confidence"]).all():
        raise ValueError(
            f"{split}: non-finite confidence"
        )

    if not np.array_equal(
        arrays["correct"],
        arrays["preds"] == arrays["labels"],
    ):
        raise ValueError(
            f"{split}: correctness metadata mismatch"
        )

    if len(np.unique(arrays["sample_ids"])) != n:
        raise ValueError(
            f"{split}: sample IDs are not unique"
        )

    if np.any(
        (arrays["confidence"] < 0.0)
        | (arrays["confidence"] > 1.0)
    ):
        raise ValueError(
            f"{split}: confidence outside [0, 1]"
        )

    return arrays


def load_selected_som():
    """Load and validate the frozen selected 20x20 SOM."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    observed_hash = sha256_file(MODEL_PATH)

    if observed_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Selected SOM hash mismatch.\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    som = SOMPlots(
        dimensions=(GRID_HEIGHT, GRID_WIDTH)
    )

    loaded = som.load_pickle(
        MODEL_PATH.name,
        str(MODEL_PATH.parent) + os.sep,
    )

    # Support either an in-place loader or a loader that returns
    # the loaded SOM object.
    if (
        loaded is not None
        and hasattr(loaded, "cluster_data")
    ):
        som = loaded

    weights = np.asarray(som.w)

    if weights.shape != (
        NUM_NEURONS,
        FEATURE_DIM,
    ):
        raise RuntimeError(
            "Loaded SOM weight shape mismatch: "
            f"{weights.shape}"
        )

    if int(som.numNeurons) != NUM_NEURONS:
        raise RuntimeError(
            "Loaded SOM neuron-count mismatch: "
            f"{som.numNeurons}"
        )

    if bool(som.sim_flag):
        raise RuntimeError(
            "Loaded SOM is marked untrained."
        )

    return som


def som_coordinates(som) -> np.ndarray:
    """Return SOM positions as shape (2, num_neurons)."""
    positions = np.asarray(som.pos)

    if positions.shape == (2, NUM_NEURONS):
        return positions

    if positions.shape == (NUM_NEURONS, 2):
        return positions.T

    raise RuntimeError(
        "Unexpected SOM position shape: "
        f"{positions.shape}"
    )


def reconstruct_assignments(
    clusters,
    cluster_distances,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert NNSOM cluster lists into per-sample BMU arrays."""
    if len(clusters) != NUM_NEURONS:
        raise RuntimeError(
            "Unexpected number of cluster lists: "
            f"{len(clusters)}"
        )

    if len(cluster_distances) != NUM_NEURONS:
        raise RuntimeError(
            "Unexpected number of distance lists: "
            f"{len(cluster_distances)}"
        )

    bmu = np.full(
        n_samples,
        -1,
        dtype=np.int64,
    )

    distance = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    assigned = np.zeros(
        n_samples,
        dtype=bool,
    )

    for neuron_id, (
        sample_indices,
        distances,
    ) in enumerate(
        zip(
            clusters,
            cluster_distances,
        )
    ):
        sample_indices = np.asarray(
            sample_indices,
            dtype=np.int64,
        )

        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        if sample_indices.shape != distances.shape:
            raise RuntimeError(
                f"Neuron {neuron_id}: index/distance "
                "shape mismatch."
            )

        if np.any(
            (sample_indices < 0)
            | (sample_indices >= n_samples)
        ):
            raise RuntimeError(
                f"Neuron {neuron_id}: invalid sample index."
            )

        if np.any(assigned[sample_indices]):
            raise RuntimeError(
                "A sample was assigned to multiple BMUs."
            )

        bmu[sample_indices] = neuron_id
        distance[sample_indices] = distances
        assigned[sample_indices] = True

    if not assigned.all():
        missing = int((~assigned).sum())

        raise RuntimeError(
            f"{missing} samples have no BMU assignment."
        )

    if np.any(bmu < 0):
        raise RuntimeError(
            "Negative BMU index after assignment."
        )

    if not np.isfinite(distance).all():
        raise RuntimeError(
            "Non-finite BMU distance detected."
        )

    return bmu, distance


def write_sample_assignments(
    path: Path,
    split: str,
    arrays: dict[str, np.ndarray],
    bmu: np.ndarray,
    distance: np.ndarray,
    coordinates: np.ndarray,
) -> None:
    """Write traceable per-sample BMU assignments."""
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

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i in range(len(bmu)):
            neuron_id = int(bmu[i])

            writer.writerow(
                {
                    "split": split,
                    "split_index": i,
                    "sample_id": int(
                        arrays["sample_ids"][i]
                    ),
                    "label": int(
                        arrays["labels"][i]
                    ),
                    "prediction": int(
                        arrays["preds"][i]
                    ),
                    "correct": int(
                        arrays["correct"][i]
                    ),
                    "confidence": float(
                        arrays["confidence"][i]
                    ),
                    "bmu_index": neuron_id,
                    "bmu_pos_0": float(
                        coordinates[0, neuron_id]
                    ),
                    "bmu_pos_1": float(
                        coordinates[1, neuron_id]
                    ),
                    "bmu_distance": float(
                        distance[i]
                    ),
                }
            )


def build_neuron_rows(
    split: str,
    arrays: dict[str, np.ndarray],
    bmu: np.ndarray,
    distance: np.ndarray,
    coordinates: np.ndarray,
) -> list[dict]:
    """Build RQ1 and validation RQ2 statistics per neuron."""
    rows = []

    for neuron_id in range(NUM_NEURONS):
        idx = np.flatnonzero(
            bmu == neuron_id
        )

        support = int(idx.size)

        counts = np.bincount(
            arrays["labels"][idx],
            minlength=CLASS_COUNT,
        ).astype(np.int64)

        if support > 0:
            dominant_class = int(
                np.argmax(counts)
            )

            purity = float(
                counts[dominant_class]
                / support
            )

            entropy = normalized_entropy(
                counts
            )

            mean_distance = safe_mean(
                distance[idx]
            )
        else:
            dominant_class = None
            purity = float("nan")
            entropy = float("nan")
            mean_distance = float("nan")

        row = {
            "split": split,
            "neuron_id": neuron_id,
            "pos_0": float(
                coordinates[0, neuron_id]
            ),
            "pos_1": float(
                coordinates[1, neuron_id]
            ),
            "support": support,
            "dominant_class": dominant_class,
            "purity": purity,
            "normalized_entropy": entropy,
            "mean_bmu_distance": mean_distance,
        }

        for class_id in range(CLASS_COUNT):
            row[
                f"class_{class_id}_count"
            ] = int(counts[class_id])

        if split == "val":
            correct = arrays["correct"][idx]
            error_mask = ~correct

            correct_count = int(
                correct.sum()
            )

            error_count = (
                support
                - correct_count
            )

            error_rate = (
                error_count / support
                if support > 0
                else float("nan")
            )

            row.update(
                {
                    "correct_count": correct_count,
                    "error_count": error_count,
                    "error_rate": error_rate,
                    "error_rate_wilson_lower_95": (
                        wilson_lower_bound(
                            error_count,
                            support,
                        )
                    ),
                    "mean_confidence": (
                        safe_mean(
                            arrays[
                                "confidence"
                            ][idx]
                        )
                    ),
                    "mean_confidence_correct": (
                        safe_mean(
                            arrays[
                                "confidence"
                            ][idx][correct]
                        )
                    ),
                    "mean_confidence_error": (
                        safe_mean(
                            arrays[
                                "confidence"
                            ][idx][
                                error_mask
                            ]
                        )
                    ),
                    "mean_bmu_distance_correct": (
                        safe_mean(
                            distance[idx][correct]
                        )
                    ),
                    "mean_bmu_distance_error": (
                        safe_mean(
                            distance[idx][
                                error_mask
                            ]
                        )
                    ),
                }
            )

        rows.append(row)

    return rows


def write_rows(
    path: Path,
    rows: list[dict],
) -> None:
    """Write dictionaries to CSV."""
    if not rows:
        raise ValueError(
            "Cannot write empty table."
        )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def matrix_from_neuron_values(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Package native NNSOM neuron coordinates with one value per neuron.

    NNSOM's topology positions are not guaranteed to form a rectangular
    Cartesian matrix with exactly grid_width unique x values. The selected
    20x20 SOM uses staggered native positions, so forcing som.pos through
    an imshow-style rectangular coordinate lookup is incorrect.

    Preserve the actual NNSOM coordinates and render neurons directly.
    """
    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if coordinates.shape != (
        2,
        NUM_NEURONS,
    ):
        raise RuntimeError(
            "Unexpected SOM coordinate shape: "
            f"{coordinates.shape}"
        )

    if values.shape != (
        NUM_NEURONS,
    ):
        raise RuntimeError(
            "Unexpected neuron-value shape: "
            f"{values.shape}"
        )

    unique_positions = np.unique(
        coordinates.T,
        axis=0,
    )

    if unique_positions.shape[0] != NUM_NEURONS:
        raise RuntimeError(
            "SOM neuron positions are not unique: "
            f"{unique_positions.shape[0]} unique positions "
            f"for {NUM_NEURONS} neurons."
        )

    return coordinates, values


def save_heatmap(
    path_base: Path,
    matrix: tuple[np.ndarray, np.ndarray],
    title: str,
    colorbar_label: str,
    *,
    categorical: bool = False,
) -> None:
    """Save a SOM neuron map using the model's native topology positions.

    The historical function name is retained to minimize unrelated code
    changes, but rendering is coordinate-based rather than imshow-based.
    """
    coordinates, values = matrix

    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(values)

    if not finite.any():
        raise RuntimeError(
            f"No finite neuron values available for figure: {title}"
        )

    fig = plt.figure(
        figsize=(8.2, 7.2)
    )

    ax = fig.add_axes(
        [0.11, 0.11, 0.72, 0.80]
    )

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
            cmap=plt.get_cmap(
                "tab10",
                CLASS_COUNT,
            ),
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
    ax.set_xlabel(
        "NNSOM topology coordinate 0"
    )
    ax.set_ylabel(
        "NNSOM topology coordinate 1"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        colorbar_label
    )

    if categorical:
        colorbar.set_ticks(
            list(range(CLASS_COUNT))
        )

    fig.savefig(
        path_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        path_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def finite_spearman(
    x: np.ndarray,
    y: np.ndarray,
):
    """Return exploratory Spearman rho or null."""
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if int(mask.sum()) < 3:
        return None

    if (
        np.unique(x[mask]).size < 2
        or np.unique(y[mask]).size < 2
    ):
        return None

    result = spearmanr(
        x[mask],
        y[mask],
    )

    rho = float(
        result.statistic
    )

    if not math.isfinite(rho):
        return None

    return rho


def summarize_rq1(
    rows: list[dict],
) -> dict:
    """Return sample-weighted representation summaries."""
    occupied = [
        row
        for row in rows
        if row["support"] > 0
    ]

    support = np.asarray(
        [
            row["support"]
            for row in occupied
        ],
        dtype=np.float64,
    )

    purity = np.asarray(
        [
            row["purity"]
            for row in occupied
        ],
        dtype=np.float64,
    )

    entropy = np.asarray(
        [
            row[
                "normalized_entropy"
            ]
            for row in occupied
        ],
        dtype=np.float64,
    )

    return {
        "occupied_neurons": len(
            occupied
        ),
        "total_neurons": NUM_NEURONS,
        "occupancy_rate": (
            len(occupied)
            / NUM_NEURONS
        ),
        "sample_weighted_mean_purity": float(
            np.average(
                purity,
                weights=support,
            )
        ),
        "sample_weighted_mean_normalized_entropy": float(
            np.average(
                entropy,
                weights=support,
            )
        ),
    }


def main() -> None:
    """Run the frozen development-stage RQ1/RQ2 analysis."""
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_DIR}"
        )

    if TEMP_DIR.exists():
        raise FileExistsError(
            "Previous temporary analysis directory exists: "
            f"{TEMP_DIR}. Inspect before retrying."
        )

    OUTPUT_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DIR.mkdir()

    assignments_dir = (
        TEMP_DIR / "assignments"
    )

    tables_dir = (
        TEMP_DIR / "tables"
    )

    figures_dir = (
        TEMP_DIR / "figures"
    )

    metadata_dir = (
        TEMP_DIR / "metadata"
    )

    for directory in (
        assignments_dir,
        tables_dir,
        figures_dir,
        metadata_dir,
    ):
        directory.mkdir()

    try:
        model_hash = sha256_file(
            MODEL_PATH
        )

        som = load_selected_som()

        coordinates = som_coordinates(
            som
        )

        frozen = json.loads(
            FROZEN_METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

        if (
            frozen.get(
                "test_evaluated"
            )
            is not False
        ):
            raise RuntimeError(
                "Frozen selected metrics indicate "
                "test evaluation."
            )

        input_hashes = {
            "selected_model": model_hash,
            "frozen_metrics": (
                sha256_file(
                    FROZEN_METRICS_PATH
                )
            ),
        }

        manifest_path = (
            EMBEDDINGS_DIR
            / "embedding_manifest.json"
        )

        if manifest_path.is_file():
            input_hashes[
                "embedding_manifest"
            ] = sha256_file(
                manifest_path
            )

        split_results = {}

        for split in (
            "train",
            "val",
        ):
            print(
                f"\nAnalyzing {split} split..."
            )

            arrays = load_split(
                split
            )

            for suffix in sorted(
                SPLIT_FILES
            ):
                input_hashes[
                    f"{split}_{suffix}"
                ] = sha256_file(
                    EMBEDDINGS_DIR
                    / f"{split}_{suffix}.npy"
                )

            (
                clusters,
                cluster_distances,
                _max_distances,
                cluster_sizes,
            ) = som.cluster_data(
                arrays["features"]
            )

            sizes = np.asarray(
                cluster_sizes,
                dtype=np.int64,
            )

            if int(sizes.sum()) != len(
                arrays["features"]
            ):
                raise RuntimeError(
                    f"{split}: cluster-size sum "
                    "does not equal sample count."
                )

            bmu, distance = (
                reconstruct_assignments(
                    clusters,
                    cluster_distances,
                    len(
                        arrays["features"]
                    ),
                )
            )

            qe = float(
                som.quantization_error(
                    cluster_distances
                )
            )

            occupied = int(
                np.count_nonzero(
                    sizes
                )
            )

            frozen_metrics = frozen[
                (
                    "train_metrics"
                    if split == "train"
                    else "validation_metrics"
                )
            ]

            expected_qe = float(
                frozen_metrics[
                    "quantization_error"
                ]
            )

            expected_occupied = int(
                frozen_metrics[
                    "occupied_neurons"
                ]
            )

            if not np.isclose(
                qe,
                expected_qe,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{split}: QE regression mismatch. "
                    f"Expected {expected_qe}, "
                    f"observed {qe}"
                )

            if occupied != expected_occupied:
                raise RuntimeError(
                    f"{split}: occupancy regression "
                    f"mismatch. Expected "
                    f"{expected_occupied}, observed "
                    f"{occupied}"
                )

            write_sample_assignments(
                assignments_dir
                / f"{split}_bmu_assignments.csv",
                split,
                arrays,
                bmu,
                distance,
                coordinates,
            )

            neuron_rows = (
                build_neuron_rows(
                    split,
                    arrays,
                    bmu,
                    distance,
                    coordinates,
                )
            )

            write_rows(
                tables_dir
                / f"{split}_neurons.csv",
                neuron_rows,
            )

            split_results[split] = {
                "arrays": arrays,
                "bmu": bmu,
                "distance": distance,
                "neuron_rows": neuron_rows,
                "qe": qe,
                "occupied": occupied,
            }

            print(
                f"PASS: {split} assignments = "
                f"{len(bmu)}"
            )

            print(
                f"PASS: {split} QE regression = "
                f"{qe:.12f}"
            )

            print(
                f"PASS: {split} occupied neurons = "
                f"{occupied}"
            )

        train_rows = (
            split_results["train"][
                "neuron_rows"
            ]
        )

        val_rows = (
            split_results["val"][
                "neuron_rows"
            ]
        )

        # ----------------------------------------------------
        # RQ1 figures
        # ----------------------------------------------------

        train_support = np.asarray(
            [
                row["support"]
                for row in train_rows
            ],
            dtype=np.float64,
        )

        train_dominant = np.asarray(
            [
                (
                    np.nan
                    if row[
                        "dominant_class"
                    ]
                    is None
                    else row[
                        "dominant_class"
                    ]
                )
                for row in train_rows
            ],
            dtype=np.float64,
        )

        val_purity = np.asarray(
            [
                row["purity"]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        val_entropy = np.asarray(
            [
                row[
                    "normalized_entropy"
                ]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        save_heatmap(
            figures_dir
            / "rq1_train_hit_count",
            matrix_from_neuron_values(
                coordinates,
                np.log1p(
                    train_support
                ),
            ),
            (
                "RQ1 — Training SOM Hit Map "
                "(log1p count)"
            ),
            "log1p(sample hits)",
        )

        save_heatmap(
            figures_dir
            / "rq1_train_dominant_class",
            matrix_from_neuron_values(
                coordinates,
                train_dominant,
            ),
            (
                "RQ1 — Dominant Ground-Truth "
                "Class per Training Neuron"
            ),
            "MNIST class",
            categorical=True,
        )

        save_heatmap(
            figures_dir
            / "rq1_validation_purity",
            matrix_from_neuron_values(
                coordinates,
                val_purity,
            ),
            (
                "RQ1 — Validation Class "
                "Purity"
            ),
            "purity",
        )

        save_heatmap(
            figures_dir
            / "rq1_validation_entropy",
            matrix_from_neuron_values(
                coordinates,
                val_entropy,
            ),
            (
                "RQ1 — Validation Normalized "
                "Class Entropy"
            ),
            "normalized entropy",
        )

        # ----------------------------------------------------
        # RQ2 tables + figures
        # ----------------------------------------------------

        occupied_val_rows = [
            row
            for row in val_rows
            if row["support"] > 0
        ]

        hotspot_rows = sorted(
            occupied_val_rows,
            key=lambda row: (
                -row[
                    "error_rate_wilson_lower_95"
                ],
                -row[
                    "error_count"
                ],
                -row[
                    "support"
                ],
                row[
                    "neuron_id"
                ],
            ),
        )

        top_hotspots = (
            hotspot_rows[:20]
        )

        write_rows(
            tables_dir
            / "validation_hotspot_top20.csv",
            top_hotspots,
        )

        error_count = np.asarray(
            [
                row["error_count"]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        error_rate = np.asarray(
            [
                row["error_rate"]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        wilson_lower = np.asarray(
            [
                row[
                    "error_rate_wilson_lower_95"
                ]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        mean_confidence = np.asarray(
            [
                row[
                    "mean_confidence"
                ]
                for row in val_rows
            ],
            dtype=np.float64,
        )

        save_heatmap(
            figures_dir
            / "rq2_validation_error_count",
            matrix_from_neuron_values(
                coordinates,
                error_count,
            ),
            (
                "RQ2 — Validation Error "
                "Count by SOM Neuron"
            ),
            "error count",
        )

        save_heatmap(
            figures_dir
            / "rq2_validation_error_rate",
            matrix_from_neuron_values(
                coordinates,
                error_rate,
            ),
            (
                "RQ2 — Validation Error "
                "Rate by SOM Neuron"
            ),
            "error rate",
        )

        save_heatmap(
            figures_dir
            / "rq2_validation_wilson_lower95",
            matrix_from_neuron_values(
                coordinates,
                wilson_lower,
            ),
            (
                "RQ2 — Support-Aware Validation "
                "Error Priority"
            ),
            (
                "Wilson 95% lower bound "
                "of error rate"
            ),
        )

        save_heatmap(
            figures_dir
            / "rq2_validation_mean_confidence",
            matrix_from_neuron_values(
                coordinates,
                mean_confidence,
            ),
            (
                "RQ2 — Mean Baseline Confidence "
                "by Validation Neuron"
            ),
            "mean confidence",
        )

        # ----------------------------------------------------
        # Overall summaries
        # ----------------------------------------------------

        val_arrays = (
            split_results["val"][
                "arrays"
            ]
        )

        val_distance = (
            split_results["val"][
                "distance"
            ]
        )

        val_correct = (
            val_arrays["correct"]
        )

        total_errors = int(
            (~val_correct).sum()
        )

        overall_error_rate = (
            total_errors
            / len(val_correct)
        )

        by_error_count = sorted(
            occupied_val_rows,
            key=lambda row: (
                -row["error_count"],
                -row["error_rate"],
                -row["support"],
                row["neuron_id"],
            ),
        )

        top10_error_count = sum(
            int(row["error_count"])
            for row in by_error_count[:10]
        )

        top10_error_fraction = (
            top10_error_count
            / total_errors
            if total_errors > 0
            else 0.0
        )

        val_error_rates = np.asarray(
            [
                row["error_rate"]
                for row in occupied_val_rows
            ],
            dtype=np.float64,
        )

        val_entropies = np.asarray(
            [
                row[
                    "normalized_entropy"
                ]
                for row in occupied_val_rows
            ],
            dtype=np.float64,
        )

        val_purities = np.asarray(
            [
                row["purity"]
                for row in occupied_val_rows
            ],
            dtype=np.float64,
        )

        val_mean_distances = np.asarray(
            [
                row[
                    "mean_bmu_distance"
                ]
                for row in occupied_val_rows
            ],
            dtype=np.float64,
        )

        summary = {
            "run_id": RUN_ID,
            "stage": "development",
            "selected_grid": [
                GRID_HEIGHT,
                GRID_WIDTH,
            ],
            "seed": 42,
            "selected_model": str(
                MODEL_PATH.relative_to(
                    REPO_ROOT
                )
            ),
            "selected_model_sha256": (
                model_hash
            ),
            "git_commit": subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                text=True,
            ).strip(),
            "nnsom_version": version(
                "NNSOM"
            ),
            "test_evaluated": False,
            "inputs": input_hashes,
            "regression_checks": {
                "train_quantization_error": (
                    split_results[
                        "train"
                    ]["qe"]
                ),
                "validation_quantization_error": (
                    split_results[
                        "val"
                    ]["qe"]
                ),
                "train_occupied_neurons": (
                    split_results[
                        "train"
                    ]["occupied"]
                ),
                "validation_occupied_neurons": (
                    split_results[
                        "val"
                    ]["occupied"]
                ),
                "status": "PASS",
            },
            "rq1": {
                "train": summarize_rq1(
                    train_rows
                ),
                "validation": summarize_rq1(
                    val_rows
                ),
            },
            "rq2": {
                "validation_samples": int(
                    len(val_correct)
                ),
                "validation_errors": (
                    total_errors
                ),
                "overall_validation_error_rate": (
                    float(
                        overall_error_rate
                    )
                ),
                "top10_error_count": (
                    top10_error_count
                ),
                "fraction_of_validation_errors_in_top10_error_count_neurons": (
                    float(
                        top10_error_fraction
                    )
                ),
                "mean_bmu_distance_correct": (
                    json_float(
                        safe_mean(
                            val_distance[
                                val_correct
                            ]
                        )
                    )
                ),
                "mean_bmu_distance_error": (
                    json_float(
                        safe_mean(
                            val_distance[
                                ~val_correct
                            ]
                        )
                    )
                ),
                "median_bmu_distance_correct": (
                    json_float(
                        safe_median(
                            val_distance[
                                val_correct
                            ]
                        )
                    )
                ),
                "median_bmu_distance_error": (
                    json_float(
                        safe_median(
                            val_distance[
                                ~val_correct
                            ]
                        )
                    )
                ),
                "mean_confidence_correct": (
                    json_float(
                        safe_mean(
                            val_arrays[
                                "confidence"
                            ][
                                val_correct
                            ]
                        )
                    )
                ),
                "mean_confidence_error": (
                    json_float(
                        safe_mean(
                            val_arrays[
                                "confidence"
                            ][
                                ~val_correct
                            ]
                        )
                    )
                ),
                "exploratory_spearman": {
                    "error_rate_vs_normalized_entropy": (
                        finite_spearman(
                            val_error_rates,
                            val_entropies,
                        )
                    ),
                    "error_rate_vs_purity": (
                        finite_spearman(
                            val_error_rates,
                            val_purities,
                        )
                    ),
                    "error_rate_vs_mean_bmu_distance": (
                        finite_spearman(
                            val_error_rates,
                            val_mean_distances,
                        )
                    ),
                },
                "hotspot_ranking": (
                    "Wilson 95% lower bound descending; "
                    "error count descending; "
                    "support descending; "
                    "neuron index ascending."
                ),
            },
        }

        (
            metadata_dir
            / "analysis_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # Input/traceability manifest.
        (
            metadata_dir
            / "input_hashes.json"
        ).write_text(
            json.dumps(
                input_hashes,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Final artifact hashes before atomic promotion.
        artifact_hashes = {}

        for path in sorted(
            TEMP_DIR.rglob("*")
        ):
            if path.is_file():
                artifact_hashes[
                    str(
                        path.relative_to(
                            TEMP_DIR
                        )
                    )
                ] = sha256_file(
                    path
                )

        (
            metadata_dir
            / "artifact_hashes.json"
        ).write_text(
            json.dumps(
                artifact_hashes,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Re-hash artifact-hashes file itself is intentionally
        # unnecessary; it is the hash manifest, not an input result.

        TEMP_DIR.rename(
            OUTPUT_DIR
        )

        print()
        print("=" * 72)
        print(
            "RQ1/RQ2 ANALYSIS COMPLETE"
        )
        print("=" * 72)

        print(
            f"Output: {OUTPUT_DIR}"
        )

        print(
            "Test evaluated: NO"
        )

        print(
            "Regression checks: PASS"
        )

        print(
            "Validation errors: "
            f"{total_errors}/"
            f"{len(val_correct)} "
            f"({overall_error_rate:.6f})"
        )

        print(
            "Top-10 error-count neurons contain: "
            f"{top10_error_count}/"
            f"{total_errors} validation errors "
            f"({top10_error_fraction:.4f})"
        )

    except Exception:
        print(
            "\nANALYSIS FAILED."
        )

        print(
            "Temporary artifacts preserved for inspection:"
        )

        print(
            TEMP_DIR
        )

        raise


if __name__ == "__main__":
    main()
