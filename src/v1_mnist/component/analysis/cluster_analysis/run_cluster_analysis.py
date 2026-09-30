"""Week 4 SOM cluster analysis.

Consumes the frozen SOM and frozen CNN embeddings read-only.
Does not train or modify the CNN, embeddings, or SOM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.v1_mnist.component.analysis.io import (
    load_selected_som,
    load_split,
    reconstruct_assignments,
)
from src.v1_mnist.component.cluster_analysis.cluster_metrics import (
    class_neuron_intersection,
    compute_cluster_metrics,
    weighted_cluster_purity,
)


REPO_ROOT = Path(__file__).resolve().parents[4]

EMBEDDINGS_DIR = (
    REPO_ROOT / "outputs/v1_mnist/cnn_baseline/embeddings/mnist"
)

MODEL_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/som_models"
    / "som_15x15_seed42_final"
)

OUTPUT_DIR = REPO_ROOT / "outputs/v1_mnist/cluster_analysis"

NUM_NEURONS = 225
NUM_CLASSES = 10


def write_cluster_table(path: Path, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "neuron_id",
        "support",
        "dominant_class",
        "dominant_count",
        "purity",
        "normalized_entropy",
    ] + [f"class_{i}_count" for i in range(NUM_CLASSES)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for metric in metrics:
            row = {
                "neuron_id": metric.neuron_id,
                "support": metric.support,
                "dominant_class": metric.dominant_class,
                "dominant_count": metric.dominant_count,
                "purity": metric.purity,
                "normalized_entropy": metric.normalized_entropy,
            }

            for class_id, count in enumerate(metric.class_counts):
                row[f"class_{class_id}_count"] = count

            writer.writerow(row)


def analyze_split(split: str, som) -> dict:
    print(f"\nAnalyzing {split}...")

    arrays = load_split(split, embeddings_dir=EMBEDDINGS_DIR)

    clusters, cluster_distances, _, cluster_sizes = som.cluster_data(
        arrays["features"]
    )

    bmu, distances = reconstruct_assignments(
        clusters,
        cluster_distances,
        len(arrays["features"]),
    )

    metrics = compute_cluster_metrics(
        arrays["labels"],
        bmu,
        num_neurons=NUM_NEURONS,
        num_classes=NUM_CLASSES,
    )

    intersection = class_neuron_intersection(
        arrays["labels"],
        bmu,
        num_neurons=NUM_NEURONS,
        num_classes=NUM_CLASSES,
    )

    qe = float(som.quantization_error(cluster_distances))
    occupied = int(np.count_nonzero(np.asarray(cluster_sizes)))

    weighted_purity = weighted_cluster_purity(metrics)

    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_cluster_table(
        tables_dir / f"{split}_cluster_metrics.csv",
        metrics,
    )

    np.savetxt(
        tables_dir / f"{split}_class_neuron_intersection.csv",
        intersection,
        delimiter=",",
        fmt="%d",
    )

    np.save(
        OUTPUT_DIR / f"{split}_bmu.npy",
        bmu,
    )

    np.save(
        OUTPUT_DIR / f"{split}_bmu_distance.npy",
        distances,
    )

    summary = {
        "split": split,
        "samples": int(len(arrays["labels"])),
        "occupied_neurons": occupied,
        "total_neurons": NUM_NEURONS,
        "occupancy_rate": float(occupied / NUM_NEURONS),
        "quantization_error": qe,
        "weighted_cluster_purity": weighted_purity,
    }

    print(f"Samples:             {summary['samples']}")
    print(f"Occupied neurons:    {occupied}/{NUM_NEURONS}")
    print(f"Quantization error:  {qe:.12f}")
    print(f"Weighted purity:     {weighted_purity:.6f}")

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen SOM...")
    som = load_selected_som(MODEL_PATH)

    results = {
        "analysis": "week4_cluster_analysis",
        "development_only": True,
        "test_used": False,
        "som": "som_15x15_seed42_final",
        "splits": {},
    }

    for split in ("train", "val"):
        results["splits"][split] = analyze_split(split, som)

    with (OUTPUT_DIR / "cluster_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)

    print("\nWeek 4 cluster analysis COMPLETE.")
    print(f"Results: {OUTPUT_DIR}")
    print("Held-out test split was NOT used.")


if __name__ == "__main__":
    main()
