"""Week 4 SOM cluster analysis.

Consumes the frozen SOM and frozen CNN embeddings read-only.
Does not train or modify the CNN, embeddings, or SOM.

The SOM (grid size, model, embeddings) comes from a SOM config; the model's
SHA-256 must match run_manifest.csv (see analysis/run_settings.py).
Outputs: outputs/v1_mnist/cluster_analysis/<selected_model>/

Usage (from the code root):
  python src/v1_mnist/component/analysis/cluster_analysis/run_cluster_analysis.py [--config CONFIG] [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.analysis.io import (
    load_selected_som,
    load_split,
    reconstruct_assignments,
)
from src.v1_mnist.component.analysis.cluster_analysis.cluster_metrics import (
    class_neuron_intersection,
    compute_cluster_metrics,
    weighted_cluster_purity,
)
from src.v1_mnist.component.analysis.run_settings import (
    DEFAULT_CONFIG,
    RQSettings,
    cluster_output_dir,
    load_rq_settings,
    verify_selected_model,
)
from src.v1_mnist.component.utils.logging import get_logger


NUM_CLASSES = 10

logger = get_logger("v1_mnist.analysis.cluster_analysis.run_cluster_analysis")


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


def analyze_split(split: str, som, settings: RQSettings, output_dir: Path) -> dict:
    logger.info("Analyzing %s...", split)

    arrays = load_split(split, embeddings_dir=settings.embeddings_dir)

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
        num_neurons=settings.num_neurons,
        num_classes=NUM_CLASSES,
    )

    intersection = class_neuron_intersection(
        arrays["labels"],
        bmu,
        num_neurons=settings.num_neurons,
        num_classes=NUM_CLASSES,
    )

    qe = float(som.quantization_error(cluster_distances))
    occupied = int(np.count_nonzero(np.asarray(cluster_sizes)))

    weighted_purity = weighted_cluster_purity(metrics)

    tables_dir = output_dir / "tables"
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
        output_dir / f"{split}_bmu.npy",
        bmu,
    )

    np.save(
        output_dir / f"{split}_bmu_distance.npy",
        distances,
    )

    summary = {
        "split": split,
        "samples": int(len(arrays["labels"])),
        "occupied_neurons": occupied,
        "total_neurons": settings.num_neurons,
        "occupancy_rate": float(occupied / settings.num_neurons),
        "quantization_error": qe,
        "weighted_cluster_purity": weighted_purity,
    }

    logger.info("Samples:             %d", summary["samples"])
    logger.info("Occupied neurons:    %d/%d", occupied, settings.num_neurons)
    logger.info("Quantization error:  %.12f", qe)
    logger.info("Weighted purity:     %.6f", weighted_purity)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 SOM cluster analysis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the grid and selected model.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing outputs for this model.")
    args = parser.parse_args()

    settings = load_rq_settings(args.config, REPO_ROOT)
    model_hash = verify_selected_model(settings)
    logger.info("Model integrity check passed: %s (SHA-256 %s)", settings.selected_model, model_hash)

    output_dir = cluster_output_dir(REPO_ROOT, settings.selected_model)
    summary_path = output_dir / "cluster_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading frozen SOM...")
    som = load_selected_som(settings.model_path, settings.grid_height, settings.grid_width)

    results = {
        "analysis": "week4_cluster_analysis",
        "development_only": True,
        "test_used": False,
        "som": settings.selected_model,
        "model_sha256": model_hash,
        "som_grid": [settings.grid_height, settings.grid_width],
        "config": settings.config_path.name,
        "splits": {},
    }

    for split in ("train", "val"):
        results["splits"][split] = analyze_split(split, som, settings, output_dir)

    with summary_path.open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2)

    logger.info("Week 4 cluster analysis COMPLETE.")
    logger.info("Results: %s", output_dir)
    logger.info("Held-out test split was NOT used.")


if __name__ == "__main__":
    main()
