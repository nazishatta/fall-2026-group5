"""Week 5 SOM error-geography analysis.

Development-stage analysis using validation data only.
The held-out test split is intentionally excluded.

This module consumes frozen embeddings and previously generated BMU
assignments. It does not modify or retrain the CNN or SOM.

Reads the Week 4 outputs for the SOM named in a config (default som.yaml):
outputs/v1_mnist/cluster_analysis/<selected_model>/ (run run_cluster_analysis.py first).

Usage (from the code root):
  python src/v1_mnist/component/analysis/error_patterns/error_patterns.py [--config CONFIG]
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

from src.v1_mnist.component.analysis.io import load_split
from src.v1_mnist.component.analysis.run_settings import (
    DEFAULT_CONFIG,
    cluster_output_dir,
    load_rq_settings,
    verify_selected_model,
)
from src.v1_mnist.component.utils.logging import get_logger


NUM_CLASSES = 10

# Development-stage hotspot definition.
# A neuron must have adequate validation support and elevated error evidence.
MIN_SUPPORT = 20
WILSON_Z = 1.959963984540054

logger = get_logger("v1_mnist.analysis.error_patterns.error_patterns")


def wilson_lower_bound(errors: int, total: int) -> float:
    """95% Wilson lower confidence bound for a binomial error rate."""

    if total <= 0:
        return float("nan")

    p = errors / total
    z2 = WILSON_Z**2

    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)

    adjustment = WILSON_Z * np.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * total)) / total
    )

    return float((centre - adjustment) / denominator)


def safe_mean(values: np.ndarray) -> float:
    """Return mean or NaN for an empty array."""

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def build_error_rows(
    labels: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    confidence: np.ndarray,
    bmu: np.ndarray,
    distance: np.ndarray,
    num_neurons: int,
) -> list[dict]:
    """Calculate error-geography statistics for every SOM neuron."""

    rows = []

    global_error_rate = float(np.mean(~correct))

    for neuron_id in range(num_neurons):
        idx = np.flatnonzero(bmu == neuron_id)
        support = int(idx.size)

        if support == 0:
            rows.append(
                {
                    "neuron_id": neuron_id,
                    "support": 0,
                    "correct_count": 0,
                    "error_count": 0,
                    "error_rate": float("nan"),
                    "wilson_lower_95": float("nan"),
                    "mean_confidence": float("nan"),
                    "mean_confidence_correct": float("nan"),
                    "mean_confidence_error": float("nan"),
                    "mean_bmu_distance": float("nan"),
                    "mean_bmu_distance_correct": float("nan"),
                    "mean_bmu_distance_error": float("nan"),
                    "error_rate_ratio_vs_global": float("nan"),
                    "hotspot": False,
                }
            )
            continue

        local_correct = correct[idx]
        local_error = ~local_correct

        correct_count = int(local_correct.sum())
        error_count = int(local_error.sum())
        error_rate = float(error_count / support)
        lower = wilson_lower_bound(error_count, support)

        # Support-aware development hotspot:
        # enough samples AND the 95% lower confidence bound exceeds
        # the overall validation error rate.
        hotspot = bool(
            support >= MIN_SUPPORT
            and lower > global_error_rate
        )

        rows.append(
            {
                "neuron_id": neuron_id,
                "support": support,
                "correct_count": correct_count,
                "error_count": error_count,
                "error_rate": error_rate,
                "wilson_lower_95": lower,
                "mean_confidence": safe_mean(confidence[idx]),
                "mean_confidence_correct": safe_mean(
                    confidence[idx][local_correct]
                ),
                "mean_confidence_error": safe_mean(
                    confidence[idx][local_error]
                ),
                "mean_bmu_distance": safe_mean(distance[idx]),
                "mean_bmu_distance_correct": safe_mean(
                    distance[idx][local_correct]
                ),
                "mean_bmu_distance_error": safe_mean(
                    distance[idx][local_error]
                ),
                "error_rate_ratio_vs_global": (
                    float(error_rate / global_error_rate)
                    if global_error_rate > 0
                    else float("nan")
                ),
                "hotspot": hotspot,
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def confusion_pairs(
    labels: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    bmu: np.ndarray,
    hotspot_neurons: set[int],
) -> list[dict]:
    """Summarize true→predicted confusion pairs inside hotspot neurons."""

    error_idx = np.flatnonzero(~correct)

    counts: dict[tuple[int, int], int] = {}
    hotspot_counts: dict[tuple[int, int], int] = {}

    for i in error_idx:
        pair = (int(labels[i]), int(preds[i]))
        counts[pair] = counts.get(pair, 0) + 1

        if int(bmu[i]) in hotspot_neurons:
            hotspot_counts[pair] = hotspot_counts.get(pair, 0) + 1

    rows = []

    for pair, count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        rows.append(
            {
                "true_class": pair[0],
                "predicted_class": pair[1],
                "validation_error_count": count,
                "hotspot_error_count": hotspot_counts.get(pair, 0),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 SOM error-geography analysis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the grid and selected model.")
    args = parser.parse_args()

    settings = load_rq_settings(args.config, REPO_ROOT)
    model_hash = verify_selected_model(settings)
    cluster_dir = cluster_output_dir(REPO_ROOT, settings.selected_model)
    tables_dir = cluster_dir / "tables"

    summary_path = cluster_dir / "cluster_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{summary_path} not found: run run_cluster_analysis.py with the same config first."
        )
    week4 = json.loads(summary_path.read_text(encoding="utf-8"))
    if week4.get("model_sha256") != model_hash:
        raise RuntimeError(
            f"Week 4 outputs in {cluster_dir} were made from a different model "
            f"({week4.get('model_sha256')}) than {settings.selected_model} ({model_hash})."
        )

    arrays = load_split("val", embeddings_dir=settings.embeddings_dir)

    bmu = np.load(cluster_dir / "val_bmu.npy")
    distance = np.load(cluster_dir / "val_bmu_distance.npy")

    n = len(arrays["labels"])

    if bmu.shape != (n,) or distance.shape != (n,):
        raise RuntimeError("Validation BMU artifacts do not match validation data.")

    rows = build_error_rows(
        labels=arrays["labels"],
        preds=arrays["preds"],
        correct=arrays["correct"],
        confidence=arrays["confidence"],
        bmu=bmu,
        distance=distance,
        num_neurons=settings.num_neurons,
    )

    write_rows(tables_dir / "val_error_geography.csv", rows)

    hotspot_rows = [row for row in rows if row["hotspot"]]

    hotspot_rows = sorted(
        hotspot_rows,
        key=lambda row: (
            row["wilson_lower_95"],
            row["error_count"],
        ),
        reverse=True,
    )

    if hotspot_rows:
        write_rows(tables_dir / "val_hotspots.csv", hotspot_rows)

    hotspot_neurons = {
        int(row["neuron_id"])
        for row in hotspot_rows
    }

    pair_rows = confusion_pairs(
        arrays["labels"],
        arrays["preds"],
        arrays["correct"],
        bmu,
        hotspot_neurons,
    )

    if pair_rows:
        write_rows(
            tables_dir / "val_confusion_pairs.csv",
            pair_rows,
        )

    # Save individual misclassified validation samples for later
    # representative-image retrieval.
    error_idx = np.flatnonzero(~arrays["correct"])

    sample_rows = []

    for i in error_idx:
        sample_rows.append(
            {
                "split_index": int(i),
                "sample_id": int(arrays["sample_ids"][i]),
                "true_class": int(arrays["labels"][i]),
                "predicted_class": int(arrays["preds"][i]),
                "confidence": float(arrays["confidence"][i]),
                "bmu": int(bmu[i]),
                "bmu_distance": float(distance[i]),
                "hotspot": bool(int(bmu[i]) in hotspot_neurons),
            }
        )

    if sample_rows:
        write_rows(
            tables_dir / "val_misclassified_samples.csv",
            sample_rows,
        )

    global_errors = int((~arrays["correct"]).sum())
    global_error_rate = float(np.mean(~arrays["correct"]))

    hotspot_error_total = sum(
        int(row["error_count"])
        for row in hotspot_rows
    )

    summary = {
        "analysis": "week5_error_geography",
        "som": settings.selected_model,
        "model_sha256": model_hash,
        "som_grid": [settings.grid_height, settings.grid_width],
        "development_split": "val",
        "test_used": False,
        "validation_samples": n,
        "validation_errors": global_errors,
        "validation_error_rate": global_error_rate,
        "min_hotspot_support": MIN_SUPPORT,
        "hotspot_rule": (
            "support >= 20 and 95% Wilson lower bound "
            "> global validation error rate"
        ),
        "hotspot_neuron_count": len(hotspot_rows),
        "hotspot_error_count": hotspot_error_total,
        "fraction_of_validation_errors_in_hotspots": (
            float(hotspot_error_total / global_errors)
            if global_errors
            else float("nan")
        ),
    }

    with (cluster_dir / "error_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Week 5 Error Geography COMPLETE")
    logger.info("Validation samples: %d", n)
    logger.info("Validation errors: %d", global_errors)
    logger.info("Global error rate: %.6f", global_error_rate)
    logger.info("Hotspot neurons: %d", len(hotspot_rows))
    logger.info("Errors inside hotspots: %d", hotspot_error_total)

    if global_errors:
        logger.info(
            "Fraction of errors captured by hotspots: %.4f",
            hotspot_error_total / global_errors,
        )

    logger.info("Top hotspots:")

    for row in hotspot_rows[:10]:
        logger.info(
            "neuron=%3d support=%3d errors=%2d rate=%.3f wilson=%.3f",
            row["neuron_id"],
            row["support"],
            row["error_count"],
            row["error_rate"],
            row["wilson_lower_95"],
        )

    logger.info("Held-out test split was NOT used.")


if __name__ == "__main__":
    main()
