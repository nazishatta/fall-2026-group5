"""Week 5 SOM error-geography analysis.

Development-stage analysis using validation data only.
The held-out test split is intentionally excluded.

This module consumes frozen embeddings and previously generated BMU
assignments. It does not modify or retrain the CNN or SOM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.v1_mnist.component.analysis.io import load_split


REPO_ROOT = Path(__file__).resolve().parents[4]

EMBEDDINGS_DIR = (
    REPO_ROOT / "outputs/v1_mnist/cnn_baseline/embeddings/mnist"
)

CLUSTER_DIR = REPO_ROOT / "outputs/v1_mnist/cluster_analysis"
TABLES_DIR = CLUSTER_DIR / "tables"

NUM_NEURONS = 225
NUM_CLASSES = 10

# Development-stage hotspot definition.
# A neuron must have adequate validation support and elevated error evidence.
MIN_SUPPORT = 20
WILSON_Z = 1.959963984540054


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
) -> list[dict]:
    """Calculate error-geography statistics for every SOM neuron."""

    rows = []

    global_error_rate = float(np.mean(~correct))

    for neuron_id in range(NUM_NEURONS):
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
    arrays = load_split("val", embeddings_dir=EMBEDDINGS_DIR)

    bmu = np.load(CLUSTER_DIR / "val_bmu.npy")
    distance = np.load(CLUSTER_DIR / "val_bmu_distance.npy")

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
    )

    write_rows(TABLES_DIR / "val_error_geography.csv", rows)

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
        write_rows(TABLES_DIR / "val_hotspots.csv", hotspot_rows)

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
            TABLES_DIR / "val_confusion_pairs.csv",
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
            TABLES_DIR / "val_misclassified_samples.csv",
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

    with (CLUSTER_DIR / "error_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nWeek 5 Error Geography COMPLETE")
    print(f"Validation samples: {n}")
    print(f"Validation errors: {global_errors}")
    print(f"Global error rate: {global_error_rate:.6f}")
    print(f"Hotspot neurons: {len(hotspot_rows)}")
    print(f"Errors inside hotspots: {hotspot_error_total}")

    if global_errors:
        print(
            "Fraction of errors captured by hotspots: "
            f"{hotspot_error_total / global_errors:.4f}"
        )

    print("\nTop hotspots:")

    for row in hotspot_rows[:10]:
        print(
            f"  neuron={row['neuron_id']:3d} "
            f"support={row['support']:3d} "
            f"errors={row['error_count']:2d} "
            f"rate={row['error_rate']:.3f} "
            f"wilson={row['wilson_lower_95']:.3f}"
        )

    print("\nHeld-out test split was NOT used.")


if __name__ == "__main__":
    main()
