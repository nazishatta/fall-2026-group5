"""Publication-quality figures for Week 4 and Week 5 SOM analysis."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE = REPO_ROOT / "outputs/v1_mnist/cluster_analysis"
TABLES = BASE / "tables"

WEEK4 = BASE / "week4_cluster_analysis"
WEEK5 = BASE / "week5_error_patterns"

GRID = 15


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix(values):
    arr = np.asarray(values, dtype=float)
    if arr.size != GRID * GRID:
        raise ValueError(f"Expected 225 neuron values, got {arr.size}")
    return arr.reshape(GRID, GRID)


def save_heatmap(values, title, label, path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(matrix(values), origin="upper", aspect="equal")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)

    ax.set_title(title)
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")

    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))

    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def week4():
    rows = read_csv(TABLES / "val_cluster_metrics.csv")

    purity = [float(r["purity"]) for r in rows]
    entropy = [float(r["normalized_entropy"]) for r in rows]
    support = [float(r["support"]) for r in rows]
    dominant = [
        np.nan if r["dominant_class"] in ("", "None")
        else float(r["dominant_class"])
        for r in rows
    ]

    figdir = WEEK4 / "figures"

    save_heatmap(
        support,
        "Validation SOM Hit Count",
        "Samples per neuron",
        figdir / "week4_val_hit_count",
    )

    save_heatmap(
        purity,
        "Validation SOM Cluster Purity",
        "Purity",
        figdir / "week4_val_cluster_purity",
    )

    save_heatmap(
        entropy,
        "Validation SOM Normalized Class Entropy",
        "Normalized entropy",
        figdir / "week4_val_class_entropy",
    )

    save_heatmap(
        dominant,
        "Validation SOM Dominant True Class",
        "Digit class",
        figdir / "week4_val_dominant_class",
    )


def week5():
    rows = read_csv(TABLES / "val_error_geography.csv")

    error_rate = [float(r["error_rate"]) for r in rows]
    error_count = [float(r["error_count"]) for r in rows]
    wilson = [float(r["wilson_lower_95"]) for r in rows]

    hotspot = [
        1.0 if r["hotspot"].lower() == "true" else 0.0
        for r in rows
    ]

    figdir = WEEK5 / "figures"

    save_heatmap(
        error_rate,
        "Validation Error Rate Across SOM",
        "Error rate",
        figdir / "week5_val_error_rate",
    )

    save_heatmap(
        error_count,
        "Validation Error Count Across SOM",
        "Misclassified samples",
        figdir / "week5_val_error_count",
    )

    save_heatmap(
        wilson,
        "Validation Error Rate: 95% Wilson Lower Bound",
        "Wilson lower bound",
        figdir / "week5_val_error_wilson_lower",
    )

    save_heatmap(
        hotspot,
        "Support-Aware Validation Error Hotspots",
        "Hotspot indicator",
        figdir / "week5_val_hotspot_map",
    )


def main():
    week4()
    week5()

    print("Week 4 and Week 5 figures generated.")
    print(f"Week 4 figures: {WEEK4 / 'figures'}")
    print(f"Week 5 figures: {WEEK5 / 'figures'}")


if __name__ == "__main__":
    main()
