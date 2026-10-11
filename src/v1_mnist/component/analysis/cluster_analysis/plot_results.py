"""Publication-quality figures for Week 4 and Week 5 SOM analysis.

Reads the tables for the SOM named in a config (default som.yaml) from
outputs/v1_mnist/cluster_analysis/<selected_model>/ and takes the grid size
from that run's cluster_summary.json.

Usage (from the code root):
  python src/v1_mnist/component/analysis/cluster_analysis/plot_results.py [--config CONFIG]
"""

import argparse
from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.analysis.run_settings import (
    DEFAULT_CONFIG,
    cluster_output_dir,
    selected_model_from_config,
)
from src.v1_mnist.component.utils.logging import get_logger


# Set in configure() from the config.
BASE: Path = Path()
TABLES: Path = Path()
WEEK4: Path = Path()
WEEK5: Path = Path()
GRID_H = 0
GRID_W = 0


def configure(config_path=None) -> None:
    """Point the module at one SOM's outputs and read its grid size."""
    global BASE, TABLES, WEEK4, WEEK5, GRID_H, GRID_W
    BASE = cluster_output_dir(REPO_ROOT, selected_model_from_config(config_path, REPO_ROOT))
    TABLES = BASE / "tables"
    WEEK4 = BASE / "week4_cluster_analysis"
    WEEK5 = BASE / "week5_error_patterns"
    summary = json.loads((BASE / "cluster_summary.json").read_text(encoding="utf-8"))
    GRID_H, GRID_W = (int(v) for v in summary["som_grid"])

logger = get_logger("v1_mnist.analysis.cluster_analysis.plot_results")


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix(values):
    arr = np.asarray(values, dtype=float)
    if arr.size != GRID_H * GRID_W:
        raise ValueError(f"Expected {GRID_H * GRID_W} neuron values, got {arr.size}")
    return arr.reshape(GRID_H, GRID_W)


def save_heatmap(values, title, label, path, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(matrix(values), origin="upper", aspect="equal")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)

    ax.set_title(title)
    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")

    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))

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
    parser = argparse.ArgumentParser(description="Week 4 and Week 5 SOM figures.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    configure(parser.parse_args().config)

    week4()
    week5()

    logger.info("Week 4 and Week 5 figures generated.")
    logger.info("Week 4 figures: %s", WEEK4 / "figures")
    logger.info("Week 5 figures: %s", WEEK5 / "figures")


if __name__ == "__main__":
    main()
