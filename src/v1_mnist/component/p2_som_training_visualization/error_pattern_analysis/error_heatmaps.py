"""Week 5 heatmaps of the SOM (Error Pattern Analysis).

Reads the per-neuron error table written by error_patterns.py and draws one
value per neuron at the neuron's position on the map (NNSOM topology):
validation error count, error rate, 95% Wilson lower bound of the error rate,
mean CNN confidence, and the hotspot map (1 = hotspot, by the project's one
hotspot rule).

Output: outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/<selected_model>/figures/pdf/
Week 4 heatmaps: cluster_analysis/cluster_heatmaps.py.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/error_heatmaps.py [--config CONFIG]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.heatmaps import (  # noqa: E402
    column,
    matrix_from_neuron_values,
    read_neuron_table,
    save_heatmap,
)
from src.v1_mnist.component.utils.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    error_output_dir,
)
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402

logger = get_logger("v1_mnist.analysis.error_patterns.error_heatmaps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 SOM heatmaps (error pattern analysis).")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    args = parser.parse_args()

    base = error_output_dir(args.config, REPO_ROOT)
    table = base / "tables" / "val_error_geography.csv"
    if not table.is_file():
        raise FileNotFoundError(f"{table} not found: run error_patterns.py first.")
    figdir = base / "figures"

    coordinates, rows = read_neuron_table(table)
    occupied = column(rows, "support") > 0
    hotspot = np.where(
        occupied,
        np.asarray([1.0 if r["hotspot"].lower() == "true" else 0.0 for r in rows]),
        np.nan,
    )

    figures = [
        (column(rows, "error_count"), "validation_error_count",
         "Validation Error Count by SOM Neuron", "error count"),
        (column(rows, "error_rate"), "validation_error_rate",
         "Validation Error Rate by SOM Neuron", "error rate"),
        (column(rows, "wilson_lower_95"), "validation_wilson_lower95",
         "Support-Aware Validation Error Priority", "Wilson 95% lower bound of error rate"),
        (column(rows, "mean_confidence"), "validation_mean_confidence",
         "Mean Baseline Confidence by Validation Neuron", "mean confidence"),
        (hotspot, "validation_hotspot_map",
         "Validation Error Hotspots (project hotspot rule)", "hotspot (1) / other (0)"),
    ]
    for values, name, title, label in figures:
        save_heatmap(figdir / name, matrix_from_neuron_values(coordinates, values), title, label)

    logger.info("Week 5 heatmaps (%d) written to %s", len(figures), figdir)


if __name__ == "__main__":
    main()
