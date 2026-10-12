"""Week 4 heatmaps of the SOM (Cluster Analysis).

Reads the per-neuron tables written by run_cluster_analysis.py and draws one
value per neuron at the neuron's position on the map (NNSOM topology):
training hit count (log scale), training dominant class, validation hit
count, validation purity, validation class entropy and validation dominant
class. Class count and names come from the config.

Output: outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/<selected_model>/figures/pdf/
Week 5 heatmaps: error_pattern_analysis/error_heatmaps.py.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/cluster_heatmaps.py [--config CONFIG]
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
    cluster_output_dir,
    load_dataset_info,
)
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402

logger = get_logger("v1_mnist.analysis.cluster_analysis.cluster_heatmaps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 SOM heatmaps (cluster analysis).")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    args = parser.parse_args()

    info = load_dataset_info(args.config, REPO_ROOT)
    class_axis = f"{info.display_name} class"
    base = cluster_output_dir(args.config, REPO_ROOT)
    tables = base / "tables"
    if not (tables / "val_cluster_metrics.csv").is_file():
        raise FileNotFoundError(f"{tables} has no Week 4 tables: run run_cluster_analysis.py first.")
    figdir = base / "figures"

    train_xy, train_rows = read_neuron_table(tables / "train_cluster_metrics.csv")
    val_xy, val_rows = read_neuron_table(tables / "val_cluster_metrics.csv")
    table_classes = sum(1 for name in val_rows[0] if name.startswith("class_") and name.endswith("_count"))
    if table_classes != info.num_classes:
        raise RuntimeError(f"Week 4 tables have {table_classes} classes; the config says {info.num_classes}.")

    figures = [
        (train_xy, np.log1p(column(train_rows, "support")), "train_hit_count",
         "Training SOM Hit Map (log1p count)", "log1p(sample hits)", False),
        (train_xy, column(train_rows, "dominant_class"), "train_dominant_class",
         "Dominant Ground-Truth Class per Training Neuron", class_axis, True),
        (val_xy, column(val_rows, "support"), "validation_hit_count",
         "Validation SOM Hit Count", "Samples per neuron", False),
        (val_xy, column(val_rows, "purity"), "validation_purity",
         "Validation Class Purity", "Purity", False),
        (val_xy, column(val_rows, "normalized_entropy"), "validation_entropy",
         "Validation Normalized Class Entropy", "Normalized entropy", False),
        (val_xy, column(val_rows, "dominant_class"), "validation_dominant_class",
         "Validation Dominant True Class", class_axis, True),
    ]
    for coordinates, values, name, title, label, categorical in figures:
        save_heatmap(
            figdir / name,
            matrix_from_neuron_values(coordinates, values),
            title,
            label,
            categorical=categorical,
            num_classes=info.num_classes if categorical else None,
        )

    logger.info("Week 4 heatmaps (%d) written to %s", len(figures), figdir)


if __name__ == "__main__":
    main()
