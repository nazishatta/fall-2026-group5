"""BMU, representation-structure, and error-geography analysis.

This module provides the top-level CLI and exports modularized components:
  - io.py: file hashing, data loading, SOM reconstruction, CSV writing
  - metrics.py: entropy, Wilson confidence bound, Spearman correlation
  - plotting.py: coordinate packaging, SVG/PDF heatmap rendering
  - rq1.py: neuron row aggregation, RQ1 summaries, and execution pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.v1_mnist.component.analysis.io import (
    FEATURE_DIM,
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_NEURONS,
    SPLIT_FILES,
    json_float,
    load_selected_som,
    load_split,
    reconstruct_assignments,
    sha256_file,
    som_coordinates,
    write_rows,
    write_sample_assignments,
)
from src.v1_mnist.component.analysis.metrics import (
    CLASS_COUNT,
    finite_spearman,
    normalized_entropy,
    safe_mean,
    safe_median,
    wilson_lower_bound,
)
from src.v1_mnist.component.analysis.plotting import (
    matrix_from_neuron_values,
    save_heatmap,
)
from src.v1_mnist.component.analysis.rq1 import (
    build_neuron_rows,
    run_rq1_analysis,
    summarize_rq1,
)
from src.v1_mnist.component.som.data import resolve_embeddings_dir
from src.v1_mnist.component.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "som_15x15_seed42_rq1_rq2_v1"
EMBEDDINGS_DIR = resolve_embeddings_dir(
    "outputs/v1_mnist/cnn_baseline/embeddings/mnist",
    REPO_ROOT,
)
MODEL_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/som_models"
    / "som_15x15_seed42_final"
)
FROZEN_METRICS_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/logs"
    / "som_15x15_seed42_final_metrics.json"
)
OUTPUT_BASE = REPO_ROOT / "outputs/v1_mnist/som/analysis"
OUTPUT_DIR = OUTPUT_BASE / RUN_ID
TEMP_DIR = OUTPUT_BASE / f".{RUN_ID}.tmp"

logger = get_logger("v1_mnist.analysis.som_bmu_rq_analysis")

__all__ = [
    "CLASS_COUNT",
    "EMBEDDINGS_DIR",
    "FEATURE_DIM",
    "FROZEN_METRICS_PATH",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "MODEL_PATH",
    "NUM_NEURONS",
    "OUTPUT_BASE",
    "OUTPUT_DIR",
    "REPO_ROOT",
    "RUN_ID",
    "SPLIT_FILES",
    "TEMP_DIR",
    "build_neuron_rows",
    "finite_spearman",
    "json_float",
    "load_selected_som",
    "load_split",
    "main",
    "matrix_from_neuron_values",
    "normalized_entropy",
    "reconstruct_assignments",
    "run_rq1_analysis",
    "safe_mean",
    "safe_median",
    "save_heatmap",
    "sha256_file",
    "som_coordinates",
    "summarize_rq1",
    "wilson_lower_bound",
    "write_rows",
    "write_sample_assignments",
]


def main() -> None:
    """Run the frozen development-stage RQ1/RQ2 analysis."""
    run_rq1_analysis(output_dir=OUTPUT_DIR, repo_root=REPO_ROOT)


if __name__ == "__main__":
    main()
