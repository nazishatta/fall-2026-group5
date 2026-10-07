"""BMU, representation-structure, and error-geography analysis.

The grid and the model come from a SOM config (default configs/som.yaml);
the model's SHA-256 must match run_manifest.csv (see run_settings.py).

This module provides the top-level CLI and exports modularized components:
  - io.py: file hashing, data loading, SOM reconstruction, CSV writing
  - metrics.py: entropy, Wilson confidence bound, Spearman correlation
  - plotting.py: coordinate packaging, SVG/PDF heatmap rendering
  - rq1.py: neuron row aggregation, RQ1 summaries, and execution pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.v1_mnist.component.analysis.run_settings import (
    DEFAULT_CONFIG,
    load_rq_settings,
    verify_selected_model,
)
from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.analysis.som_bmu_rq_analysis")

__all__ = [
    "CLASS_COUNT",
    "DEFAULT_CONFIG",
    "FEATURE_DIM",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "NUM_NEURONS",
    "REPO_ROOT",
    "SPLIT_FILES",
    "build_neuron_rows",
    "finite_spearman",
    "json_float",
    "load_rq_settings",
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
    "verify_selected_model",
    "wilson_lower_bound",
    "write_rows",
    "write_sample_assignments",
]


def main() -> None:
    """Run the RQ1/RQ2 analysis on the SOM named in a config.

    Examples (from the code root):
      python src/v1_mnist/component/analysis/som_bmu_rq_analysis.py
          -> 15x15 headline model (configs/som.yaml)
      python src/v1_mnist/component/analysis/som_bmu_rq_analysis.py \
          --config src/v1_mnist/component/configs/som_20x20_robustness.yaml
    Output goes to outputs/v1_mnist/som/analysis/<selected_model>_rq1_rq2/.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the grid and selected model.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing analysis folder for this model.")
    args = parser.parse_args()
    run_rq1_analysis(repo_root=REPO_ROOT, config_path=args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
