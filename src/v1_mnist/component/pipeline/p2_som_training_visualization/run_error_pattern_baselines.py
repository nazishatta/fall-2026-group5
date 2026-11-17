"""Error pattern baselines: is the SOM's error-geography result special, or do simple alternatives find it too?

Stages (each saves to <output_dir>/work/, so a long run can be done in short steps):
  som      hotspot analysis of the headline SOM, checked against the project's own result
  kmeans   k-means cells for the given seeds (default: the seeds in the config) and their hotspot analysis
  final    error detection scores, bootstrap intervals, results.json, summary.csv and the LaTeX table (in the run's output folder)
  table    rewrite summary.csv and the LaTeX table (in the run's output folder) from an existing results.json
  all      som, kmeans and final in order

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_pattern_baselines.py \
      --stage all [--config CONFIG] [--seeds 0 1 2]

Validation split only. Stops if this run's results already exist (results are never overwritten).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.baseline_stages import (  # noqa: E402
    stage_final,
    stage_kmeans,
    stage_som,
    stage_table,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_baseline_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Error pattern baselines (k-means, kNN impurity, max-softmax).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["som", "kmeans", "final", "table", "all"], required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="k-means seeds for the kmeans stage.")
    args = parser.parse_args()

    settings, analysis = load_baseline_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist.error_pattern_baselines", settings.output_dir / "logs" / f"{args.stage}.log")

    if args.stage in ("som", "all"):
        stage_som(settings, analysis, REPO_ROOT)
    if args.stage in ("kmeans", "all"):
        stage_kmeans(settings, analysis, REPO_ROOT, args.seeds or settings.kmeans_seeds)
    if args.stage in ("final", "all"):
        stage_final(settings, analysis, REPO_ROOT)
    if args.stage == "table":
        stage_table(settings)


if __name__ == "__main__":
    main()
