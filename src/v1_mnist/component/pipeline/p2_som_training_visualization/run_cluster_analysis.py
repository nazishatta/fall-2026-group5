"""Cluster analysis of the trained SOM.

Steps:
  1. run_cluster_analysis: assign every train/val sample to its BMU (model and
     embedding checks first); per-neuron class counts (NNSOM), purity, entropy and
     the class-neuron intersection table; each image's BMU and BMU distance.
  2. cluster_figures: class maps, purity map, class pie and stem.
  3. cluster_heatmaps: cluster heatmaps (hit count, purity, entropy, dominant class).

Results go to outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_cluster_analysis.py \
      [--config CONFIG] [--dry-run]

Needs the trained SOM (run_som_nnsom_training.py). Stops if this SOM's cluster
results already exist (results are never overwritten; move or delete them to redo).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    P2, SOM_CONFIG, Step, base_parser, check_not_done, load, resolve, run_steps,
)


def main() -> None:
    parser = base_parser("Cluster analysis of the trained SOM.", SOM_CONFIG)
    args = parser.parse_args()

    config = load(args.config)
    model = str(config.visualization.selected_model)
    cluster_dir = resolve(config.paths.cluster_dir) / model
    check_not_done([cluster_dir / "cluster_summary.json"], args.dry_run, hint=f"Folder: {cluster_dir}")

    folder = P2 / "cluster_analysis"
    steps = [
        Step("BMU assignment, purity, entropy, class-neuron intersection",
             folder / "run_cluster_analysis.py", ["--config", args.config]),
        Step("Class maps, purity map, class pie and stem",
             folder / "cluster_figures.py", ["--config", args.config, "--model-name", model]),
        Step("Cluster heatmaps", folder / "cluster_heatmaps.py", ["--config", args.config]),
    ]
    run_steps(f"Cluster analysis ({model})", steps, args.dry_run)


if __name__ == "__main__":
    main()
