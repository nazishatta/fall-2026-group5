"""SOM training and visualization with NNSOM.

Steps:
  1. train_som: train the SOM on the frozen train embeddings, tracking the
     quantization error every epoch (skipped with --skip-training).
  2. visualize_som: topology, hit histograms, U-Matrix,
     component planes, feature maps and the other SOM figures.
  3. The three convergence charts (QE, train/validation quality, neighbourhood radius).
  4. grid_ablation: the 10x10 / 15x15 / 20x20 comparison table.

Results go to outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training/
(the convergence charts to its convergence_plots/ folder). demo/ is a hand-picked
showcase: copy the charts you want to show there yourself.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_som_nnsom_training.py \
      [--config CONFIG] [--run-id RUN_ID] [--skip-training] [--dry-run]

--run-id defaults to visualization.selected_model in the config; the figures and
convergence charts are drawn for that run. If a model with
that run ID already exists the runner stops: add --skip-training to redraw everything
from the existing model, or train a new model under a new --run-id. Models are never
overwritten.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    P2, SOM_CONFIG, Step, base_parser, check_not_done, load, resolve, run_steps,
)


def main() -> None:
    parser = base_parser("SOM training and visualization: train the SOM and draw its figures.", SOM_CONFIG)
    parser.add_argument("--run-id", default=None,
                        help="SOM run ID (default: visualization.selected_model in the config).")
    parser.add_argument("--skip-training", action="store_true",
                        help="Use the existing model for this run ID; only redraw figures and tables.")
    args = parser.parse_args()

    config = load(args.config)
    run_id = args.run_id or str(config.visualization.selected_model)
    if not args.skip_training:
        check_not_done(
            [resolve(config.paths.som_models_dir) / run_id], args.dry_run,
            hint="Add --skip-training to redraw the figures from the existing model, or use a new --run-id.",
        )

    folder = P2 / "som_nnsom_training"
    charts = folder / "convergence_plots"
    chart_args = ["--config", args.config, "--run-id", run_id]
    steps = []
    if not args.skip_training:
        train_args = ["--config", args.config, "--run-id", run_id]
        steps.append(Step(f"Train SOM {run_id} (QE tracked every epoch)", folder / "train_som.py", train_args))
    steps += [
        Step("SOM figures: topology, hit histograms, U-Matrix, component planes, feature maps",
             folder / "visualize_som.py",
             ["--config", args.config, "--model-name", run_id]),
        Step("Convergence chart: quantization error per epoch",
             charts / "generate_som_qe_interactive.py", chart_args),
        Step("Convergence chart: train vs. validation quality",
             charts / "generate_som_quality_interactive.py", chart_args),
        Step("Convergence chart: neighbourhood radius per epoch",
             charts / "generate_som_radius_interactive.py", chart_args),
        Step("Grid-size comparison table (10x10 / 15x15 / 20x20)", folder / "grid_ablation.py", ["--config", args.config]),
    ]
    run_steps(f"SOM training and visualization ({run_id})", steps, args.dry_run)


if __name__ == "__main__":
    main()
