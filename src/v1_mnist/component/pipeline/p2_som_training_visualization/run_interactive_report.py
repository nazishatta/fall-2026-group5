"""Interactive HTML report for the trained SOM.

Steps:
  1. visualize_som_interactive: one self-contained HTML page with the SOM map,
     class composition, errors and the samples behind each neuron.

Result: outputs/v1_mnist/p2_som_training_visualization/interactive_report/interactive_report.html

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_interactive_report.py \
      [--config CONFIG] [--split train|val] [--dry-run]

The report is a visualization, so it is redrawn when run again.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    P2, SOM_CONFIG, Step, base_parser, load, run_steps,
)


def main() -> None:
    parser = base_parser("Interactive HTML report for the trained SOM.", SOM_CONFIG)
    parser.add_argument("--split", choices=["train", "val"], default=None,
                        help="Split used for the detail panel (default: from the config).")
    args = parser.parse_args()

    config = load(args.config)
    model = str(config.visualization.selected_model)
    report_args = ["--config", args.config, "--model-name", model] + (["--split", args.split] if args.split else [])
    steps = [Step("Interactive HTML report", P2 / "interactive_report" / "visualize_som_interactive.py", report_args)]
    run_steps(f"Interactive report ({model})", steps, args.dry_run)


if __name__ == "__main__":
    main()
