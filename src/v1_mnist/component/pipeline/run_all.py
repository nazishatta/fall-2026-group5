"""Run the step runners in proposal order.

  Phase 1  data_pipeline            p1_model_training_baseline/run_data_pipeline.py
           cnn_baseline_training    p1_model_training_baseline/run_cnn_baseline_training.py
  Phase 2  som_nnsom_training       p2_som_training_visualization/run_som_nnsom_training.py
           cluster_analysis         p2_som_training_visualization/run_cluster_analysis.py
           error_pattern_analysis   p2_som_training_visualization/run_error_pattern_analysis.py
           (report)                 p2_som_training_visualization/run_interactive_report.py,
                                    after error_pattern_analysis

By default it starts at som_nnsom_training: data_pipeline and cnn_baseline_training
produce the frozen checkpoint and embeddings that every SOM result depends on.
Use --from data_pipeline to run everything.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/run_all.py [--from STEP] [--to STEP]
      [--cnn-config CFG] [--som-config CFG] [--skip-training] [--no-report] [--dry-run]

Each runner keeps its own protection: a step whose results already exist stops
the whole run (results are never overwritten; move or delete them to redo a step).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    CNN_CONFIG, CNN_STEPS, REPORT_RUNNER, SOM_CONFIG, STEP_RUNNERS, Step, run_steps,
)

STEP_NAMES = list(STEP_RUNNERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the step runners in proposal order.")
    parser.add_argument("--from", dest="first", default="som_nnsom_training", choices=STEP_NAMES,
                        help="First step to run (default: som_nnsom_training).")
    parser.add_argument("--to", dest="last", default="error_pattern_analysis", choices=STEP_NAMES,
                        help=f"Last step to run (default: {STEP_NAMES[-1]}).")
    parser.add_argument("--cnn-config", default=CNN_CONFIG, help=f"Config for {' and '.join(CNN_STEPS)}.")
    parser.add_argument("--som-config", default=SOM_CONFIG, help="Config for the SOM steps and the report.")
    parser.add_argument("--skip-training", action="store_true",
                        help="som_nnsom_training: use the existing SOM instead of training a new one.")
    parser.add_argument("--no-report", action="store_true", help="Do not build the interactive report.")
    parser.add_argument("--dry-run", action="store_true", help="Print every runner's steps without running them.")
    parser.add_argument(
        "--phase3-stage",
        default="qe_novelty",
        help="Phase 3 stage when phase3_novelty is selected.",
    )
    parser.add_argument(
        "--phase3-config",
        default="src/v1_mnist/component/configs/qe_novelty.yaml",
        help="Configuration for the selected Phase 3 stage.",
    )
    args = parser.parse_args()
    start, end = STEP_NAMES.index(args.first), STEP_NAMES.index(args.last)
    if start > end:
        parser.error("--from must not come after --to")

    # Each runner gets --dry-run itself, so a dry run lists every runner's own steps.
    common = ["--dry-run"] if args.dry_run else []
    plan = []
    for name in STEP_NAMES[start:end + 1]:
        if name == "phase3_novelty":
            extra = [
                "--stage", args.phase3_stage,
                "--config", args.phase3_config,
            ]
        else:
            extra = [
                "--config",
                args.cnn_config if name in CNN_STEPS else args.som_config,
            ]
        if name == "som_nnsom_training" and args.skip_training:
            extra.append("--skip-training")
        plan.append(Step(name, STEP_RUNNERS[name], [*extra, *common]))
    if args.last == "error_pattern_analysis" and not args.no_report:
        plan.append(Step("interactive report", REPORT_RUNNER, ["--config", args.som_config, *common]))

    run_steps("run_all: " + ", ".join(step.label for step in plan), plan, dry_run=False)


if __name__ == "__main__":
    main()
