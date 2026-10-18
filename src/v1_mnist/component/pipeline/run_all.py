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

# Phase 3 configuration only; existing CNN and SOM configs are unchanged.
PHASE3_CONFIGS = {
    "qe_novelty": "src/v1_mnist/component/configs/qe_novelty.yaml",
    "qe_ood": "src/v1_mnist/component/configs/qe_ood_evaluation.yaml",
    "svhn_benchmark": "src/v1_mnist/component/configs/novelty_benchmark.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the step runners in proposal order.")
    parser.add_argument("--from", dest="first", default="som_nnsom_training", choices=STEP_NAMES,
                        help="First step to run (default: som_nnsom_training).")
    parser.add_argument("--to", dest="last", default=STEP_NAMES[-1], choices=STEP_NAMES,
                        help=f"Last step to run (default: {STEP_NAMES[-1]}).")
    parser.add_argument("--cnn-config", default=CNN_CONFIG, help=f"Config for {' and '.join(CNN_STEPS)}.")
    parser.add_argument("--som-config", default=SOM_CONFIG, help="Config for the SOM steps and the report.")
    parser.add_argument("--skip-training", action="store_true",
                        help="som_nnsom_training: use the existing SOM instead of training a new one.")
    parser.add_argument("--no-report", action="store_true", help="Do not build the interactive report.")
    parser.add_argument("--dry-run", action="store_true", help="Print every runner's steps without running them.")
    args = parser.parse_args()
    start, end = STEP_NAMES.index(args.first), STEP_NAMES.index(args.last)
    if start > end:
        parser.error("--from must not come after --to")

    # Each runner gets --dry-run itself, so a dry run lists every runner's own steps.
    common = ["--dry-run"] if args.dry_run else []
    plan = []
    for name in STEP_NAMES[start:end + 1]:
        extra = ["--config", args.cnn_config if name in CNN_STEPS else args.som_config]
        if name == "som_nnsom_training" and args.skip_training:
            extra.append("--skip-training")
        plan.append(Step(name, STEP_RUNNERS[name], [*extra, *common]))

    # Phase 3 integration. Preserve the original Phase 1–2 code above.
    phase2_end = STEP_NAMES.index("error_pattern_analysis")
    if end >= phase2_end:
        # Replace Phase 3 placeholder commands with adapter-specific commands.
        for index, step in enumerate(plan):
            if step.label in PHASE3_CONFIGS:
                plan[index] = Step(
                    step.label,
                    STEP_RUNNERS[step.label],
                    [
                        "--stage", step.label,
                        "--config", PHASE3_CONFIGS[step.label],
                        *common,
                    ],
                )

        # Keep the interactive report directly after Phase 2 when included.
        if start <= phase2_end and not args.no_report:
            report_index = next(
                i for i, step in enumerate(plan)
                if step.label == "error_pattern_analysis"
            )
            plan.insert(
                report_index + 1,
                Step(
                    "interactive report",
                    REPORT_RUNNER,
                    ["--config", args.som_config, *common],
                ),
            )

        run_steps(
            "run_all: " + ", ".join(step.label for step in plan),
            plan,
            dry_run=False,
        )
        return

    if args.last == STEP_NAMES[-1] and not args.no_report:
        plan.append(Step("interactive report", REPORT_RUNNER, ["--config", args.som_config, *common]))

    run_steps("run_all: " + ", ".join(step.label for step in plan), plan, dry_run=False)


if __name__ == "__main__":
    main()
