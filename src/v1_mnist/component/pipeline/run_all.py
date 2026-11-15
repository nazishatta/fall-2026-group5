"""Run the step runners in proposal order.

  Phase 1  data_pipeline            p1_model_training_baseline/run_data_pipeline.py
           cnn_baseline_training    p1_model_training_baseline/run_cnn_baseline_training.py
  Phase 2  som_nnsom_training       p2_som_training_visualization/run_som_nnsom_training.py
           cluster_analysis         p2_som_training_visualization/run_cluster_analysis.py
           error_pattern_analysis   p2_som_training_visualization/run_error_pattern_analysis.py
           (report)                 p2_som_training_visualization/run_interactive_report.py,
                                    after error_pattern_analysis
  Phase 3  novelty_detection        p3_novelty_detection/run_*.py (three runners, run in the
                                    order their results depend on each other; see
                                    ORDER below)

By default it starts at som_nnsom_training: data_pipeline and cnn_baseline_training
produce the frozen checkpoint and embeddings that every SOM result depends on.
Use --from data_pipeline to start earlier. Phase 3 is not part of the default range:
add --to novelty_detection to run it after the report, or --from novelty_detection to run
only Phase 3. Phase 3 steps whose results exist are skipped; the three steps that only
repeat historical results (qe_ood, qe_random_control, qe_u_matrix) run only with
--with-reproductions.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/run_all.py [--from STEP] [--to STEP]
      [--cnn-config CFG] [--som-config CFG] [--skip-training] [--no-report]
      [--with-reproductions] [--dry-run]

Each runner keeps its own protection: in Phases 1 and 2 a step whose results already exist
stops the whole run; Phase 3 skips finished steps. Results are never overwritten (move or
delete them, or use a new run ID, to redo a step).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    CNN_CONFIG, CNN_STEPS, PIPELINE, REPORT_RUNNER, SOM_CONFIG, STEP_RUNNERS, P3Step, Step, run_steps,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection import (  # noqa: E402
    run_novelty_benchmark, run_qe_novelty_detection, run_u_matrix_analysis,
)

# ---- Phase 3 order -----------------------------------------------------------
#How ``run_all.py`` runs Phase 3: the order that respects which step reads which results.
#
#Phase 3 has three runners, but their steps depend on each other across runners, so
#they cannot simply run one runner after another:
#
#  1. qe_novelty               writes the QE summary that nearly everything reads.
#  2. the whole novelty_benchmark runner
#                              svhn_benchmark writes the SVHN score arrays that the
#                              threshold sweep, the figures and the gap analysis read.
#  3. qe_threshold_sweep (and the reproduction steps, if asked for).
#  4. the whole u_matrix_analysis runner.
#
#Three steps only re-create results that already exist under the historical run
#folders (and that the ``reorg_...`` reproduction runs already repeated). They are
#left out unless ``--with-reproductions`` is given: qe_ood, qe_random_control and
#qe_u_matrix. A step whose results exist is skipped by the runner either way
#(results are never overwritten).

PHASE3_STEP = "novelty_detection"
P3_PIPELINE = PIPELINE / "p3_novelty_detection"
REPRODUCTION_STEPS = frozenset({"qe_ood", "qe_random_control", "qe_u_matrix"})

RUNNER_STEPS: dict[str, list[P3Step]] = {
    "run_qe_novelty_detection.py": run_qe_novelty_detection.STEPS,
    "run_novelty_benchmark.py": run_novelty_benchmark.STEPS,
    "run_u_matrix_analysis.py": run_u_matrix_analysis.STEPS,
}
# (runner, steps to run from it; None means all of its steps), in dependency order.
ORDER: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("run_qe_novelty_detection.py", ("qe_novelty", "qe_score_distribution_figure")),
    ("run_novelty_benchmark.py", None),
    ("run_qe_novelty_detection.py", ("qe_ood", "qe_random_control", "qe_threshold_sweep")),
    ("run_u_matrix_analysis.py", None),
)


def groups(include_reproductions: bool) -> list[tuple[str, list[str]]]:
    """(runner, step names) groups in the order ``run_all.py`` runs them; empty groups are dropped."""
    result = []
    for runner, names in ORDER:
        chosen = list(names) if names is not None else [step.name for step in RUNNER_STEPS[runner]]
        chosen = [name for name in chosen if include_reproductions or name not in REPRODUCTION_STEPS]
        if chosen:
            result.append((runner, chosen))
    return result


def phase3_steps(dry_run: bool, include_reproductions: bool = False) -> list[Step]:
    """One ``Step`` per group, each run with ``--steps`` (a runner skips steps that already have results)."""
    return [
        Step(f"Phase 3 {Path(runner).stem.removeprefix('run_')}: {', '.join(names)}",
             P3_PIPELINE / runner, ["--steps", *names, *(["--dry-run"] if dry_run else [])])
        for runner, names in groups(include_reproductions)
    ]


PHASE2_NAMES = list(STEP_RUNNERS)
STEP_NAMES = [*PHASE2_NAMES, PHASE3_STEP]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the step runners in proposal order.")
    parser.add_argument("--from", dest="first", default="som_nnsom_training", choices=STEP_NAMES,
                        help="First step to run (default: som_nnsom_training).")
    parser.add_argument("--to", dest="last", default=None, choices=STEP_NAMES,
                        help=f"Last step to run (default: {PHASE2_NAMES[-1]}, or {PHASE3_STEP} when --from is "
                             f"{PHASE3_STEP}; {PHASE3_STEP} adds Phase 3).")
    parser.add_argument("--cnn-config", default=CNN_CONFIG, help=f"Config for {' and '.join(CNN_STEPS)}.")
    parser.add_argument("--som-config", default=SOM_CONFIG, help="Config for the SOM steps and the report.")
    parser.add_argument("--skip-training", action="store_true",
                        help="som_nnsom_training: use the existing SOM instead of training a new one.")
    parser.add_argument("--no-report", action="store_true", help="Do not build the interactive report.")
    parser.add_argument("--with-reproductions", action="store_true",
                        help="Phase 3: also run qe_ood, qe_random_control and qe_u_matrix, which repeat historical results.")
    parser.add_argument("--dry-run", action="store_true", help="Print every runner's steps without running them.")
    args = parser.parse_args()
    last = args.last or (PHASE3_STEP if args.first == PHASE3_STEP else PHASE2_NAMES[-1])
    start, end = STEP_NAMES.index(args.first), STEP_NAMES.index(last)
    if start > end:
        parser.error("--from must not come after --to")

    # Each runner gets --dry-run itself, so a dry run lists every runner's own steps.
    common = ["--dry-run"] if args.dry_run else []
    plan = []
    for name in STEP_NAMES[start:end + 1]:
        if name == PHASE3_STEP:
            continue
        extra = ["--config", args.cnn_config if name in CNN_STEPS else args.som_config]
        if name == "som_nnsom_training" and args.skip_training:
            extra.append("--skip-training")
        plan.append(Step(name, STEP_RUNNERS[name], [*extra, *common]))
    if start <= len(PHASE2_NAMES) - 1 <= end and not args.no_report:
        plan.append(Step("interactive report", REPORT_RUNNER, ["--config", args.som_config, *common]))
    if end >= STEP_NAMES.index(PHASE3_STEP):
        plan.extend(phase3_steps(args.dry_run, args.with_reproductions))

    run_steps("run_all: " + ", ".join(step.label for step in plan), plan, dry_run=False)


if __name__ == "__main__":
    main()
