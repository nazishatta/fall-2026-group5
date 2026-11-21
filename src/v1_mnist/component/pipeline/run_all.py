"""Run the step runners in proposal order.

  Phase 1  data_pipeline            p1_model_training_baseline/run_data_pipeline.py
           cnn_baseline_training    p1_model_training_baseline/run_cnn_baseline_training.py
  Phase 2  som_nnsom_training       p2_som_training_visualization/run_som_nnsom_training.py
           cluster_analysis         p2_som_training_visualization/run_cluster_analysis.py
           error_pattern_analysis   p2_som_training_visualization/run_error_pattern_analysis.py
           (report)                 p2_som_training_visualization/run_interactive_report.py,
                                    after error_pattern_analysis
           error_pattern_baselines, error_detector_budget, som_optuna, feature_layer_ablation
                                    (study_steps.py)
  Phase 3  novelty_detection        p3_novelty_detection/run_*.py (three runners, run in the
                                    order their results depend on each other; see
                                    ORDER below)
           held_out_class_novelty   (study_steps.py)
  Phase 4  guided_retraining        pools, transform check, both experiments (study_steps.py)
           guided_retraining_test   the single test pass and everything after it (study_steps.py)

By default it starts at som_nnsom_training: data_pipeline and cnn_baseline_training
produce the frozen checkpoint and embeddings that every SOM result depends on.
Use --from data_pipeline to start earlier. The default range ends at error_pattern_analysis
and the report; --to STEP runs on to any later step, and --from STEP alone (for a step after
the report) runs just that step. Phase 3 steps whose results exist are skipped; for MNIST, the
three steps that only repeat historical results (qe_ood, qe_random_control, qe_u_matrix) run only
with --with-reproductions.

Stop points: guided_retraining_test opens the test split, so it runs only when asked for by name
(--to or --from). Before it, a person fills in the macro-F1 guard and freezes the protocol with
p4_guided_retraining/run_protocol_stamp.py --config ...; the runner does neither, and the test
stage refuses to run without the stamp.

The dataset is chosen by the folder of configs (--config-dir): every config in it merges that
folder's base.yaml, which names the dataset and the output folder. The default is the MNIST
folder; src/v1_fashion_mnist/component/pipeline/run_all.py calls this script with the
Fashion-MNIST folder.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/run_all.py [--config-dir DIR] [--from STEP] [--to STEP]
      [--cnn-config CFG] [--som-config CFG] [--skip-training] [--no-report]
      [--with-reproductions] [--dry-run]

Each runner keeps its own protection: in Phases 1 and 2 a step whose results already exist
stops the whole run; Phase 3 skips finished steps. Results are never overwritten (move or
delete them, or use a new run ID, to redo a step).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    CNN_STEPS,
    CONFIG_DIR,
    PIPELINE,
    REPO_ROOT,
    REPORT_RUNNER,
    STEP_RUNNERS,
    P3Step,
    Step,
    log,
    run_steps,
    show,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection import (  # noqa: E402
    run_novelty_benchmark,
    run_qe_novelty_detection,
    run_u_matrix_analysis,
)
from src.v1_mnist.component.pipeline.study_steps import (  # noqa: E402
    LATER_STUDY_STEPS,
    PHASE2_STUDY_STEPS,
    STUDY_STEPS,
    StudyStep,
)
from src.v1_mnist.component.utils.in_distribution import in_distribution  # noqa: E402

# ---- Phase 3 order -----------------------------------------------------------
#How ``run_all.py`` runs Phase 3: the order that respects which step reads which results.
#
#Phase 3 has three runners, but their steps depend on each other across runners, so
#they cannot simply run one runner after another:
#
#  1. qe_novelty               writes the QE summary that nearly everything reads; then
#     qe_ood (QE on SVHN), whose metrics svhn_benchmark must reproduce.
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
#
#Those three are reproductions only for MNIST, the one dataset with historical runs. For any
#other dataset they are first runs of the same analyses, so they are always planned. Which steps
#belong to a dataset's study is decided by its config folder: the runner skips, and says so, any
#step whose config is not in that folder.

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
    ("run_qe_novelty_detection.py", ("qe_novelty", "qe_score_distribution_figure", "qe_ood")),
    ("run_novelty_benchmark.py", None),
    ("run_qe_novelty_detection.py", ("qe_random_control", "qe_threshold_sweep")),
    ("run_u_matrix_analysis.py", None),
)


def optional_steps(config_dir: str = CONFIG_DIR) -> frozenset[str]:
    """Steps left out unless asked for: the reproductions, which exist only for MNIST."""
    return REPRODUCTION_STEPS if in_distribution(REPO_ROOT / config_dir / "base.yaml").is_mnist else frozenset()


def groups(include_reproductions: bool, config_dir: str = CONFIG_DIR) -> list[tuple[str, list[str]]]:
    """(runner, step names) groups in the order ``run_all.py`` runs them; empty groups are dropped."""
    optional = optional_steps(config_dir)
    result = []
    for runner, names in ORDER:
        chosen = list(names) if names is not None else [step.name for step in RUNNER_STEPS[runner]]
        chosen = [name for name in chosen if include_reproductions or name not in optional]
        if chosen:
            result.append((runner, chosen))
    return result


def phase3_steps(dry_run: bool, include_reproductions: bool = False, config_dir: str = CONFIG_DIR) -> list[Step]:
    """One ``Step`` per group, each run with ``--steps`` (a runner skips steps that already have results)."""
    return [
        Step(f"Phase 3 {Path(runner).stem.removeprefix('run_')}: {', '.join(names)}",
             P3_PIPELINE / runner,
             ["--steps", *names, "--config-dir", config_dir, *(["--dry-run"] if dry_run else [])])
        for runner, names in groups(include_reproductions, config_dir)
    ]


def study_plan(step: StudyStep, config_dir: str, dry_run: bool) -> list[Step]:
    """The commands of one study step, or none (with a message) if its configs are not in the folder.

    The study scripts have no --dry-run of their own, so in a dry run they are listed, not run.
    """
    missing = step.missing(config_dir)
    if missing:
        names = ", ".join(path.name for path in missing)
        log.info(f"Skipping {step.name}: not part of this dataset's study (no {names} in {config_dir})")
        return []
    done = step.finished(config_dir)
    if done:
        log.info(f"Skipping {step.name}: results already exist ({', '.join(show(path) for path in done)})")
        return []
    return [replace(command, listed_only=dry_run) for command in step.commands(config_dir)]


PHASE2_NAMES = list(STEP_RUNNERS)
STEP_NAMES = [*PHASE2_NAMES, *(s.name for s in PHASE2_STUDY_STEPS), PHASE3_STEP,
              *(s.name for s in LATER_STUDY_STEPS)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the step runners in proposal order.")
    parser.add_argument("--config-dir", default=CONFIG_DIR,
                        help=f"Folder of configs, one per dataset (default: {CONFIG_DIR}).")
    parser.add_argument("--from", dest="first", default="som_nnsom_training", choices=STEP_NAMES,
                        help="First step to run (default: som_nnsom_training).")
    parser.add_argument("--to", dest="last", default=None, choices=STEP_NAMES,
                        help=f"Last step to run (default: {PHASE2_NAMES[-1]}, or the --from step itself when it "
                             f"comes after {PHASE2_NAMES[-1]}; {PHASE3_STEP} adds Phase 3).")
    parser.add_argument("--cnn-config", default=None,
                        help=f"Config for {' and '.join(CNN_STEPS)} (default: cnn_baseline.yaml in --config-dir).")
    parser.add_argument("--som-config", default=None,
                        help="Config for the SOM steps and the report (default: som.yaml in --config-dir).")
    parser.add_argument("--skip-training", action="store_true",
                        help="som_nnsom_training: use the existing SOM instead of training a new one.")
    parser.add_argument("--no-report", action="store_true", help="Do not build the interactive report.")
    parser.add_argument("--with-reproductions", action="store_true",
                        help="Phase 3: also run qe_ood, qe_random_control and qe_u_matrix, which repeat historical results.")
    parser.add_argument("--dry-run", action="store_true", help="Print every runner's steps without running them.")
    args = parser.parse_args(argv)
    config_dir = args.config_dir.rstrip("/")
    cnn_config = args.cnn_config or f"{config_dir}/cnn_baseline.yaml"
    som_config = args.som_config or f"{config_dir}/som.yaml"
    after_phase2 = STEP_NAMES.index(args.first) > STEP_NAMES.index(PHASE2_NAMES[-1])
    last = args.last or (args.first if after_phase2 else PHASE2_NAMES[-1])
    start, end = STEP_NAMES.index(args.first), STEP_NAMES.index(last)
    if start > end:
        parser.error("--from must not come after --to")

    # Each runner gets --dry-run itself, so a dry run lists every runner's own steps.
    common = ["--dry-run"] if args.dry_run else []
    plan = []
    for name in STEP_NAMES[start:end + 1]:
        if name == PHASE3_STEP:
            plan.extend(phase3_steps(args.dry_run, args.with_reproductions, config_dir))
        elif name in STUDY_STEPS:
            plan.extend(study_plan(STUDY_STEPS[name], config_dir, args.dry_run))
        else:
            extra = ["--config", cnn_config if name in CNN_STEPS else som_config]
            if name == "som_nnsom_training" and args.skip_training:
                extra.append("--skip-training")
            plan.append(Step(name, STEP_RUNNERS[name], [*extra, *common]))
            if name == PHASE2_NAMES[-1] and not args.no_report:
                plan.append(Step("interactive report", REPORT_RUNNER, ["--config", som_config, *common]))

    if not plan:
        log.info("run_all: nothing to run (every step in the range is finished or not part of this dataset's study).")
        return
    run_steps("run_all: " + ", ".join(step.label for step in plan), plan, dry_run=False)


if __name__ == "__main__":
    main()
