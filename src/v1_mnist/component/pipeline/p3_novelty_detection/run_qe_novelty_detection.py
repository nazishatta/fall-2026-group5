"""QE novelty detection: quantization-error scoring and external OOD evaluation.

Steps (scripts in p3_novelty_detection/qe_novelty_detection/):
  1. qe_novelty: score MNIST samples by quantization error (QE), with the
     decision threshold frozen from the training QE.
  2. qe_ood: evaluate the QE score on MNIST validation versus SVHN (AUROC with a
     bootstrap interval). The official MNIST final-test split is never used.
  3. qe_random_control: compare the QE flags with random selection.
  4. qe_threshold_sweep: the QE score at several training-percentile cutoffs (90 to 99.5),
     with threshold-free AUROC and a label-shuffle control.

Results go to outputs/v1_mnist/p3_novelty_detection/qe_novelty_detection/<run_id>/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p3_novelty_detection/run_qe_novelty_detection.py \
      [--only STEP [--config CONFIG]] [--dry-run]

Every step has its own config in src/v1_mnist/component/configs/ (shown by --dry-run).
Needs the trained SOM and its cluster-analysis distances (Phase 2). Step 4 also reads
the SVHN QE score array written by the svhn_benchmark step of run_novelty_benchmark.py.
A step whose results folder already exists is skipped (results are never overwritten);
to repeat a step, give it a new run_id in a new config and run it with --only STEP --config.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_steps import P3Step, run_phase3  # noqa: E402

FOLDER = "qe_novelty_detection"
STEPS = [
    P3Step("qe_novelty", "QE novelty detection with the frozen training-QE threshold",
           f"{FOLDER}/run_qe_novelty.py", "qe_novelty.yaml", ("--save",)),
    P3Step("qe_ood", "QE evaluation: MNIST validation versus SVHN",
           f"{FOLDER}/run_qe_ood.py", "qe_ood_evaluation.yaml", ("--evaluate", "--save")),
    P3Step("qe_random_control", "Random-selection control for the QE flags",
           f"{FOLDER}/run_qe_random_control.py", "qe_random_control.yaml", ("--save",)),
    P3Step("qe_threshold_sweep", "QE at several percentile cutoffs, with threshold-free AUROC",
           f"{FOLDER}/run_qe_threshold_sweep.py", "qe_threshold_sweep.yaml"),
]


def main() -> None:
    run_phase3("QE novelty detection", "QE novelty detection and external OOD evaluation.", STEPS)


if __name__ == "__main__":
    main()
