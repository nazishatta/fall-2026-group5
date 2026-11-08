"""U-Matrix analysis of the frozen SOM as a novelty signal.

Steps (scripts in p3_novelty_detection/u_matrix_analysis/):
  1. qe_u_matrix: reproduce the MNIST validation QE and U-Matrix statistics with
     the frozen SOM.
  2. qe_umatrix_validation_figure: QE versus U-Matrix on MNIST validation (PDF).
  3. u_matrix_grid_figure: the U-Matrix drawn on the SOM's hexagonal grid (PDF).
  4. proposal_gap_analysis: descriptive analysis of the QE and U-Matrix signals.
  5. svhn_som_overlay: where SVHN images land on the SOM grid and where the QE detector
     misses them (saves the SVHN-to-SOM assignments).

Results go to outputs/v1_mnist/p3_novelty_detection/u_matrix_analysis/<run_id>/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p3_novelty_detection/run_u_matrix_analysis.py \
      [--only STEP [--config CONFIG]] [--dry-run]

Every step has its own config in src/v1_mnist/component/configs/ (shown by --dry-run).
Needs the trained SOM (Phase 2) and the QE summary from run_qe_novelty_detection.py.
Steps 2 and 4 also read the SVHN score arrays written by the svhn_benchmark step
of run_novelty_benchmark.py. Step 5 scores the SVHN test images itself, so it also needs
PyTorch, NNSOM, the frozen CNN and the SVHN test file (DVC). A step whose results folder
already exists is skipped (results are never overwritten); to repeat a step, give it a
new run_id in a new config and run it with --only STEP --config.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_steps import P3Step, run_phase3  # noqa: E402

FOLDER = "u_matrix_analysis"
STEPS = [
    P3Step("qe_u_matrix", "QE and U-Matrix statistics on MNIST validation",
           f"{FOLDER}/run_qe_u_matrix.py", "qe_u_matrix_reproduction.yaml", ("--save",)),
    P3Step("qe_umatrix_validation_figure", "Figure: QE versus U-Matrix on MNIST validation",
           f"{FOLDER}/run_qe_umatrix_validation_figure.py", "qe_umatrix_validation_figure.yaml"),
    P3Step("u_matrix_grid_figure", "Figure: U-Matrix on the hexagonal SOM grid",
           f"{FOLDER}/run_u_matrix_grid_figure.py", "u_matrix_grid_figure.yaml"),
    P3Step("proposal_gap_analysis", "Descriptive analysis of the QE and U-Matrix signals",
           f"{FOLDER}/run_proposal_gap_analysis.py", "phase3_proposal_gap_analysis.yaml"),
    P3Step("svhn_som_overlay", "SVHN on the SOM grid: where the QE detector misses",
           f"{FOLDER}/run_svhn_som_overlay.py", "svhn_som_overlay.yaml"),
]


def main() -> None:
    run_phase3("U-Matrix analysis", "U-Matrix analysis of the frozen SOM as a novelty signal.", STEPS)


if __name__ == "__main__":
    main()
