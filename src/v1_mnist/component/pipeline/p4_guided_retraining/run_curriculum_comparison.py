"""Curriculum comparison (Phase 4, hard example mining): standard training against curricula.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_curriculum_comparison.py --stage train
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_curriculum_comparison.py --stage summary
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_curriculum_comparison.py --stage final_test

Options: --arm NAME, --seed N (run part of the work), --smoke (quick 2-epoch trial in a separate
folder). Training and summary use validation images only; final_test needs the frozen protocol.
Finished runs are never overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum_arms import (  # noqa: E402
    make_curriculum_builder,
)
from src.v1_mnist.component.utils.retraining_cli import run_experiment_cli  # noqa: E402

if __name__ == "__main__":
    run_experiment_cli("curriculum", "Curriculum comparison: standard vs curriculum arms.",
                       make_curriculum_builder, REPO_ROOT)
