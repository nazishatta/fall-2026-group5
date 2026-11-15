"""Targeted augmentation (Phase 4): add altered copies of the pool images and retrain.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_targeted_augmentation.py --stage check
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_targeted_augmentation.py --stage train
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_targeted_augmentation.py --stage summary
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_targeted_augmentation.py --stage final_test

``check`` looks at the altered images (contact sheet, Phase 1 CNN accuracy on altered validation
images). Options: --arm NAME, --seed N, --smoke. Same rules as the curriculum comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.augmentation_arms import (  # noqa: E402
    make_augmentation_builder,
)
from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.augmentation_check import (
    run_check,  # noqa: E402
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.retraining_cli import run_experiment_cli  # noqa: E402
from src.v1_mnist.component.utils.retraining_data import load_retraining_data  # noqa: E402
from src.v1_mnist.component.utils.retraining_settings import load_experiment_settings  # noqa: E402


def main() -> None:
    if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] == "check":
        settings = load_experiment_settings("augmentation", None, REPO_ROOT)
        setup_logger("v1_mnist", settings.output_dir / "logs" / "check.log")
        run_check(settings, load_retraining_data(settings, REPO_ROOT))
        return
    run_experiment_cli("augmentation", "Targeted augmentation: standard vs augmented arms.",
                       make_augmentation_builder, REPO_ROOT)


if __name__ == "__main__":
    main()
