"""Same-budget comparison of the error detectors (exploratory): kNN impurity and max-softmax flag the same share
of validation images as the SOM's and k-means' hotspot cells; how many of the CNN's mistakes do they catch?

Usage (from the code root, the folder that holds src/ and outputs/; needs the finished error-pattern baseline run):
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_error_detector_budget.py

Validation split only. Writes a new run folder and stops if its results already exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.budget_stage import (  # noqa: E402
    DEFAULT_CONFIG,
    load_budget_settings,
    run_budget,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_budget_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist.error_detector_budget", settings.output_dir / "logs" / "budget.log")
    run_budget(settings, REPO_ROOT)


if __name__ == "__main__":
    main()
