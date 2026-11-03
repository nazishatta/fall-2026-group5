"""Hard example mining (Phase 4): which training images are hard, before any retraining.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_hard_example_mining.py [--config CONFIG]

Training images and Phase 2 records only; the test split is never loaded. Stops if this run's
results already exist (results are never overwritten): set a new run.revision in the config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.build_pools import (
    build_pools,  # noqa: E402
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_pool_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard example mining: SOM, max-softmax and random pools.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    settings, analysis = load_pool_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist.hard_example_mining", settings.output_dir / "logs" / "hard_example_mining.log")
    build_pools(settings, analysis, REPO_ROOT)


if __name__ == "__main__":
    main()
