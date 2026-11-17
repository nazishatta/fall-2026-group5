"""Novelty detection on held-out classes (Phase 3, near novelty): train on 6 digits, treat the other 4 as novel.

Protocol: src/v1_mnist/docs/p3_novelty_detection/held_out_class_novelty/HELD_OUT_CLASS_NOVELTY_PROTOCOL.md

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p3_novelty_detection/run_held_out_class_novelty.py --stage run
  python src/v1_mnist/component/pipeline/p3_novelty_detection/run_held_out_class_novelty.py --stage compare

``run`` trains and scores every split (a finished split is skipped, so it can be re-run after an
interruption); ``--split K`` runs only that split. ``compare`` writes results.json and the tables once all
splits are done. Only training and validation images are read; the MNIST test split is never loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_held_out_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402

LOGGER = logging.getLogger("v1_mnist.held_out_class_novelty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("run", "compare"), required=True)
    parser.add_argument("--split", type=int, action="append", help="Only this split, 1 to 5 (repeatable).")
    args = parser.parse_args()
    settings = load_held_out_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist", settings.output_dir / "logs" / f"{args.stage}.log")
    LOGGER.info("Run %s, stage %s", settings.run_id, args.stage)
    if args.stage == "run":
        from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.split_run import run_splits
        run_splits(settings, args.split)
    else:
        from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.comparison import run_compare
        results = run_compare(settings)
        for name, s in results["statements"].items():
            if isinstance(s, dict) and "verdict" in s:
                LOGGER.info("%s: %s", name, s["verdict"])
        LOGGER.info("H3 (not worse by the margin): %s", results["statements"]["H3_som_not_worse_by_margin"])


if __name__ == "__main__":
    main()
