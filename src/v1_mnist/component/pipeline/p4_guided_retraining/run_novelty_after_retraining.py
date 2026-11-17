"""Novelty detection after retraining (Phase 4): Phase 3's novelty scores on the retrained networks.

Protocol: src/v1_mnist/docs/p4_guided_retraining/novelty_after_retraining/NOVELTY_AFTER_RETRAINING_PROTOCOL.md

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_novelty_after_retraining.py --stage score
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_novelty_after_retraining.py --stage compare

``score`` scores the reference and every retrained cell (a finished cell is skipped, so it can be
re-run after an interruption); ``--cell NAME`` scores only that cell. ``compare`` writes results.json and
the tables once every cell is scored. MNIST validation is the normal set, SVHN test the novel set; the
MNIST test split is never read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_novelty_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402

LOGGER = logging.getLogger("v1_mnist.novelty_after_retraining")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("score", "compare"), required=True)
    parser.add_argument("--cell", action="append", help="Only this cell, e.g. standard_seed1 (repeatable).")
    args = parser.parse_args()
    settings = load_novelty_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist", settings.output_dir / "logs" / f"{args.stage}.log")
    LOGGER.info("Run %s, stage %s", settings.run_id, args.stage)
    if args.stage == "score":
        from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.cell_scoring import run_scores
        run_scores(settings, args.cell)
    else:
        from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.comparison import run_compare
        results = run_compare(settings)
        for name, s in results["statements"].items():
            if isinstance(s, dict) and "verdict" in s:
                LOGGER.info("%s: %s", name, s["verdict"])


if __name__ == "__main__":
    main()
