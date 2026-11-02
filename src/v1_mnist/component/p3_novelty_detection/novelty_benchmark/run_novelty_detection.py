"""Dedicated pipeline entry point for Phase 3 novelty detection.

Usage:
    python src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/run_novelty_detection.py \
        --stage qe_novelty --config CONFIG
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.run_phase3_stage import (
    STAGES,
)


def main() -> None:
    """Dispatch a selected Phase 3 stage through its existing adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()

    if forwarded and args.stage != "ood_comparison":
        parser.error("Additional arguments are only allowed for ood_comparison")

    adapter = (
        REPO_ROOT / "src/v1_mnist/component/p3_novelty_detection/"
        "novelty_benchmark/run_phase3_stage.py"
    )

    if not adapter.is_file():
        raise FileNotFoundError(adapter)

    config = args.config
    if not config.is_absolute():
        config = REPO_ROOT / config

    command = [
        sys.executable,
        str(adapter),
        "--stage",
        args.stage,
        "--config",
        str(config.resolve()),
        *forwarded,
    ]

    if args.dry_run:
        command.append("--dry-run")

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
