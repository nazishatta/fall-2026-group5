"""Adapter connecting Phase 3 scripts to the shared pipeline interface."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Allow direct execution from the shared pipeline.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("v1_mnist.pipeline.phase3")

PIPELINE_DIR = Path(__file__).resolve().parent

STAGES = {
    "qe_novelty": (
        "run_qe_novelty.py",
        ["--save"],
    ),
    "qe_ood": (
        "run_qe_ood.py",
        ["--evaluate", "--save"],
    ),
    "svhn_benchmark": (
        "run_svhn_benchmark.py",
        [],
    ),
}

BENCHMARK_CONFIG = (
    REPO_ROOT
    / "src/v1_mnist/component/configs/novelty_benchmark.yaml"
)


def build_command(stage: str, config: Path) -> list[str]:
    """Build one Phase 3 stage command without executing it."""
    script, extra = STAGES[stage]
    resolved_config = (
        config if config.is_absolute() else REPO_ROOT / config
    ).resolve()

    if stage == "svhn_benchmark":
        # Historical benchmark currently reads this fixed YAML location.
        if resolved_config != BENCHMARK_CONFIG.resolve():
            raise ValueError(
                "SVHN benchmark currently requires novelty_benchmark.yaml"
            )
        return [sys.executable, str(PIPELINE_DIR / script)]

    return [
        sys.executable,
        str(PIPELINE_DIR / script),
        "--config",
        str(resolved_config),
        *extra,
    ]


def main() -> None:
    """Execute or preview a configured Phase 3 stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    command = build_command(args.stage, args.config)
    LOGGER.info("Phase 3 stage: %s", args.stage)
    LOGGER.info("Command: %s", " ".join(command))

    if args.dry_run:
        LOGGER.info("Dry run: no experiment executed")
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
