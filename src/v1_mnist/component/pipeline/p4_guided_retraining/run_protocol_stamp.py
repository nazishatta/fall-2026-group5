"""Freeze the retraining protocol (run once, after every [TBD] is filled).

Usage (from the code root):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_protocol_stamp.py [--config CONFIG]

--config is the dataset's retraining.yaml (default: the MNIST one); it names the protocol file
and the stamp file.

Refuses while RETRAINING_PROTOCOL.md still has an open [TBD] or evaluation.macro_f1_guard in
configs/retraining.yaml is empty. After the stamp exists, the final test stages check that neither
file changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.protocol_stamp import create_stamp  # noqa: E402
from src.v1_mnist.component.utils.retraining_settings import (  # noqa: E402
    load_experiment_settings,
)


def main(argv: list[str] | None = None) -> Path:
    """Write the stamp for the protocol named in the config; refuses if it is already frozen."""
    parser = argparse.ArgumentParser(description="Freeze the retraining protocol.")
    parser.add_argument("--config", default=None,
                        help="The dataset's retraining.yaml (default: src/v1_mnist/component/configs/retraining.yaml).")
    args = parser.parse_args(argv)
    log = setup_logger("v1_mnist.protocol_stamp")
    path = create_stamp(load_experiment_settings("curriculum", args.config, REPO_ROOT))
    log.info("Protocol frozen: %s", path)
    return path


if __name__ == "__main__":
    main()
