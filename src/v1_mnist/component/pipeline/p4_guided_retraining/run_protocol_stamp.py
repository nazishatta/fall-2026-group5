"""Freeze the retraining protocol (run once, after every [TBD] is filled).

Usage (from the code root):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_protocol_stamp.py

Refuses while RETRAINING_PROTOCOL.md still has an open [TBD] or evaluation.macro_f1_guard in
configs/retraining.yaml is empty. After the stamp exists, the final test stages check that neither
file changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402
from src.v1_mnist.component.utils.protocol_stamp import create_stamp  # noqa: E402
from src.v1_mnist.component.utils.retraining_settings import load_experiment_settings  # noqa: E402

if __name__ == "__main__":
    log = setup_logger("v1_mnist.protocol_stamp")
    path = create_stamp(load_experiment_settings("curriculum", None, REPO_ROOT))
    log.info("Protocol frozen: %s", path)
