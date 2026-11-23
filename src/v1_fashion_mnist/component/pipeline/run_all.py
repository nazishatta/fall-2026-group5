"""Run the pipeline on Fashion-MNIST: the shared step runners, in proposal order, with the
Fashion-MNIST settings.

Nothing here is specific to a step. It calls ``src/v1_mnist/component/pipeline/run_all.py`` with
``--config-dir src/v1_fashion_mnist/component/configs``; that folder's ``base.yaml`` names the
dataset (fashion_mnist) and sends every result to ``outputs/v1_fashion_mnist/``. Every option of
the shared ``run_all.py`` works here too (--from, --to, --skip-training, --no-report, --dry-run).

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_fashion_mnist/component/pipeline/run_all.py --from data_pipeline [--dry-run]
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]          # code root (holds src/ and outputs/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.pipeline import run_all as shared_run_all  # noqa: E402

CONFIG_DIR = "src/v1_fashion_mnist/component/configs"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared pipeline with the Fashion-MNIST configs (a --config-dir given here wins)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--config-dir" not in args:
        args = ["--config-dir", CONFIG_DIR, *args]
    shared_run_all.main(args)


if __name__ == "__main__":
    main()
