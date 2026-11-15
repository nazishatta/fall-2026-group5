"""SOM re-analysis (Phase 4): re-train the SOM on the retrained CNNs' embeddings and compare the maps.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_som_reanalysis.py --stage embeddings
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_som_reanalysis.py --stage train_som
  python src/v1_mnist/component/pipeline/p4_guided_retraining/run_som_reanalysis.py --stage analyze

``embeddings`` needs PyTorch and the retraining checkpoints; ``train_som`` and ``analyze`` do not.
Training and validation images only; the test split is never read. A run is never overwritten:
finished cells are skipped, and any change to the protocol, the configs or a checkpoint stops the run
(set a new run.revision in configs/som_reanalysis.yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.run_state import (
    start_run,
    verify_run,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    DEFAULT_CONFIG,
    load_reanalysis_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger

STAGES = ("embeddings", "train_som", "analyze")


def main() -> None:
    parser = argparse.ArgumentParser(description="SOM re-analysis of the retrained CNNs.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    settings = load_reanalysis_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist.som_reanalysis", settings.output_dir / "logs" / f"{args.stage}.log")
    if args.stage == "embeddings":
        from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.cell_embeddings import (
            run_embeddings,
        )

        start_run(settings)
        run_embeddings(settings, REPO_ROOT)
    elif args.stage == "train_som":
        from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_cell import (
            run_train_som,
        )

        verify_run(settings)
        run_train_som(settings)
    else:
        from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.analysis import (
            run_analysis,
        )
        from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.reanalysis_figures import (
            make_figures,
        )

        verify_run(settings)
        make_figures(settings, run_analysis(settings))


if __name__ == "__main__":
    main()
