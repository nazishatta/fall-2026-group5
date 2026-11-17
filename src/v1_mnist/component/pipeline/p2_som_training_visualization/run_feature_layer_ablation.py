"""Feature-layer ablation (Phase 2): SOMs on fc1 and fc2 of the Phase 1 LeNet-5, compared on A1 to A3.

Protocol: src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training/FEATURE_LAYER_ABLATION_PROTOCOL.md

Usage (from the code root, the folder that holds src/ and outputs/), in this order:
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_feature_layer_ablation.py --stage extract
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_feature_layer_ablation.py --stage som
  python src/v1_mnist/component/pipeline/p2_som_training_visualization/run_feature_layer_ablation.py --stage compare

``extract`` reads both layers of the frozen network (needs PyTorch) and checks fc2 against Phase 1. ``som``
trains the 10 maps (a finished map is skipped, so it can be re-run after an interruption; ``--layer fc1``
runs one layer only). ``compare`` writes results.json and the tables. The MNIST test split is never loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (  # noqa: E402
    DEFAULT_CONFIG,
    load_ablation_settings,
)
from src.v1_mnist.component.utils.logging import setup_logger  # noqa: E402

LOGGER = logging.getLogger("v1_mnist.feature_layer_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("extract", "som", "compare"), required=True)
    parser.add_argument("--layer", choices=("fc1", "fc2"), help="som stage only: run this layer only.")
    args = parser.parse_args()
    settings = load_ablation_settings(args.config, REPO_ROOT)
    setup_logger("v1_mnist", settings.output_dir / "logs" / f"{args.stage}.log")
    LOGGER.info("Run %s, stage %s", settings.run_id, args.stage)
    if args.stage == "extract":
        from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.extraction import (
            run_extraction,
        )
        run_extraction(settings)
    elif args.stage == "som":
        from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.som_runs import (
            run_soms,
        )
        run_soms(settings, args.layer)
    else:
        from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.comparison import (
            run_compare,
        )
        run_compare(settings)


if __name__ == "__main__":
    main()
