"""CNN baseline training: train LeNet-5 and extract embeddings.

Steps:
  1. train_baseline: train LeNet-5, save the best checkpoint, test metrics,
     training curves and test confusion matrix.
  2. extract_embeddings: save the 84-D fc2 embeddings (plus labels, predictions,
     confidence, sample IDs) for train, val and test, using the checkpoint from step 1.

Results go to outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/.

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/pipeline/p1_model_training_baseline/run_cnn_baseline_training.py [--config CONFIG] [--dry-run]

The checkpoint and embeddings are frozen inputs for every SOM result. Each run
has a run ID (run.id in configs/cnn_baseline.yaml); the training and extraction
scripts refuse to overwrite an existing run. A new or
re-tuned run needs a new run.id (and a new paths.embeddings_dir).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    CNN_CONFIG, P1, REPO_ROOT, Step, base_parser, check_not_done, load, resolve, run_steps,
)
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.run_ids import (  # noqa: E402
    cnn_run_paths,
)


def main() -> None:
    parser = base_parser("CNN baseline training: train LeNet-5 and extract embeddings.", CNN_CONFIG)
    args = parser.parse_args()

    config = load(args.config)
    checkpoint = resolve(cnn_run_paths(config).checkpoint)
    embeddings_dir = resolve(config.paths.embeddings_dir)
    check_not_done(
        [checkpoint, embeddings_dir / "train_features.npy"], args.dry_run,
        hint="Every SOM result depends on these files. For a new run, set a new run.id and paths.embeddings_dir.",
    )

    folder = P1 / "cnn_baseline_training"
    steps = [
        Step("Train LeNet-5, record metrics, curves and confusion matrix",
             folder / "train_baseline.py", ["--config", args.config]),
        Step("Extract fc2 embeddings for train, val and test",
             folder / "extract_embeddings.py",
             ["--config", args.config, "--checkpoint", checkpoint.relative_to(REPO_ROOT).as_posix()]),
    ]
    run_steps("CNN baseline training", steps, args.dry_run)


if __name__ == "__main__":
    main()
