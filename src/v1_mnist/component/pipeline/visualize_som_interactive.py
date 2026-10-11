"""Build the interactive SOM report (single self-contained HTML page).

This is the only file you need to run. It wires together the modules in
``src/v1_mnist/component/interactive/som_interactive_plots/``:

    load config + data  ->  load SOM  ->  per-neuron stats  ->  payload
    ->  fill HTML template  ->  write interactive_report.html

The report is written to (path relative to the repo root):

    outputs/v1_mnist/som/som_interactive/interactive_report.html

Usage (from the repo root):
    python src/v1_mnist/component/pipeline/visualize_som_interactive.py
    python src/v1_mnist/component/pipeline/visualize_som_interactive.py \
        --config src/v1_mnist/component/configs/som.yaml \
        --model-name som_15x15_nb11_ep250_s42_v2 --split val

Uses train + validation only; the test split is never touched.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.som.data import load_som_data
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.interactive.som_interactive_plots import (
    build_payload,
    load_model_and_grid,
    write_report,
)
from src.v1_mnist.component.interactive.som_interactive_plots.settings import (
    REPORT_FILENAME,
)
from src.v1_mnist.component.visualization.som_visualizations import (
    top_variance_features,
)

OUTPUT_SUBDIR = "som_interactive"                         # under paths.output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive, self-contained HTML report for a trained MNIST SOM."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/v1_mnist/component/configs/som.yaml",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Saved SOM filename/run-id. Defaults to visualization.selected_model.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val"],
        help=(
            "Split used for the click-to-inspect detail panel. "
            "Defaults to visualization.class_map_split."
        ),
    )
    return parser.parse_args()


def resolve_path(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def as_split(s) -> dict:
    """EmbeddingSplit dataclass -> plain dict for build_payload."""
    return {
        "features": s.features,
        "labels": s.labels,
        "preds": s.preds,
        "sample_ids": s.sample_ids,
        "confidence": s.confidence,
    }


def main() -> None:
    args = parse_args()
    warnings.simplefilter("ignore")

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = (REPO_ROOT / config_path).resolve()

    config = load_config(str(config_path))

    model_name = args.model_name or config.visualization.selected_model
    detail_split = args.split or str(config.visualization.class_map_split)

    models_dir = resolve_path(config.paths.som_models_dir)
    logs_dir = resolve_path(config.paths.logs_dir)
    embeddings_dir = config.paths.embeddings_dir

    destinations = [
        resolve_path(config.paths.output_dir) / OUTPUT_SUBDIR / REPORT_FILENAME
    ]

    log_file = logs_dir / f"{model_name}_visualize_interactive_html.log"
    logger = setup_logger("visualize_som_interactive", log_file=log_file)

    # Data Preparation
    data = load_som_data(embeddings_dir=embeddings_dir, expected_feature_dim=84)

    # Load som instance
    som, model_metrics = load_model_and_grid(model_name, models_dir, logs_dir)

    # Reattach the train-fitted scaler (same as visualize_som.py)
    som.norm_func = data.scaler.transform

    grid = (
        int(model_metrics["som"]["grid_height"]),
        int(model_metrics["som"]["grid_width"]),
    )

    # Same top-variance fc2 dimensions visualize_som.py uses
    scaled_train = data.scaler.transform(data.train.features)
    n_top = int(config.visualization.top_embedding_features)
    top_features = top_variance_features(scaled_train, n_features=n_top)

    logger.info("=" * 72)
    logger.info("MNIST v1 SOM Interactive HTML Report")
    logger.info("=" * 72)
    logger.info("Model: %s", model_name)
    logger.info("Grid:  %dx%d", *grid)
    logger.info("Detail split: %s", detail_split)
    logger.info("Top-variance fc2 dimensions: %s", top_features)

    # Test split is intentionally NOT used (same rule as visualize_som.py).
    payload = build_payload(
        som,
        model_name=model_name,
        grid=grid,
        splits={"train": as_split(data.train), "val": as_split(data.val)},
        detail_split=detail_split,
        top_features=top_features,
    )

    for path in write_report(payload, destinations):
        size_mb = path.stat().st_size / 1e6
        logger.info("Wrote %s (%.2f MB)", path, size_mb)
        print(f"Open in a browser: {path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
