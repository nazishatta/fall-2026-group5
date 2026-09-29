from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.data.mnist_dataset import (
    compute_class_distribution,
    get_dataloaders,
)
from src.v1_mnist.component.utils.config import load_config, set_seed
from src.v1_mnist.component.utils.logging import get_logger, setup_logger
from src.v1_mnist.component.visualization.plots import (
    plot_class_distribution,
    plot_sample_images,
)


def prepare_data(
    config_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Prepare MNIST dataset splits, verify distributions, and emit vector figures.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary of computed dataset split statistics.
    """
    if config_path is None:
        config_path = (
            PROJECT_ROOT
            / "src"
            / "v1_mnist"
            / "component"
            / "configs"
            / "cnn_baseline.yaml"
        )

    config = load_config(str(config_path))
    set_seed(getattr(getattr(config, "training", None), "seed", 42))

    output_dir = Path(
        getattr(config.paths, "output_dir", "./outputs/v1_mnist/cnn_baseline")
    )
    logs_dir = Path(getattr(config.paths, "logs_dir", output_dir / "logs"))
    log_file = logs_dir / "prepare_data.log"

    logger = setup_logger("prepare_data", log_file=log_file)
    logger.info("Initializing MNIST data preparation using config: %s", config_path)

    logger.info("Loading MNIST raw data and creating deterministic train/val/test splits...")
    (
        train_loader,
        val_loader,
        test_loader,
        train_labels,
        val_labels,
        test_labels,
    ) = get_dataloaders(config)

    logger.info("Computing class distributions across splits...")
    train_dist = compute_class_distribution(train_labels, "train")
    val_dist = compute_class_distribution(val_labels, "val")
    test_dist = compute_class_distribution(test_labels, "test")

    figures_dir = Path(
        getattr(
            config.paths,
            "figures_dir",
            output_dir / "figures",
        )
    )
    logger.info("Emitting class distribution plots (SVG + PDF)...")
    plot_class_distribution(
        train_dist,
        "Training Set Class Distribution",
        figures_dir / "train_dist.svg",
    )
    plot_class_distribution(
        val_dist,
        "Validation Set Class Distribution",
        figures_dir / "val_dist.svg",
    )
    plot_class_distribution(
        test_dist,
        "Test Set Class Distribution",
        figures_dir / "test_dist.svg",
    )

    logger.info("Plotting sample images from training set (vector SVG)...")
    plot_sample_images(
        train_loader.dataset,
        5,
        figures_dir / "sample_images.svg",
    )

    metrics_dir = Path(
        getattr(
            config.paths,
            "metrics_dir",
            output_dir / "metrics",
        )
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "test_samples": len(test_labels),
        "train_dist": train_dist,
        "val_dist": val_dist,
        "test_dist": test_dist,
    }

    metrics_path = metrics_dir / "split_stats.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    logger.info("Split statistics written to %s", metrics_path)
    logger.info(
        "Dataset preparation complete: train=%d, val=%d, test=%d",
        len(train_labels),
        len(val_labels),
        len(test_labels),
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            PROJECT_ROOT
            / "src"
            / "v1_mnist"
            / "component"
            / "configs"
            / "cnn_baseline.yaml"
        ),
        help="Path to config file (default: cnn_baseline.yaml)",
    )
    args = parser.parse_args()
    prepare_data(args.config)


if __name__ == "__main__":
    main()
