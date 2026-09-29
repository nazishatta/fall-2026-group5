from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.data.mnist_dataset import get_dataloaders
from src.v1_mnist.component.features.extractor import FeatureExtractor, save_embeddings
from src.v1_mnist.component.models.lenet5 import LeNet5
from src.v1_mnist.component.utils.config import get_device, load_config, set_seed
from src.v1_mnist.component.utils.logging import get_logger, setup_logger


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 digest string for file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(
    split_name: str,
    arrays: Sequence[np.ndarray],
    expected_rows: int,
    expected_dim: int,
) -> dict[str, Any]:
    """Validate extracted arrays for completeness, finiteness, and correctness."""
    features, labels, preds, sample_ids, confidence, correct = arrays
    expected_shapes = {
        "features": (expected_rows, expected_dim),
        "labels": (expected_rows,),
        "preds": (expected_rows,),
        "sample_ids": (expected_rows,),
        "confidence": (expected_rows,),
        "correct": (expected_rows,),
    }
    observed = {
        "features": features,
        "labels": labels,
        "preds": preds,
        "sample_ids": sample_ids,
        "confidence": confidence,
        "correct": correct,
    }
    for name, array in observed.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{split_name}_{name} shape {array.shape}; expected {expected_shapes[name]}"
            )

    if not np.isfinite(features).all():
        raise ValueError(f"{split_name} features contain NaN or Inf")
    if not np.isfinite(confidence).all():
        raise ValueError(f"{split_name} confidence contains NaN or Inf")
    if not np.isin(labels, np.arange(10)).all():
        raise ValueError(f"{split_name} labels are outside 0..9")
    if not np.isin(preds, np.arange(10)).all():
        raise ValueError(f"{split_name} predictions are outside 0..9")
    if len(np.unique(sample_ids)) != expected_rows:
        raise ValueError(f"{split_name} sample IDs are not unique")
    if not np.array_equal(correct, preds == labels):
        raise ValueError(f"{split_name} correctness array is inconsistent")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError(f"{split_name} confidence is outside [0, 1]")

    return {
        "rows": expected_rows,
        "feature_dim": expected_dim,
        "accuracy": float(correct.mean()),
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "mean_confidence": float(confidence.mean()),
        "features_finite": True,
        "sample_ids_unique": True,
        "label_range": [int(labels.min()), int(labels.max())],
        "prediction_range": [int(preds.min()), int(preds.max())],
    }


def extract_embeddings(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Extract deterministic penultimate embeddings from trained LeNet-5 model.

    Args:
        checkpoint_path: Path to best LeNet-5 weights (.pth).
        config_path: Path to configuration file.

    Returns:
        Manifest dictionary with extraction metadata and sha256 checksums.
    """
    import torch

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

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
    set_seed(config.training.seed)
    device = get_device(config)

    embeddings_dir = Path(config.paths.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(getattr(config.paths, "output_dir", "./outputs/v1_mnist/cnn_baseline"))
    logs_dir = Path(getattr(config.paths, "logs_dir", output_dir / "logs"))
    log_file = logs_dir / "extract_embeddings.log"
    logger = setup_logger("extract_embeddings", log_file=log_file)

    logger.info("Using device: %s", device)
    # Official embeddings must keep a fixed row order and carry the original MNIST
    # sample IDs, so every row can be traced back to its image.
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        config, train_shuffle=False, return_sample_ids=True
    )

    model = LeNet5(num_classes=config.dataset.num_classes).to(device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    extractor = FeatureExtractor(model, layer_name=model.feature_layer_name)
    logger.info("Extracting embeddings from penultimate layer '%s'", model.feature_layer_name)

    loaders = [
        ("train", train_loader, 49000),
        ("val", val_loader, 10500),
        ("test", test_loader, 10500),
    ]

    split_manifest: dict[str, Any] = {}
    saved_files: dict[str, str] = {}

    for split_name, loader, expected_rows in loaders:
        logger.info("Processing %s split...", split_name)
        features, labels, preds, sample_ids, confidence, correct = (
            extractor.extract(loader, device, return_metadata=True)
        )

        validation_stats = validate_split(
            split_name,
            (features, labels, preds, sample_ids, confidence, correct),
            expected_rows,
            model.feature_dim,
        )

        paths = save_embeddings(
            features,
            labels,
            preds,
            split_name,
            embeddings_dir,
            sample_ids,
            confidence,
            correct,
        )

        split_manifest[split_name] = {
            "validation": validation_stats,
            "files": {
                name: {
                    "path": str(Path(path).resolve().relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
        }

        for path in paths.values():
            saved_files[str(Path(path).resolve().relative_to(PROJECT_ROOT))] = sha256_file(path)

        logger.info(
            "Extracted %s: %d samples, accuracy: %.4f",
            split_name,
            expected_rows,
            validation_stats["accuracy"],
        )

    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_file.resolve().relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "target_layer": model.feature_layer_name,
        "feature_dim": model.feature_dim,
        "num_classes": config.dataset.num_classes,
        "splits": split_manifest,
        "manifest_files": saved_files,
    }

    manifest_path = embeddings_dir / "embedding_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info("Saved deterministic embedding manifest to %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deterministic, traceable LeNet-5 embeddings"
    )
    parser.add_argument(
        "--config",
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
    parser.add_argument("--checkpoint", required=True, help="Best model checkpoint")
    args = parser.parse_args()
    extract_embeddings(args.checkpoint, args.config)


if __name__ == "__main__":
    main()
