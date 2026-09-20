import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.component.utils.config import load_config, set_seed, get_device
from src.component.data.mnist_dataset import get_dataloaders
from src.component.models.lenet5 import LeNet5
from src.component.features.extractor import (
    FeatureExtractor,
    save_embeddings
)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(split_name, arrays, expected_rows, expected_dim):
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
                f"{split_name}_{name} shape {array.shape}; "
                f"expected {expected_shapes[name]}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract deterministic, traceable LeNet-5 embeddings"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / 'src' / 'component' / 'configs' / 'week_2_baseline.yaml'),
        help="Path to config file (default: week_2_baseline.yaml)"
    )
    parser.add_argument("--checkpoint", required=True, help="Best model checkpoint")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = load_config(args.config)
    seed = int(config.training.seed)
    set_seed(seed)
    device = get_device(config)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        config,
        train_shuffle=False,
        return_sample_ids=True,
    )

    model = LeNet5(num_classes=config.dataset.num_classes).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    feature_layer = config.model.feature_layer
    feature_dim = int(config.model.feature_dim)
    extractor = FeatureExtractor(model, feature_layer)
    output_dir = os.path.join(config.paths.embeddings_dir, config.dataset.name)
    os.makedirs(output_dir, exist_ok=True)

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    validation = {}

    for split_name, loader in loaders.items():
        arrays = extractor.extract(loader, device, return_metadata=True)
        validation[split_name] = validate_split(
            split_name,
            arrays,
            expected_rows=len(loader.dataset),
            expected_dim=feature_dim,
        )
        features, labels, preds, sample_ids, confidence, correct = arrays
        save_embeddings(
            features,
            labels,
            preds,
            split_name,
            output_dir,
            sample_ids=sample_ids,
            confidence=confidence,
            correct=correct,
        )
        print(
            f"Saved {split_name}: features={features.shape}, "
            f"accuracy={validation[split_name]['accuracy']:.6f}"
        )

    all_ids = np.concatenate(
        [
            np.load(os.path.join(output_dir, f"{split}_sample_ids.npy"))
            for split in ("train", "val", "test")
        ]
    )
    if len(all_ids) != 70000 or len(np.unique(all_ids)) != 70000:
        raise ValueError("Sample IDs do not form a unique partition of all 70,000 samples")

    checkpoint_hash = sha256_file(args.checkpoint)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset.name,
        "split_protocol": "combined MNIST 70000, stratified 70/15/15",
        "seed": seed,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "feature_layer": feature_layer,
        "feature_definition": "fc2 linear output captured before functional ReLU",
        "feature_dim": feature_dim,
        "train_shuffle_during_extraction": False,
        "sample_id_definition": (
            "index in combined original MNIST train-then-test array; 0..69999"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "validation": validation,
        "all_sample_ids_unique_and_complete": True,
    }

    manifest_path = os.path.join(output_dir, "embedding_manifest.json")
    validation_path = os.path.join(output_dir, "embedding_validation.json")
    with open(manifest_path, "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
    with open(validation_path, "w", encoding="utf-8") as file_handle:
        json.dump(validation, file_handle, indent=2)

    print(f"Checkpoint SHA-256: {checkpoint_hash}")
    print(f"All validated embeddings saved to {output_dir}")


if __name__ == "__main__":
    main()
