"""Frozen-CNN SVHN versus intensity-inverted SVHN norm diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.qe_norm_dependence import (
    embedding_norms,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.qe_norm_dependence import (
    norm_summary as summarize_norms,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.run_ids import check_run_id
from src.v1_mnist.component.utils.shared_provenance import (
    ROOT,
    load_manifest,
    verify_shared_inputs,
)
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = get_logger(__name__)


def norm_summary(features: np.ndarray) -> dict[str, float | int]:
    """Summarize L2 norms of finite, two-dimensional embeddings."""
    return summarize_norms(embedding_norms(features))


def compare_norms(
    original: np.ndarray, inverted: np.ndarray
) -> dict[str, Any]:
    """Calculate aligned embedding-norm comparisons."""
    if original.shape != inverted.shape:
        raise ValueError("Original and inverted embeddings must align")
    first = norm_summary(original)
    second = norm_summary(inverted)
    delta = (
        np.linalg.norm(inverted.astype(np.float64), axis=1)
        - np.linalg.norm(original.astype(np.float64), axis=1)
    )
    return {
        "original": first,
        "inverted": second,
        "paired_delta_mean": float(np.mean(delta)),
        "paired_delta_median": float(np.median(delta)),
        "inverted_greater_fraction": float(np.mean(delta > 0)),
    }


def run(config_path: Path) -> Path:
    """Run a new non-overwriting diagnostic on the frozen CNN."""
    import torch
    from torchvision import transforms

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
        svhn_test_loader,
    )

    config = load_config(config_path)
    manifest = load_manifest()
    run_id = str(config.run_id)
    check_run_id(run_id)

    destination = ROOT / str(config.output_root) / run_id
    if destination.exists():
        raise FileExistsError(f"Run already exists: {destination}")

    checkpoint_rel = str(manifest["inputs"]["checkpoint"])
    svhn_rel = str(manifest["svhn_file"])
    verify_shared_inputs(
        {"checkpoint": checkpoint_rel, "svhn_file": svhn_rel},
        required_keys=("checkpoint", "svhn_file"),
    )

    checkpoint = ROOT / checkpoint_rel
    svhn_file = ROOT / svhn_rel
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.set_num_threads(1)

    model = LeNet5(num_classes=int(config.classes))
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    size = (int(config.image_size), int(config.image_size))
    original_transform = image_transform("svhn", size)

    inverted_transform = transforms.Compose([
        original_transform,
        transforms.Lambda(lambda pixels: 1.0 - pixels),
    ])

    def extract(transform: Any) -> np.ndarray:
        loader = svhn_test_loader(svhn_file, size, int(config.batch_size), transform=transform)
        with torch.no_grad():
            extracted = FeatureExtractor(model, "fc2").extract(
                loader, torch.device("cpu")
            )
        features = np.asarray(extracted[0])
        if features.shape != (
            len(loader.dataset), int(config.feature_dim)
        ):
            raise RuntimeError("Unexpected frozen CNN embedding shape")
        return features

    LOGGER.info("Extracting original SVHN embeddings")
    original = extract(original_transform)
    LOGGER.info("Extracting inverted SVHN embeddings")
    inverted = extract(inverted_transform)

    metrics = compare_norms(original, inverted)
    payload = {
        "run_id": run_id,
        "analysis_status": "post_hoc_exploratory",
        "dataset": "SVHN test",
        "sample_count": len(original),
        "feature_layer": "fc2",
        "feature_dim": int(config.feature_dim),
        "inversion": "1.0 - grayscale tensor intensity",
        "metrics": metrics,
        "input_sha256": {
            "checkpoint": sha256_file(checkpoint),
            "svhn_file": sha256_file(svhn_file),
        },
        "config_sha256": sha256_file(config_path),
        "seed": int(config.seed),
        "leakage_control": (
            "Frozen CNN evaluated only. No model fitting, threshold "
            "selection, MNIST final-test access or benchmark retuning."
        ),
        "limitations": (
            "Exploratory comparison of embedding magnitude, not "
            "novelty-detection accuracy or a causal explanation."
        ),
    }

    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "embedding_norm_metrics.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    LOGGER.info("Saved diagnostic to %s", output)
    return output


def main() -> None:
    """Execute an explicitly configured new diagnostic run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
