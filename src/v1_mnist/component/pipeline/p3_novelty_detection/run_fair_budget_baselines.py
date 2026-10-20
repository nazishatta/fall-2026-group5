"""Post-hoc equal-training-pool novelty baseline diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
    FeatureExtractor,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_baseline import (
    fit_autoencoder,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.classical_baselines import (
    fit_classical_baselines,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.shared_provenance import (
    verify_shared_inputs,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
    image_transform,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    prepare,
    read_settings,
)
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]
METHODS = ("isolation_forest", "one_class_svm", "autoencoder")


def repository_path(value: str) -> Path:
    """Resolve an input within the repository."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject inconsistent training budgets and seed protocols."""
    seeds = config["seed_list"]
    budget = int(config["training_sample_count"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("At least five distinct seeds are required")
    if budget <= 0:
        raise ValueError("Training sample budget must be positive")
    if int(config["one_class_svm"]["max_training_samples"]) != budget:
        raise ValueError("SVM training budget differs from common budget")
    if config["analysis_status"] != "post_hoc_exploratory":
        raise ValueError("This experiment must be labeled post-hoc")


def evaluate_seed(
    train: np.ndarray,
    validation: np.ndarray,
    svhn: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Fit three methods on identical MNIST subset indices."""
    budget = int(config["training_sample_count"])
    selected = np.random.default_rng(seed).choice(
        len(train), size=budget, replace=False
    )
    subset = train[selected]
    joined = np.concatenate((validation, svhn), axis=0)
    cut = len(validation)
    settings = {
        **config,
        "seed": seed,
    }

    classical = fit_classical_baselines(subset, joined, settings)
    ae_scores, losses = fit_autoencoder(subset, joined, settings)
    scores = {
        **classical,
        "autoencoder": ae_scores,
    }

    labels = np.concatenate(
        (np.zeros(cut, dtype=int), np.ones(len(svhn), dtype=int))
    )
    metrics = {}
    for method in METHODS:
        values = np.asarray(scores[method], dtype=np.float64)
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid evaluation scores: {method}")
        metrics[method] = {
            "auroc": float(roc_auc_score(labels, values)),
            "aupr": float(average_precision_score(labels, values)),
        }

    subset_hash = hashlib.sha256(
        np.ascontiguousarray(selected, dtype=np.int64).tobytes()
    ).hexdigest()

    return {
        "seed": seed,
        "training_samples_per_method": budget,
        "subset_indices_sha256": subset_hash,
        "methods": metrics,
        "autoencoder_initial_loss": float(losses[0]),
        "autoencoder_final_loss": float(losses[-1]),
    }


def run(config_path: Path) -> Path:
    """Execute fresh equal-pool fits, without overwriting old results."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_protocol(config)

    run_id = str(config["run_id"])
    if not run_id or not all(
        char.isalnum() or char in "_-" for char in run_id
    ):
        raise ValueError("Invalid run ID")

    destination = repository_path(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run must not be reused: {destination}")

    verify_shared_inputs({
        "train_features": str(config["embeddings"]["train"]),
        "validation_features": str(config["embeddings"]["validation"]),
        "svhn_file": str(config["svhn_file"]),
    })

    train_path = repository_path(config["embeddings"]["train"])
    val_path = repository_path(config["embeddings"]["validation"])
    hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(val_path),
    }
    if hashes != config["expected_sha256"]:
        raise RuntimeError("Frozen MNIST embedding fingerprints mismatch")

    train = np.load(train_path, allow_pickle=False)
    validation = np.load(val_path, allow_pickle=False)
    expected = config["expected"]
    width = int(expected["feature_dim"])

    if train.shape != (int(expected["mnist_train_count"]), width):
        raise RuntimeError("Unexpected MNIST training embeddings")
    if validation.shape != (
        int(expected["mnist_validation_count"]), width
    ):
        raise RuntimeError("Unexpected MNIST validation embeddings")
    if int(config["training_sample_count"]) > len(train):
        raise ValueError("Training sample budget exceeds available data")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Nonfinite frozen embeddings")

    # Follow the historical SVHN preprocessing and frozen CNN protocol.
    svhn_relative = Path(str(config["svhn_file"]))
    if (
        svhn_relative.is_absolute()
        or svhn_relative.parts != ("data", "ood_svhn", "test_32x32.mat")
    ):
        raise ValueError("Unexpected SVHN dataset location")
    svhn_file = ROOT / svhn_relative
    svhn_hash = sha256_file(svhn_file)

    ood_settings = read_settings(
        repository_path(config["qe_evaluation_config"])
    )
    if (
        tuple(ood_settings["image_size"]) != (28, 28)
        or ood_settings["feature_layer"] != "fc2"
    ):
        raise RuntimeError("Frozen CNN feature protocol mismatch")

    model, _, _, provenance = prepare(ood_settings)
    if float(provenance["qe_threshold"]) != float(expected["qe_threshold"]):
        raise RuntimeError("Frozen QE protocol mismatch")

    dataset = SVHN(
        root=str(svhn_file.parent),
        split="test",
        download=False,
        transform=image_transform("svhn", (28, 28)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(ood_settings["batch_size"]),
        shuffle=False,
    )
    torch.set_num_threads(1)
    svhn, _, _ = FeatureExtractor(model, "fc2").extract(
        loader, torch.device("cpu")
    )
    del model, loader, dataset
    if svhn.shape != (int(expected["svhn_count"]), width):
        raise RuntimeError("Unexpected frozen SVHN embeddings")

    results = []
    for seed in config["seed_list"]:
        LOGGER.info("Fair-budget seed %s: fitting three methods", seed)
        outcome = evaluate_seed(
            train, validation, svhn, config, int(seed)
        )
        results.append(outcome)
        for method in METHODS:
            LOGGER.info(
                "Seed %s %s AUROC %.6f AUPR %.6f",
                seed, method,
                outcome["methods"][method]["auroc"],
                outcome["methods"][method]["aupr"],
            )

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ("auroc", "aupr"):
            values = np.array(
                [r["methods"][method][metric] for r in results],
                dtype=float,
            )
            summary[method][metric] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "per_seed": values.tolist(),
            }

    output = {
        "run_id": run_id,
        "analysis_status": "post_hoc_exploratory",
        "training_sample_count": int(config["training_sample_count"]),
        "seeds": config["seed_list"],
        "per_seed_results": results,
        "summary": summary,
        "input_sha256": {
            **hashes, "svhn_mat": svhn_hash
        },
        "config_sha256": sha256_file(config_path),
        "leakage_control": (
            "All three baselines trained only on the same 10,000 MNIST "
            "training examples per seed. MNIST validation and SVHN test "
            "used for evaluation, not fitting; final MNIST test untouched."
        ),
        "limitation": (
            "Post-hoc analysis after prior SVHN result inspection. "
            "Training-pool sizes are equal, but computational budgets, "
            "model preprocessing, and optimization costs are not equal. "
            "Seed spread is not an independent-data confidence interval."
        ),
    }

    destination.mkdir(parents=True, exist_ok=False)
    metrics_path = destination / "fair_budget_metrics.json"
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write(chr(10))

    LOGGER.info("Saved fair-budget metrics: %s", metrics_path)
    return metrics_path


def main() -> None:
    """Run the configuration-driven fair-budget experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.config)


if __name__ == "__main__":
    main()
