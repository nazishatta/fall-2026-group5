"""Additive Phase 3 diagnostic: train-scaled autoencoder on frozen CNN embeddings.

Never changes historical models, code, thresholds, or results. This is an
exploratory follow-up, NOT a replacement for the preregistered baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
    FeatureExtractor,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_baseline import (
    fit_autoencoder,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    evaluate_scores,
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


def check_file(path: Path, expected_hash: str) -> None:
    """Fail closed when a historical immutable input has changed."""
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {path}: {observed}")


def scaled_embeddings(
    train: np.ndarray, validation: np.ndarray, svhn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler on MNIST train only, transform both evaluation sets."""
    if train.ndim != 2 or validation.ndim != 2 or svhn.ndim != 2:
        raise ValueError("Expected 2D features")
    if train.shape[1] != validation.shape[1] or train.shape[1] != svhn.shape[1]:
        raise ValueError("Embedding dimension mismatch")
    if any(not np.isfinite(a).all() for a in (train, validation, svhn)):
        raise ValueError("Nonfinite feature values")
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train).astype(np.float32)
    scoring_scaled = scaler.transform(np.concatenate((validation, svhn))).astype(np.float32)
    return train_scaled, scoring_scaled


def run(config_path: Path) -> Path:
    """Run exactly once with a new ID, preserving historical outputs."""
    config_path = config_path.resolve()
    settings: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_shared_inputs(settings["inputs"])
    run_id = str(settings["run_id"])
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run ID")
    out_dir = ROOT / str(settings["output_root"]) / run_id
    if out_dir.exists():
        raise FileExistsError(f"STOP: run already exists: {out_dir}")

    paths = {key: ROOT / str(value) for key, value in settings["inputs"].items()}
    for key, expected_hash in settings["expected_sha256"].items():
        check_file(paths[key], str(expected_hash))

    qecfg = read_settings(ROOT / str(settings["qe_evaluation_config"]))
    if tuple(qecfg["image_size"]) != (28, 28) or qecfg["feature_layer"] != "fc2":
        raise RuntimeError("Frozen CNN feature/preprocessing protocol mismatch")
    if int(settings["seed"]) != int(qecfg["bootstrap"]["seed"]):
        raise RuntimeError("Seed differs from historical evaluation")

    train = np.load(paths["train_features"], allow_pickle=False)
    val = np.load(paths["validation_features"], allow_pickle=False)
    expected = settings["expected"]
    dim = int(expected["feature_dim"])
    if train.shape != (int(expected["training_samples"]), dim):
        raise RuntimeError("Unexpected MNIST training shape")
    if val.shape != (int(expected["validation_samples"]), dim):
        raise RuntimeError("Unexpected MNIST validation shape")

    torch.set_num_threads(int(settings["inference"]["thread_count"]))
    # Historical preflight verifies the frozen CNN, SOM, threshold, and CNN checkpoint.
    model, _, frozen_id_qe, provenance = prepare(qecfg)
    if len(frozen_id_qe) != len(val):
        raise RuntimeError("Frozen validation features not aligned")
    dataset = SVHN(
        root=str(paths["svhn_file"].parent), split="test", download=False,
        transform=image_transform("svhn", (28, 28)),
    )
    loader = DataLoader(
        dataset, batch_size=int(settings["inference"]["batch_size"]), shuffle=False
    )
    svhn, labels, _ = FeatureExtractor(model, "fc2").extract(
        loader, torch.device("cpu")
    )
    if svhn.shape != (int(expected["svhn_samples"]), dim) or len(labels) != len(svhn):
        raise RuntimeError("Unexpected SVHN feature count/order")
    del model, dataset, loader

    train_scaled, scoring_scaled = scaled_embeddings(train, val, svhn)
    config = {"seed": int(settings["seed"]), "autoencoder": settings["autoencoder"]}
    scores, losses = fit_autoencoder(train_scaled, scoring_scaled, config)
    id_scores, ood_scores = scores[: len(val)], scores[len(val) :]
    metric = evaluate_scores(
        id_scores, ood_scores,
        bootstrap_repeats=int(settings["evaluation"]["bootstrap_iterations"]),
        seed=int(settings["evaluation"]["bootstrap_seed"]),
        target_tpr=float(settings["evaluation"]["target_tpr"]),
    )
    report: dict[str, Any] = {
        "run_id": run_id,
        "method": "autoencoder_train_standardized_diagnostic",
        "id_dataset": "mnist", "id_split": "validation",
        "ood_dataset": "svhn", "ood_split": "test",
        "metrics": metric,
        "losses": {"initial": losses[0], "final": losses[-1]},
        "preprocessing": "StandardScaler fitted on MNIST training embeddings ONLY",
        "config_sha256": sha256_file(config_path),
        "input_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "frozen_provenance": provenance,
        "leakage_control": (
            "Frozen CNN and SOM. Train-fitted scaler and autoencoder see MNIST "
            "training features only. MNIST validation and SVHN test used only "
            "for evaluation. No threshold tuning or MNIST final test access."
        ),
        "limitation": (
            "Exploratory follow-up after observing anomalous historical OOD AUROC. "
            "Do not replace historical result or claim independent test-based selection."
        ),
    }
    # Save only after full successful evaluation. The preflight forbids reuse.
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "standardized_autoencoder_metrics.json").open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    LOGGER.info("Run %s: AUROC=%.6f AUPR=%.6f FPR95=%.6f", run_id, metric["auroc"], metric["aupr"], metric["fpr_at_95_tpr"])
    LOGGER.info("Saved %s", out_dir)
    return out_dir


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.config)


if __name__ == "__main__":
    main()
