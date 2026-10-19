"""Reproduce saved Fashion-MNIST kNN/Energy/Mahalanobis evidence, read-only.

Same frozen CNN, MNIST train embeddings, MNIST validation and Fashion-MNIST
samples used by the original console measurements. No tuning or file writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
)

ROOT = Path(__file__).resolve().parents[5]
LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = (
    ROOT / "src/v1_mnist/component/configs/fashion_required_baselines.yaml"
)


def safe_input(path: str) -> Path:
    """Resolve inputs inside this worktree, rejecting external symlinks."""
    candidate = (ROOT / path).resolve()
    if not candidate.is_relative_to(ROOT) or not candidate.is_file():
        raise ValueError(f"Required regular file within repository not found: {path}")
    return candidate


def sha256_file(path: Path) -> str:
    """Hash frozen inputs for provenance without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auc_interval(
    normal: np.ndarray, novel: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    """Match the stratified 2,000-resample console bootstrap exactly."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(len(normal), dtype=np.int8), np.ones(len(novel), dtype=np.int8)]
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        ids = rng.integers(len(normal), size=len(normal))
        oods = rng.integers(len(novel), size=len(novel))
        values[index] = roc_auc_score(labels, np.r_[normal[ids], novel[oods]])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def calculate_metrics(
    normal: np.ndarray, novel: np.ndarray, *, samples: int, seed: int, tpr: float
) -> dict[str, Any]:
    """Use the same scores, directions and FPR interpolation as saved evidence."""
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    novel = np.asarray(novel, dtype=np.float64).reshape(-1)
    if not len(normal) or not len(novel) or not np.isfinite(normal).all() or not np.isfinite(novel).all():
        raise ValueError("Nonempty finite ID and OOD scores required")
    labels = np.r_[np.zeros(len(normal), dtype=np.int8), np.ones(len(novel), dtype=np.int8)]
    scores = np.r_[normal, novel]
    fpr, actual_tpr, _ = roc_curve(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auroc_ci95": list(auc_interval(normal, novel, samples=samples, seed=seed)),
        "aupr_ood": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": float(np.interp(tpr, actual_tpr, fpr)),
    }


def verify_metrics(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail rather than rewrite evidence when any measured metric differs."""
    for key in ("auroc", "auroc_ci95", "aupr_ood", "fpr_at_95_tpr"):
        if key not in expected or not np.allclose(actual[key], expected[key], atol=1e-10, rtol=0):
            raise RuntimeError(f"Reproduction mismatch: {key}: {actual[key]} versus {expected.get(key)}")


def verify(config_path: Path) -> dict[str, Any]:
    """Read-only reproduction of all three saved scientific comparisons."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import FashionMNIST

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    embeddings = str(config["embeddings_dir"])
    inputs = {
        "checkpoint": safe_input(str(config["checkpoint"])),
        "train_features": safe_input(f"{embeddings}/train_features.npy"),
        "train_labels": safe_input(f"{embeddings}/train_labels.npy"),
        "validation_features": safe_input(f"{embeddings}/val_features.npy"),
    }
    recorded = {
        name: json.loads(safe_input(str(path)).read_text(encoding="utf-8"))
        for name, path in config["results"].items()
    }
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    # Prevent evaluating with different frozen inputs than those used originally.
    knn_hashes = recorded["knn"]["input_sha256"]
    em_hashes = recorded["energy_mahalanobis"]["input_sha256"]
    for key in ("checkpoint", "train_features", "validation_features"):
        if hashes[key] != knn_hashes[key] or hashes[key] != em_hashes[key]:
            raise RuntimeError(f"Frozen input SHA-256 mismatch: {key}")
    if hashes["train_labels"] != em_hashes["train_labels"]:
        raise RuntimeError("Frozen training labels SHA-256 mismatch")

    train = np.load(inputs["train_features"], allow_pickle=False)
    labels = np.load(inputs["train_labels"], allow_pickle=False)
    validation = np.load(inputs["validation_features"], allow_pickle=False)
    dimension = int(config["feature_dim"])
    if train.shape != (int(config["expected_train_count"]), dimension):
        raise RuntimeError("Unexpected MNIST training features")
    if labels.shape != (len(train),) or validation.shape != (
        int(config["expected_validation_count"]), dimension
    ):
        raise RuntimeError("Unaligned MNIST data")

    torch.set_num_threads(int(config["threads"]))
    model = LeNet5(num_classes=int(config["num_classes"]))
    model.load_state_dict(torch.load(inputs["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    fashion = FashionMNIST(
        root=str(ROOT / str(config["fashion_root"])), train=False, download=False,
        transform=image_transform("fashion_mnist", (int(config["image_size"]),) * 2),
    )
    loader = DataLoader(fashion, batch_size=int(config["batch_size"]), shuffle=False)
    features, _, _ = FeatureExtractor(model, str(config["feature_layer"])).extract(
        loader, torch.device("cpu")
    )
    if features.shape != (int(config["expected_fashion_count"]), dimension):
        raise RuntimeError("Unexpected Fashion-MNIST embeddings")

    joined = np.concatenate([validation, features], axis=0)
    cut = len(validation)
    with torch.inference_mode():
        val_logits = model.fc3(torch.relu(torch.from_numpy(validation).float())).numpy()
        fashion_logits = model.fc3(torch.relu(torch.from_numpy(features).float())).numpy()
    temperature = float(config["energy_temperature"])
    scoring = {
        "knn": knn_novelty(train, joined, neighbors=int(config["neighbors"])),
        "energy": np.r_[energy_novelty(val_logits, temperature), energy_novelty(fashion_logits, temperature)],
        "mahalanobis": mahalanobis_novelty(train, labels, joined),
    }
    verified: dict[str, Any] = {}
    for name, score in scoring.items():
        metrics = calculate_metrics(
            score[:cut], score[cut:], samples=int(config["bootstrap_iterations"]),
            seed=int(config["bootstrap_seed"]), tpr=float(config["target_tpr"]),
        )
        record = recorded["knn"]["metrics"] if name == "knn" else recorded["energy_mahalanobis"][name]
        verify_metrics(metrics, record)
        LOGGER.info("PASS %s: AUROC %.8f", name, metrics["auroc"])
        verified[name] = metrics
    LOGGER.info("PASS: Three baselines reproduce frozen evidence; no output files written")
    return verified


def main() -> None:
    """Run explicit read-only verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify", action="store_true", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    verify(args.config)


if __name__ == "__main__":
    main()
