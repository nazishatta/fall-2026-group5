"""Independent development-only MNIST-to-OOD QE evaluation; no MNIST test access."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_scoring import (
    score_embeddings,
)
from src.v1_mnist.component.utils.config import load_config

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]


def resolved(path: str) -> Path:
    """Resolve a configured path under the checkout, rejecting escapes."""
    candidate = (ROOT / path).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {path}")
    return candidate


def read_settings(config_path: Path) -> dict[str, Any]:
    """Validate the development-only configuration without reading test samples."""
    shared = load_config(config_path)
    config = vars(shared).copy()
    config['ood_datasets'] = [vars(entry).copy() for entry in shared.ood_datasets]
    config['bootstrap'] = vars(shared.bootstrap).copy()
    config['image_size'] = [int(shared.dataset.image_size)] * 2
    config['feature_layer'] = str(shared.model.feature_layer)
    if not isinstance(config, dict) or config.get("id_dataset") != "mnist":
        raise ValueError("This runner currently supports the MNIST ID model only")
    if config.get("id_split") != "val":
        raise ValueError("Development runner must use MNIST validation, never test")
    entries = config.get("ood_datasets")
    if not isinstance(entries, list) or {e.get("name") for e in entries} != {
        "svhn",
        "fashion_mnist",
    }:
        raise ValueError("Expected separate SVHN and Fashion-MNIST experiments")
    if any(e.get("split") != "test" for e in entries):
        raise ValueError("Unexpected OOD evaluation split")
    run_ids = [e.get("run_id") for e in entries]
    if any(not isinstance(x, str) or not x or "/" in x or ".." in x for x in run_ids):
        raise ValueError("Invalid output run ID")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run IDs")
    if tuple(config["image_size"]) != (28, 28) or config["feature_layer"] != "fc2":
        raise ValueError("This frozen CNN requires fc2 and grayscale 28x28")
    for key in (
        "som_config",
        "qe_config",
        "cnn_checkpoint",
        "output_dir",
        "data_root",
        "training_distance_path",
        "validation_distance_path",
    ):
        resolved(str(config[key]))
    if int(config["batch_size"]) <= 0:
        raise ValueError("Batch size must be positive")
    if not 0 < float(config["threshold_percentile"]) < 100:
        raise ValueError("Invalid training QE percentile")
    return config


def prepare(config: dict[str, Any]) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    """Verify frozen inputs, reconstruct training-only scaler; never load MNIST test."""
    import torch

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )
    from src.v1_mnist.component.utils.run_settings import (
        load_analysis_settings,
        verify_selected_model,
    )
    from src.v1_mnist.component.utils.som_io import (
        load_selected_som,
        load_split,
        sha256_file,
    )

    settings = load_analysis_settings(resolved(config["som_config"]), ROOT)
    som_sha = verify_selected_model(settings)
    train_path = settings.embeddings_dir / "train_features.npy"
    val_distance_path = resolved(config["validation_distance_path"])
    train_distance_path = resolved(config["training_distance_path"])
    train_hash = sha256_file(train_path)
    expected_train = settings.expected_inputs.get("train_features_sha256")
    if expected_train and train_hash != expected_train:
        raise RuntimeError("MNIST training embeddings SHA-256 mismatch")
    train = load_split("train", settings.embeddings_dir, settings.feature_dim)
    # The validation metadata is checked without ever opening the final-test split.
    load_split("val", settings.embeddings_dir, settings.feature_dim)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(train["features"])
    som = load_selected_som(
        settings.model_path,
        settings.grid_height,
        settings.grid_width,
        settings.feature_dim,
    )
    som.norm_func = scaler.transform

    qe_config = vars(load_config(resolved(config["qe_config"])))
    qe_summary = (
        resolved(qe_config["output_dir"]) / qe_config["run_id"] / "qe_summary.json"
    )
    prior = json.loads(qe_summary.read_text(encoding="utf-8"))
    if prior["model_sha256"] != som_sha or prior["som_run"] != settings.selected_model:
        raise RuntimeError("QE threshold provenance does not match frozen SOM")
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.quantization_novelty import (
        calculate_threshold,
        load_distances,
    )
    from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_novelty import (
        require_hash,
    )

    hashes = prior["input_sha256"]
    require_hash(train_distance_path, hashes["train_bmu_distance"])
    require_hash(val_distance_path, hashes["val_bmu_distance"])
    train_scores = load_distances(train_distance_path, int(prior["train_count"]))
    threshold = calculate_threshold(train_scores, float(config["threshold_percentile"]))
    if threshold != float(prior["threshold"]):
        raise RuntimeError("QE threshold differs from saved calibration")
    validation = load_distances(val_distance_path, int(prior["validation_count"]))

    ckpt = resolved(config["cnn_checkpoint"])
    checkpoint_sha = sha256_file(ckpt)
    if checkpoint_sha != str(config["cnn_sha256"]):
        raise RuntimeError("CNN checkpoint SHA-256 mismatch")
    model = LeNet5(num_classes=settings.num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return (
        model,
        som,
        validation,
        {
            "qe_threshold": threshold,
            "som_model": settings.selected_model,
            "som_sha256": som_sha,
            "cnn_sha256": checkpoint_sha,
            "train_features_sha256": train_hash,
            "train_qe_sha256": hashes["train_bmu_distance"],
            "val_qe_sha256": hashes["val_bmu_distance"],
        },
    )


def evaluate(config: dict[str, Any], save: bool = False) -> list[dict[str, Any]]:
    """Evaluate offline-cached OOD datasets without downloads or final MNIST test."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )

    output_root = resolved(config["output_dir"])
    if any(
        output_root == resolved(config[key])
        or output_root in resolved(config[key]).parents
        for key in (
            "som_config",
            "qe_config",
            "cnn_checkpoint",
            "training_distance_path",
            "validation_distance_path",
        )
    ):
        raise ValueError("Output directory overlaps protected upstream artifacts")
    for entry in config["ood_datasets"]:
        if (output_root / entry["run_id"]).exists():
            raise FileExistsError(f"Existing run: {entry['run_id']}")
    model, som, validation, provenance = prepare(config)
    results: list[dict[str, Any]] = []
    for entry in config["ood_datasets"]:
        name = entry["name"]
        transform = image_transform(name, tuple(config["image_size"]))
        data_root = resolved(config["data_root"])
        if name == "svhn":
            dataset = datasets.SVHN(
                str(data_root), split="test", transform=transform, download=False
            )
        else:
            dataset = datasets.FashionMNIST(
                str(data_root), train=False, transform=transform, download=False
            )
        loader = DataLoader(
            dataset, batch_size=int(config["batch_size"]), shuffle=False
        )
        features, _, _ = FeatureExtractor(model, str(config["feature_layer"])).extract(
            loader, torch.device("cpu")
        )
        if features.ndim != 2 or features.shape[1] != som.w.shape[1]:
            raise ValueError("OOD embedding dimension mismatch")
        scores = score_embeddings(som, features)
        report = ood_metrics(
            validation, scores, provenance["qe_threshold"], float(config["target_tpr"])
        )
        low, high = stratified_auroc_interval(
            validation,
            scores,
            iterations=int(config["bootstrap"]["iterations"]),
            seed=int(config["bootstrap"]["seed"]),
            confidence=float(config["bootstrap"]["confidence"]),
        )
        results.append(
            {
                "run_id": entry["run_id"],
                "id_dataset": "mnist",
                "id_split": "val",
                "ood_dataset": name,
                "ood_split": "test",
                "development_only": True,
                "final_mnist_test_used": False,
                "metrics": report,
                "auroc_bootstrap_interval": [low, high],
                "provenance": provenance,
            }
        )
    if save:
        # Repeat the preflight to guard against paths created during evaluation.
        for item in results:
            if (output_root / item["run_id"]).exists():
                raise FileExistsError(f"Existing run: {item['run_id']}")
        for item in results:
            folder = output_root / item["run_id"]
            folder.mkdir(parents=True, exist_ok=False)
            with (folder / "qe_ood_metrics.json").open("x", encoding="utf-8") as handle:
                json.dump(item, handle, indent=2)
                handle.write("\n")
            LOGGER.info("Saved %s", folder)
    return results


def main() -> None:
    """Validate config by default; OOD evaluation requires an explicit flag."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/v1_mnist/component/configs/qe_ood_evaluation.yaml"),
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    from src.v1_mnist.component.utils.logging import get_logger

    global LOGGER
    LOGGER = get_logger("v1_mnist.p3.qe_ood")
    if args.save and not args.evaluate:
        parser.error("--save requires --evaluate")
    config = read_settings(args.config)
    if args.evaluate:
        LOGGER.info(
            "Evaluation finished: %s", json.dumps(evaluate(config, save=args.save))
        )
    else:
        LOGGER.info("Configuration validated; OOD datasets not opened")


if __name__ == "__main__":
    main()
