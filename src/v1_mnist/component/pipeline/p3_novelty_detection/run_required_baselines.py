"""Frozen-CNN max-softmax, energy, Mahalanobis and optional kNN evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    evaluate_scores,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
    max_softmax_novelty,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.paired_ood_comparison import (
    evaluate_paired_qe,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file as fingerprint

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("v1_mnist.p3.required_baselines")




def input_path(value: str) -> Path:
    """Resolve and restrict a configured path to the repository."""
    configured = Path(value)
    if configured.is_absolute() or ".." in configured.parts:
        raise ValueError(f"Invalid repository-relative input path: {value}")

    path = ROOT / configured
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


from src.v1_mnist.component.pipeline.p3_novelty_detection.cnn_score_inference import (
    reconstruct_logits,
    verify_predictions,
)


def run(config_path: Path, *, save: bool = False) -> dict[str, Any]:
    """Evaluate frozen samples without fitting or overwriting evaluation data."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import SVHN

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )

    config = load_config(config_path)
    run_id = str(config.run_id)
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run ID")

    output_root = ROOT / str(config.output_root)
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run must not be reused: {destination}")

    inputs = {
        key: input_path(str(getattr(config.inputs, key)))
        for key in (
            "checkpoint",
            "train_features",
            "train_labels",
            "validation_features",
            "validation_predictions",
            "validation_confidence",
            "svhn_file",
        )
    }
    hashes = {key: fingerprint(path) for key, path in inputs.items()}

    for key in ("checkpoint", "train_features", "validation_features"):
        expected_hash = str(getattr(config.expected_sha256, key))
        if hashes[key] != expected_hash:
            raise RuntimeError(f"Frozen SHA-256 mismatch: {key}")

    train = np.load(inputs["train_features"], allow_pickle=False)
    train_labels = np.load(inputs["train_labels"], allow_pickle=False)
    validation = np.load(inputs["validation_features"], allow_pickle=False)
    val_preds = np.load(inputs["validation_predictions"], allow_pickle=False)
    val_confidence = np.load(
        inputs["validation_confidence"], allow_pickle=False
    )

    expected = config.expected
    if train.shape != (
        int(expected.training_samples), int(expected.feature_dim)
    ):
        raise RuntimeError("Unexpected training embeddings")
    if validation.shape != (
        int(expected.validation_samples), int(expected.feature_dim)
    ):
        raise RuntimeError("Unexpected validation embeddings")
    if train_labels.shape != (len(train),):
        raise RuntimeError("Training labels are not aligned")

    torch.set_num_threads(int(config.inference.thread_count))
    model = LeNet5(num_classes=int(expected.classes))
    state = torch.load(
        inputs["checkpoint"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    val_logits = reconstruct_logits(model, validation)
    verify_predictions(
        val_logits, val_preds, val_confidence,
        float(config.inference.confidence_tolerance),
    )
    LOGGER.info("Frozen MNIST validation predictions verified")

    image_size = tuple(int(x) for x in config.inference.image_size)
    dataset = SVHN(
        root=str(inputs["svhn_file"].parent),
        split="test",
        download=False,
        transform=image_transform("svhn", image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.inference.batch_size),
        shuffle=False,
    )
    extracted = FeatureExtractor(
        model, str(config.inference.feature_layer)
    ).extract(loader, torch.device("cpu"), return_metadata=True)

    svhn_features, _, svhn_predictions, _, svhn_confidence, _ = extracted
    if svhn_features.shape != (
        int(expected.svhn_samples), int(expected.feature_dim)
    ):
        raise RuntimeError("Unexpected SVHN embedding shape")

    svhn_logits = reconstruct_logits(model, svhn_features)
    verify_predictions(
        svhn_logits, svhn_predictions, svhn_confidence,
        float(config.inference.confidence_tolerance),
    )
    LOGGER.info("Frozen SVHN predictions verified")

    temperature = float(config.evaluation.energy_temperature)

    scores = {
        "max_softmax": (
            max_softmax_novelty(val_logits),
            max_softmax_novelty(svhn_logits),
        ),
        "energy": (
            energy_novelty(val_logits, temperature),
            energy_novelty(svhn_logits, temperature),
        ),
    }

    if bool(getattr(config, "export_scores_only", False)):
        from sklearn.metrics import roc_auc_score

        neighbors = int(config.evaluation.knn_neighbors)
        scores["knn"] = (
            knn_novelty(train, validation, neighbors=neighbors),
            knn_novelty(train, svhn_features, neighbors=neighbors),
        )

        exported = {}
        labels = np.r_[
            np.zeros(len(validation), dtype=int),
            np.ones(len(svhn_features), dtype=int),
        ]
        for name in ("energy", "knn"):
            id_scores, ood_scores = scores[name]
            if not (
                len(id_scores) == len(validation)
                and len(ood_scores) == len(svhn_features)
                and np.isfinite(id_scores).all()
                and np.isfinite(ood_scores).all()
            ):
                raise RuntimeError(f"STOP: Invalid {name} score arrays")
            observed_auc = float(
                roc_auc_score(labels, np.r_[id_scores, ood_scores])
            )
            expected_auc = float(
                getattr(config.expected_export_auroc, name)
            )
            if not np.isclose(
                observed_auc, expected_auc, atol=1e-6, rtol=0
            ):
                raise RuntimeError(f"STOP: Historical {name} AUROC mismatch")
            exported[name] = observed_auc

        destination.mkdir(parents=True, exist_ok=False)
        for name in ("energy", "knn"):
            for split, values in zip(
                ("mnist_val", "svhn_test"), scores[name], strict=True
            ):
                output_path = destination / f"{name}_{split}_scores.npy"
                with output_path.open("xb") as stream:
                    np.save(stream, np.asarray(values, dtype=np.float64),
                            allow_pickle=False)

        metadata = {
            "run_id": run_id,
            "mode": "frozen_score_export_only",
            "auroc": exported,
            "n_id": len(validation),
            "n_ood": len(svhn_features),
            "knn_neighbors": neighbors,
            "energy_temperature": temperature,
            "input_sha256": hashes,
            "config_sha256": fingerprint(config_path),
            "leakage_control": (
                "Frozen CNN, original MNIST train/validation and SVHN test. "
                "No model retraining, threshold selection, or MNIST final test."
            ),
            "alignment_limitation": (
                "Unshuffled torchvision SVHN test ordering; historical QE "
                "arrays do not contain independently verifiable sample IDs."
            ),
        }
        with (destination / "score_export_metrics.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\\n")
        LOGGER.info("Exported energy and kNN scores to %s", destination)
        return metadata

    # Fit only on MNIST training embeddings; score both evaluation splits.
    joined = np.concatenate([validation, svhn_features], axis=0)
    mahalanobis = mahalanobis_novelty(train, train_labels, joined)
    scores["mahalanobis"] = (
        mahalanobis[: len(validation)],
        mahalanobis[len(validation) :],
    )

    if hasattr(config.evaluation, "knn_neighbors"):
        neighbors = int(config.evaluation.knn_neighbors)
        scores['knn'] = (
            knn_novelty(train, validation, neighbors=neighbors),
            knn_novelty(train, svhn_features, neighbors=neighbors),
        )

    results: dict[str, Any] = {}
    for name, (id_scores, ood_scores) in scores.items():
        results[name] = evaluate_scores(
            id_scores,
            ood_scores,
            bootstrap_repeats=int(config.evaluation.bootstrap_iterations),
            seed=int(config.evaluation.bootstrap_seed),
            target_tpr=float(config.evaluation.target_tpr),
        )
        LOGGER.info(
            "%s: AUROC=%.6f AUPR=%.6f FPR@95TPR=%.6f",
            name,
            results[name]["auroc"],
            results[name]["aupr"],
            results[name]["fpr_at_95_tpr"],
        )


    paired_result = evaluate_paired_qe(
        config=config,
        scores=scores,
        run_id=run_id,
        hashes=hashes,
        config_path=config_path,
        input_path=input_path,
        fingerprint=fingerprint,
        logger=LOGGER,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "id_dataset": "mnist",
        "id_split": "validation",
        "ood_dataset": "svhn",
        "ood_split": "test",
        "training_samples": len(train),
        "id_samples": len(validation),
        "ood_samples": len(svhn_features),
        "methods": results,
        "energy_temperature": temperature,
        "input_sha256": hashes,
        "config_sha256": fingerprint(config_path),
        "leakage_control": (
            "Frozen CNN. Mahalanobis fitted on MNIST training embeddings "
            "only. MNIST validation and SVHN used for evaluation, not tuning. "
            "No MNIST final-test samples accessed."
        ),
        "limitations": [
            "Single CNN checkpoint and single evaluation dataset.",
            "Energy temperature fixed at 1.0 for this new comparison.",
            "Mahalanobis uses shared LedoitWolf covariance shrinkage.",
            "These method settings are not claimed to be historical preregistration.",
        ],
    }

    if save:
        destination.mkdir(parents=True, exist_ok=False)
        output_path = destination / "required_baseline_metrics.json"
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        LOGGER.info("Saved %s", output_path)

        if paired_result is not None:
            paired_file = destination / "qe_msp_paired_metrics.json"
            with paired_file.open("x", encoding="utf-8") as stream:
                json.dump(paired_result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            LOGGER.info("Saved %s", paired_file)

    return result


def main() -> None:
    """Evaluate read-only by default; --save creates a separate run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "src/v1_mnist/component/configs/required_ood_baselines.yaml"
        ),
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = run(args.config, save=args.save)
    LOGGER.info("Evaluation completed: %s", result["run_id"])


if __name__ == "__main__":
    main()
