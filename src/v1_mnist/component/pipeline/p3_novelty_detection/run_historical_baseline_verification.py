"""Recompute historical MNIST-only validation summaries in a NEW run folder.

This verifies numerical reproduction, not a byte-identical historical run:
the original autoencoder checkpoint and environment manifest are unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_baseline import (
    fit_autoencoder,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.classical_baselines import (
    fit_classical_baselines,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("v1_mnist.p3.historical_baseline_verification")


def resolve_repo_path(relative: str) -> Path:
    """Resolve a repository-relative path while preventing directory escape."""
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {relative}")
    return candidate


def compare_summaries(
    actual: dict[str, Any], historical: dict[str, Any], *, atol: float, rtol: float
) -> dict[str, dict[str, float | bool]]:
    """Compare numeric leaves explicitly; never modify historical evidence."""
    comparisons: dict[str, dict[str, float | bool]] = {}
    for field, observed in actual.items():
        reference = historical
        for part in field.split("."):
            reference = reference[part]
        measured = float(observed)
        expected = float(reference)
        comparisons[field] = {
            "observed": measured,
            "historical": expected,
            "absolute_difference": abs(measured - expected),
            "agrees": bool(np.isclose(measured, expected, atol=atol, rtol=rtol)),
        }
    return comparisons


def run(config_path: Path, *, save: bool = False) -> dict[str, Any]:
    """Verify frozen inputs, rerun both methods, optionally save new evidence."""
    settings = load_config(config_path)
    benchmark_path = resolve_repo_path(str(settings.benchmark_config))
    config = load_config(benchmark_path)
    run_id = str(settings.run_id)
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run ID")
    destination = resolve_repo_path(str(settings.output_root)) / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run must not be overwritten: {destination}")

    from src.v1_mnist.component.utils.som_io import sha256_file

    train_path = resolve_repo_path(str(config.embeddings.train))
    validation_path = resolve_repo_path(str(config.embeddings.validation))
    input_hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    for name, observed in input_hashes.items():
        expected = str(getattr(config.expected_sha256, name))
        if observed != expected:
            raise RuntimeError(f"Frozen {name} embeddings SHA-256 mismatch")

    train = np.load(train_path, allow_pickle=False)
    validation = np.load(validation_path, allow_pickle=False)
    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("Expected 2D frozen embedding matrices")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("Embedding dimension mismatch")
    if train.shape[0] != int(config.expected.mnist_train_count):
        raise ValueError("Frozen training count mismatch")
    if validation.shape[0] != int(config.expected.mnist_validation_count):
        raise ValueError("Frozen validation count mismatch")
    if train.shape[1] != int(config.expected.feature_dim):
        raise ValueError("Frozen feature dimension mismatch")

    classical_path = resolve_repo_path(str(settings.historical.classical))
    autoencoder_path = resolve_repo_path(str(settings.historical.autoencoder))
    historical_classical = json.loads(classical_path.read_text(encoding="utf-8"))
    historical_autoencoder = json.loads(autoencoder_path.read_text(encoding="utf-8"))
    historical_hashes = {
        "classical": sha256_file(classical_path),
        "autoencoder": sha256_file(autoencoder_path),
    }
    config_values = {
        "seed": int(config.seed),
        "isolation_forest": vars(config.isolation_forest),
        "one_class_svm": vars(config.one_class_svm),
        "autoencoder": vars(config.autoencoder),
    }
    classical_scores = fit_classical_baselines(train, validation, config_values)
    reconstruction, losses = fit_autoencoder(train, validation, config_values)
    comparisons = {
        "classical": compare_summaries(
            {
                "isolation_forest.mean_novelty_score": float(np.mean(classical_scores["isolation_forest"])),
                "isolation_forest.std_novelty_score": float(np.std(classical_scores["isolation_forest"])),
                "one_class_svm.mean_novelty_score": float(np.mean(classical_scores["one_class_svm"])),
                "one_class_svm.std_novelty_score": float(np.std(classical_scores["one_class_svm"])),
            },
            historical_classical,
            atol=float(settings.agreement.absolute_tolerance),
            rtol=float(settings.agreement.relative_tolerance),
        ),
        "autoencoder": compare_summaries(
            {
                "initial_training_loss": losses[0],
                "final_training_loss": losses[-1],
                "validation_mean_reconstruction_error": float(np.mean(reconstruction)),
            },
            historical_autoencoder,
            atol=float(settings.agreement.absolute_tolerance),
            rtol=float(settings.agreement.relative_tolerance),
        ),
    }
    agrees = all(item["agrees"] for group in comparisons.values() for item in group.values())
    payload: dict[str, Any] = {
        "run_id": run_id,
        "scope": "training-fit and MNIST-validation-only comparison; no final test or OOD access",
        "historical_input_sha256": historical_hashes,
        "frozen_embedding_sha256": input_hashes,
        "comparison": comparisons,
        "all_numeric_metrics_agree": agrees,
        "limitations": [
            "Recomputed summaries are not proof of an identical original checkpoint.",
            "Original historical package versions were not preserved.",
            "Method configuration is based on currently committed settings.",
            "The original historical JSON artifacts remain immutable.",
        ],
    }
    if save:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "verification_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("Saved new validation recheck: %s", destination)
    else:
        LOGGER.info("Validation recheck completed without writing outputs")
    return payload


def main() -> None:
    """Recompute on demand; use --save only for a new, unused run ID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "src/v1_mnist/component/configs/historical_baseline_verification.yaml",
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = run(args.config, save=args.save)
    LOGGER.info("All historical numeric comparisons agree: %s", result["all_numeric_metrics_agree"])


if __name__ == "__main__":
    main()
