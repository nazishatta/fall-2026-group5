"""Single-run, nine-detector exploratory MNIST-validation versus SVHN benchmark."""
from __future__ import annotations

import argparse
import json
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
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.classical_baselines import (
    fit_classical_baselines,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
    max_softmax_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.shared_provenance import (
    load_manifest,
    verify_shared_inputs,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
    image_transform,
)
from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
    neuron_u_matrix,
    sample_u_matrix,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.cnn_score_inference import (
    reconstruct_logits,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    prepare,
    read_settings,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.run_svhn_benchmark import (
    score_som,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.unified_novelty_metrics import (
    METHODS,
    evaluate,
    seed_summary,
    stratified_bootstrap,
    subset_fingerprint,
    subset_indices,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger(__name__)
FIT_NAMES = ("isolation_forest", "one_class_svm", "autoencoder", "mahalanobis", "knn")
FROZEN_NAMES = ("qe", "u_matrix", "max_softmax", "energy")


def repo_path(value: str) -> Path:
    """Resolve only safe, repository-relative paths."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative path: {value}")
    return ROOT / relative


def check_protocol(config: dict[str, Any]) -> None:
    """Validate study design independently of its numerical findings."""
    seeds = config["fitting"]["seeds"]
    budget = int(config["fitting"]["mnist_training_budget"])
    run_id = config["run_id"]
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Five unique seeds required")
    if not 0 < budget <= int(config["expected"]["training_samples"]):
        raise ValueError("Invalid training sample budget")
    if int(config["fitting"]["one_class_svm"]["max_training_samples"]) != budget:
        raise ValueError("One-Class SVM must use the common training budget")
    if not isinstance(run_id, str) or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ValueError("Unsafe run ID")
    if float(config["scoring"]["energy_temperature"]) <= 0:
        raise ValueError("Invalid energy temperature")
    if int(config["scoring"]["bootstrap_iterations"]) < 1:
        raise ValueError("Bootstrap iterations must be positive")


def load_inputs(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, np.ndarray, dict[str, Any], dict[str, str]]:
    """Load frozen assets; verify shared manifest before touching OOD test data."""
    manifest = load_manifest()
    inputs = dict(manifest["inputs"])
    inputs["svhn_file"] = manifest["svhn_file"]
    verify_shared_inputs(inputs, required_keys=tuple(inputs))
    label_path = repo_path(
        "outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/embeddings/train_labels.npy"
    )
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    train = np.load(repo_path(inputs["train_features"]), allow_pickle=False)
    validation = np.load(repo_path(inputs["validation_features"]), allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    expected = config["expected"]
    width = int(expected["embedding_dimensions"])
    if train.shape != (int(expected["training_samples"]), width):
        raise RuntimeError("MNIST training embedding shape mismatch")
    if validation.shape != (int(expected["validation_samples"]), width):
        raise RuntimeError("MNIST validation embedding shape mismatch")
    if labels.shape != (len(train),) or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError("MNIST label mismatch")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise RuntimeError("Nonfinite frozen embeddings")
    ood_config = read_settings(repo_path(config["qe_config"]))
    model, som, validation_qe, provenance = prepare(ood_config)
    if tuple(som.w.shape) != (int(expected["som_units"]), width):
        raise RuntimeError("Frozen SOM mismatch")
    if len(validation_qe) != len(validation):
        raise RuntimeError("Frozen QE validation length mismatch")
    if provenance["qe_threshold"] != float(expected["frozen_training_qe_threshold"]):
        raise RuntimeError("Frozen QE threshold mismatch")
    hashes = {key: sha256_file(repo_path(path)) for key, path in inputs.items()}
    hashes["train_labels"] = sha256_file(label_path)
    hashes["som_model"] = provenance["som_sha256"]
    return train, validation, labels, model, som, validation_qe, provenance, hashes


def get_frozen_scores(
    config: dict[str, Any], model: Any, som: Any, validation: np.ndarray, validation_qe: np.ndarray,
    svhn: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute four frozen-reference scores directly, once per unified run."""
    id_bmu, calculated_qe = score_som(som, validation)
    ood_bmu, ood_qe = score_som(som, svhn)
    if not np.allclose(calculated_qe, validation_qe, rtol=0, atol=1e-4):
        raise RuntimeError("Frozen QE validation reconstruction mismatch")
    u_values = neuron_u_matrix(som.w, som.neuron_dist)
    id_logits = reconstruct_logits(model, validation)
    ood_logits = reconstruct_logits(model, svhn)
    temperature = float(config["scoring"]["energy_temperature"])
    return {
        "qe": (validation_qe, ood_qe),
        "u_matrix": (sample_u_matrix(u_values, id_bmu), sample_u_matrix(u_values, ood_bmu)),
        "max_softmax": (max_softmax_novelty(id_logits), max_softmax_novelty(ood_logits)),
        "energy": (energy_novelty(id_logits, temperature), energy_novelty(ood_logits, temperature)),
    }


def fit_seed(
    config: dict[str, Any], seed: int, train: np.ndarray, labels: np.ndarray,
    joined: np.ndarray,
) -> tuple[dict[str, np.ndarray], str, dict[str, float]]:
    """Fit five methods on the exact same selected MNIST training rows."""
    indices = subset_indices(len(train), int(config["fitting"]["mnist_training_budget"]), seed)
    subset = train[indices]
    joined = np.asarray(joined)
    settings = {
        "seed": seed,
        "isolation_forest": config["fitting"]["isolation_forest"],
        "one_class_svm": config["fitting"]["one_class_svm"],
        "autoencoder": config["fitting"]["autoencoder"],
    }
    classical = fit_classical_baselines(subset, joined, settings)
    # Scale AE using training-only statistics; preserve its historical implementation.
    scaler = StandardScaler().fit(subset)
    ae_scores, losses = fit_autoencoder(
        scaler.transform(subset).astype(np.float32),
        scaler.transform(joined).astype(np.float32), settings,
    )
    mahalanobis = mahalanobis_novelty(subset, labels[indices], joined)
    knn = knn_novelty(subset, joined, neighbors=int(config["fitting"]["knn_neighbors"]))
    scores = {**classical, "autoencoder": ae_scores, "mahalanobis": mahalanobis, "knn": knn}
    return scores, subset_fingerprint(indices), {
        "autoencoder_initial_training_loss": float(losses[0]),
        "autoencoder_final_training_loss": float(losses[-1]),
    }


def run(config_path: Path) -> Path:
    """Execute all nine methods; refuse overwrites and preserve frozen artifacts."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    check_protocol(config)
    output = repo_path(config["output_root"]) / config["run_id"]
    if output.exists():
        raise FileExistsError(f"STOP: Existing run: {output}")
    train, validation, train_labels, model, som, val_qe, provenance, hashes = load_inputs(config)
    manifest = load_manifest()
    svhn_path = repo_path(manifest["svhn_file"])
    ood_settings = read_settings(repo_path(config["qe_config"]))
    dataset = SVHN(
        root=str(svhn_path.parent), split="test", download=False,
        transform=image_transform("svhn", tuple(ood_settings["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=int(config["scoring"]["batch_size"]), shuffle=False)
    torch.set_num_threads(1)
    svhn, _, _ = FeatureExtractor(model, "fc2").extract(loader, torch.device("cpu"))
    if svhn.shape != (int(config["expected"]["svhn_samples"]), int(config["expected"]["embedding_dimensions"])):
        raise RuntimeError("SVHN embeddings shape mismatch")
    joined = np.concatenate((validation, svhn), axis=0)
    n_id = len(validation)
    frozen = get_frozen_scores(config, model, som, validation, val_qe, svhn)
    del model, som, loader, dataset
    seeds = [int(seed) for seed in config["fitting"]["seeds"]]
    rows: list[dict[str, Any]] = []
    frozen_metrics: dict[str, Any] = {}
    for name, (id_values, ood_values) in frozen.items():
        point = evaluate(id_values, ood_values, float(config["scoring"]["target_tpr"]))
        frozen_metrics[name] = {
            **point,
            "confidence_intervals_95": stratified_bootstrap(
                id_values, ood_values, float(config["scoring"]["target_tpr"]),
                int(config["scoring"]["bootstrap_iterations"]),
                int(config["scoring"]["bootstrap_seed"]),
            ),
        }
        LOGGER.info("Frozen %s AUROC %.6f", name, point["auroc"])
    for seed in seeds:
        LOGGER.info("Fitting five common-budget methods for seed %d", seed)
        fitted, subset_hash, losses = fit_seed(config, seed, train, train_labels, joined)
        metrics = {}
        for name, values in fitted.items():
            if len(values) != len(joined):
                raise RuntimeError(f"Invalid score size: {name}")
            metrics[name] = evaluate(
                values[:n_id], values[n_id:], float(config["scoring"]["target_tpr"]),
            )
            LOGGER.info("Seed %d %s AUROC %.6f", seed, name, metrics[name]["auroc"])
        rows.append({"seed": seed, "training_subset_sha256": subset_hash, "metrics": metrics, **losses})
    if set(frozen_metrics) | set(rows[0]["metrics"]) != set(METHODS):
        raise RuntimeError("Not all nine methods evaluated")
    # Label-independent random-score null: sanity-check the evaluation protocol.
    rng_null = np.random.default_rng(int(config["scoring"]["bootstrap_seed"]))
    null_metrics = evaluate(
        rng_null.random(n_id), rng_null.random(len(svhn)),
        float(config["scoring"]["target_tpr"]),
    )
    payload = {
        "run_id": config["run_id"], "study_type": "post_hoc_exploratory",
        "data": {"id_split": "mnist_validation", "ood_split": "svhn_official_test", "mnist_training_samples": len(train), "mnist_validation_samples": n_id, "svhn_test_samples": len(svhn)},
        "methods": list(METHODS),
        "random_score_null_control": null_metrics,
        "reference_frozen_metrics": frozen_metrics,
        "fitted_methods_per_seed": rows,
        "fitted_methods_seed_summary": seed_summary(rows),
        "fitting": {
            "common_training_rows_per_seed": int(config["fitting"]["mnist_training_budget"]),
            "same_training_row_indices_within_seed": True,
            "frozen_reference_methods": list(FROZEN_NAMES),
            "freshly_fitted_methods": list(FIT_NAMES),
            "autoencoder_input_standardization": "StandardScaler fitted on selected MNIST train subset only",
            "methods_have_equal_compute_cost": False,
            "isolation_forest_internal_max_samples_per_tree": int(config["fitting"]["isolation_forest"]["max_samples"]),
        },
        "provenance": {"input_sha256": hashes, "config_sha256": sha256_file(config_path), "frozen_som": provenance},
        "metrics_protocol": "OOD positive; higher novelty; empirical FPR at target TPR on labeled evaluation, not a training-frozen deployment threshold",
        "leakage_control": "All freshly fitted methods use MNIST training subsets only; scalers fit on training only. Validation and SVHN are evaluation-only. SVHN previously examined, therefore post-hoc exploratory. MNIST final test never accessed.",
        "limitations": [
            "Frozen CNN and SOM were trained on historical complete training data, not retrained for each 10k subset.",
            "Five fitted methods share exact selected MNIST training rows, but not computational budgets or model families.",
            "Isolation Forest uses 256 samples per tree; total training pool has 10k rows.",
            "Bootstrap intervals of frozen methods condition on one frozen CNN/SOM; seed spread of fitted methods is not a bootstrap confidence interval.",
            "All methods use SVHN as external OOD; this evaluation was informed by prior SVHN observations.",
            "Frozen SOM preprocessing and model selection follow historical settings; no new selection or tuning performed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "unified_nine_method_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write(chr(10))
    LOGGER.info("Saved nine-method results to %s", output)
    return output


def main() -> None:
    """Run a new immutable post-hoc nine-method comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
