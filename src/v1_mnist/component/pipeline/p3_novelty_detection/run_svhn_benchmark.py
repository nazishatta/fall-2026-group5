"""Independent SVHN comparison using frozen MNIST models and baseline settings.

Does not modify existing source, configs, models or QE experiment results.
Run from the isolated project root, with PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
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
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    analyze_misses,
    evaluate_scores,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
    image_transform,
)
from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
    neuron_u_matrix,
    sample_u_matrix,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.run_qe_ood import (
    prepare,
    read_settings,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import reconstruct_assignments, sha256_file

LOGGER = get_logger("phase3_svhn_comparison")
ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/novelty_benchmark.yaml"


def _to_mapping(value: Any) -> Any:
    """Convert nested shared configuration namespaces to dictionaries."""
    if isinstance(value, SimpleNamespace):
        return {key: _to_mapping(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_mapping(item) for item in value]
    return value


def score_som(som: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample BMUs and their original NNSOM QE distances."""
    clusters, distances, _, _ = som.cluster_data(features)
    bmu, qe = reconstruct_assignments(clusters, distances, len(features))
    return bmu, qe


def score_method(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    repeats: int,
    seed: int,
    target_tpr: float,
) -> dict[str, object]:
    """Evaluate fixed scores with stratified confidence intervals."""
    return evaluate_scores(
        id_scores, ood_scores, bootstrap_repeats=repeats, seed=seed,
        target_tpr=target_tpr
    )


def main() -> None:
    """Run the preregistered five-method MNIST vs SVHN comparison."""
    benchmark = _to_mapping(load_config(CONFIG_PATH))
    run_id = benchmark["run_id"]
    if not isinstance(run_id, str) or not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("Invalid benchmark run ID")
    output = ROOT / benchmark["output_root"] / run_id
    qe_config_path = ROOT / benchmark["qe_evaluation_config"]
    svhn_file = ROOT / benchmark["svhn_file"]
    qe_result = ROOT / benchmark["qe_reference_metrics"]

    if output.exists():
        raise FileExistsError(f"STOP: Existing run must not be overwritten: {output}")
    for path in (qe_config_path, svhn_file, qe_result):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input missing: {path}")

    ood_config = read_settings(qe_config_path)
    if benchmark["dataset"] != "mnist" or int(benchmark["seed"]) != int(ood_config["bootstrap"]["seed"]):
        raise RuntimeError("Benchmark configuration differs from recorded protocol")
    if (tuple(ood_config["image_size"]) != (28, 28)
            or ood_config["feature_layer"] != "fc2"):
        raise RuntimeError("CNN preprocessing or feature layer protocol changed")
    train_path = ROOT / benchmark["embeddings"]["train"]
    val_path = ROOT / benchmark["embeddings"]["validation"]
    expected_hashes = benchmark["expected_sha256"]
    hashes = {"train": sha256_file(train_path), "validation": sha256_file(val_path)}
    if hashes != expected_hashes:
        raise RuntimeError("STOP: CNN embeddings do not match original fingerprints")
    qe_saved = json.loads(qe_result.read_text(encoding="utf-8"))
    if qe_saved.get("metrics", {}).get("ood_count") != benchmark["expected"]["svhn_count"]:
        raise RuntimeError("STOP: QE reference run does not have 26,032 SVHN samples")

    # Verify checkpoint, SOM and frozen QE calibration via existing preflight.
    model, som, frozen_id_qe, provenance = prepare(ood_config)
    if provenance["qe_threshold"] != benchmark["expected"]["qe_threshold"]:
        raise RuntimeError("STOP: Frozen QE threshold unexpectedly changed")
    if len(frozen_id_qe) != benchmark["expected"]["mnist_validation_count"] or tuple(som.w.shape) != (benchmark["expected"]["som_neurons"], benchmark["expected"]["feature_dim"]):
        raise RuntimeError("STOP: MNIST reference or SOM dimensions changed")
    train = np.load(train_path, allow_pickle=False)
    validation = np.load(val_path, allow_pickle=False)
    if len(train) != benchmark["expected"]["mnist_train_count"] or len(validation) != len(frozen_id_qe):
        raise RuntimeError("STOP: Original MNIST split sizes changed")

    # SVHN is used for scoring ONLY; no fitting, threshold calibration or selection.
    dataset = SVHN(
        root=str(svhn_file.parent), split="test", download=False,
        transform=image_transform("svhn", (28, 28)),
    )
    loader = DataLoader(dataset, batch_size=int(ood_config["batch_size"]), shuffle=False)
    torch.set_num_threads(1)
    features, labels, _ = FeatureExtractor(model, "fc2").extract(loader, torch.device("cpu"))
    del model, loader, dataset
    if features.shape != (benchmark["expected"]["svhn_count"], benchmark["expected"]["feature_dim"]):
        raise RuntimeError(f"Unexpected SVHN embeddings: {features.shape}")
    if len(labels) != len(features):
        raise RuntimeError("SVHN label order mismatch")
    LOGGER.info("Frozen SVHN embeddings: %s", features.shape)

    id_bmu, id_qe = score_som(som, validation)
    ood_bmu, ood_qe = score_som(som, features)
    qe_error = float(np.max(np.abs(id_qe - frozen_id_qe)))
    if qe_error > benchmark["tolerances"]["qe_reconstruction"]:
        raise RuntimeError(f"STOP: QE validation reconstruction mismatch: {qe_error}")

    som_neuron_scores = neuron_u_matrix(som.w, som.neuron_dist)
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "qe": (frozen_id_qe, ood_qe),
        "u_matrix": (
            sample_u_matrix(som_neuron_scores, id_bmu),
            sample_u_matrix(som_neuron_scores, ood_bmu),
        ),
    }

    # Both ID and OOD passed as scoring samples; models fit only on 49k MNIST train.
    together = np.concatenate([validation, features], axis=0)
    cut = len(validation)
    classical = fit_classical_baselines(train, together, benchmark)
    for name, values in classical.items():
        scores[name] = (values[:cut], values[cut:])
    ae_scores, losses = fit_autoencoder(train, together, benchmark)
    scores["autoencoder"] = (ae_scores[:cut], ae_scores[cut:])

    # Cross-check QE results against the independent, already frozen evaluation.
    qe_eval = score_method(*scores["qe"], repeats=0, target_tpr=float(ood_config["target_tpr"]), seed=int(benchmark["seed"]))
    for key, old_key in (("auroc", "auroc"), ("aupr", "aupr_ood")):
        difference = abs(float(qe_eval[key]) - float(qe_saved["metrics"][old_key]))
        if difference > benchmark["tolerances"]["metric_agreement"]:
            raise RuntimeError(f"STOP: QE {key} disagrees with frozen run: {difference}")

    iterations = int(ood_config["bootstrap"]["iterations"])
    seed = int(ood_config["bootstrap"]["seed"])
    results = {
        name: score_method(id_s, ood_s, repeats=iterations, seed=seed,
                           target_tpr=float(ood_config["target_tpr"]))
        for name, (id_s, ood_s) in scores.items()
    }
    results["qe"]["frozen_threshold_misses"] = analyze_misses(
        scores["qe"][1], threshold=provenance["qe_threshold"],
        sample_ids=np.arange(len(features)),
        max_examples=int(ood_config["max_missed_examples"]),
    )
    result: dict[str, object] = {
        "run_id": run_id,
        "id_dataset": "mnist", "id_split": "validation",
        "ood_dataset": "svhn", "ood_split": "test",
        "sample_counts": {"mnist_train": len(train), "mnist_validation": cut,
                          "svhn_test": len(features)},
        "methods": results,
        "qe_validation_max_absolute_difference": qe_error,
        "autoencoder_initial_training_loss": losses[0],
        "autoencoder_final_training_loss": losses[-1],
        "settings": benchmark,
        "provenance": {**provenance, "config_sha256": sha256_file(CONFIG_PATH),
                       "svhn_mat_sha256": sha256_file(svhn_file),
                       "train_embeddings_sha256": hashes["train"],
                       "validation_embeddings_sha256": hashes["validation"],
                       "previous_qe_metrics_sha256": sha256_file(qe_result)},
        "leakage_control": (
            "CNN and SOM remain frozen. Baselines fit only on MNIST train. "
            "MNIST validation and SVHN test used only for scoring; OOD scores "
            "not used for tuning. FPR@95%TPR is a ROC operating point measured "
            "on labeled evaluation samples, not a deployable frozen threshold."
        ),
        "limitations": [
            ("One-Class SVM fitted on 10,000 MNIST train examples; other "
             "baselines used 49,000, so fitting budgets differ."),
            "This is a single-seed, single-OOD-dataset comparison.",
            "Sample-level miss IDs alone do not establish causes of failures.",
            ("U-Matrix scores indicate local SOM prototype separation, not "
             "direct sample-to-BMU distance."),
        ],
    }
    # Output directory is created only after all scorers/metrics succeed.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison_metrics.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    # Save sample order and scores for reproducible miss analysis; outputs/ is ignored.
    with (output / "svhn_labels.npy").open("xb") as file:
        np.save(file, labels, allow_pickle=False)
    for name, (id_s, ood_s) in scores.items():
        for split, values in (("mnist_val", id_s), ("svhn_test", ood_s)):
            with (output / f"{name}_{split}_scores.npy").open("xb") as file:
                np.save(file, np.asarray(values, dtype=np.float64), allow_pickle=False)
    LOGGER.info("SAVED %s", output / "comparison_metrics.json")
    for method, metrics in results.items():
        LOGGER.info("%s: AUROC %.6f, AUPR %.6f, FPR@95TPR %.6f", method,
                    metrics["auroc"], metrics["aupr"], metrics["fpr_at_95_tpr"])


if __name__ == "__main__":
    main()
