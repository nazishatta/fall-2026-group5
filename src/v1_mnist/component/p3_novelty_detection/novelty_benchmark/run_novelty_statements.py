"""Score all six novelty scores on the same images in one run and judge statements O1 to O4.

For a dataset whose Phase 3 protocol states O1 to O4 in advance (Fashion-MNIST:
src/v1_fashion_mnist/docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md, section 3). MNIST has
no config for this step: its protocol did not state these statements before its results were known.

Usage (from the code root):
  python src/v1_mnist/component/p3_novelty_detection/novelty_benchmark/run_novelty_statements.py --config CONFIG

The frozen CNN and SOM are checked exactly as in the other Phase 3 steps (shared provenance file,
input SHA-256, the frozen QE threshold, the network's own predictions). The six scores are computed
with the MNIST Phase 3 functions on the validation split and the outside test split, in file order;
nothing is fitted on the outside images and the dataset's test split is never opened. Results go to
a new folder: never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.cnn_score_inference import (
    reconstruct_logits,
    verify_predictions,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.knn_baseline import (
    knn_novelty,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.novelty_statements import (
    SCORE_IDS,
    SCORES,
    StatementSettings,
    judge,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.required_baselines import (
    energy_novelty,
    mahalanobis_novelty,
    max_softmax_novelty,
)
from src.v1_mnist.component.utils.in_distribution import in_distribution
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.repo_paths import resolve_inside
from src.v1_mnist.component.utils.run_ids import check_run_id
from src.v1_mnist.component.utils.shared_provenance import verify_shared_inputs
from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("v1_mnist.p3.novelty_statements")
INPUTS = ("checkpoint", "train_features", "train_labels", "validation_features",
          "validation_predictions", "validation_confidence", "svhn_file")


def read_config(config_path: Path) -> dict[str, Any]:
    """The step's config; refuses a bad run ID or missing sections."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    check_run_id(config.get("run_id"))
    for section in ("inputs", "expected_sha256", "expected", "inference", "evaluation", "qe_config"):
        if section not in config:
            raise KeyError(f"config has no {section!r}")
    return config


def statement_settings(evaluation: dict[str, Any]) -> StatementSettings:
    """The protocol's bootstrap, control and margin settings from the config's evaluation section."""
    defaults = StatementSettings()
    values = {key: type(getattr(defaults, key))(evaluation[key])
              for key in StatementSettings.__dataclass_fields__ if key in evaluation}
    return StatementSettings(**values)


def load_inputs(config: dict[str, Any], config_path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Check the inputs against the dataset's provenance file and the config; load the arrays."""
    verify_shared_inputs({key: config["inputs"][key] for key in ("checkpoint", "train_features",
                                                                 "validation_features", "svhn_file")},
                         required_keys=("checkpoint", "train_features", "validation_features", "svhn_file"),
                         config_path=config_path)
    paths = {key: resolve_inside(ROOT, config["inputs"][key]) for key in INPUTS}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    for key, expected in config["expected_sha256"].items():
        if hashes[key] != str(expected):
            raise RuntimeError(f"STOP: frozen SHA-256 mismatch: {key}")
    arrays = {key: np.load(path, allow_pickle=False) for key, path in paths.items() if key.endswith(
        ("features", "labels", "predictions", "confidence"))}
    expected = config["expected"]
    width = int(expected["feature_dim"])
    if arrays["train_features"].shape != (int(expected["training_samples"]), width):
        raise RuntimeError("Unexpected training embeddings")
    if arrays["validation_features"].shape != (int(expected["validation_samples"]), width):
        raise RuntimeError("Unexpected validation embeddings")
    if arrays["train_labels"].shape != (len(arrays["train_features"]),):
        raise RuntimeError("Training labels are not aligned")
    return {**arrays, "svhn_file": paths["svhn_file"], "checkpoint": paths["checkpoint"]}, hashes


def compute_scores(config: dict[str, Any], data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The six scores on validation and SVHN test, in file order (needs torch, nnsom, the SVHN file)."""
    import torch

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.run_svhn_benchmark import (
        score_som,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        svhn_test_loader,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.run_qe_ood import (
        prepare,
        read_settings,
    )
    from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
        neuron_u_matrix,
        sample_u_matrix,
    )

    inference, expected = config["inference"], config["expected"]
    ood_config = read_settings(resolve_inside(ROOT, config["qe_config"]))
    model, som, frozen_val_qe, provenance = prepare(ood_config)
    torch.set_num_threads(int(inference["thread_count"]))
    tolerance = float(inference["confidence_tolerance"])
    validation = data["validation_features"]
    val_logits = reconstruct_logits(model, validation)
    verify_predictions(val_logits, data["validation_predictions"], data["validation_confidence"], tolerance)

    loader = svhn_test_loader(data["svhn_file"], tuple(int(x) for x in inference["image_size"]),
                              int(inference["batch_size"]), channels=int(ood_config["channels"]))
    svhn, _, svhn_preds, _, svhn_conf, _ = FeatureExtractor(model, str(inference["feature_layer"])).extract(
        loader, torch.device("cpu"), return_metadata=True)
    if svhn.shape != (int(expected["svhn_samples"]), int(expected["feature_dim"])):
        raise RuntimeError(f"Unexpected SVHN embeddings: {svhn.shape}")
    svhn_logits = reconstruct_logits(model, svhn)
    verify_predictions(svhn_logits, svhn_preds, svhn_conf, tolerance)

    val_bmu, val_qe = score_som(som, validation)
    svhn_bmu, svhn_qe = score_som(som, svhn)
    gap = float(np.max(np.abs(val_qe - frozen_val_qe)))
    if gap > float(config.get("tolerances", {}).get("qe_reconstruction", 1e-4)):
        raise RuntimeError(f"STOP: validation QE differs from the frozen QE by {gap}")
    neuron_scores = neuron_u_matrix(som.w, som.neuron_dist)
    evaluation = config["evaluation"]
    train, joined = data["train_features"], np.concatenate([validation, svhn], axis=0)
    mahalanobis = mahalanobis_novelty(train, data["train_labels"], joined)
    neighbors = int(evaluation["knn_neighbors"])
    scores = {
        "qe": (np.asarray(frozen_val_qe, dtype=np.float64), np.asarray(svhn_qe, dtype=np.float64)),
        "u_matrix": (sample_u_matrix(neuron_scores, val_bmu), sample_u_matrix(neuron_scores, svhn_bmu)),
        "max_softmax": (max_softmax_novelty(val_logits), max_softmax_novelty(svhn_logits)),
        "energy": (energy_novelty(val_logits, float(evaluation["energy_temperature"])),
                   energy_novelty(svhn_logits, float(evaluation["energy_temperature"]))),
        "mahalanobis": (mahalanobis[: len(validation)], mahalanobis[len(validation):]),
        "knn": (knn_novelty(train, validation, neighbors=neighbors), knn_novelty(train, svhn, neighbors=neighbors)),
    }
    return scores, {**provenance, "qe_reconstruction_max_abs_difference": gap}


def write_run(destination: Path, config: dict[str, Any], config_path: Path, scores: dict[str, Any],
              judged: dict[str, Any], hashes: dict[str, str], provenance: dict[str, Any]) -> dict[str, Any]:
    """The JSON, two small tables and the score arrays, into a brand-new folder."""
    dataset = in_distribution(config_path)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("tables", "scores"):
        (destination / name).mkdir()
    for name in SCORES:
        for split, values in zip((dataset.val_tag, "svhn_test"), scores[name], strict=True):
            with (destination / "scores" / f"{name}_{split}_scores.npy").open("xb") as stream:
                np.save(stream, np.asarray(values, dtype=np.float64), allow_pickle=False)
    with (destination / "tables" / "scores.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["score_id", "score", "auroc", "auroc_ci_low", "auroc_ci_high", "aupr_out",
                         "aupr_chance_level", "fpr_at_95_tpr"])
        for name in SCORES:
            row = judged["scores"][name]
            writer.writerow([SCORE_IDS[name], name, row["auroc"], *row["confidence_intervals_95"]["auroc"],
                             row["aupr"], row["aupr_chance_level"], row["fpr_at_95_tpr"]])
    with (destination / "tables" / "paired_against_qe.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "qe_auroc", "other_auroc", "qe_minus_other", "ci_low", "ci_high"])
        for row in judged["paired_against_qe"].values():
            writer.writerow([row["comparison"], row["qe_auroc"], row["other_auroc"], row["qe_minus_other_auroc"],
                             *row["difference_ci_95"]])
    result = {
        "run_id": config["run_id"],
        "id_dataset": dataset.name, "id_split": "validation", "ood_dataset": "svhn", "ood_split": "test",
        **judged,
        "provenance": {"config_sha256": sha256_file(config_path), "input_sha256": hashes, "frozen_som": provenance},
        "test_split_read": False,
        "leakage_control": dataset.wording(
            "Frozen CNN and SOM. Mahalanobis and nearest-neighbour references fitted on MNIST training "
            "embeddings only. MNIST validation and SVHN test used for scoring only; nothing is tuned or "
            "chosen on SVHN. The MNIST test split is never opened."),
        "limitations": [
            "One network and one map, so no claim of 'better' is made (the proposal requires 5 seeds for that).",
            "One outside set (SVHN): the statements speak to far novelty only.",
            "Paired comparisons assume the identical, unshuffled image order of every score, which this run ensures.",
        ],
    }
    with (destination / "novelty_statements.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def run(config_path: Path) -> dict[str, Any]:
    """Score, judge and save under a new run ID; refuses an existing one."""
    config = read_config(config_path)
    destination = resolve_inside(ROOT, config["output_root"]) / str(config["run_id"])
    if destination.exists():
        raise FileExistsError(f"STOP: existing run must not be overwritten: {destination}")
    data, hashes = load_inputs(config, config_path)
    scores, provenance = compute_scores(config, data)
    judged = judge(scores, statement_settings(config["evaluation"]))
    result = write_run(destination, config, config_path, scores, judged, hashes, provenance)
    for key, item in result["statements"].items():
        verdicts = item.get("verdict") or {n: v["verdict"] for n, v in item["per_score"].items()}
        LOGGER.info("%s: %s", key, verdicts)
    LOGGER.info("Saved %s", destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
