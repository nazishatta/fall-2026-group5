"""Is part of the QE novelty result an embedding-norm effect?

The norm-only score is already saved as the length-only control of the autoencoder
scaling diagnostic. This step measures how closely QE follows the fc2 embedding norm
and whether QE still separates SVHN test from MNIST validation among images whose
norm lies inside a training-norm band. Every reference value comes from MNIST
training only. Exploratory, because SVHN was examined in earlier runs. The held-out
MNIST test split is never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.qe_norm_dependence import (
    analyze,
)
from src.v1_mnist.component.utils.file_hash import sha256_file
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.repo_paths import resolve_inside
from src.v1_mnist.component.utils.run_ids import check_run_id

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/qe_norm_dependence.yaml"
LOGGER = get_logger(__name__)
METRICS_FILE = "qe_norm_dependence_metrics.json"


def repo_path(value: str) -> Path:
    """Resolve a configured path inside the repository."""
    return resolve_inside(ROOT, value)


def read_config(config_path: Path) -> dict[str, Any]:
    """Load the config and reject unsafe run IDs or invalid settings."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    check_run_id(config.get("run_id"))
    low, high = (float(value) for value in config["norm_matched_band"])
    if not 0 <= low < high <= 100:
        raise ValueError("Percentiles must lie between 0 and 100")
    if (
        int(config["bootstrap_iterations"]) < 2
        or not 0 < float(config["confidence"]) < 1
    ):
        raise ValueError("Invalid bootstrap settings")
    return config


def load_scores(config: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen embeddings, embed SVHN with the frozen CNN and score QE."""
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
    from src.v1_mnist.component.utils.shared_provenance import (
        load_manifest,
        verify_shared_inputs,
    )

    expected = config["expected"]
    manifest = load_manifest()
    shared = dict(manifest["inputs"])
    shared["svhn_file"] = manifest["svhn_file"]
    verify_shared_inputs(shared, required_keys=tuple(shared))
    settings = read_settings(repo_path(config["qe_config"]))
    model, som, validation_qe, provenance = prepare(settings)
    train = np.load(repo_path(shared["train_features"]), allow_pickle=False)
    validation = np.load(repo_path(shared["validation_features"]), allow_pickle=False)
    loader = svhn_test_loader(
        repo_path(shared["svhn_file"]),
        tuple(settings["image_size"]),
        int(config["batch_size"]),
    )
    torch.set_num_threads(1)
    svhn, _, _ = FeatureExtractor(model, "fc2").extract(loader, torch.device("cpu"))
    _bmu, svhn_qe = score_som(som, svhn)
    dim = int(expected["feature_dim"])
    if (
        train.shape != (int(expected["train_samples"]), dim)
        or validation.shape != (int(expected["validation_samples"]), dim)
        or svhn.shape != (int(expected["svhn_samples"]), dim)
        or len(validation_qe) != len(validation)
    ):
        raise RuntimeError("STOP: Unexpected embedding or QE shapes")
    if float(provenance["qe_threshold"]) != float(
        expected["frozen_training_qe_threshold"]
    ):
        raise RuntimeError("STOP: Frozen QE threshold mismatch")
    return {
        "train": train,
        "validation": validation,
        "svhn": np.asarray(svhn),
        "validation_qe": np.asarray(validation_qe, dtype=np.float64),
        "svhn_qe": np.asarray(svhn_qe, dtype=np.float64),
        "qe_threshold": float(provenance["qe_threshold"]),
        "provenance": provenance,
        "hashes": {key: sha256_file(repo_path(value)) for key, value in shared.items()},
    }


def check_qe_reference(analysis: dict[str, Any], config: dict[str, Any]) -> None:
    """Stop if QE no longer reproduces the recorded headline AUROC."""
    expected = config["expected"]
    gap = abs(
        float(analysis["qe_reference"]["auroc"]) - float(expected["qe_reference_auroc"])
    )
    if gap > float(expected["qe_auroc_tolerance"]):
        raise RuntimeError(
            f"STOP: QE AUROC differs from the recorded value by {gap:.3g}"
        )


def write_run(
    destination: Path,
    config: dict[str, Any],
    config_sha256: str,
    data: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Write the metrics JSON and one summary table into a brand-new folder."""
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "tables").mkdir()
    band = analysis["norm_matched_qe"]
    rows = [
        (
            "all images",
            analysis["qe_reference"],
            len(data["validation"]),
            len(data["svhn"]),
        )
    ]
    rows.append(
        (
            f"norm inside training p{band['band_percentiles'][0]:g}-p{band['band_percentiles'][1]:g}",
            band["qe"],
            band["id_count"],
            band["ood_count"],
        )
    )
    with (destination / "tables" / "qe_auroc_by_norm_subset.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "subset",
                "mnist_validation_images",
                "svhn_images",
                "qe_auroc",
                "auroc_ci_low",
                "auroc_ci_high",
            ]
        )
        for name, metrics, n_id, n_ood in rows:
            if metrics is None:
                writer.writerow([name, n_id, n_ood, "", "", ""])
            else:
                writer.writerow(
                    [name, n_id, n_ood, metrics["auroc"], *metrics["auroc_ci"]]
                )
    result = {
        "run_id": config["run_id"],
        "analysis_status": "post_hoc_exploratory",
        "id_set": "MNIST validation",
        "ood_set": "SVHN test",
        "feature_layer": "fc2",
        "seed": int(config["seed"]),
        **analysis,
        "provenance": {
            "config_sha256": config_sha256,
            "input_sha256": data["hashes"],
            "frozen_som": data["provenance"],
        },
        "leakage_control": (
            "The QE threshold and the norm band come from MNIST training "
            "embeddings only. MNIST validation and SVHN test are scored only. Nothing is "
            "fitted or tuned. The held-out MNIST test split is never opened."
        ),
        "limitations": [
            "SVHN was examined in earlier runs, so this check is exploratory.",
            "Fashion-MNIST and other OOD sets are not included.",
            "Correlation with the norm does not show that the norm causes the QE ranking.",
            "The norm band keeps only SVHN images with MNIST-like norms, which may be unusual SVHN images.",
        ],
    }
    with (destination / METRICS_FILE).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def run(config_path: Path) -> Path:
    """Run the QE-norm check into a new folder; never overwrite an existing run."""
    config = read_config(config_path)
    destination = repo_path(config["output_root"]) / config["run_id"]
    if destination.exists():
        raise FileExistsError(f"STOP: Existing run: {destination}")
    data = load_scores(config)
    analysis = analyze(
        data["train"],
        data["validation"],
        data["svhn"],
        data["validation_qe"],
        data["svhn_qe"],
        qe_threshold=data["qe_threshold"],
        settings=config,
    )
    check_qe_reference(analysis, config)
    write_run(destination, config, sha256_file(config_path), data, analysis)
    LOGGER.info("Saved QE-norm dependence check to %s", destination)
    return destination


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
