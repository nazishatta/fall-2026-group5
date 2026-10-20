"""Reconstruct frozen QE missed-SVHN examples from historical IDs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]


def repository_path(value: str) -> Path:
    """Resolve a configured repository-contained path."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path escapes repository")
    return path


def run(config_path: Path) -> Path:
    """Verify original examples and save a provenance-tracked PDF."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_id = str(config["run_id"])
    if not run_id or not all(
        character.isalnum() or character in "_-" for character in run_id
    ):
        raise ValueError("Invalid run ID")

    destination = repository_path(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run: {destination}")

    historical = repository_path(config["historical_run"])
    scores_path = historical / "score_arrays/qe_svhn_test_scores.npy"
    labels_path = historical / "score_arrays/svhn_labels.npy"
    comparison_path = historical / "comparison_metrics.json"
    # The verified SVHN file may be a symlink to external data storage.
    # Restrict the configured location and retain the SHA-256 check below.
    svhn_relative = Path(str(config["svhn_file"]))
    if (
        svhn_relative.is_absolute()
        or ".." in svhn_relative.parts
        or svhn_relative.parts != ("data", "ood_svhn", "test_32x32.mat")
    ):
        raise ValueError("Unexpected SVHN dataset path")
    svhn_path = ROOT / svhn_relative

    if sha256_file(scores_path) != config["qe_scores_sha256"]:
        raise RuntimeError("Frozen QE score fingerprint mismatch")
    if sha256_file(svhn_path) != config["svhn_sha256"]:
        raise RuntimeError("SVHN source fingerprint mismatch")

    scores = np.load(scores_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    historical_metrics = json.loads(
        comparison_path.read_text(encoding="utf-8")
    )

    record = historical_metrics["methods"]["qe"]["frozen_threshold_misses"]
    threshold = float(record["threshold"])
    examples = record["lowest_scoring_misses"]
    expected_count = int(config["expected_samples"])
    count = int(config["example_count"])

    if (
        scores.shape != (expected_count,)
        or labels.shape != scores.shape
        or not np.isfinite(scores).all()
        or len(examples) != count
    ):
        raise RuntimeError("Invalid or incomplete frozen records")

    missed = np.flatnonzero(scores <= threshold)
    if (
        len(missed) != int(config["expected_misses"])
        or len(missed) != int(record["missed"])
    ):
        raise RuntimeError("Frozen missed count mismatch")

    expected_indices = missed[
        np.argsort(scores[missed], kind="stable")[:count]
    ]
    recorded_indices = np.array(
        [int(example["sample_id"]) for example in examples],
        dtype=int,
    )

    if not np.array_equal(expected_indices, recorded_indices):
        raise RuntimeError("Historical missed-example IDs disagree")

    for index, example in zip(recorded_indices, examples, strict=True):
        if not np.isclose(
            scores[index],
            float(example["score"]),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError("Historical QE score mismatch")

    dataset = loadmat(
        svhn_path, variable_names=["X", "y"]
    )
    images = dataset["X"]
    original_labels = dataset["y"].reshape(-1)
    normalized_labels = np.where(
        original_labels == 10, 0, original_labels
    )

    if images.shape != (32, 32, 3, expected_count):
        raise RuntimeError("Unexpected SVHN image shape")
    if not np.array_equal(labels, normalized_labels):
        raise RuntimeError("SVHN label ordering mismatch")

    columns = 5
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(12, 2.8 * rows)
    )
    flat_axes = np.asarray(axes).reshape(-1)

    for axis, index in zip(flat_axes, recorded_indices, strict=True):
        # Match torchvision SVHN orientation for the original .mat array.
        image = images[:, :, :, index].transpose(1, 0, 2)
        axis.imshow(image)
        axis.set_title(
            f"ID {index} | digit {int(labels[index])}"
            f"\nQE {scores[index]:.3f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(
        "20 lowest-QE missed SVHN examples",
        fontsize=14,
    )
    fig.tight_layout()

    name = str(config["figure_name"])
    if Path(name).name != name or not name.endswith(".pdf"):
        raise ValueError("Invalid figure filename")

    destination.mkdir(parents=True, exist_ok=False)
    pdf = save_pdf(fig, destination / name)

    result = {
        "run_id": run_id,
        "figure_name": name,
        "figure_sha256": sha256_file(pdf),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            "svhn_source": sha256_file(svhn_path),
            "qe_scores": sha256_file(scores_path),
            "saved_labels": sha256_file(labels_path),
            "historical_comparison": sha256_file(comparison_path),
        },
        "threshold": threshold,
        "svhn_sample_count": expected_count,
        "missed_count": len(missed),
        "detection_rate": float(record["detection_rate"]),
        "sample_indices": recorded_indices.tolist(),
        "sample_qe_scores": scores[recorded_indices].tolist(),
        "sample_labels": labels[recorded_indices].tolist(),
        "leakage_control": (
            "Frozen historical SVHN evaluation only; no refitting, "
            "threshold selection, or final MNIST test access."
        ),
        "reproduction_note": (
            "Uses historical missed-image IDs and scores with "
            "original SVHN images; visual layout may differ."
        ),
    }

    with (destination / "figure_metrics.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write(chr(10))

    LOGGER.info("Saved missed-SVHN PDF: %s", pdf)
    return pdf


def main() -> None:
    """Run the configured figure reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    run(args.config)


if __name__ == "__main__":
    main()
