"""Reproduce the MNIST validation QE versus U-Matrix visualization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import spearmanr

from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]


def repository_path(value: str) -> Path:
    """Resolve a configured path within the repository."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path outside repository")
    return path


def run(config_path: Path) -> Path:
    """Verify frozen validation scores and create a new PDF result."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_id = str(config["run_id"])
    if not run_id or not all(
        char.isalnum() or char in "_-" for char in run_id
    ):
        raise ValueError("Unsafe run ID")

    destination = repository_path(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"STOP: Run already exists: {destination}")

    folder = repository_path(config["scores_directory"])
    arrays = {}
    hashes = {}

    for name in ("qe", "u_matrix"):
        source = folder / f"{name}_mnist_val_scores.npy"
        actual = sha256_file(source)
        if actual != config["score_sha256"][name]:
            raise RuntimeError(f"SHA-256 mismatch: {source}")

        values = np.load(source, allow_pickle=False)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores: {source}")
        arrays[name] = values
        hashes[name] = actual

    qe = arrays["qe"]
    um = arrays["u_matrix"]
    summary_path = repository_path(config["source_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if qe.shape != um.shape or len(qe) != int(summary["validation_samples"]):
        raise RuntimeError("Validation sample counts mismatch")

    qe_mean = float(np.mean(qe))
    um_mean = float(np.mean(um))
    correlation = float(spearmanr(qe, um).statistic)

    mean_tol = float(config["mean_tolerance"])
    corr_tol = float(config["correlation_tolerance"])

    if not np.isclose(qe_mean, summary["qe_mean"], atol=mean_tol, rtol=0):
        raise RuntimeError("Historical QE mean mismatch")
    if not np.isclose(um_mean, summary["u_matrix_mean"], atol=mean_tol, rtol=0):
        raise RuntimeError("Historical U-Matrix mean mismatch")
    if not np.isclose(
        correlation, summary["spearman_correlation"],
        atol=corr_tol, rtol=0
    ):
        raise RuntimeError("Historical correlation mismatch")

    figure_name = str(config["figure_name"])
    if Path(figure_name).name != figure_name or not figure_name.endswith(".pdf"):
        raise ValueError("Invalid PDF filename")

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    ax.scatter(
        qe, um, s=5, alpha=0.28, linewidths=0, rasterized=True
    )
    ax.set_xlabel("Quantization error (distance to BMU)")
    ax.set_ylabel("U-Matrix novelty score")
    ax.set_title("MNIST validation: QE versus U-Matrix")
    ax.text(
        0.03, 0.97,
        f"n = {len(qe):,} | Spearman rho = {correlation:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()

    destination.mkdir(parents=True, exist_ok=False)
    figure_path = save_pdf(fig, destination / figure_name)

    metadata = {
        "run_id": run_id,
        "figure": figure_name,
        "sample_count": len(qe),
        "qe_mean": qe_mean,
        "u_matrix_mean": um_mean,
        "spearman_correlation": correlation,
        "input_sha256": hashes,
        "source_summary_sha256": sha256_file(summary_path),
        "config_sha256": sha256_file(config_path),
        "figure_sha256": sha256_file(figure_path),
        "leakage_control": (
            "Frozen MNIST validation scores only; no model fitting, "
            "threshold tuning, SVHN use, or final MNIST test access."
        ),
        "reproduction_note": (
            "Reproduces historical summary statistics and score data, "
            "not necessarily the original figure's visual layout."
        ),
    }

    with (destination / "figure_metrics.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write(chr(10))

    LOGGER.info("Saved verified PDF: %s", figure_path)
    return figure_path


def main() -> None:
    """Run the configured figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.config)


if __name__ == "__main__":
    main()
