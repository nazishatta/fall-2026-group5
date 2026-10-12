"""Run records for SOM training: run-ID checks, file fingerprints, the run manifest,
the never-overwrite check, the deterministic smoke-test subset and the QE history.

Split from train_som.py (2026-10-10); train_som.py runs the training itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.v1_mnist.component.utils.figures import save_pdf  # noqa: E402


def validate_run_id(run_id: str) -> str:
    """Validate run_id string format."""
    if not run_id:
        raise ValueError("run_id must be a non-empty string.")

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError(
            "run_id may contain only letters, numbers, '.', '_', "
            "and '-', and must begin with a letter or number."
        )

    return run_id


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_run_manifest(manifest_path: Path, row: dict[str, Any]) -> None:
    """Append one run to the SOM run manifest (CSV), writing a header if new."""
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def preflight_scientific_outputs(
    run_id: str,
    logs_dir: Path,
    models_dir: Path,
) -> Tuple[Path, Path]:
    """Verify that this run ID has no model or metrics yet (runs are never overwritten)."""
    metrics_path = logs_dir / f"{run_id}_metrics.json"
    model_path = models_dir / run_id

    existing_paths = [path for path in (metrics_path, model_path) if path.exists()]

    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Scientific run_id '{run_id}' already has existing artifact(s): {existing}. "
            "Runs are never overwritten: use a new --run-id."
        )

    return metrics_path, model_path


def deterministic_subset(
    x: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Extract a reproducible random subset of samples."""
    if n_samples >= len(x):
        return x

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=n_samples, replace=False)
    return x[indices]


def save_qe_history(
    history: Any,
    run_id: str,
    logs_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, str]:
    """Save epoch-wise QE history as JSON, CSV and a PDF figure."""
    pdf_convergence_dir = figures_dir / "pdf" / "convergence"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = logs_dir / f"{run_id}_qe_history.json"
    csv_path = tables_dir / f"{run_id}_qe_history.csv"
    figure_pdf = pdf_convergence_dir / f"{run_id}_qe_curve.pdf"

    payload = history.to_dict()
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "quantization_error", "neighborhood_radius"])
        writer.writerows(
            zip(
                history.epoch,
                history.quantization_error,
                history.neighborhood_radius,
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.quantization_error, marker="o", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NNSOM-style quantization error")
    ax.set_title("SOM Quantization Error Across Training")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    save_pdf(fig, figure_pdf)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "figure_pdf": str(figure_pdf),
    }
