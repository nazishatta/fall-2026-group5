"""Frozen inputs of the baseline check: validation and training embeddings, and the SOM's cells.

Only the training and validation splits are ever loaded (the project's loader refuses anything else).
The SOM model and the embedding files are checked against their recorded SHA-256 hashes first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
    BaselineSettings,
)
from src.v1_mnist.component.utils.run_settings import (
    AnalysisSettings,
    cluster_output_dir,
    verify_inputs,
    verify_selected_model,
)
from src.v1_mnist.component.utils.som_io import load_split, sha256_file


@dataclass(frozen=True)
class ValidationInputs:
    """Arrays the check works on, plus the hashes of the files they came from."""

    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_confidence: np.ndarray
    val_errors: np.ndarray
    val_cells: np.ndarray
    sha256: dict[str, str]


def load_inputs(settings: BaselineSettings, analysis: AnalysisSettings, repo_root: Path) -> ValidationInputs:
    """Verify the SOM model and the embeddings, then load them (validation and training only)."""
    model_hash = verify_selected_model(analysis)
    train = load_split("train", analysis.embeddings_dir, analysis.feature_dim)
    val = load_split("val", analysis.embeddings_dir, analysis.feature_dim)

    hashes = {
        "train_features": sha256_file(analysis.embeddings_dir / "train_features.npy"),
        "val_features": sha256_file(analysis.embeddings_dir / "val_features.npy"),
    }
    verify_inputs(analysis, hashes)

    cells_path = cluster_output_dir(settings.som_config, repo_root) / "assignments" / "val_bmu.npy"
    cells = np.load(cells_path).astype(int)
    if cells.shape != val["labels"].shape or cells.min() < 0 or cells.max() >= settings.num_cells:
        raise ValueError(f"{cells_path}: expected one cell index in [0, {settings.num_cells}) per validation image.")
    hashes["som_model"] = model_hash
    hashes["val_cells_som"] = sha256_file(cells_path)

    return ValidationInputs(
        train_features=train["features"],
        train_labels=train["labels"],
        val_features=val["features"],
        val_labels=val["labels"],
        val_confidence=val["confidence"].astype(float),
        val_errors=~val["correct"].astype(bool),
        val_cells=cells,
        sha256=hashes,
    )
