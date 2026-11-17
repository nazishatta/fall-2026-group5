"""Frozen inputs of the hard example mining step: training images as the SOM sees them.

Only training images are used. The SOM model and the embedding files are checked against their
recorded SHA-256 hashes first, and the Phase 2 assignment table is checked against the saved
training sample IDs and labels, so the table cannot belong to another split or another order.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
    PoolSettings,
)
from src.v1_mnist.component.utils.run_provenance import hash_files
from src.v1_mnist.component.utils.run_settings import (
    AnalysisSettings,
    cluster_output_dir,
    error_output_dir,
    verify_inputs,
    verify_selected_model,
)


@dataclass(frozen=True)
class TrainInputs:
    """Per-training-image arrays (one entry per image, in saved order) and per-neuron tables."""

    sample_id: np.ndarray
    label: np.ndarray
    correct: np.ndarray
    confidence: np.ndarray
    bmu: np.ndarray
    bmu_distance: np.ndarray
    purity_by_neuron: np.ndarray
    hotspot_neurons: np.ndarray
    sha256: dict[str, str]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 2 file: {path}. Run the Phase 2 steps first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _purity_table(rows: list[dict[str, str]], num_neurons: int) -> np.ndarray:
    """Training purity per neuron; neurons that hold no training image stay NaN."""
    purity = np.full(num_neurons, np.nan)
    for row in rows:
        if int(row["support"]) > 0:
            purity[int(row["neuron_id"])] = float(row["purity"])
    return purity


def load_train_inputs(
    settings: PoolSettings, analysis: AnalysisSettings, repo_root: Path
) -> TrainInputs:
    """Verify the SOM model and embeddings, then load the training table and the hotspot list."""
    model_hash = verify_selected_model(analysis)
    emb = analysis.embeddings_dir
    files = {
        "train_features": emb / "train_features.npy",
        "val_features": emb / "val_features.npy",
        "train_sample_ids": emb / "train_sample_ids.npy",
        "train_labels": emb / "train_labels.npy",
    }
    cluster_dir = cluster_output_dir(settings.som_config, repo_root)
    files["train_bmu_assignments"] = cluster_dir / "assignments" / "train_bmu_assignments.csv"
    files["train_cluster_metrics"] = cluster_dir / "tables" / "train_cluster_metrics.csv"
    files["val_hotspots"] = error_output_dir(settings.som_config, repo_root) / "tables" / "val_hotspots.csv"
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")

    hashes = hash_files(files)
    verify_inputs(analysis, hashes)
    hashes["som_model"] = model_hash

    rows = _read_rows(files["train_bmu_assignments"])
    sample_id = np.array([int(r["sample_id"]) for r in rows])
    label = np.array([int(r["label"]) for r in rows])
    saved_ids = np.load(files["train_sample_ids"])
    saved_labels = np.load(files["train_labels"])
    if not (np.array_equal(sample_id, saved_ids) and np.array_equal(label, saved_labels)):
        raise ValueError(
            "train_bmu_assignments.csv does not match the saved training sample IDs and labels "
            "(different split or different order)."
        )

    bmu = np.array([int(r["bmu_index"]) for r in rows])
    if bmu.min() < 0 or bmu.max() >= analysis.num_neurons:
        raise ValueError(f"bmu_index must be in [0, {analysis.num_neurons}).")

    hotspot_rows = [r for r in _read_rows(files["val_hotspots"]) if r["hotspot"] == "True"]
    hotspot_neurons = np.array(sorted(int(r["neuron_id"]) for r in hotspot_rows))
    if len(hotspot_neurons) != settings.num_hotspots:
        raise ValueError(
            f"val_hotspots.csv lists {len(hotspot_neurons)} hotspots; the hotspot rule gives "
            f"{settings.num_hotspots}."
        )

    return TrainInputs(
        sample_id=sample_id,
        label=label,
        correct=np.array([int(r["correct"]) == 1 for r in rows]),
        confidence=np.array([float(r["confidence"]) for r in rows]),
        bmu=bmu,
        bmu_distance=np.array([float(r["bmu_distance"]) for r in rows]),
        purity_by_neuron=_purity_table(_read_rows(files["train_cluster_metrics"]), analysis.num_neurons),
        hotspot_neurons=hotspot_neurons,
        sha256=hashes,
    )
