"""Everything a retraining run reads: the frozen split, the hard-image pools and the hotspot mask.

Training and validation images only. The test split is loaded by the final test stage alone.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.v1_mnist.component.utils.frozen_split import (
    SplitImages,
    load_train_val_images,
)
from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings
from src.v1_mnist.component.utils.run_provenance import hash_files
from src.v1_mnist.component.utils.run_settings import cluster_output_dir


@dataclass(frozen=True)
class RetrainingData:
    """Images of the frozen split plus what the hard example mining step decided."""

    train: SplitImages
    val: SplitImages
    val_in_hotspot: np.ndarray
    hotspot_neurons: np.ndarray
    train_in_hotspot: np.ndarray
    som_pool: np.ndarray
    msp_pool: np.ndarray
    rules_satisfied: np.ndarray
    bmu_distance: np.ndarray
    confidence: np.ndarray
    sha256: dict[str, str]


def _csv_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {c: np.array([float(r[c]) for r in rows]) for c in columns}


def load_retraining_data(settings: ExperimentSettings, repo_root: Path) -> RetrainingData:
    """Load and cross-check the split, the pool table and the validation hotspot mask."""
    results_path = settings.pool_dir / "results.json"
    difficulty_path = settings.pool_dir / "difficulty" / "train_difficulty.csv"
    for path in (results_path, difficulty_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run the hard example mining step first (and `dvc pull` if the "
                "difficulty table is tracked with DVC)."
            )
    pools = json.loads(results_path.read_text(encoding="utf-8"))
    hotspots = np.array(pools["hotspot_neurons"], dtype=np.int64)

    train, val = load_train_val_images(settings.data_root, settings.embeddings_dir)

    table = _csv_columns(
        difficulty_path,
        ["sample_id", "in_hotspot", "low_purity", "qe_flag", "in_som_pool", "in_msp_pool",
         "bmu_distance", "confidence"],
    )
    if not np.array_equal(table["sample_id"].astype(np.int64), train.sample_ids):
        raise ValueError("train_difficulty.csv is not in the order of the saved training sample IDs.")

    cells_path = cluster_output_dir(settings.som_config, repo_root) / "assignments" / "val_bmu_assignments.csv"
    cells = _csv_columns(cells_path, ["sample_id", "bmu_index"])
    if not np.array_equal(cells["sample_id"].astype(np.int64), val.sample_ids):
        raise ValueError("val_bmu_assignments.csv is not in the order of the saved validation sample IDs.")

    hashes = hash_files({
        "pool_results": results_path,
        "train_difficulty": difficulty_path,
        "val_bmu_assignments": cells_path,
    })
    hashes.update({f"pools_{k}": v for k, v in pools["sha256"].items()})
    return RetrainingData(
        train=train,
        val=val,
        val_in_hotspot=np.isin(cells["bmu_index"].astype(np.int64), hotspots),
        hotspot_neurons=hotspots,
        train_in_hotspot=table["in_hotspot"].astype(bool),
        som_pool=table["in_som_pool"].astype(bool),
        msp_pool=table["in_msp_pool"].astype(bool),
        rules_satisfied=(table["in_hotspot"] + table["low_purity"] + table["qe_flag"]).astype(np.int64),
        bmu_distance=table["bmu_distance"],
        confidence=table["confidence"],
        sha256=hashes,
    )
