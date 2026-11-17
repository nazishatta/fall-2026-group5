"""File outputs of the cluster analysis: the per-neuron table and the run metadata.

Split from run_cluster_analysis.py on 2026-11-17 (keep files small, one responsibility each).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.cluster_analysis.cluster_metrics import (
    ClusterMetrics,
)
from src.v1_mnist.component.utils.som_io import sha256_file


def write_cluster_table(
    path: Path, metrics: list[ClusterMetrics], coordinates: np.ndarray, num_classes: int,
) -> None:
    """One row per neuron, in neuron order, with the neuron's position on the map."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "neuron_id",
        "pos_0",
        "pos_1",
        "support",
        "dominant_class",
        "dominant_count",
        "purity",
        "normalized_entropy",
    ] + [f"class_{i}_count" for i in range(num_classes)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for metric in metrics:
            row = {
                "neuron_id": metric.neuron_id,
                "pos_0": float(coordinates[0, metric.neuron_id]),
                "pos_1": float(coordinates[1, metric.neuron_id]),
                "support": metric.support,
                "dominant_class": metric.dominant_class,
                "dominant_count": metric.dominant_count,
                "purity": metric.purity,
                "normalized_entropy": metric.normalized_entropy,
            }

            for class_id, count in enumerate(metric.class_counts):
                row[f"class_{class_id}_count"] = count

            writer.writerow(row)


def git_commit(repo_root: Path) -> str:
    """The current commit, or 'unknown' outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_metadata(folder: Path, input_hashes: dict[str, str]) -> None:
    """metadata/input_hashes.json and metadata/artifact_hashes.json (every other file in folder)."""
    metadata_dir = folder / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    artifact_hashes = {
        p.relative_to(folder).as_posix(): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    (metadata_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
