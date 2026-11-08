"""Stage 3: measures, paired comparisons, statements R1 to R5 and CKA. NumPy only; validation only."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.cka import (
    cka_matrix,
    group_means,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.comparison import (
    paired_summary,
    statements,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    cell_name,
    embeddings_folder,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_measures import (
    all_measures,
)
from src.v1_mnist.component.utils.run_provenance import write_json

log = logging.getLogger("v1_mnist.som_reanalysis")

COMPARED = ("weighted_purity", "low_purity_share", "class_scatter_index", "topographic_error_pct",
            "hotspot_error_rate", "hotspot_error_share", "overall_error")


@dataclass(frozen=True)
class CellData:
    """What the measures need from one cell (validation images only)."""

    bmu: np.ndarray
    neuron_dist: np.ndarray
    predictions: np.ndarray
    topographic_error_pct: float
    quantization_error: float


def measure_cell(cell: CellData, labels: np.ndarray, settings: ReanalysisSettings) -> dict[str, Any]:
    """All per-cell numbers of the protocol (M1 to M7; quantization error is reported only)."""
    r = settings.recipe
    out = all_measures(
        cell.bmu, labels, cell.predictions, cell.neuron_dist, settings.retraining.train.num_classes,
        settings.purity_cutoff, settings.adjacency, r.hotspot_fraction, r.min_hotspots,
    )
    out["topographic_error_pct"] = cell.topographic_error_pct
    out["quantization_error"] = cell.quantization_error
    return out


def compare_arms(table: dict[str, dict[str, Any]], settings: ReanalysisSettings) -> dict[str, dict[str, Any]]:
    """Paired summaries for every configured pair and compared measure."""
    out: dict[str, dict[str, Any]] = {}
    for first, second in settings.pairs:
        key = f"{first}_vs_{second}"
        out[key] = {}
        for measure in COMPARED:
            a = np.array([table[cell_name(first, s)][measure] for s in settings.seeds])
            b = np.array([table[cell_name(second, s)][measure] for s in settings.seeds])
            out[key][measure] = paired_summary(a, b, settings.confidence)
    return out


def load_cell(settings: ReanalysisSettings, cell: str, predictions: np.ndarray) -> CellData:
    npz = np.load(settings.folder("assignments") / f"{cell}.npz")
    record = json.loads((settings.folder("metrics") / f"{cell}.json").read_text(encoding="utf-8"))
    quality = record["validation_quality"]
    return CellData(
        bmu=npz["val_bmu"], neuron_dist=npz["neuron_dist"], predictions=predictions,
        topographic_error_pct=float(quality["topological_error_1st_order_pct"]),
        quantization_error=float(quality["quantization_error"]),
    )


def _write_cell_table(path: Path, table: dict[str, dict[str, Any]]) -> None:
    columns = ["weighted_purity", "low_purity_share", "class_scatter_index", "topographic_error_pct",
               "hotspot_error_rate", "hotspot_error_share", "overall_error", "quantization_error",
               "occupied_neurons", "hotspot_images"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", *columns])
        for cell, row in table.items():
            writer.writerow([cell, *[row[c] for c in columns]])


def _write_comparison_table(path: Path, comparisons: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "measure", "mean_difference", "interval_low", "interval_high",
                         "share_first_larger", "interval_excludes_zero", "seeds"])
        for pair, measures in comparisons.items():
            for measure, s in measures.items():
                writer.writerow([pair, measure, s["mean_difference"], *s["interval"], s["share_a_larger"],
                                 s["excludes_zero"], s["seeds"]])


def run_analysis(settings: ReanalysisSettings) -> dict[str, Any]:
    """Write tables/ from the stored assignments, embeddings and predictions."""
    ana = settings.retraining.analysis
    labels = np.load(ana.embeddings_dir / "val_labels.npy").astype(np.int64)
    table: dict[str, dict[str, Any]] = {}
    ref_pred = np.load(ana.embeddings_dir / "val_preds.npy").astype(np.int64)
    table[settings.reference] = measure_cell(load_cell(settings, settings.reference, ref_pred), labels, settings)
    for cell in settings.cells:
        pred = np.load(embeddings_folder(settings, cell) / "val_preds.npy")
        table[cell] = measure_cell(load_cell(settings, cell, pred), labels, settings)

    comparisons = compare_arms(table, settings)
    first, second = settings.pairs[0]
    random_arm = settings.pairs[1][1]
    verdicts = statements(comparisons, settings.tolerance_pct, second, random_arm, first)

    features = {cell: np.load(embeddings_folder(settings, cell) / "val_features.npy") for cell in settings.cells}
    names, mat = cka_matrix(features)
    arm_of = {cell: cell.rsplit("_seed", 1)[0] for cell in names}
    cka = {"cells": names, "matrix": mat.tolist(), "group_means": group_means(names, mat, arm_of)}

    tables = settings.folder("tables")
    _write_cell_table(tables / "cell_measures.csv", table)
    _write_comparison_table(tables / "paired_comparisons.csv", comparisons)
    write_json(tables / "cell_measures.json", table)
    write_json(tables / "paired_comparisons.json", comparisons)
    write_json(tables / "statements.json", verdicts)
    write_json(tables / "cka.json", cka)
    for name, v in verdicts.items():
        log.info("%s: %s", name, "MET" if v["met"] else "NOT MET")
    return {"table": table, "comparisons": comparisons, "statements": verdicts, "cka": cka}
