"""Stage 3: fc1 - fc2 per SOM seed, t-intervals and the three-way verdicts of L1 to L3 (protocol section 4)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.provenance import (
    verify_run,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
    AblationSettings,
    cell_name,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.comparison import paired_summary
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

log = logging.getLogger("v1_mnist.feature_layer_ablation")
STATEMENTS = {"L1_error_cell_auroc": "A1_error_cell_auroc", "L2_weighted_purity": "A2_weighted_purity",
              "L3_qe_novelty_auroc": "A3_qe_novelty_auroc"}
REPORTED = ("hotspot_error_share", "hotspot_error_rate", "topographic_error_pct", "occupied_neurons")


def value(metrics: dict[str, Any], measure: str) -> float:
    """One number per map: A1 is the mean over the half-splits; TE comes from the SOM quality record."""
    if measure == "A1_error_cell_auroc":
        return float(metrics[measure]["mean"])
    if measure == "topographic_error_pct":
        return float(metrics["validation_quality"]["topological_error_1st_order_pct"])
    return float(metrics[measure])


def verdict(interval: list[float], first: str, second: str) -> str:
    """Lower end above 0: first better; upper end below 0: second better; otherwise inconclusive."""
    if interval[0] > 0:
        return f"{first} better"
    if interval[1] < 0:
        return f"{second} better"
    return "inconclusive"


def compare(per_layer: dict[str, list[dict[str, Any]]], layers: tuple[str, str], confidence: float) -> dict[str, Any]:
    """Statements L1 to L3 and the reported measures, from the metrics of each layer in seed order."""
    first, second = layers
    out: dict[str, Any] = {"statements": {}, "reported": {}, "means": {}}
    for name, measure in STATEMENTS.items():
        a = np.array([value(m, measure) for m in per_layer[first]])
        b = np.array([value(m, measure) for m in per_layer[second]])
        s = paired_summary(a, b, confidence)
        out["statements"][name] = {**s, "verdict": verdict(s["interval"], first, second), "primary": name.startswith("L1")}
    for measure in (*STATEMENTS.values(), *REPORTED, "val_qe_mean"):
        out["means"][measure] = {layer: float(np.mean([value(m, measure) for m in per_layer[layer]]))
                                 for layer in layers}
    for measure in REPORTED:
        a = np.array([value(m, measure) for m in per_layer[first]])
        b = np.array([value(m, measure) for m in per_layer[second]])
        out["reported"][measure] = paired_summary(a, b, confidence)
    return out


def _write_tables(folder: Path, per_layer: dict[str, list[dict[str, Any]]], result: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cols = (*STATEMENTS.values(), *REPORTED)
    with (folder / "cell_measures.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "layer", "som_seed", *cols])
        for layer, rows in per_layer.items():
            for m in rows:
                w.writerow([m["cell"], layer, m["som_seed"], *(value(m, c) for c in cols)])
    with (folder / "statements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["statement", "mean_difference", "interval_low", "interval_high", "share_fc1_larger", "verdict"])
        for name, s in result["statements"].items():
            w.writerow([name, s["mean_difference"], *s["interval"], s["share_a_larger"], s["verdict"]])


def run_compare(settings: AblationSettings) -> dict[str, Any]:
    """Read every cell, write results.json and the tables (once; never overwritten)."""
    ensure_not_done(settings.output_dir / "results.json")
    verify_run(settings)
    per_layer: dict[str, list[dict[str, Any]]] = {}
    for layer in settings.layers:
        rows = []
        for seed in settings.som_seeds:
            path = settings.metrics_path(layer, seed)
            if not path.is_file():
                raise FileNotFoundError(f"{cell_name(layer, seed)} is not done: run the 'som' stage first.")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        per_layer[layer] = rows
    result = compare(per_layer, (settings.layers[0], settings.layers[1]), settings.confidence)
    extraction = json.loads(settings.extraction_path.read_text(encoding="utf-8"))
    result.update({"run_id": settings.run_id, "test_used": False, "som_seeds": list(settings.som_seeds),
                   "fc2_reproduction_max_abs_difference": extraction["fc2_max_abs_difference"]})
    write_json(settings.output_dir / "results.json", result)
    _write_tables(settings.folder("tables"), per_layer, result)
    for name, s in result["statements"].items():
        log.info("%s: %+.4f [%+.4f, %+.4f] -> %s", name, s["mean_difference"], *s["interval"], s["verdict"])
    return result
