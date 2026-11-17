"""Same-budget comparison of the error detectors (exploratory; added after the baseline results were seen).

Reads the finished error-pattern baseline run (the SOM and k-means hotspot shares) and the same frozen
validation inputs, recomputes kNN impurity and max-softmax exactly as that run did, and asks: if each score
flags the same share of images as the hotspot cells hold, what share of the CNN's mistakes does it catch?
Writes a new run folder; the baseline run is only read. Validation split only.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.budget import (
    bootstrap_caught,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.inputs import (
    load_inputs,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.partitions import (
    knn_impurity,
    scale_features,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
    load_baseline_settings,
    resolve_path,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json
from src.v1_mnist.component.utils.scores import max_softmax_score
from src.v1_mnist.component.utils.som_io import sha256_file

logger = get_logger("v1_mnist.error_detector_budget")
DEFAULT_CONFIG = "src/v1_mnist/component/configs/error_detector_budget.yaml"
CELL_METHODS = ("som_cell", "kmeans_cell")


@dataclass(frozen=True)
class BudgetSettings:
    config_path: Path
    baselines_config: str
    run_id: str
    output_dir: Path
    resamples: int
    seed: int


def load_budget_settings(config_path: str | Path | None, repo_root: Path) -> BudgetSettings:
    cfg_path = resolve_path(repo_root, config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base, analysis = load_baseline_settings(raw["inputs"]["baselines_config"], repo_root)
    revision = raw["run"].get("revision") or ""
    run_id = f"{raw['run']['name']}_som_{analysis.grid_height}x{analysis.grid_width}" + (f"_{revision}" if revision else "")
    return BudgetSettings(cfg_path, str(raw["inputs"]["baselines_config"]), run_id,
                          resolve_path(repo_root, raw["paths"]["output_root"]) / run_id,
                          int(raw["bootstrap"]["resamples"]), int(raw["bootstrap"]["seed"]))


def hotspot_shares(baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Image and mistake shares of the hotspot cells (mean over the held-out halves; k-means: mean over seeds)."""
    out = {}
    for method in CELL_METHODS:
        h = baseline["hotspot"][method]
        h = h.get("hotspot_shares", h)
        out[method] = {"image_share": float(h["mean_image_share_in_hotspots"]),
                       "error_share": float(h["mean_error_share_in_hotspots"])}
    return out


def run_budget(settings: BudgetSettings, repo_root: Path) -> dict[str, Any]:
    results_path = settings.output_dir / "results.json"
    ensure_not_done(results_path)
    base, analysis = load_baseline_settings(settings.baselines_config, repo_root)
    baseline = json.loads(base.results_path.read_text(encoding="utf-8"))
    data = load_inputs(base, analysis, repo_root)
    if data.sha256 != baseline["inputs_sha256"]:
        raise RuntimeError("The frozen inputs differ from the ones the baseline run recorded.")
    train_scaled, val_scaled = scale_features(data.train_features, data.val_features, base.feature_range)
    scores = {
        "knn_impurity": knn_impurity(train_scaled, data.train_labels, val_scaled, base.knn_neighbors, analysis.num_classes),
        "max_softmax": max_softmax_score(data.val_confidence),
    }
    shares = hotspot_shares(baseline)
    budgets = {}
    for method, s in shares.items():
        budgets[f"same_share_as_{method}"] = {
            "cells": {**s, "ratio": s["error_share"] / s["image_share"]},
            "scores": bootstrap_caught(scores, data.val_errors, s["image_share"], settings.resamples, settings.seed),
        }
        for name, r in budgets[f"same_share_as_{method}"]["scores"].items():
            logger.info("%s at %.2f%% of images: %.1f%% of mistakes (cells: %.1f%%)", name,
                        100 * s["image_share"], 100 * r["error_share"], 100 * s["error_share"])
    results = {
        "run_id": settings.run_id, "split": "val", "test_used": False, "exploratory": True,
        "validation_samples": int(data.val_errors.size), "validation_errors": int(data.val_errors.sum()),
        "inputs_sha256": data.sha256, "config_sha256": sha256_file(settings.config_path),
        "baseline_results_sha256": sha256_file(base.results_path), "baseline_run_id": baseline["run_id"],
        "settings": {"bootstrap_resamples": settings.resamples, "bootstrap_seed": settings.seed,
                     "knn_neighbors": base.knn_neighbors},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "budgets": budgets,
    }
    write_json(results_path, results)
    _write_csv(settings.output_dir / "summary.csv", budgets)
    return results


def _write_csv(path: Path, budgets: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["budget", "method", "image_share", "error_share", "error_share_low", "error_share_high", "ratio"])
        for budget, b in budgets.items():
            c = b["cells"]
            w.writerow([budget, budget.replace("same_share_as_", ""), c["image_share"], c["error_share"], "", "", c["ratio"]])
            for name, r in b["scores"].items():
                w.writerow([budget, name, r["image_share"], r["error_share"], r["error_share_low"],
                            r["error_share_high"], r["ratio"]])
