"""The stages of the error pattern baseline check (hotspot analysis, k-means seeds, final results).

Each stage saves to <output_dir>/work/ so a long run can be done in short steps; the final stage
writes results.json, the summary CSV and the LaTeX table, and refuses to overwrite an existing run.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.detection import (
    bootstrap_auroc,
    half_split_detection,
    summarize,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.hotspot_comparison import (
    analyse_partition,
    check_reproduction,
    summarize_seeds,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.inputs import (
    ValidationInputs,
    load_inputs,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.partitions import (
    fit_kmeans_cells,
    knn_impurity,
    scale_features,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.result_tables import (
    write_report_outputs,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.run_preflight import (
    ensure_not_done,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
    BaselineSettings,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.run_settings import AnalysisSettings, robustness_dir
from src.v1_mnist.component.utils.scores import max_softmax_score

logger = get_logger("v1_mnist.error_pattern_baselines")

SOM_WORK_FILE = "hotspot_som.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_som(settings: BaselineSettings, analysis: AnalysisSettings, repo_root: Path) -> None:
    """Hotspot analysis of the headline SOM; must reproduce the project's own numbers exactly."""
    ensure_not_done(settings.results_path)
    out = settings.work_dir / SOM_WORK_FILE
    if out.exists():
        logger.info("SOM hotspot analysis already done: %s", out.name)
        return
    data = load_inputs(settings, analysis, repo_root)
    logger.info("Hotspot analysis of the %dx%d SOM (%d validation errors)", *settings.grid, int(data.val_errors.sum()))
    result = analyse_partition(data.val_cells, data.val_errors, data.val_labels, settings.num_cells,
                               analysis.num_classes, settings.robustness)
    reference = robustness_dir(settings.som_config, repo_root) / f"{analysis.selected_model}_error_pattern_robustness.json"
    check = check_reproduction(result, reference)
    if not check["reproduced_exactly"]:
        raise RuntimeError(f"Our SOM hotspot numbers differ from {reference.name}: {check}")
    logger.info("Reproduction check against %s: exact", reference.name)
    _write_json(out, {"analysis": result, "reproduction_check": check})


def stage_kmeans(
    settings: BaselineSettings, analysis: AnalysisSettings, repo_root: Path, seeds: Sequence[int]
) -> None:
    """Fit k-means on the training embeddings for each seed and run the hotspot analysis on its cells."""
    ensure_not_done(settings.results_path)
    data = load_inputs(settings, analysis, repo_root)
    train_scaled, val_scaled = scale_features(data.train_features, data.val_features, settings.feature_range)
    for seed in seeds:
        out = settings.work_dir / f"hotspot_kmeans_s{seed}.json"
        if out.exists():
            logger.info("k-means seed %d already done", seed)
            continue
        cells, inertia = fit_kmeans_cells(train_scaled, val_scaled, settings.num_cells, seed, settings.kmeans_n_init)
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        np.save(settings.work_dir / f"kmeans_cells_s{seed}.npy", cells)
        result = analyse_partition(cells, data.val_errors, data.val_labels, settings.num_cells,
                                   analysis.num_classes, settings.robustness)
        result["inertia_train"] = inertia
        _write_json(out, result)
        conc = result["hotspot_concentration_held_out"]
        logger.info("k-means seed %d: median ratio %.3f, chance %.3f", seed, conc["median_ratio"], conc["chance_median_ratio"])


def _detection_results(
    settings: BaselineSettings, analysis: AnalysisSettings, data: ValidationInputs, kmeans_cells: dict[int, np.ndarray]
) -> dict[str, Any]:
    train_scaled, val_scaled = scale_features(data.train_features, data.val_features, settings.feature_range)
    fixed = {
        "knn_impurity": knn_impurity(train_scaled, data.train_labels, val_scaled, settings.knn_neighbors, analysis.num_classes),
        "max_softmax": max_softmax_score(data.val_confidence),
    }
    scores = half_split_detection(
        {"som_cell": [data.val_cells], "kmeans_cell": [kmeans_cells[s] for s in settings.kmeans_seeds]},
        fixed, data.val_errors, settings.num_cells, settings.detection_repeats, settings.robustness.split_seed_base,
    )
    methods: dict[str, Any] = {}
    for name, score in scores.items():
        entry: dict[str, Any] = {"auroc": summarize(score.auroc), "average_precision": summarize(score.average_precision)}
        if name != "som_cell":
            diff = score.auroc - scores["som_cell"].auroc
            entry["auroc_minus_som"] = {**summarize(diff), "share_of_repeats_better_than_som": float((diff > 0).mean())}
        if name == "kmeans_cell":
            entry["mean_auroc_per_seed"] = {str(s): float(v) for s, v in zip(settings.kmeans_seeds, scores[name].auroc_per_partition.mean(axis=1), strict=True)}
        methods[name] = entry
    bootstrap = bootstrap_auroc(data.val_errors, fixed, settings.bootstrap_resamples, settings.bootstrap_seed)
    return {"methods": methods, "bootstrap": bootstrap, "error_base_rate": float(data.val_errors.mean())}


def stage_final(settings: BaselineSettings, analysis: AnalysisSettings, repo_root: Path) -> dict[str, Any]:
    """Error detection scores, bootstrap intervals, results.json, summary CSV and the LaTeX table."""
    ensure_not_done(settings.results_path)
    data = load_inputs(settings, analysis, repo_root)
    som = _read_json(settings.work_dir / SOM_WORK_FILE)
    per_seed = {s: _read_json(settings.work_dir / f"hotspot_kmeans_s{s}.json") for s in settings.kmeans_seeds}
    kmeans_cells = {s: np.load(settings.work_dir / f"kmeans_cells_s{s}.npy") for s in settings.kmeans_seeds}

    results: dict[str, Any] = {
        "run_id": settings.run_id,
        "split": "val",
        "test_used": False,
        "validation_samples": int(data.val_errors.size),
        "validation_errors": int(data.val_errors.sum()),
        "inputs_sha256": data.sha256,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "settings": {
            "grid": list(settings.grid), "num_cells": settings.num_cells,
            "kmeans_seeds": list(settings.kmeans_seeds), "kmeans_n_init": settings.kmeans_n_init,
            "knn_neighbors": settings.knn_neighbors, "detection_repeats": settings.detection_repeats,
            "bootstrap_resamples": settings.bootstrap_resamples, "bootstrap_seed": settings.bootstrap_seed,
            "split_seed_base": settings.robustness.split_seed_base, "shuffle_seed_base": settings.robustness.perm_seed_base,
            "split_repeats": settings.robustness.n_splits, "shuffles": settings.robustness.n_permutations,
        },
        "reproduction_check": som["reproduction_check"],
        "hotspot": {"som_cell": som["analysis"],
                    "kmeans_cell": {"per_seed": {str(s): r for s, r in per_seed.items()}, **summarize_seeds(per_seed)}},
        "error_detection": _detection_results(settings, analysis, data, kmeans_cells),
    }
    _write_json(settings.results_path, results)
    write_report_outputs(results, settings.output_dir)
    logger.info("Results written to %s", settings.output_dir)
    return results


def stage_table(settings: BaselineSettings) -> None:
    """Rewrite the summary CSV and the LaTeX table from an existing results.json (results are untouched)."""
    results = _read_json(settings.results_path)
    write_report_outputs(results, settings.output_dir)
    logger.info("Summary CSV and LaTeX table rewritten in %s", settings.output_dir)
