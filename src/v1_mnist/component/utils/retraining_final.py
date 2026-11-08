"""The single test pass: evaluate every finished run on the test split, once, after the protocol is frozen.

Nothing is trained or chosen here. Each saved checkpoint (SHA-256 checked) is evaluated on the test
images; arms are compared with a paired bootstrap over test images (seeds averaged first); the
criteria of RETRAINING_PROTOCOL.md section 5 are applied to the numbers. Stops if the protocol is
not frozen, if a run is missing, or if test results already exist (never overwritten).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.v1_mnist.component.utils.cnn_retraining import (
    load_model,
    predict,
    resolve_device,
)
from src.v1_mnist.component.utils.frozen_split import load_test_images
from src.v1_mnist.component.utils.paired_stats import (
    paired_bootstrap,
    seed_average_error,
    seed_spread,
)
from src.v1_mnist.component.utils.protocol_stamp import verify_stamp
from src.v1_mnist.component.utils.retraining_figures import plot_confusion_comparison
from src.v1_mnist.component.utils.retraining_metrics import (
    confusion,
    evaluation_metrics,
)
from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings
from src.v1_mnist.component.utils.retraining_summary import (
    comparison_pairs,
    load_run_record,
)
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json
from src.v1_mnist.component.utils.som_cells import assign_cells
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger("v1_mnist.retraining_final")
MASKS = ("overall", "hotspot", "non_hotspot")


def _evaluate_arm(
    settings: ExperimentSettings, arm: str, test_x: np.ndarray, test_y: np.ndarray,
    in_hotspot: np.ndarray, device: torch.device,
) -> dict[str, Any]:
    wrong, metrics, matrices = [], [], []
    for seed in settings.seeds:
        record = load_run_record(settings, arm, seed)
        checkpoint = settings.checkpoint_path(arm, seed)
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"{checkpoint.name}: checkpoint differs from the one recorded after training.")
        model = load_model(torch.load(checkpoint, map_location="cpu"), settings.train.num_classes, device)
        pred, conf = predict(model, test_x, device)
        np.savez_compressed(settings.prediction_path(arm, seed, "test"), pred=pred, confidence=conf)
        wrong.append(pred != test_y)
        metrics.append(evaluation_metrics(test_y, pred, in_hotspot, settings.train.num_classes))
        matrices.append(confusion(test_y, pred, settings.train.num_classes))
    return {"wrong": np.array(wrong), "metrics": metrics, "confusion_mean": np.mean(matrices, axis=0)}


def _summarize(evaluated: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in MASKS:
        rates = np.array([m[key]["error_rate"] for m in evaluated["metrics"]])
        out[f"{key}_error"] = seed_spread(rates)
        out[f"{key}_error"]["per_seed"] = rates.tolist()
        out[f"{key}_error"]["images"] = evaluated["metrics"][0][key]["images"]
    f1 = np.array([m["macro_f1"] for m in evaluated["metrics"]])
    out["macro_f1"] = {**seed_spread(f1), "per_seed": f1.tolist()}
    out["per_class_error_mean"] = np.mean([m["per_class_error"] for m in evaluated["metrics"]], axis=0).tolist()
    return out


def _compare(
    settings: ExperimentSettings, guided: dict[str, Any], control: dict[str, Any], in_hotspot: np.ndarray
) -> dict[str, Any]:
    a, b = seed_average_error(guided["wrong"]), seed_average_error(control["wrong"])
    masks = {"overall": np.ones_like(in_hotspot), "hotspot": in_hotspot, "non_hotspot": ~in_hotspot}
    result = {
        key: paired_bootstrap(a, b, mask, settings.bootstrap_resamples, settings.bootstrap_seed)
        for key, mask in masks.items()
    }
    f1_a = np.mean([m["macro_f1"] for m in guided["metrics"]])
    f1_b = np.mean([m["macro_f1"] for m in control["metrics"]])
    result["macro_f1_difference"] = float(f1_a - f1_b)
    guard = settings.macro_f1_guard or 0.0
    result["macro_f1_guard_holds"] = bool(f1_a - f1_b >= -guard)
    hot, overall = result["hotspot"], result["overall"]
    result["meets_criterion"] = bool(hot["mean_difference"] < 0 and hot["excludes_zero"]
                                     and result["macro_f1_guard_holds"])
    result["meets_overall_criterion"] = bool(overall["mean_difference"] < 0 and overall["excludes_zero"]
                                             and result["macro_f1_guard_holds"])
    result["no_damage_elsewhere"] = bool(result["non_hotspot"]["ci_95"][1] <= settings.damage_tolerance)
    return result


def run_final_test(settings: ExperimentSettings, repo_root: Path) -> dict[str, Any]:
    """Evaluate all arms and seeds on the test split and write test_results.json and tables."""
    results_path = settings.output_dir / "test_results.json"
    ensure_not_done(results_path)
    verify_stamp(settings)
    for arm in settings.arms:
        for seed in settings.seeds:
            load_run_record(settings, arm.name, seed)  # stops if any run is unfinished

    test = load_test_images(settings.data_root, settings.embeddings_dir)
    features = np.load(settings.embeddings_dir / "test_features.npy")
    if not np.array_equal(np.load(settings.embeddings_dir / "test_sample_ids.npy"), test.sample_ids):
        raise ValueError("Test sample IDs differ between the images and the embeddings.")
    cells, _ = assign_cells(settings.analysis, features)
    pools = json.loads((settings.pool_dir / "results.json").read_text(encoding="utf-8"))
    hotspots = np.array(pools["hotspot_neurons"])
    in_hotspot = np.isin(cells, hotspots)
    LOGGER.info("Test split: %d images, %d in hotspot neurons.", len(test.labels), int(in_hotspot.sum()))

    device = resolve_device(settings.device)
    evaluated = {a.name: _evaluate_arm(settings, a.name, test.images, test.labels, in_hotspot, device)
                 for a in settings.arms}
    arms = {name: _summarize(ev) for name, ev in evaluated.items()}
    comparisons = {
        f"{g} minus {c}": _compare(settings, evaluated[g], evaluated[c], in_hotspot)
        for g, c in comparison_pairs(settings.arms)
    }
    for guided, control in comparison_pairs(settings.arms):
        if control == "standard":
            plot_confusion_comparison(
                evaluated[control]["confusion_mean"], evaluated[guided]["confusion_mean"],
                control, guided, settings.output_dir / "figures" / f"confusion_{control}_vs_{guided}",
            )
    results = {
        "run_id": settings.run_id, "split": "test", "test_images": len(test.labels),
        "hotspot_test_images": int(in_hotspot.sum()), "seeds": list(settings.seeds),
        "macro_f1_guard": settings.macro_f1_guard, "arms": arms, "comparisons": comparisons,
        "confusion_mean": {n: ev["confusion_mean"].tolist() for n, ev in evaluated.items()},
        "stamp_sha256": sha256_file(settings.stamp_path),
    }
    write_json(results_path, results)
    _write_tables(settings, results)
    LOGGER.info("Wrote %s", results_path)
    return results


def _write_tables(settings: ExperimentSettings, results: dict[str, Any]) -> None:
    settings.tables_dir.mkdir(parents=True, exist_ok=True)
    with (settings.tables_dir / "test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "overall_error", "overall_std", "hotspot_error", "hotspot_std",
                         "non_hotspot_error", "non_hotspot_std", "macro_f1", "macro_f1_std"])
        for name, a in results["arms"].items():
            writer.writerow([name] + [round(a[k][s], 6) for k in
                            ("overall_error", "hotspot_error", "non_hotspot_error", "macro_f1")
                            for s in ("mean", "std")])
    with (settings.tables_dir / "test_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "subset", "mean_difference", "ci_low", "ci_high", "excludes_zero"])
        for name, c in results["comparisons"].items():
            for key in MASKS:
                writer.writerow([name, key, round(c[key]["mean_difference"], 6), round(c[key]["ci_95"][0], 6),
                                 round(c[key]["ci_95"][1], 6), c[key]["excludes_zero"]])
