"""Exploratory breakdown of the test results: errors fixed per seed, and error per class.

Added 2026-11-17, after the single test pass (Reviews #8 and #9). It changes no setting and no
criterion, trains nothing and runs no model: it reads the test predictions that final_test saved
(predictions/<arm>_seed<N>_test.npz) and the test labels. Results are descriptive, not success
criteria (RETRAINING_PROTOCOL.md section 5 is unchanged).

- Errors fixed: for each arm and seed, the standard arm's test errors (same seed) that the arm gets
  right, and the new errors it makes that standard got right, overall and on hotspot images.
- Per class: test error per digit for every arm, and the same arm-against-arm comparisons as the
  test pass (paired bootstrap over images of that digit, seeds averaged first). No arm defines the
  classes, so this view does not favour SOM-guided arms the way hotspot error can. Ten classes are
  tested without a multiple-comparison correction, so single intervals are read with care.
"""

from __future__ import annotations

import csv
import json
import logging
from typing import Any

import numpy as np

from src.v1_mnist.component.utils.paired_stats import (
    paired_bootstrap,
    seed_average_error,
    seed_spread,
)
from src.v1_mnist.component.utils.protocol_stamp import verify_stamp
from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings
from src.v1_mnist.component.utils.retraining_summary import comparison_pairs
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json
from src.v1_mnist.component.utils.som_cells import assign_cells

LOGGER = logging.getLogger("v1_mnist.retraining_breakdown")
BASELINE = "standard"


def errors_fixed(base_wrong: np.ndarray, arm_wrong: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    """Counts on the masked images: errors of the baseline the arm fixes, and errors it adds."""
    fixed = int((base_wrong & ~arm_wrong & mask).sum())
    new = int((arm_wrong & ~base_wrong & mask).sum())
    return {"baseline_errors": int((base_wrong & mask).sum()), "arm_errors": int((arm_wrong & mask).sum()),
            "fixed": fixed, "new": new, "net_fixed": fixed - new}


def per_class_error(wrong: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Error rate per class for one seed's 0/1 error vector."""
    return np.array([wrong[labels == c].mean() for c in range(num_classes)])


def _load_wrong(settings: ExperimentSettings, arm: str, labels: np.ndarray) -> np.ndarray:
    """Seeds x images array of 0/1 test errors from the saved predictions."""
    rows = []
    for seed in settings.seeds:
        pred = np.load(settings.prediction_path(arm, seed, "test"))["pred"]
        if pred.shape != labels.shape:
            raise ValueError(f"{arm} seed {seed}: {pred.shape[0]} predictions for {labels.shape[0]} test labels.")
        rows.append(pred != labels)
    return np.array(rows)


def _hotspot_mask(settings: ExperimentSettings) -> np.ndarray:
    """Test images on hotspot neurons, found exactly as in the test pass."""
    features = np.load(settings.embeddings_dir / "test_features.npy")
    cells, _ = assign_cells(settings.analysis, features)
    pools = json.loads((settings.pool_dir / "results.json").read_text(encoding="utf-8"))
    return np.isin(cells, np.array(pools["hotspot_neurons"]))


def run_breakdown(settings: ExperimentSettings) -> dict[str, Any]:
    """Write test_breakdown.json and four tables; stops if the breakdown already exists."""
    out_path = settings.output_dir / "test_breakdown.json"
    ensure_not_done(out_path)
    verify_stamp(settings)
    test_results = json.loads((settings.output_dir / "test_results.json").read_text(encoding="utf-8"))
    labels = np.load(settings.embeddings_dir / "test_labels.npy")
    in_hotspot = _hotspot_mask(settings)
    if int(in_hotspot.sum()) != test_results["hotspot_test_images"]:
        raise RuntimeError("Hotspot test images differ from the test pass.")
    k = settings.train.num_classes
    wrong = {a.name: _load_wrong(settings, a.name, labels) for a in settings.arms}
    for name, w in wrong.items():  # the predictions must be the ones the test pass scored
        recorded = test_results["arms"][name]
        if not (np.allclose(w.mean(axis=1), recorded["overall_error"]["per_seed"])
                and np.allclose(w[:, in_hotspot].mean(axis=1), recorded["hotspot_error"]["per_seed"])):
            raise RuntimeError(f"{name}: saved predictions or hotspot images do not reproduce test_results.json.")

    masks = {"overall": np.ones_like(in_hotspot), "hotspot": in_hotspot}
    fixed = {
        arm: {str(seed): {m: errors_fixed(wrong[BASELINE][i], w[i], mask) for m, mask in masks.items()}
              for i, seed in enumerate(settings.seeds)}
        for arm, w in wrong.items() if arm != BASELINE
    }
    per_class = {arm: np.array([per_class_error(w[i], labels, k) for i in range(len(settings.seeds))])
                 for arm, w in wrong.items()}
    class_comparisons = {
        f"{g} minus {c}": {
            str(cls): paired_bootstrap(seed_average_error(wrong[g]), seed_average_error(wrong[c]),
                                       labels == cls, settings.bootstrap_resamples, settings.bootstrap_seed)
            for cls in range(k)
        }
        for g, c in comparison_pairs(settings.arms)
    }
    results = {
        "run_id": settings.run_id, "exploratory": True, "added": "2026-11-17, after the test pass",
        "seeds": list(settings.seeds), "test_images": int(labels.size),
        "hotspot_test_images": int(in_hotspot.sum()),
        "images_per_class": [int((labels == c).sum()) for c in range(k)],
        "errors_fixed_vs_standard": fixed,
        "per_class_error": {a: {"mean": v.mean(axis=0).tolist(), "std": v.std(axis=0, ddof=1).tolist()}
                            for a, v in per_class.items()},
        "per_class_comparisons": class_comparisons,
        "stamp_sha256": test_results["stamp_sha256"],
    }
    write_json(out_path, results)
    _write_tables(settings, results, k)
    LOGGER.info("Wrote %s", out_path)
    return results


def _write_tables(settings: ExperimentSettings, results: dict[str, Any], k: int) -> None:
    tables = settings.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "test_errors_fixed.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["arm", "seed", "subset", "standard_errors", "arm_errors", "fixed", "new", "net_fixed"])
        for arm, by_seed in results["errors_fixed_vs_standard"].items():
            for seed, by_mask in by_seed.items():
                for subset, c in by_mask.items():
                    w.writerow([arm, seed, subset, c["baseline_errors"], c["arm_errors"], c["fixed"], c["new"],
                                c["net_fixed"]])
    with (tables / "test_errors_fixed_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["arm", "subset", "fixed_mean", "new_mean", "net_fixed_mean", "net_fixed_std"])
        for arm, by_seed in results["errors_fixed_vs_standard"].items():
            for subset in ("overall", "hotspot"):
                vals = {f: np.array([s[subset][f] for s in by_seed.values()]) for f in ("fixed", "new", "net_fixed")}
                net = seed_spread(vals["net_fixed"])
                w.writerow([arm, subset, round(vals["fixed"].mean(), 2), round(vals["new"].mean(), 2),
                            round(net["mean"], 2), round(net["std"], 2)])
    with (tables / "test_per_class_error.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["arm"] + [f"class_{c}" for c in range(k)])
        for arm, v in results["per_class_error"].items():
            w.writerow([arm] + [round(x, 6) for x in v["mean"]])
    with (tables / "test_per_class_comparisons.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["comparison", "class", "images", "mean_difference", "ci_low", "ci_high", "excludes_zero"])
        for name, by_class in results["per_class_comparisons"].items():
            for cls, c in by_class.items():
                w.writerow([name, cls, c["images"], round(c["mean_difference"], 6), round(c["ci_95"][0], 6),
                            round(c["ci_95"][1], 6), c["excludes_zero"]])
