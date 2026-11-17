"""Stage 2: statements H1 to H4 over the five splits (protocol section 7), results.json and tables.

Reads the per-split metrics only. A difference is taken per split; the mean and its t-interval come from the
SOM re-analysis function. Each statement is "shown", "opposite shown" or "inconclusive".
"""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.provenance import verify_run
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import HeldOutSettings
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.split_metrics import SCORES
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.comparison import (
    INCONCLUSIVE,
    OPPOSITE,
    SHOWN,
    margin_verdict,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.comparison import paired_summary
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

__all__ = ["INCONCLUSIVE", "OPPOSITE", "SHOWN", "statements", "score_summary", "run_compare"]


def superiority_verdict(interval: list[float]) -> str:
    """Lower end above 0: shown; upper end below 0: opposite shown; otherwise inconclusive."""
    low, high = interval
    if low > 0:
        return SHOWN
    if high < 0:
        return OPPOSITE
    return INCONCLUSIVE


def _auroc(metrics: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.array([m["scores"][name]["auroc"] for m in metrics], dtype=float)


def statements(metrics: list[dict[str, Any]], confidence: float, margin: float) -> dict[str, Any]:
    """H1 to H4 from the per-split metrics (in split order) and the summaries behind them."""
    qe = _auroc(metrics, "qe")
    summary = {other: paired_summary(qe, _auroc(metrics, other), confidence) for other in SCORES[1:]}
    failing = [m["split"] for m in metrics if not m["scores"]["qe"]["auroc_interval"][0] > 0.5]
    return {
        "H1_som_detects_held_out_digits": {"verdict": SHOWN if not failing else OPPOSITE,
                                           "splits_not_above_0_5": failing},
        "H2_som_better_than_max_softmax": {"verdict": superiority_verdict(summary["msp"]["interval"]),
                                           "primary": True},
        "H3_som_not_worse_by_margin": {other: margin_verdict(summary[other]["interval"], margin)
                                       for other in ("msp", "energy", "mahalanobis")},
        "H4_map_adds_to_nearest_neighbour": {"verdict": superiority_verdict(summary["knn"]["interval"])},
        "qe_minus_score_over_splits": summary,
        "margin": margin,
    }


def score_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean, SD and per-split AUROC of every score, plus the controls and descriptive numbers."""
    out: dict[str, Any] = {}
    for name in (*SCORES, "length"):
        values = _auroc(metrics, name)
        out[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "per_split": values.tolist()}
    out["closed_set_accuracy_known"] = [m["closed_set_accuracy_known"] for m in metrics]
    out["qe_length_spearman"] = [m["qe_length_spearman"] for m in metrics]
    out["qe_label_shuffle_mean"] = [m["qe_label_shuffle_control"]["shuffled_mean"] for m in metrics]
    out["competitive_rq4_splits"] = int(sum(bool(m["competitive_rq4"]) for m in metrics))
    out["cells_holding_share"] = [m["cells_holding_share"] for m in metrics]
    return out


def run_compare(settings: HeldOutSettings) -> dict[str, Any]:
    """Write results.json and the tables; stops unless every split is done and the inputs are unchanged."""
    results_path = settings.output_dir / "results.json"
    ensure_not_done(results_path)
    verify_run(settings)
    missing = [s for s in settings.split_ids if not settings.metrics_path(s).is_file()]
    if missing:
        raise FileNotFoundError(f"Splits not done yet: {missing}")
    metrics = [json.loads(settings.metrics_path(s).read_text(encoding="utf-8")) for s in settings.split_ids]
    results = {"run_id": settings.run_id, "unknown_digits": [list(u) for u in settings.unknown_digits],
               "statements": statements(metrics, settings.confidence, settings.auroc_margin),
               "scores": score_summary(metrics), "mnist_test_read": False}
    write_json(results_path, results)
    _write_tables(settings, metrics)
    return results


def _write_tables(settings: HeldOutSettings, metrics: list[dict[str, Any]]) -> None:
    tables = settings.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "split_scores.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["split", "unknown_digits", "score", "auroc", "auroc_low", "auroc_high", "aupr_ood",
                    "fpr_at_95_tpr"])
        for m in metrics:
            for name in (*SCORES, "length"):
                s = m["scores"][name]
                w.writerow([m["split"], " ".join(map(str, m["unknown_digits"])), name, round(s["auroc"], 6),
                            round(s["auroc_interval"][0], 6), round(s["auroc_interval"][1], 6),
                            round(s["aupr_ood"], 6), round(s["fpr_at_target_tpr"], 6)])
    with (tables / "qe_minus_scores.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["split", "score", "qe_minus_score_auroc", "ci_low", "ci_high"])
        for m in metrics:
            for name, d in m["qe_minus"].items():
                w.writerow([m["split"], name, round(d["difference"], 6), round(d["interval"][0], 6),
                            round(d["interval"][1], 6)])
