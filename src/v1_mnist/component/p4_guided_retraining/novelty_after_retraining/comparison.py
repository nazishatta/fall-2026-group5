"""Stage 2: compare the arms and decide statements N1 to N4 (protocol section 5).

Reads the per-cell metrics only. Between arms, differences are taken per seed with a t-interval
(the SOM re-analysis function, unchanged). Each statement is "shown", "opposite shown" or "inconclusive".
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.provenance import (
    verify_run,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.scoring import (
    DETECTORS,
    DIAGNOSTIC,
    competitive,
)
from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.settings import (
    NoveltySettings,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.comparison import (
    paired_summary,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    cell_name,
)
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

SHOWN, OPPOSITE, INCONCLUSIVE = "shown", "opposite shown", "inconclusive"
STANDARD, GUIDED, RANDOM = "standard", "augmented_som", "augmented_random"


def margin_verdict(interval: list[float], margin: float) -> str:
    """Non-inferiority with three outcomes: lower end above -margin, upper end below -margin, or neither."""
    low, high = interval
    if low > -margin:
        return SHOWN
    if high < -margin:
        return OPPOSITE
    return INCONCLUSIVE


def detection_verdict(intervals: dict[str, list[float]]) -> tuple[str, list[str]]:
    """N1 as written in the protocol: every interval above 0.5, otherwise the cells that are not."""
    failing = sorted(cell for cell, (low, _high) in intervals.items() if not low > 0.5)
    return (SHOWN if not failing else OPPOSITE), failing


def _per_seed(metrics: dict[str, dict[str, Any]], arm: str, seeds: tuple[int, ...],
              value: Callable[[dict[str, Any]], float]) -> np.ndarray:
    return np.array([value(metrics[cell_name(arm, s)]) for s in seeds], dtype=float)


def qe(m: dict[str, Any]) -> float:
    """S1 AUROC of one cell."""
    return float(m["scores"]["qe"]["auroc"])


def gap_msp(m: dict[str, Any]) -> float:
    """S1 minus S2 AUROC of one cell."""
    return float(m["scores"]["qe"]["auroc"] - m["scores"]["msp"]["auroc"])


def statements(metrics: dict[str, dict[str, Any]], settings: NoveltySettings) -> dict[str, Any]:
    """N1 to N4 and the paired summaries behind them."""
    seeds, margin, conf = settings.seeds, settings.auroc_margin, settings.confidence
    summaries = {
        "qe_auroc_som_vs_standard": paired_summary(_per_seed(metrics, GUIDED, seeds, qe),
                                                   _per_seed(metrics, STANDARD, seeds, qe), conf),
        "qe_auroc_som_vs_random": paired_summary(_per_seed(metrics, GUIDED, seeds, qe),
                                                 _per_seed(metrics, RANDOM, seeds, qe), conf),
        "qe_minus_msp_som_vs_standard": paired_summary(_per_seed(metrics, GUIDED, seeds, gap_msp),
                                                       _per_seed(metrics, STANDARD, seeds, gap_msp), conf),
    }
    n1, failing = detection_verdict({cell_name(a, s): metrics[cell_name(a, s)]["scores"]["qe"]["auroc_interval"]
                                     for a in (GUIDED, RANDOM) for s in seeds})
    return {
        "N1_som_still_detects_novelty": {"verdict": n1, "cells_not_above_0_5": failing},
        "N2_guided_retraining_does_not_harm": {
            "verdict": margin_verdict(summaries["qe_auroc_som_vs_standard"]["interval"], margin)},
        "N3_guidance_does_not_harm": {
            "verdict": margin_verdict(summaries["qe_auroc_som_vs_random"]["interval"], margin)},
        "N4_gap_to_max_softmax_does_not_widen": {
            "verdict": margin_verdict(summaries["qe_minus_msp_som_vs_standard"]["interval"], margin)},
        "margin": margin, "summaries": summaries,
    }


def arm_summary(metrics: dict[str, dict[str, Any]], settings: NoveltySettings) -> dict[str, Any]:
    """Mean and SD over seeds per arm and score, the competitive count and the length check."""
    out: dict[str, Any] = {}
    for arm in settings.arms:
        cells = [metrics[cell_name(arm, s)] for s in settings.seeds]
        out[arm] = {
            name: {"mean": float(np.mean([c["scores"][name]["auroc"] for c in cells])),
                   "std": float(np.std([c["scores"][name]["auroc"] for c in cells], ddof=1))}
            for name in (*DETECTORS, DIAGNOSTIC)
        }
        out[arm]["competitive_cells"] = int(sum(competitive(c["qe_minus"], c["scores"], settings.auroc_margin)
                                                for c in cells))
        out[arm]["qe_length_spearman_mean"] = float(np.mean([c["qe_length_spearman"] for c in cells]))
    return out


def run_compare(settings: NoveltySettings) -> dict[str, Any]:
    """Write results.json and the tables; stops unless every cell is scored and inputs are unchanged."""
    results_path = settings.output_dir / "results.json"
    ensure_not_done(results_path)
    verify_run(settings)
    names = (settings.reference, *settings.cells)
    missing = [c for c in names if not (settings.folder("metrics") / f"{c}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Cells not scored yet: {missing}")
    metrics = {c: json.loads((settings.folder("metrics") / f"{c}.json").read_text(encoding="utf-8")) for c in names}
    results = {"run_id": settings.run_id, "statements": statements(metrics, settings),
               "arms": arm_summary(metrics, settings), "reference": metrics[settings.reference]["scores"],
               f"{settings.reanalysis.retraining.dataset.name}_test_read": False}
    write_json(results_path, results)
    _write_tables(settings, metrics)
    return results


def _write_tables(settings: NoveltySettings, metrics: dict[str, dict[str, Any]]) -> None:
    tables = settings.folder("tables")
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "cell_scores.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["cell", "score", "auroc", "auroc_low", "auroc_high", "aupr_ood", "fpr_at_95_tpr",
                    "detection_rate", "false_alarm_rate"])
        for cell, m in metrics.items():
            for name in (*DETECTORS, DIAGNOSTIC):
                s = m["scores"][name]
                w.writerow([cell, name, round(s["auroc"], 6), round(s["auroc_interval"][0], 6),
                            round(s["auroc_interval"][1], 6), round(s["aupr_ood"], 6),
                            round(s["fpr_at_target_tpr"], 6), round(s["ood_detection_rate"], 6),
                            round(s["id_false_positive_rate"], 6)])
    with (tables / "qe_minus_detectors.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["cell", "detector", "qe_minus_detector_auroc", "ci_low", "ci_high"])
        for cell, m in metrics.items():
            for name, d in m["qe_minus"].items():
                w.writerow([cell, name, round(d["difference"], 6), round(d["interval"][0], 6),
                            round(d["interval"][1], 6)])
