"""Validation-only summary of an experiment: mean and spread over seeds, and arm-against-arm differences.

Used to choose nothing on the test split: this is what the macro-F1 guard is measured from.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np

from src.v1_mnist.component.utils.retraining_settings import ArmSpec, ExperimentSettings
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json

METRICS = {
    "overall_error": lambda v: v["overall"]["error_rate"],
    "hotspot_error": lambda v: v["hotspot"]["error_rate"],
    "non_hotspot_error": lambda v: v["non_hotspot"]["error_rate"],
    "macro_f1": lambda v: v["macro_f1"],
}


def load_run_record(settings: ExperimentSettings, arm: str, seed: int) -> dict[str, Any]:
    """The saved record of one finished run."""
    path = settings.metrics_path(arm, seed)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the {arm} run for seed {seed} is not finished.")
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_pairs(arms: tuple[ArmSpec, ...]) -> list[tuple[str, str]]:
    """(guided arm, control arm) pairs.

    Every arm against standard; SOM-guided against its controls (random, max-softmax; hotspot against
    random_hotspot) among arms that share the same pacing; a paced arm against the same arm unpaced.
    """
    names = {a.name for a in arms}
    by_key = {(a.kind, a.pool, a.schedule): a.name for a in arms}
    pairs: list[tuple[str, str]] = []
    if "standard" in names:
        pairs += [(a.name, "standard") for a in arms if a.kind != "standard"]
    groups = {(a.kind, a.schedule) for a in arms if a.kind != "standard"}
    for kind, schedule in sorted(groups):
        for guided, control in (("som", "random"), ("som", "msp"), ("hotspot", "random_hotspot")):
            if (kind, guided, schedule) in by_key and (kind, control, schedule) in by_key:
                pairs.append((by_key[(kind, guided, schedule)], by_key[(kind, control, schedule)]))
    for kind, pool, schedule in sorted(by_key):
        if schedule and (kind, pool, "") in by_key:
            pairs.append((by_key[(kind, pool, schedule)], by_key[(kind, pool, "")]))
    return pairs


def per_seed_values(settings: ExperimentSettings, arm: str) -> dict[str, np.ndarray]:
    """Each validation metric of one arm, one value per seed (in seed order)."""
    records = [load_run_record(settings, arm, s)["validation"] for s in settings.seeds]
    return {name: np.array([get(r) for r in records]) for name, get in METRICS.items()}


def summarize_validation(settings: ExperimentSettings) -> dict[str, Any]:
    """Mean and standard deviation over seeds per arm, plus paired differences between arms."""
    values = {a.name: per_seed_values(settings, a.name) for a in settings.arms}
    arms = {
        name: {m: {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "per_seed": v.tolist()}
               for m, v in metrics.items()}
        for name, metrics in values.items()
    }
    differences = {}
    for guided, control in comparison_pairs(settings.arms):
        diffs = {m: values[guided][m] - values[control][m] for m in METRICS}
        differences[f"{guided} minus {control}"] = {
            m: {"mean": float(d.mean()), "std": float(d.std(ddof=1)), "seeds_below_zero": int((d < 0).sum())}
            for m, d in diffs.items()
        }
    summary: dict[str, Any] = {
        "run_id": settings.run_id, "split": "validation", "seeds": list(settings.seeds),
        "arms": arms, "differences": differences,
    }
    if "standard" in arms:
        summary["macro_f1_guard_suggestion"] = 2.0 * arms["standard"]["macro_f1"]["std"]
    return summary


def write_validation_summary(settings: ExperimentSettings) -> dict[str, Any]:
    """Write validation_summary.json and .csv once (never overwritten)."""
    json_path = settings.tables_dir / "validation_summary.json"
    ensure_not_done(json_path)
    summary = summarize_validation(settings)
    write_json(json_path, summary)
    header = ["arm"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    with (settings.tables_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for name, metrics in summary["arms"].items():
            writer.writerow([name] + [round(metrics[m][s], 6) for m in METRICS for s in ("mean", "std")])
    return summary
