"""Training-stability monitoring: how validation accuracy moves from epoch to epoch.

Reports (no pass/fail criterion): the largest single-epoch fall in validation accuracy, the same for
the images outside the hotspots (a fall there means the easy images are being forgotten), and how far
the last epoch ends below the best one. Validation images only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings
from src.v1_mnist.component.utils.retraining_summary import load_run_record

LOGGER = logging.getLogger("v1_mnist")

CURVES = (
    ("val_accuracy", "All validation images"),
    ("val_hotspot_accuracy", "Hotspot images"),
    ("val_non_hotspot_accuracy", "Images outside the hotspots"),
)


def _curve(history: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([h[key] for h in history], dtype=float)


def largest_drop(curve: np.ndarray) -> float:
    """The biggest fall from one epoch to the next (0 when accuracy never fell)."""
    if len(curve) < 2:
        return 0.0
    return float(max(0.0, -np.diff(curve).min()))


def run_stability(history: list[dict[str, Any]]) -> dict[str, float]:
    """Stability numbers of one run."""
    overall, easy = _curve(history, "val_accuracy"), _curve(history, "val_non_hotspot_accuracy")
    return {
        "largest_drop_overall": largest_drop(overall),
        "largest_drop_non_hotspot": largest_drop(easy),
        "last_epoch_below_best": float(overall.max() - overall[-1]),
        "non_hotspot_last_below_peak": float(easy.max() - easy[-1]),
    }


def stability_summary(settings: ExperimentSettings) -> dict[str, dict[str, dict[str, float]]]:
    """Per arm, the mean and standard deviation over seeds of each stability number."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for arm in settings.arms:
        runs = [run_stability(load_run_record(settings, arm.name, s)["history"]) for s in settings.seeds]
        summary[arm.name] = {
            key: {"mean": float(np.mean([r[key] for r in runs])),
                  "std": float(np.std([r[key] for r in runs], ddof=1)) if len(runs) > 1 else 0.0}
            for key in runs[0]
        }
    return summary


def plot_validation_curves(settings: ExperimentSettings, output_path: Path) -> Path:
    """Validation accuracy by epoch (mean over seeds) for every arm: all, hotspot, outside hotspots."""
    fig, axes = plt.subplots(1, len(CURVES), figsize=(5 * len(CURVES), 4), sharex=True)
    for ax, (key, title) in zip(axes, CURVES, strict=True):
        for arm in settings.arms:
            curves = [_curve(load_run_record(settings, arm.name, s)["history"], key) for s in settings.seeds]
            mean = np.mean(curves, axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, label=arm.name, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
    axes[-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    return save_pdf(fig, output_path)
