"""The results figure: held-out AUROC per grid for both targets, and the primary difference over repeats.

Ported from the original ``figure.py``; saved as PDF only (project rule), never PNG or SVG.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (  # noqa: E402
    GridSettings,
)
from src.v1_mnist.component.utils.figures import save_pdf  # noqa: E402

log = logging.getLogger("v1_mnist.grid_comparison")
INK, MUTED, ACCENT, BAND = "#222222", "#6b6b6b", "#2a6fb0", "#e8eef6"


def run_figure(settings: GridSettings) -> Path:
    """Draw the figure from results.json and the repeats table."""
    if settings.figure_path.exists():
        raise FileExistsError(f"{settings.figure_path} already exists; results are never overwritten.")
    res = json.loads(settings.results_path.read_text(encoding="utf-8"))
    with (settings.tables_dir / "endpoint_reps.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n_err, n_low = res["n_errors"], res["n_lowconf"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    for ax, metric, title in ((axes[0], "auroc_err", f"Predicting CNN mistakes ({n_err} errors)\nprimary measure"),
                              (axes[1], "auroc_lowconf", f"Predicting low-confidence images ({n_low})\nsecondary measure")):
        for i, grid in enumerate(settings.grids):
            p = res["per_grid"][str(grid)][metric]
            ax.plot([i, i], [p["p2_5"], p["p97_5"]], color=ACCENT, lw=2, solid_capstyle="round")
            ax.plot(i, p["mean"], "o", ms=9, color=ACCENT, mec="white", mew=2)
            ax.text(i + 0.12, p["mean"], f"{p['mean']:.3f}", va="center", fontsize=10, color=INK)
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.text(len(settings.grids) - 0.55, 0.5, "chance", fontsize=8.5, color=MUTED, va="bottom", ha="right")
        ax.set_xticks(range(len(settings.grids)), [f"{g}×{g}" for g in settings.grids])
        ax.set_xlim(-0.5, len(settings.grids) - 0.5)
        ax.set_ylabel("held-out AUROC (mean, 95% range)", color=INK)
        ax.set_title(title, fontsize=11, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
    ax = axes[2]
    c = settings.primary
    delta = np.array([float(r[f"{c.metric}_{c.big}"]) - float(r[f"{c.metric}_{c.small}"]) for r in rows])
    p = res["primary"]
    ax.axvspan(-res["margin"], res["margin"], color=BAND, zorder=0)
    ax.hist(delta, bins=30, color=ACCENT, alpha=0.85, edgecolor="white")
    for x in (p["ci_low"], p["ci_high"]):
        ax.axvline(x, color=INK, lw=1.2, ls="--")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel(f"AUROC difference: {c.big}×{c.big} minus {c.small}×{c.small} ({len(rows)} repeats)",
                  color=INK)
    ax.set_ylabel("number of repeats", color=INK)
    ax.set_title(f"Mean difference {p['mean_delta']:+.3f}, 95% interval [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]\n"
                 f"shaded = ±{res['margin']} 'too small to matter' zone → verdict: {p['verdict']}",
                 fontsize=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle("Pre-registered grid comparison: does a finer map predict where the CNN fails, on images it has "
                 "not seen?", fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    path = save_pdf(fig, settings.figure_path)
    log.info("Saved %s", path)
    return path
