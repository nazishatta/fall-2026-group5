"""Hotspot concentration for any partition of the validation images, with the project's own rule.

The statistics are computed by the project's error pattern analysis (analyse), so a SOM and a
k-means partition are judged by exactly the same code, rule, splits and shuffled-error baseline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.error_pattern_robustness import (
    RobustnessSettings,
    analyse,
    stratified_halves,
)
from src.v1_mnist.component.utils.hotspots import n_hotspots, top_hotspots

REPRODUCTION_TOLERANCE = 1e-9


def hotspot_shares(
    cells: np.ndarray, errors: np.ndarray, num_cells: int, cfg: RobustnessSettings
) -> dict[str, float]:
    """Mean share of images and of errors that fall in the hotspot cells (held-out halves).

    The held-out ratio is errors' share divided by images' share, so smaller hotspots give larger ratios.
    """
    err = errors.astype(float)
    k = n_hotspots(num_cells, cfg.hotspot_fraction, cfg.min_hotspots)
    image_share: list[float] = []
    error_share: list[float] = []
    for rep in range(cfg.n_splits):
        half_a, half_b = stratified_halves(errors, rep, cfg.split_seed_base)
        for fit, test in ((half_a, half_b), (half_b, half_a)):
            hot = top_hotspots(cells[fit], err[fit], num_cells, k)
            inside = np.isin(cells[test], hot)
            image_share.append(float(inside.mean()))
            error_share.append(float(err[test][inside].sum() / err[test].sum()))
    return {"mean_image_share_in_hotspots": float(np.mean(image_share)),
            "mean_error_share_in_hotspots": float(np.mean(error_share))}


def analyse_partition(
    cells: np.ndarray, errors: np.ndarray, labels: np.ndarray, num_cells: int, num_classes: int,
    cfg: RobustnessSettings,
) -> dict[str, Any]:
    """The project's hotspot and overlap statistics plus the hotspot size shares."""
    result = analyse(cells, errors, labels, num_cells, num_classes, cfg)
    result["hotspot_shares"] = hotspot_shares(cells, errors, num_cells, cfg)
    return result


def check_reproduction(som_result: Mapping[str, Any], reference_path: Path) -> dict[str, Any]:
    """Compare our SOM numbers with the project's own result file; they must agree exactly."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ours = som_result["hotspot_concentration_held_out"]
    theirs = reference["hotspot_concentration_held_out"]
    pairs = [
        (ours["median_ratio"], theirs["median_ratio"]),
        (ours["chance_median_ratio"], theirs["chance_median_ratio"]),
        (som_result["overlap_regions"]["spearman_error_rate_vs_purity"],
         reference["overlap_regions"]["spearman_error_rate_vs_purity"]),
    ]
    exact = all(abs(a - b) < REPRODUCTION_TOLERANCE for a, b in pairs)
    return {
        "reference_file": reference_path.name,
        "reference_median_ratio": theirs["median_ratio"],
        "this_run_median_ratio": ours["median_ratio"],
        "reproduced_exactly": bool(exact),
    }


def summarize_seeds(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Median, range and mean hotspot size over the k-means seeds."""
    ratios = [r["hotspot_concentration_held_out"]["median_ratio"] for r in per_seed.values()]
    chance = [r["hotspot_concentration_held_out"]["chance_median_ratio"] for r in per_seed.values()]
    p_values = [r["hotspot_concentration_held_out"]["p_value_vs_chance"] for r in per_seed.values()]
    shares = [r["hotspot_shares"] for r in per_seed.values()]
    return {
        "median_ratio_over_seeds": float(np.median(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "mean_chance_median_ratio": float(np.mean(chance)),
        "max_p_value_vs_chance": float(np.max(p_values)),
        "mean_image_share_in_hotspots": float(np.mean([s["mean_image_share_in_hotspots"] for s in shares])),
        "mean_error_share_in_hotspots": float(np.mean([s["mean_error_share_in_hotspots"] for s in shares])),
    }
