"""Paired comparison of two arms over CNN seeds (protocol section 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

HIGHER_IS_BETTER = {"weighted_purity": True}


def paired_summary(a: np.ndarray, b: np.ndarray, confidence: float) -> dict[str, Any]:
    """Mean of (a - b) over seeds with a t-interval, plus how often a is larger than b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("Need two equally long vectors with at least 2 seeds.")
    diff = a - b
    n = diff.size
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * sd / np.sqrt(n))
    low, high = mean - half, mean + half
    return {
        "seeds": int(n),
        "mean_difference": mean,
        "std_difference": sd,
        "interval": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "share_a_larger": float(np.mean(diff > 0)),
        "share_a_smaller": float(np.mean(diff < 0)),
    }


def decide(measure: str, summary: dict[str, Any]) -> bool:
    """True if the first arm is better and the interval excludes zero."""
    mean = summary["mean_difference"]
    better = mean > 0 if HIGHER_IS_BETTER.get(measure, False) else mean < 0
    return bool(better and summary["excludes_zero"])


def statements(
    summaries: dict[str, dict[str, dict[str, Any]]], tolerance_pct: float, standard: str, random_arm: str,
    first: str,
) -> dict[str, dict[str, Any]]:
    """R1 to R5 of the protocol.

    ``summaries[f"{first}_vs_{other}"][measure]`` holds paired_summary output.
    """
    vs_std = summaries[f"{first}_vs_{standard}"]
    vs_rnd = summaries[f"{first}_vs_{random_arm}"]
    r5_upper = vs_std["topographic_error_pct"]["interval"][1]
    return {
        "R1_boundaries_sharpen_vs_standard": {"met": decide("weighted_purity", vs_std["weighted_purity"]),
                                              "measure": "weighted_purity"},
        "R2_hotspots_shrink_vs_standard": {"met": decide("hotspot_error_rate", vs_std["hotspot_error_rate"]),
                                           "measure": "hotspot_error_rate"},
        "R3_guidance_sharpens_vs_random": {"met": decide("weighted_purity", vs_rnd["weighted_purity"]),
                                           "measure": "weighted_purity"},
        "R4_guidance_shrinks_hotspots_vs_random": {"met": decide("hotspot_error_rate", vs_rnd["hotspot_error_rate"]),
                                                   "measure": "hotspot_error_rate"},
        "R5_map_not_less_organised_vs_standard": {"met": bool(r5_upper <= tolerance_pct),
                                                  "measure": "topographic_error_pct",
                                                  "upper_end": float(r5_upper), "tolerance": tolerance_pct},
    }
