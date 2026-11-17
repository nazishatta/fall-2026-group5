"""Same-budget comparison: how many CNN mistakes each per-image score catches when it flags the same
share of validation images as the SOM's (or k-means') hotspot cells. NumPy only.

Ties at the cut are shared out by their expected value (as if broken at random), so a score with many
equal values (kNN impurity) is not helped or hurt by the order of the images.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def caught_at_share(score: np.ndarray, errors: np.ndarray, share: float) -> dict[str, float]:
    """Flag the ``share`` of images with the highest scores; return the share of all mistakes they hold.

    ``ratio`` is that share of mistakes divided by the share of images (1 = chance), the same quantity as
    the hotspot concentration ratio of the SOM cells.
    """
    score = np.asarray(score, dtype=float)
    err = np.asarray(errors, dtype=float)
    if score.shape != err.shape or score.ndim != 1 or not 0 < share < 1:
        raise ValueError("score and errors must be equally long vectors and share must be in (0, 1).")
    total = err.sum()
    if total == 0:
        raise ValueError("There are no mistakes to catch.")
    budget = share * score.size
    cut = np.sort(score)[::-1][min(int(np.ceil(budget)) - 1, score.size - 1)]   # score of the last flagged image
    above, tied = score > cut, score == cut
    used = float(above.sum())
    # Expected mistakes among the tied images that are still flagged (ties broken at random).
    caught = err[above].sum() + max(budget - used, 0.0) * err[tied].mean()
    return {"image_share": float(share), "error_share": float(caught / total),
            "ratio": float(caught / total / share), "images_flagged": float(budget)}


def bootstrap_caught(scores: Mapping[str, np.ndarray], errors: np.ndarray, share: float, resamples: int,
                     seed: int) -> dict[str, dict[str, float]]:
    """Share of mistakes caught at ``share``, with a 95% bootstrap interval over images (same resamples for all)."""
    err = np.asarray(errors, dtype=bool)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in scores}
    for _ in range(resamples):
        idx = rng.integers(0, err.size, err.size)
        if not err[idx].any():
            continue
        for name, score in scores.items():
            draws[name].append(caught_at_share(np.asarray(score)[idx], err[idx], share)["error_share"])
    out: dict[str, dict[str, float]] = {}
    for name, score in scores.items():
        point = caught_at_share(score, err, share)
        low, high = np.percentile(draws[name], [2.5, 97.5])
        out[name] = {**point, "error_share_low": float(low), "error_share_high": float(high)}
    return out
