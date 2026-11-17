"""Uncertainty for the arm-against-arm comparison.

Arms are trained with the same seeds, so each image has one error value per seed and arm. The
comparison averages over seeds first, then bootstraps over images (paired: the same resampled
images are used for both arms).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def seed_average_error(errors_by_seed: np.ndarray) -> np.ndarray:
    """Mean error per image over seeds. Input shape: seeds x images (0/1 values)."""
    if errors_by_seed.ndim != 2:
        raise ValueError("Expected an array of shape (seeds, images).")
    return errors_by_seed.mean(axis=0)


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    mask: np.ndarray,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Mean of (a - b) over the masked images with a percentile bootstrap interval.

    Negative values mean arm ``a`` makes fewer mistakes than arm ``b``.
    """
    diff = (errors_a - errors_b)[mask]
    if diff.size == 0:
        raise ValueError("The mask selects no images.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "images": int(diff.size),
        "mean_difference": float(diff.mean()),
        "ci_95": [float(low), float(high)],
        "excludes_zero": bool(low > 0 or high < 0),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def seed_spread(values: np.ndarray) -> dict[str, float]:
    """Mean and sample standard deviation over seeds of one number (for example macro-F1)."""
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {"mean": float(values.mean()), "std": std, "n": int(values.size)}
