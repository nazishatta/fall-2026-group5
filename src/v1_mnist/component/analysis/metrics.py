from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from scipy.stats import spearmanr

CLASS_COUNT = 10


def normalized_entropy(counts: np.ndarray | Sequence[int | float]) -> float:
    """Normalized Shannon entropy over the ten MNIST classes.

    Returns:
        Entropy normalized to [0, 1] by dividing by log(10), or NaN if total count is <= 0.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    total = float(counts_arr.sum())

    if total <= 0:
        return float("nan")

    probabilities = counts_arr[counts_arr > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    return entropy / math.log(CLASS_COUNT)


def wilson_lower_bound(
    errors: int,
    total: int,
    z: float = 1.959963984540054,
) -> float:
    """95% Wilson score lower bound for a binomial proportion.

    Args:
        errors: Number of error samples.
        total: Total number of samples.
        z: Normal distribution quantile (default ~1.96 for 95% CI).

    Returns:
        Lower bound float in [0, 1], or NaN if total is <= 0.
    """
    if total <= 0:
        return float("nan")

    p_hat = errors / total
    z2 = z * z

    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator

    half_width = (
        z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total))
        )
        / denominator
    )

    return max(0.0, center - half_width)


def safe_mean(values: np.ndarray | Sequence[float]) -> float:
    """Arithmetic mean or NaN for an empty array."""
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_median(values: np.ndarray | Sequence[float]) -> float:
    """Median or NaN for an empty array."""
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def finite_spearman(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> float | None:
    """Return exploratory Spearman rho or None if invalid / constant."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return None

    if np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return None

    result = spearmanr(x_arr[mask], y_arr[mask])
    rho = float(result.statistic)

    if not math.isfinite(rho):
        return None

    return rho
