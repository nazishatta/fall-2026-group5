from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from scipy.stats import spearmanr


def normalized_entropy(counts: np.ndarray | Sequence[int | float]) -> float:
    """Normalized Shannon entropy of one neuron's class counts, in [0, 1].

    Divided by log(number of classes), e.g. log(10) for ten classes.
    This is the project's only entropy function (Weeks 4 and 5 and the
    robustness check all use it).

    Returns:
        Normalized entropy; NaN if the total count is <= 0; 0.0 if there is
        only one class.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional.")

    total = float(counts_arr.sum())

    if total <= 0:
        return float("nan")

    if counts_arr.size <= 1:
        return 0.0

    probabilities = counts_arr[counts_arr > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    return entropy / math.log(counts_arr.size)


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


Z95 = 1.959963984540054


def wilson_interval(
    successes: int,
    total: int,
    z: float = Z95,
) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a binomial proportion.

    Same formula as wilson_lower_bound; used for headline rates such as the
    CNN's test accuracy (successes = correct images).

    Returns:
        (lower, upper) in [0, 1], or (NaN, NaN) if total is <= 0.
    """
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def wilson_lower_bound_array(errors: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Vectorized 95% Wilson lower bound (same formula as wilson_lower_bound).

    One value per neuron. Neurons with no images get -inf, so they always
    rank last when neurons are sorted by this bound.
    """
    n = np.maximum(total, 1).astype(float)
    p = errors / n
    z2 = Z95 * Z95
    den = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / den
    half = Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return np.where(total > 0, np.maximum(0.0, center - half), -np.inf)


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
