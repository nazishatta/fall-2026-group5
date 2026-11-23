"""The pre-registered held-out endpoint, the map metrics and the decision rule (sections 6 and 7).

Ported unchanged in its maths from the original ``evaluate.py`` and ``analyze.py``.
"""

from __future__ import annotations

import functools

import numpy as np
from NNSOM.plots import SOMPlots
from sklearn.metrics import average_precision_score, roc_auc_score

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    Contrast,
)


@functools.lru_cache(maxsize=None)
def neuron_dist(grid: int) -> np.ndarray:
    """Grid distances between neurons of a square map, as NNSOM computes them."""
    return np.asarray(SOMPlots(dimensions=(grid, grid)).neuron_dist)


def bmus(weights: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First and second best-matching neurons and the distance to the first (``xs`` already scaled)."""
    score = (weights * weights).sum(1)[:, None] - 2.0 * (weights @ xs.T)
    order = np.argpartition(score, 1, axis=0)[:2]
    top = np.take_along_axis(score, order, axis=0)
    swap = top[0] > top[1]
    first = np.where(swap, order[1], order[0])
    second = np.where(swap, order[0], order[1])
    dist = np.sqrt(np.maximum(score[first, np.arange(xs.shape[0])] + (xs * xs).sum(1), 0))
    return first, second, dist


def map_metrics(weights: np.ndarray, grid: int, xs: np.ndarray, labels: np.ndarray,
                num_classes: int) -> dict[str, float]:
    """Descriptive measures of one map on the validation set (section 7)."""
    nd = neuron_dist(grid)
    num_neurons = weights.shape[0]
    first, second, dist = bmus(weights, xs)
    hits = np.bincount(first, minlength=num_neurons)
    occupied = hits > 0
    per_neuron = np.bincount(first, weights=dist, minlength=num_neurons)
    counts = np.zeros((num_neurons, num_classes))
    np.add.at(counts, (first, labels), 1)
    majority = counts.argmax(1)
    ii, jj = np.where(np.triu(nd == 1))
    both = occupied[ii] & occupied[jj]
    topo = nd[first, second]
    return {
        "qe_nnsom": float(np.where(occupied, per_neuron / np.maximum(hits, 1), 0).mean()),
        "qe_per_image": float(dist.mean()),
        "te1": float(100 * np.mean(topo > 1.1)),
        "te12": float(100 * np.mean(topo > 2.1)),
        "occupancy": float(occupied.mean()),
        "purity": float(100 * counts.max(1)[occupied].sum() / hits.sum()),
        "neighbour_disagreement": float(100 * np.mean(majority[ii][both] != majority[jj][both])),
    }


def cell_rate_scores(cells_fit: np.ndarray, y_fit: np.ndarray, cells_score: np.ndarray,
                     num_neurons: int) -> np.ndarray:
    """Error rate per cell from the fit half; empty cells get the fit half's overall rate."""
    n = np.bincount(cells_fit, minlength=num_neurons).astype(float)
    k = np.bincount(cells_fit, weights=y_fit.astype(float), minlength=num_neurons)
    rate = np.where(n > 0, k / np.maximum(n, 1), y_fit.mean())
    return rate[cells_score]


def crossfit(cells: np.ndarray, y: np.ndarray, half_a: np.ndarray, half_b: np.ndarray,
             num_neurons: int) -> tuple[float, float]:
    """Mean of the A-to-B and B-to-A AUROC and average precision."""
    out = []
    for fit, score in ((half_a, half_b), (half_b, half_a)):
        s = cell_rate_scores(cells[fit], y[fit], cells[score], num_neurons)
        out.append((roc_auc_score(y[score], s), average_precision_score(y[score], s)))
    mean = np.mean(out, 0)
    return float(mean[0]), float(mean[1])


def split_indices(err: np.ndarray, rep: int, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
    """Repeat ``rep``: halves stratified by CNN error, then a bootstrap resample within each half."""
    rng = np.random.default_rng(seed_base + rep)
    halves: list[list[np.ndarray]] = [[], []]
    for value in (True, False):
        idx = rng.permutation(np.flatnonzero(err == value))
        h = len(idx) // 2
        halves[0].append(idx[:h])
        halves[1].append(idx[h:])
    a, b = np.concatenate(halves[0]), np.concatenate(halves[1])
    if np.intersect1d(a, b).size:
        raise RuntimeError(f"Repeat {rep}: the two halves share images.")
    return rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)


def verdict(low: float, high: float, margin: float, big: int, small: int) -> str:
    """The pre-registered decision rule, applied in this order (section 6)."""
    if -margin <= low and high <= margin:
        return "practically equivalent"
    if low > 0:
        return f"{big}x{big} better"
    if high < 0:
        return f"{small}x{small} better"
    return "inconclusive"


def summarise(rows: list[dict[str, float]], contrast: Contrast, margin: float) -> dict[str, object]:
    """Mean difference over the repeats, its 2.5-97.5 percentile range and the verdict."""
    delta = np.array([r[f"{contrast.metric}_{contrast.big}"] - r[f"{contrast.metric}_{contrast.small}"]
                      for r in rows])
    low, high = np.percentile(delta, [2.5, 97.5])
    return {
        "contrast": f"{contrast.big}x{contrast.big} minus {contrast.small}x{contrast.small}",
        "metric": contrast.metric,
        "mean_delta": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_reps_big_better": float(np.mean(delta > 0)),
        "verdict": verdict(float(low), float(high), margin, contrast.big, contrast.small),
    }
