"""Fast batch-SOM trainer: the algorithm of ``train_som_with_history`` with two speed-ups.

Ported unchanged in its maths from the grid comparison's original ``fast_som.py`` (pre-registration,
section 3). The speed-ups:
  1. No dense (neurons x samples) one-hot or neighbourhood matrices: per-neuron sums are built with a
     sparse one-hot matrix, then smoothed with the (neurons x neurons) neighbourhood matrix.
  2. Distances via ||w||^2 + ||x||^2 - 2 w.x (one matrix product) instead of ``cdist``.

``rng_mode``:
  "exact": draws the same (neurons x samples) random matrix as the project trainer, so a seed
           reproduces the project trainer's map.
  "fast":  draws one random number per sample; the same 90% winner-retention rule with a different
           random stream (used for the bootstrap maps, as pre-registered).
The original's GPU branch was never used (the pre-registered runs ran on CPU) and is not ported.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable

import numpy as np
import scipy.sparse as sparse
from NNSOM.plots import SOMPlots

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.memory_safe_init import (
    economy_svd_for_nnsom_init,
)

RNG_MODES = ("exact", "fast")
RETENTION = 0.90   # NNSOM's stochastic winner retention


def winners(weights: np.ndarray, x: np.ndarray, x_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-matching neuron of each sample and its Euclidean distance.

    ||x||^2 is the same for every neuron, so it is added back for the winners only.
    """
    score = (weights * weights).sum(1)[:, None] - 2.0 * (weights @ x.T)
    win = score.argmin(0)
    dist = np.sqrt(np.maximum(score[win, np.arange(x.shape[0])] + x_sq, 0.0))
    return win, dist


def nnsom_qe(dist: np.ndarray, win: np.ndarray, num_neurons: int) -> float:
    """NNSOM's quantization error: the mean over neurons of each neuron's mean distance (0 if empty)."""
    hits = np.bincount(win, minlength=num_neurons)
    total = np.bincount(win, weights=dist, minlength=num_neurons)
    return float(np.where(hits > 0, total / np.maximum(hits, 1), 0.0).mean())


def train_som_fast(
    x_train: np.ndarray,
    grid_height: int,
    grid_width: int,
    init_neighborhood: int,
    epochs: int,
    steps: int,
    norm_func: Callable[[np.ndarray], np.ndarray],
    seed: int,
    rng_mode: str = "exact",
    record_history: bool = True,
) -> tuple[Any, dict[str, list[float]]]:
    """Train one map; return the NNSOM object (weights set) and the per-epoch QE history."""
    if rng_mode not in RNG_MODES:
        raise ValueError(f"rng_mode must be one of {RNG_MODES}, got {rng_mode!r}.")
    np.random.seed(seed)
    som = SOMPlots(dimensions=(grid_height, grid_width))
    with economy_svd_for_nnsom_init(), contextlib.redirect_stdout(io.StringIO()):
        som.init_w(x_train, norm_func=norm_func)
    x = np.ascontiguousarray(som.normalize(x_train, norm_func=norm_func).T, dtype=np.float64)  # (S, F)

    weights = np.asarray(som.w, dtype=np.float64)
    neuron_dist = np.asarray(som.neuron_dist)
    num_neurons, num_samples = weights.shape[0], x.shape[0]
    eye = np.eye(num_neurons)
    x_sq = (x * x).sum(1)
    samples = np.arange(num_samples)

    win, dist = winners(weights, x, x_sq)
    history: dict[str, list[float]] = {"quantization_error": [], "neighborhood_radius": []}
    if record_history:
        history["quantization_error"].append(nnsom_qe(dist, win, num_neurons))
        history["neighborhood_radius"].append(float(init_neighborhood))

    for step in range(epochs):
        radius = 1 + (init_neighborhood - 1) * (1 - step / steps)
        neighbourhood = (neuron_dist <= radius).astype(np.float64) + eye
        if rng_mode == "exact":
            keep = np.random.rand(num_neurons, num_samples)[win, samples] < RETENTION
        else:
            keep = np.random.rand(num_samples) < RETENTION
        kept_winners = win[keep]
        counts = np.bincount(kept_winners, minlength=num_neurons).astype(np.float64)
        kept = np.nonzero(keep)[0]
        onehot = sparse.csr_matrix((np.ones(kept.size), (kept_winners, kept)), shape=(num_neurons, num_samples))
        sums = np.asarray(onehot @ x)
        denominator = neighbourhood @ counts
        numerator = neighbourhood @ sums
        losers = denominator == 0
        updated = numerator / np.where(losers, 1.0, denominator)[:, None]
        weights = np.where(losers[:, None], weights, updated)

        win, dist = winners(weights, x, x_sq)
        if record_history:
            history["quantization_error"].append(nnsom_qe(dist, win, num_neurons))
            history["neighborhood_radius"].append(float(radius))

    som.w = weights
    som.norm_func = norm_func
    return som, history
