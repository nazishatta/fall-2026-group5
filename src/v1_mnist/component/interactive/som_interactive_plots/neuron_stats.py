"""Per-neuron statistics for the interactive report.

Everything here is plain NumPy: it turns a trained SOM plus one data split
into the numbers behind each map (hit counts, class counts, error rates,
neighbour weight distances).
"""

from __future__ import annotations

import numpy as np

from .numeric_utils import clean_values, to_numpy
from .settings import NUM_CLASSES


def assign_bmus(som, features: np.ndarray):
    """Cluster the data with NNSOM and return per-sample BMU and distance.

    ``som.cluster_data`` returns one index list per neuron, sorted by
    distance to that neuron (same 'clust' NNSOM's int_dict uses).
    """

    clust, dist, _max_dist, _sizes = som.cluster_data(features)

    n = features.shape[0]
    bmu = np.full(n, -1, dtype=np.int64)
    bmu_dist = np.full(n, np.nan, dtype=np.float64)

    for neuron, (idx, d) in enumerate(zip(clust, dist)):
        idx = to_numpy(idx).astype(np.int64).ravel()
        d = to_numpy(d).astype(np.float64).ravel()
        if idx.size:
            bmu[idx] = neuron
            bmu_dist[idx] = d[: idx.size]

    return clust, bmu, bmu_dist


def split_summary(
    num_neurons: int,
    bmu: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict:
    """Per-neuron hit counts, class counts, feature means and error stats."""

    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    wrong = labels != preds

    hits = np.bincount(bmu, minlength=num_neurons)

    # int_dict 'cat' equivalent: class counts per neuron (neurons x 10)
    cat = np.zeros((num_neurons, NUM_CLASSES), dtype=np.int64)
    np.add.at(cat, (bmu, labels), 1)

    # Mean (scaled) fc2 value per neuron (feature_maps/*_color_hist)
    sums = np.zeros((num_neurons, features.shape[1]), dtype=np.float64)
    np.add.at(sums, bmu, features)
    with np.errstate(invalid="ignore", divide="ignore"):
        feat_mean = sums / hits[:, None]
    feat_mean[hits == 0] = np.nan

    errors = np.bincount(bmu, weights=wrong.astype(np.float64), minlength=num_neurons)

    with np.errstate(invalid="ignore", divide="ignore"):
        purity = cat.max(axis=1) / hits
        error_rate = errors / hits
    purity[hits == 0] = np.nan
    error_rate[hits == 0] = np.nan

    dominant = np.where(hits > 0, cat.argmax(axis=1), -1)

    # Dominant WRONG prediction per neuron (complex_hist edge label)
    wrong_pred_counts = np.zeros((num_neurons, NUM_CLASSES), dtype=np.int64)
    np.add.at(wrong_pred_counts, (bmu[wrong], preds[wrong]), 1)
    dominant_wrong = np.where(
        wrong_pred_counts.sum(axis=1) > 0, wrong_pred_counts.argmax(axis=1), -1
    )

    # Confusion pairs (true -> pred) of the misclassified samples per neuron
    confusion = [dict() for _ in range(num_neurons)]
    for b, t, p in zip(bmu[wrong], labels[wrong], preds[wrong]):
        key = f"{t}>{p}"
        confusion[b][key] = confusion[b].get(key, 0) + 1

    total_errors = max(int(wrong.sum()), 1)

    return {
        "n": int(features.shape[0]),
        "accuracy": float(1.0 - wrong.mean()) if features.shape[0] else None,
        "hits": hits.astype(int).tolist(),
        "cat": cat.astype(int).tolist(),
        "feat_mean": clean_values(feat_mean, sig=4),
        "errors": errors.astype(int).tolist(),
        "error_share": clean_values(errors / total_errors, 4),
        "purity": clean_values(purity, 4),
        "error_rate": clean_values(error_rate, 4),
        "dominant": dominant.astype(int).tolist(),
        "dominant_wrong": dominant_wrong.astype(int).tolist(),
        "confusion": [
            sorted(([k, v] for k, v in c.items()), key=lambda kv: -kv[1])
            for c in confusion
        ],
    }


def neighbour_edges(som, weights: np.ndarray):
    """Adjacent neuron pairs and the distance between their weight vectors.

    Same neighbour rule NNSOM's neuron_dist_plot uses
    (grid distance <= 1.001).
    """

    ndist = to_numpy(som.neuron_dist)
    ii, jj = np.where(np.triu(ndist <= 1.001, k=1))
    d = np.linalg.norm(weights[ii] - weights[jj], axis=1)
    return [[int(i), int(j), round(float(x), 5)] for i, j, x in zip(ii, jj, d)]
