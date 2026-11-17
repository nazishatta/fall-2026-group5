"""Map measures of the SOM re-analysis (protocol section 4). NumPy only, validation images only.

Every function takes the neuron assignment (``bmu``) of the validation images, their labels and the
retrained CNN's predictions, so the same code measures every cell.
"""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.utils.hotspots import n_hotspots, top_hotspots


def neuron_majority(bmu: np.ndarray, labels: np.ndarray, num_neurons: int, num_classes: int) -> np.ndarray:
    """Counts matrix (neurons x classes) of true labels per neuron."""
    counts = np.zeros((num_neurons, num_classes), dtype=np.int64)
    np.add.at(counts, (bmu, labels), 1)
    return counts


def weighted_purity(counts: np.ndarray) -> float:
    """M1: share of images whose class is the majority class of their neuron."""
    total = counts.sum()
    if total == 0:
        raise ValueError("No images.")
    return float(counts.max(axis=1).sum() / total)


def low_purity_share(counts: np.ndarray, cutoff: float) -> float:
    """M2: share of images that sit in neurons with purity below ``cutoff``."""
    size = counts.sum(axis=1)
    occupied = size > 0
    purity = np.zeros(len(size))
    purity[occupied] = counts.max(axis=1)[occupied] / size[occupied]
    return float(size[occupied & (purity < cutoff)].sum() / size.sum())


def class_scatter_index(counts: np.ndarray, neuron_dist: np.ndarray, adjacency: float) -> float:
    """M3: average number of separate groups of neighbouring neurons per class.

    Each occupied neuron is labelled with its majority class. Two neurons of the same class belong to
    one group if they are neighbours (map distance below ``adjacency``) directly or through a chain of
    same-class neurons. Classes that own no neuron are left out of the average.
    """
    occupied = counts.sum(axis=1) > 0
    owner = np.where(occupied, counts.argmax(axis=1), -1)
    groups: list[int] = []
    for cls in np.unique(owner[owner >= 0]):
        members = np.flatnonzero(owner == cls)
        adj = neuron_dist[np.ix_(members, members)] < adjacency
        seen = np.zeros(len(members), dtype=bool)
        n_groups = 0
        for start in range(len(members)):
            if seen[start]:
                continue
            n_groups += 1
            stack = [start]
            seen[start] = True
            while stack:
                node = stack.pop()
                for nxt in np.flatnonzero(adj[node] & ~seen):
                    seen[nxt] = True
                    stack.append(int(nxt))
        groups.append(n_groups)
    return float(np.mean(groups))


def hotspot_measures(
    bmu: np.ndarray, error: np.ndarray, num_neurons: int, fraction: float, minimum: int
) -> dict[str, float | int | list[int]]:
    """M5, M6, M7: error rate inside the hotspot neurons, their share of all errors, overall error.

    The hotspots are picked with the Phase 2 rule on this map (top k by the Wilson lower bound).
    """
    err = np.asarray(error, dtype=float)
    k = n_hotspots(num_neurons, fraction, minimum)
    spots = top_hotspots(bmu, err, num_neurons, k)
    inside = np.isin(bmu, spots)
    n_inside = int(inside.sum())
    hotspot_rate = float(err[inside].mean()) if n_inside else float("nan")
    total_errors = float(err.sum())
    share = float(err[inside].sum() / total_errors) if total_errors > 0 else float("nan")
    return {
        "hotspot_error_rate": hotspot_rate,
        "hotspot_error_share": share,
        "overall_error": float(err.mean()),
        "hotspot_images": n_inside,
        "hotspots": [int(s) for s in spots],
    }


def all_measures(
    bmu: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    neuron_dist: np.ndarray,
    num_classes: int,
    purity_cutoff: float,
    adjacency: float,
    fraction: float,
    minimum: int,
) -> dict[str, float | int | list[int]]:
    """M1, M2, M3, M5, M6, M7 for one map (M4 comes from the SOM itself, M8 from cka.py)."""
    num_neurons = neuron_dist.shape[0]
    counts = neuron_majority(bmu, labels, num_neurons, num_classes)
    out: dict[str, float | int | list[int]] = {
        "weighted_purity": weighted_purity(counts),
        "low_purity_share": low_purity_share(counts, purity_cutoff),
        "class_scatter_index": class_scatter_index(counts, neuron_dist, adjacency),
        "occupied_neurons": int((counts.sum(axis=1) > 0).sum()),
    }
    out.update(hotspot_measures(bmu, (predictions != labels).astype(float), num_neurons, fraction, minimum))
    return out
