"""Assemble the JSON payload that the HTML page renders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import numpy as np

from src.v1_mnist.component.utils.logging import get_logger

from .neuron_stats import assign_bmus, neighbour_edges, split_summary
from .numeric_utils import clean_values, to_numpy
from .settings import TOPN

logger = get_logger("v1_mnist.interactive_report.report_payload")


def build_payload(
    som: SOMModel,
    model_name: str,
    grid: tuple[int, int],
    splits: dict,
    detail_split: str,
    top_features: list[int],
    feature_prefix: str,
    num_classes: int,
    max_detail_points: int,
    sample_seed: int,
) -> dict:
    """Assemble everything the HTML page needs.

    ``splits`` maps split name -> dict(features, labels, preds,
    sample_ids, confidence). Features are RAW embeddings; NNSOM's
    cluster_data applies som.norm_func itself. If the detail split has more
    than ``max_detail_points`` samples, that many are drawn at random with
    ``sample_seed`` (both from the config's visualization section).
    """

    weights = to_numpy(som.w).astype(np.float64)
    pos = to_numpy(som.pos).astype(np.float64)
    if pos.shape[0] != 2:
        pos = pos.T
    num_neurons, feature_dim = weights.shape

    payload = {
        "meta": {
            "model": model_name,
            "grid": [int(grid[0]), int(grid[1])],
            "num_neurons": int(num_neurons),
            "feature_dim": int(feature_dim),
            "feature_prefix": feature_prefix,
            "top_features": [int(f) for f in top_features],
            "detail_split": detail_split,
            "num_classes": int(num_classes),
            "topn": TOPN,
        },
        "pos": clean_values(pos, 5),
        "w": clean_values(weights, 4),
        "edges": neighbour_edges(som, weights),
        "splits": {},
        "points": None,
    }

    for name, s in splits.items():
        feats = np.asarray(s["features"], dtype=np.float64)
        logger.info("Clustering %s split (%d samples) ...", name, feats.shape[0])
        _clust, bmu, bmu_dist = assign_bmus(som, feats)

        # Show feature values in the SOM's own (MinMax-scaled, -1..1) input
        # space: same scale as the weights / component planes, and raw fc2
        # dims range from ~1e-4 to ~20, which makes a shared raw axis useless.
        # Map colours are unchanged (MinMax is affine per dimension).
        norm = getattr(som, "norm_func", None)
        scaled = to_numpy(norm(feats)).astype(np.float64) if callable(norm) else feats

        payload["splits"][name] = split_summary(
            num_neurons, bmu, scaled, s["labels"], s["preds"], num_classes
        )

        if name != detail_split:
            continue

        # Per-sample rows for the drill-down panel (NNSOM int_dict
        # 'data' / 'target' / 'num1' / 'num2' equivalent).
        n = feats.shape[0]
        keep = np.arange(n)
        if n > max_detail_points:
            rng = np.random.default_rng(sample_seed)
            keep = np.sort(rng.choice(n, max_detail_points, replace=False))

        payload["points"] = {
            "sampled_from": int(n),
            "bmu": bmu[keep].astype(int).tolist(),
            "dist": clean_values(bmu_dist[keep], 3),
            "label": np.asarray(s["labels"])[keep].astype(int).tolist(),
            "pred": np.asarray(s["preds"])[keep].astype(int).tolist(),
            "conf": clean_values(np.asarray(s["confidence"], dtype=np.float64)[keep], 3),
            "sample_id": np.asarray(s["sample_ids"])[keep].astype(int).tolist(),
            "feats": {
                str(f): clean_values(scaled[keep, f], sig=4) for f in top_features
            },
        }

    return payload
