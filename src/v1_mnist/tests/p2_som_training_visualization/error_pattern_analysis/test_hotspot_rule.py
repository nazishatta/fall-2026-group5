"""Week 5 (Error Pattern Analysis): the project's one hotspot rule (utils/hotspots.py)."""

import numpy as np

from NNSOM.utils import get_perc_misclassified

from src.v1_mnist.component.utils.hotspots import n_hotspots, top_hotspots
from src.v1_mnist.component.utils.stats import wilson_lower_bound, wilson_lower_bound_array
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.error_patterns import (
    build_error_rows,
)


def test_number_of_hotspots_per_grid() -> None:
    # k = max(5, round(5% of neurons)), as in ERROR_PATTERN_ROBUSTNESS_CRITERIA.md
    assert n_hotspots(100, 0.05, 5) == 5      # 10x10
    assert n_hotspots(225, 0.05, 5) == 11     # 15x15
    assert n_hotspots(400, 0.05, 5) == 20     # 20x20
    assert n_hotspots(16, 0.05, 5) == 5       # tiny grid: at least 5


def test_vectorized_wilson_matches_scalar() -> None:
    errors = np.array([0, 1, 3, 10, 0])
    total = np.array([10, 7, 20, 10, 0])
    vec = wilson_lower_bound_array(errors.astype(float), total)
    for e, t, v in zip(errors, total, vec):
        if t == 0:
            assert v == -np.inf          # empty neurons always rank last
        else:
            assert np.isclose(v, wilson_lower_bound(int(e), int(t)), rtol=1e-12, atol=1e-15)


def test_top_hotspots_ranking_and_ties() -> None:
    # Neuron 0: 2/2 wrong; neuron 1: 2/2 wrong (tie with 0 -> lower index first);
    # neuron 2: 1/10 wrong; neuron 3: empty (never a hotspot); neuron 4: 0/5 wrong.
    bmu = np.array([0, 0, 1, 1] + [2] * 10 + [4] * 5)
    err = np.array([1, 1, 1, 1] + [1] + [0] * 9 + [0] * 5, dtype=float)
    assert top_hotspots(bmu, err, num_neurons=5, k=3).tolist() == [0, 1, 2]
    assert 3 not in top_hotspots(bmu, err, num_neurons=5, k=5).tolist()


def test_error_rate_matches_nnsom_get_perc_misclassified() -> None:
    """Week 5 error rates equal NNSOM's own per-neuron misclassification percentage."""
    rng = np.random.default_rng(1)
    num_neurons = 9
    labels = rng.integers(0, 10, size=600)
    preds = np.where(rng.random(600) < 0.15, (labels + 1) % 10, labels)
    bmu = rng.integers(0, num_neurons, size=600)
    bmu[bmu == 4] = 5                                  # leave neuron 4 empty
    correct = preds == labels
    rows = build_error_rows(
        correct=correct,
        confidence=rng.random(600),
        bmu=bmu,
        coordinates=np.vstack([np.arange(num_neurons), np.zeros(num_neurons)]),
        num_neurons=num_neurons,
        hotspot_rank={},
    )
    clusters = [np.flatnonzero(bmu == n) for n in range(num_neurons)]
    nnsom_percent = get_perc_misclassified(labels, preds, clusters)
    for row, pct in zip(rows, nnsom_percent):
        if row["support"]:
            assert np.isclose(100 * row["error_rate"], pct)
        else:
            assert np.isnan(row["error_rate"]) and pct == 0
