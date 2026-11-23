"""The held-out endpoint and the decision rule (pre-registration, section 6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.endpoint import (
    cell_rate_scores,
    crossfit,
    split_indices,
    summarise,
    verdict,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    Contrast,
)


def test_split_halves_are_disjoint_stratified_and_bootstrapped() -> None:
    err = np.zeros(200, dtype=bool)
    err[:20] = True
    a, b = split_indices(err, rep=3, seed_base=5000)
    assert len(a) == len(b) == 100
    assert not set(a) & set(b)                      # the halves never share an image
    assert len(set(a)) < len(a)                     # a bootstrap resample repeats images
    a2, b2 = split_indices(err, rep=3, seed_base=5000)
    np.testing.assert_array_equal(a, a2)            # the same repeat gives the same split


def test_split_puts_half_the_errors_in_each_half() -> None:
    err = np.zeros(60, dtype=bool)
    err[::6] = True                                 # 10 errors
    a, b = split_indices(err, rep=0, seed_base=1)
    errors = np.random.default_rng(1).permutation(np.flatnonzero(err))   # the split's first draw
    assert set(a[err[a]]) <= set(errors[:5])        # A's errors come only from A's five
    assert set(b[err[b]]) <= set(errors[5:])        # B's errors come only from B's five


def test_empty_cells_get_the_fit_halfs_overall_rate() -> None:
    cells_fit, y_fit = np.array([0, 0, 1]), np.array([1, 0, 1])
    scores = cell_rate_scores(cells_fit, y_fit, np.array([0, 1, 2]), num_neurons=3)
    np.testing.assert_allclose(scores, [0.5, 1.0, 2 / 3])


def test_crossfit_scores_a_perfectly_separating_map_as_one() -> None:
    cells = np.array([0] * 10 + [1] * 10)
    y = np.array([0] * 10 + [1] * 10)
    a = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    b = np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])
    assert crossfit(cells, y, a, b, num_neurons=2) == (1.0, 1.0)


@pytest.mark.parametrize("low, high, expected", [
    (-0.01, 0.015, "practically equivalent"),
    (0.03, 0.10, "20x20 better"),
    (-0.10, -0.03, "15x15 better"),
    (-0.10, 0.09, "inconclusive"),
    (0.005, 0.03, "20x20 better"),                  # above zero but not inside the margin
])
def test_decision_rule_in_its_preregistered_order(low: float, high: float, expected: str) -> None:
    assert verdict(low, high, 0.02, 20, 15) == expected


def test_summarise_uses_big_minus_small() -> None:
    rows = [{"auroc_err_20": 0.7 + 0.001 * i, "auroc_err_15": 0.6} for i in range(200)]
    out = summarise(rows, Contrast(20, 15, "auroc_err"), margin=0.02)
    assert out["mean_delta"] == pytest.approx(0.1995)
    assert out["verdict"] == "20x20 better" and out["share_reps_big_better"] == 1.0
