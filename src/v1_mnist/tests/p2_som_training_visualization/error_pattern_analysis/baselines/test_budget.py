"""Same-budget capture of mistakes (exploratory error-detector comparison). NumPy only."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.budget import (
    bootstrap_caught,
    caught_at_share,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.budget_stage import (
    hotspot_shares,
)


def test_the_highest_scores_are_flagged_first() -> None:
    errors = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    out = caught_at_share(np.arange(10, 0, -1, dtype=float), errors, 0.2)
    assert out["error_share"] == 0.5 and out["ratio"] == 2.5 and out["images_flagged"] == 2.0


def test_ties_at_the_cut_count_by_their_expected_value() -> None:
    errors = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert caught_at_share(np.zeros(10), errors, 0.5)["error_share"] == pytest.approx(0.5)
    score = np.array([2.0, 1, 1, 1, 0, 0, 0, 0, 0, 0])    # one image above, three tied for two places
    errors = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    assert caught_at_share(score, errors, 0.3)["error_share"] == pytest.approx(0.5 * 2 / 3)


def test_a_perfect_score_catches_every_mistake_within_its_budget() -> None:
    errors = np.r_[np.ones(5), np.zeros(95)]
    out = caught_at_share(errors.copy(), errors, 0.05)
    assert out["error_share"] == 1.0 and out["ratio"] == pytest.approx(20.0)


def test_bootstrap_interval_contains_the_point_value() -> None:
    rng = np.random.default_rng(0)
    errors = rng.random(500) < 0.05
    scores = {"good": errors + rng.normal(0, 0.3, 500), "noise": rng.random(500)}
    out = bootstrap_caught(scores, errors, 0.05, resamples=200, seed=1)
    for r in out.values():
        assert r["error_share_low"] <= r["error_share"] <= r["error_share_high"]
    assert out["good"]["error_share"] > out["noise"]["error_share"]


def test_hotspot_shares_are_read_from_the_baseline_results() -> None:
    baseline = {"hotspot": {
        "som_cell": {"hotspot_shares": {"mean_image_share_in_hotspots": 0.016, "mean_error_share_in_hotspots": 0.18}},
        "kmeans_cell": {"mean_image_share_in_hotspots": 0.038, "mean_error_share_in_hotspots": 0.20}}}
    assert hotspot_shares(baseline) == {"som_cell": {"image_share": 0.016, "error_share": 0.18},
                                        "kmeans_cell": {"image_share": 0.038, "error_share": 0.20}}
