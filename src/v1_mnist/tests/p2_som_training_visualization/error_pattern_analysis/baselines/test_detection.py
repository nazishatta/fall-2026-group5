from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.detection import (
    bootstrap_auroc,
    cell_error_scores,
    half_split_detection,
    summarize,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    cells = rng.integers(0, 10, 600)
    errors = (cells < 2) & (rng.random(600) < 0.5)  # errors live in cells 0 and 1
    return cells, errors


def test_cell_error_scores_use_fit_rates_and_fall_back_for_unseen_cells() -> None:
    cells = np.array([0, 0, 1, 1, 2])
    errors = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    scores = cell_error_scores(cells, errors, np.array([0, 1, 2, 3]), np.array([4]), num_cells=3)
    assert np.isclose(scores[0], errors[[0, 1, 2, 3]].mean())  # cell 2 unseen in the fit images


def test_error_cells_beat_a_constant_score_and_a_perfect_score_scores_one() -> None:
    cells, errors = _toy_data()
    perfect = errors.astype(float)
    out = half_split_detection({"cells": [cells]}, {"perfect": perfect, "constant": np.zeros(600)},
                               errors, num_cells=10, repeats=5, split_seed_base=7000)
    assert out["perfect"].auroc.mean() == 1.0
    assert out["constant"].auroc.mean() == 0.5
    assert out["cells"].auroc.mean() > 0.7
    assert out["cells"].auroc.shape == (5,)


def test_bootstrap_is_repeatable_and_contains_the_point_estimate() -> None:
    cells, errors = _toy_data()
    score = (cells < 2).astype(float) + 0.01 * np.random.default_rng(2).random(600)
    first = bootstrap_auroc(errors, {"s": score}, resamples=50, seed=5)["s"]
    second = bootstrap_auroc(errors, {"s": score}, resamples=50, seed=5)["s"]
    assert first == second
    assert first["lower"] <= first["point"] <= first["upper"]


def test_summarize_reports_mean_and_percentile_range() -> None:
    summary = summarize(np.arange(101, dtype=float))
    assert summary["mean"] == 50.0 and summary["lower"] == 2.5 and summary["upper"] == 97.5
