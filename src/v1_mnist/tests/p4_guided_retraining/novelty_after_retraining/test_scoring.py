"""Scores and per-cell metrics of the novelty-after-retraining step (synthetic data, no models)."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.scoring import (
    DETECTORS,
    DIAGNOSTIC,
    cell_metrics,
    competitive,
    relu_logits,
    score_split,
)


def _data(rng: np.random.Generator, n: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 3, size=n)
    x = rng.normal(size=(n, 4)) + labels[:, None] * 3.0 + shift
    return x, labels


def test_relu_logits_matches_relu_then_linear() -> None:
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    w = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.5, 0.0, -1.0])
    assert np.allclose(relu_logits(x, w, b), [[0.5, 2.0, 1.0], [3.5, 0.0, 2.0]])


def test_scores_are_higher_for_novel_images() -> None:
    rng = np.random.default_rng(0)
    train_x, train_y = _data(rng, 600, 0.0)
    normal_x, _ = _data(rng, 200, 0.0)
    novel_x = rng.normal(size=(200, 4)) * 0.3 + 40.0          # far from every class
    w, b = rng.normal(size=(3, 4)), np.zeros(3)
    qe = {k: np.linalg.norm(x - x.mean(0), axis=1) for k, x in (("n", normal_x), ("o", novel_x))}
    normal = score_split(normal_x, relu_logits(normal_x, w, b), qe["n"], train_x, train_y, 1.0)
    novel = score_split(novel_x, relu_logits(novel_x, w, b), qe["o"], train_x, train_y, 1.0)
    assert set(normal) == {*DETECTORS, DIAGNOSTIC}
    assert novel["mahalanobis"].mean() > normal["mahalanobis"].mean()
    # D1 is minus the length: longer embeddings give lower scores.
    assert np.allclose(normal[DIAGNOSTIC], -np.linalg.norm(normal_x, axis=1))


def test_cell_metrics_shape_and_threshold_from_training() -> None:
    rng = np.random.default_rng(1)
    names = (*DETECTORS, DIAGNOSTIC)
    train = {n: rng.normal(size=300) for n in names}
    normal = {n: rng.normal(size=200) for n in names}
    novel = {n: rng.normal(size=200) + 3.0 for n in names}
    m = cell_metrics(train, normal, novel, percentile=95, resamples=50, seed=42, confidence=0.95)
    for n in names:
        s = m["scores"][n]
        assert s["auroc"] > 0.9 and s["auroc_interval"][0] <= s["auroc"] <= s["auroc_interval"][1]
        assert np.isclose(s["threshold_from_training"], np.percentile(train[n], 95))
    assert set(m["qe_minus"]) == {"msp", "energy", "mahalanobis"}
    assert -1.0 <= m["qe_length_spearman"] <= 1.0


def test_competitive_uses_the_best_baseline() -> None:
    scores = {"qe": {"auroc": 0.95}, "msp": {"auroc": 0.97}, "energy": {"auroc": 0.99}, "mahalanobis": {"auroc": 0.8}}
    close = {"msp": {"interval": [-0.03, -0.01]}, "energy": {"interval": [-0.005, 0.01]},
             "mahalanobis": {"interval": [0.1, 0.2]}}
    far = {**close, "energy": {"interval": [-0.06, -0.03]}}
    assert competitive(close, scores, 0.01)        # compared with energy, the best baseline
    assert not competitive(far, scores, 0.01)
