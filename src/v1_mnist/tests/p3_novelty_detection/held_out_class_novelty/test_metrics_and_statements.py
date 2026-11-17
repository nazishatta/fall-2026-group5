"""Per-split metrics and the statements H1 to H4 on made-up scores (NumPy and scikit-learn only)."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.comparison import (
    INCONCLUSIVE,
    OPPOSITE,
    SHOWN,
    score_summary,
    statements,
    superiority_verdict,
)
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.split_metrics import (
    SCORES,
    all_scores,
    per_digit_qe_auroc,
    split_metrics,
)


def _parts(seed: int = 0) -> tuple[dict, dict, dict, np.ndarray]:
    """Known images near three class centres; novel images in between and a bit farther away."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (3, 8))
    train_y = np.repeat(np.arange(3), 60)
    feats = {"train": centres[train_y] + rng.normal(0, 1, (180, 8)),
             "normal": centres[np.repeat(np.arange(3), 20)] + rng.normal(0, 1, (60, 8)),
             "novel": rng.normal(0, 1, (40, 8)) * 2.5}
    w = rng.normal(0, 1, (3, 8))
    logits = {k: np.maximum(v, 0) @ w.T for k, v in feats.items()}
    qe = {k: np.min(np.linalg.norm(v[:, None, :] - centres[None], axis=2), axis=1) for k, v in feats.items()}
    return feats, logits, qe, train_y


def test_split_metrics_report_every_score_the_control_and_the_length_check() -> None:
    feats, logits, qe, train_y = _parts()
    scores = all_scores(feats, logits, qe, train_y, temperature=1.0, neighbors=5)
    m = split_metrics(scores, 95, resamples=50, seed=42, confidence=0.95, margin=0.01,
                      null_iterations=50, null_seed=42)
    assert set(SCORES) | {"length"} <= set(m["scores"])
    assert set(m["qe_minus"]) == {"msp", "energy", "mahalanobis", "knn"}
    assert m["scores"]["qe"]["auroc"] > 0.8
    low, high = m["scores"]["knn"]["auroc_interval"]
    assert 0 <= low <= m["scores"]["knn"]["auroc"] <= high <= 1
    assert abs(m["qe_label_shuffle_control"]["shuffled_mean"] - 0.5) < 0.1
    assert isinstance(m["competitive_rq4"], bool) and -1 <= m["qe_length_spearman"] <= 1


def test_per_digit_auroc_names_each_unknown_digit() -> None:
    out = per_digit_qe_auroc(np.zeros(10), np.r_[np.ones(5), -np.ones(5)], np.r_[[3] * 5, [7] * 5])
    assert out == {"3": 1.0, "7": 0.0}


def test_superiority_verdict_has_three_outcomes() -> None:
    assert superiority_verdict([0.01, 0.05]) == SHOWN
    assert superiority_verdict([-0.05, -0.01]) == OPPOSITE
    assert superiority_verdict([-0.01, 0.02]) == INCONCLUSIVE


def _split(k: int, qe: float, others: dict[str, float]) -> dict:
    scores = {"qe": {"auroc": qe, "auroc_interval": [qe - 0.01, qe + 0.01]}, "length": {"auroc": 0.6}}
    scores.update({n: {"auroc": v} for n, v in others.items()})
    return {"split": k, "scores": scores, "closed_set_accuracy_known": 0.99, "qe_length_spearman": -0.2,
            "qe_label_shuffle_control": {"shuffled_mean": 0.5}, "competitive_rq4": True,
            "cells_holding_share": {"share": 0.9, "novel": 20, "normal": 150}}


def test_statements_when_the_som_wins_on_every_split() -> None:
    metrics = [_split(k, 0.95 + 0.002 * k, {"msp": 0.90 + 0.003 * k, "energy": 0.91, "mahalanobis": 0.80 + 0.01 * k,
                                            "knn": 0.93 + 0.001 * k}) for k in range(1, 6)]
    out = statements(metrics, confidence=0.95, margin=0.01)
    assert out["H1_som_detects_held_out_digits"]["verdict"] == SHOWN
    assert out["H2_som_better_than_max_softmax"]["verdict"] == SHOWN
    assert out["H3_som_not_worse_by_margin"] == {"msp": SHOWN, "energy": SHOWN, "mahalanobis": SHOWN}
    assert out["H4_map_adds_to_nearest_neighbour"]["verdict"] == SHOWN
    summary = score_summary(metrics)
    assert summary["competitive_rq4_splits"] == 5 and len(summary["qe"]["per_split"]) == 5


def test_statements_when_max_softmax_wins_and_one_split_fails() -> None:
    metrics = [_split(k, 0.80 + 0.01 * k, {"msp": 0.97 + 0.001 * k, "energy": 0.97, "mahalanobis": 0.9,
                                           "knn": 0.80 + 0.012 * k}) for k in range(1, 6)]
    metrics[0]["scores"]["qe"]["auroc_interval"] = [0.45, 0.9]
    out = statements(metrics, confidence=0.95, margin=0.01)
    assert out["H1_som_detects_held_out_digits"] == {"verdict": OPPOSITE, "splits_not_above_0_5": [1]}
    assert out["H2_som_better_than_max_softmax"]["verdict"] == OPPOSITE
    assert out["H3_som_not_worse_by_margin"]["msp"] == OPPOSITE
