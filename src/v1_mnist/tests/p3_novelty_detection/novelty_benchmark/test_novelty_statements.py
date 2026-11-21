"""Statements O1 to O4: the verdict rules and the judged result, on small made-up scores."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.novelty_statements import (  # noqa: E402
    INCONCLUSIVE,
    MET,
    NOT_MET,
    SCORES,
    StatementSettings,
    competitive_verdict,
    detection_verdict,
    judge,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.run_novelty_statements import (  # noqa: E402
    statement_settings,
)

FAST = StatementSettings(bootstrap_iterations=50, paired_iterations=50, shuffle_iterations=50)


@pytest.mark.parametrize(("interval", "verdict"), [
    ([0.51, 0.60], MET), ([0.50, 0.60], INCONCLUSIVE), ([0.45, 0.55], INCONCLUSIVE),
    ([0.40, 0.50], NOT_MET), ([0.30, 0.45], NOT_MET),
])
def test_detection_verdict(interval: list[float], verdict: str) -> None:
    assert detection_verdict(interval) == verdict


@pytest.mark.parametrize(("interval", "verdict"), [
    ([-0.009, 0.02], MET), ([0.01, 0.03], MET), ([-0.010, 0.02], INCONCLUSIVE),
    ([-0.03, -0.005], INCONCLUSIVE), ([-0.03, -0.011], NOT_MET),
])
def test_competitive_verdict_with_the_protocol_margin(interval: list[float], verdict: str) -> None:
    assert competitive_verdict(interval, 0.01) == verdict


def _scores(rng: np.random.Generator, n_id: int = 200, n_ood: int = 300) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """QE separates well; max-softmax is a little better; the U-Matrix is pure noise."""
    out = {}
    for name in SCORES:
        shift = {"qe": 2.0, "max_softmax": 2.2, "u_matrix": 0.0}.get(name, 1.8)
        # Kept positive: QE is a distance, and the label-shuffle control refuses negative QE.
        out[name] = (np.abs(rng.normal(size=n_id)) + 1.0, np.abs(rng.normal(size=n_ood)) + 1.0 + shift)
    return out


def test_judge_gives_every_measure_and_statement() -> None:
    result = judge(_scores(np.random.default_rng(0)), FAST)
    assert set(result["scores"]) == set(SCORES)
    assert set(result["paired_against_qe"]) == set(SCORES[1:])
    assert result["aupr_chance_level"] == pytest.approx(300 / 500)
    assert result["scores"]["qe"]["score_id"] == "S1"
    assert result["statements"]["O1"]["verdict"] == MET          # QE separates the two sets
    assert result["statements"]["O4"]["verdict"] in {INCONCLUSIVE, NOT_MET}   # U-Matrix is noise here
    assert result["statements"]["O2"]["primary"] is True
    assert set(result["statements"]["O3"]["per_score"]) == {"energy", "mahalanobis", "knn"}
    control = result["label_shuffle_control"]
    assert abs(control["shuffled_mean"] - 0.5) < 0.05 and control["iterations"] == 50


def test_judge_is_reproducible_with_the_same_seeds() -> None:
    scores = _scores(np.random.default_rng(1))
    assert judge(scores, FAST) == judge(scores, FAST)


def test_scores_must_be_aligned_and_complete() -> None:
    scores = _scores(np.random.default_rng(2))
    with pytest.raises(ValueError, match="not aligned"):
        judge({**scores, "knn": (scores["knn"][0][:-1], scores["knn"][1])}, FAST)
    with pytest.raises(ValueError, match="Missing"):
        judge({k: v for k, v in scores.items() if k != "energy"}, FAST)


def test_settings_from_the_config_keep_their_types() -> None:
    settings = statement_settings({"bootstrap_iterations": 2000, "competitive_margin": 0.01,
                                   "energy_temperature": 1.0, "knn_neighbors": 5})
    assert settings.bootstrap_iterations == 2000 and isinstance(settings.bootstrap_iterations, int)
    assert settings.competitive_margin == 0.01
