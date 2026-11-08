"""Small deterministic tests that never touch official MNIST test or OOD data."""

from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    analyze_misses,
    evaluate_scores,
    save_new_json,
)


def test_fpr_definition_matches_qe() -> None:
    """Baseline FPR must equal the preregistered QE definition."""
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
        ood_metrics,
    )

    normal = np.array([0.1, 0.4, 0.6])
    novel = np.array([0.2, 0.5, 0.7])

    qe_result = ood_metrics(normal, novel, threshold=0.5, target_tpr=0.95)
    comparison = evaluate_scores(
        normal, novel, bootstrap_repeats=10, seed=42, target_tpr=0.95
    )

    assert comparison["fpr_at_95_tpr"] == qe_result["fpr_at_target_tpr"]


def test_fpr_is_not_interpolated() -> None:
    """Choose the observed ROC point reaching the target TPR."""
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
        minimum_fpr_at_target_tpr,
    )

    fpr = np.array([0.0, 0.4, 0.6])
    tpr = np.array([0.0, 0.9, 1.0])

    assert minimum_fpr_at_target_tpr(fpr, tpr, 0.95) == 0.6


def test_perfect_separation() -> None:
    result = evaluate_scores(np.array([0.0, 0.1]), np.array([0.9, 1.0]),
                             bootstrap_repeats=10, seed=42, target_tpr=0.95)
    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["fpr_at_95_tpr"] == 0.0


def test_reproducible_bootstrap() -> None:
    normal = np.array([0.1, 0.4, 0.5])
    novel = np.array([0.3, 0.6, 0.8])
    first = evaluate_scores(normal, novel, bootstrap_repeats=25, seed=42, target_tpr=0.95)
    second = evaluate_scores(normal, novel, bootstrap_repeats=25, seed=42, target_tpr=0.95)
    assert first == second


def test_no_ood_rejected() -> None:
    with pytest.raises(ValueError, match="Both ID and OOD"):
        evaluate_scores(np.array([0.5]), np.array([]), bootstrap_repeats=0, seed=42, target_tpr=0.95)


def test_misses_are_ordered() -> None:
    result = analyze_misses(np.array([0.8, 0.2, 0.7]), threshold=0.5,
                            sample_ids=np.array(["a", "b", "c"]), max_examples=20)
    assert result["missed"] == 1
    assert result["lowest_scoring_misses"][0]["sample_id"] == "b"


def test_never_overwrite_existing_result(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    save_new_json(path, {"auroc": 0.8})
    with pytest.raises(FileExistsError):
        save_new_json(path, {"auroc": 0.9})


def test_nonhistorical_target_tpr_uses_correct_metric_name() -> None:
    """FPR at 90% TPR must never be mislabeled FPR at 95% TPR."""
    result = evaluate_scores(
        np.array([0.1, 0.2]),
        np.array([0.8, 0.9]),
        bootstrap_repeats=10,
        seed=42,
        target_tpr=0.90,
    )
    assert result["target_tpr"] == 0.90
    assert result["fpr_at_target_tpr"] == 0.0
    assert "fpr_at_95_tpr" not in result
    assert "fpr_at_target_tpr" in result["confidence_intervals_95"]


@pytest.mark.parametrize("target_tpr", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_target_tpr_rejected(target_tpr: float) -> None:
    """Only finite target TPRs greater than zero and at most one are valid."""
    with pytest.raises(ValueError, match="target_tpr"):
        evaluate_scores(
            np.array([0.1, 0.2]),
            np.array([0.8, 0.9]),
            bootstrap_repeats=0,
            seed=42,
            target_tpr=target_tpr,
        )
