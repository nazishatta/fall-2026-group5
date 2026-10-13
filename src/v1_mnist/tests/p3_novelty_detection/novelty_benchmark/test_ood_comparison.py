"""Small deterministic tests that never touch official MNIST test or OOD data."""

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    analyze_misses,
    evaluate_scores,
    save_new_json,
)


def test_perfect_separation() -> None:
    result = evaluate_scores(np.array([0.0, 0.1]), np.array([0.9, 1.0]),
                             bootstrap_repeats=10)
    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["fpr_at_95_tpr"] == 0.0


def test_reproducible_bootstrap() -> None:
    normal = np.array([0.1, 0.4, 0.5])
    novel = np.array([0.3, 0.6, 0.8])
    first = evaluate_scores(normal, novel, bootstrap_repeats=25, seed=42)
    second = evaluate_scores(normal, novel, bootstrap_repeats=25, seed=42)
    assert first == second


def test_no_ood_rejected() -> None:
    with pytest.raises(ValueError, match="Both ID and OOD"):
        evaluate_scores(np.array([0.5]), np.array([]))


def test_misses_are_ordered() -> None:
    result = analyze_misses(np.array([0.8, 0.2, 0.7]), threshold=0.5,
                            sample_ids=np.array(["a", "b", "c"]))
    assert result["missed"] == 1
    assert result["lowest_scoring_misses"][0]["sample_id"] == "b"


def test_never_overwrite_existing_result(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    save_new_json(path, {"auroc": 0.8})
    with pytest.raises(FileExistsError):
        save_new_json(path, {"auroc": 0.9})
