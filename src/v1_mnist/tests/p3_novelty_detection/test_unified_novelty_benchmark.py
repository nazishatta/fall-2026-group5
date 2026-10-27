"""Regression checks for unified nine-method evaluation protocol."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection.unified_novelty_benchmark import (
    check_protocol,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.unified_novelty_metrics import (
    METHODS,
    evaluate,
    seed_summary,
    subset_fingerprint,
    subset_indices,
)


def test_nine_distinct_methods() -> None:
    assert len(METHODS) == 9
    assert len(set(METHODS)) == 9


def test_budget_selection_deterministic_without_replacement() -> None:
    first = subset_indices(40, 10, 42)
    np.testing.assert_array_equal(first, subset_indices(40, 10, 42))
    assert len(set(first.tolist())) == 10
    assert subset_fingerprint(first) == subset_fingerprint(first.copy())
    assert subset_fingerprint(first) != subset_fingerprint(subset_indices(40, 10, 43))


def test_reject_invalid_budget() -> None:
    with pytest.raises(ValueError):
        subset_indices(10, 11, 42)


def test_metrics_direction_and_operating_point() -> None:
    result = evaluate(np.array([0.0, 0.1, 0.2]), np.array([0.8, 0.9, 1.0]), 0.95)
    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["fpr_at_target_tpr"] == 0.0


def test_score_nan_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate(np.array([np.nan]), np.array([1.0]), 0.95)


def test_seed_spread_requires_five_runs() -> None:
    with pytest.raises(ValueError):
        seed_summary([{"metrics": {"knn": {"auroc": 0.9}}}])


def test_protocol_rejects_unequal_svm_budget() -> None:
    source = Path(__file__).resolve().parents[2] / "component/configs/unified_novelty_benchmark.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    check_protocol(config)
    modified = copy.deepcopy(config)
    modified["fitting"]["one_class_svm"]["max_training_samples"] = 9000
    with pytest.raises(ValueError, match="common training budget"):
        check_protocol(modified)


def test_protocol_rejects_repeated_seed() -> None:
    source = Path(__file__).resolve().parents[2] / "component/configs/unified_novelty_benchmark.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["fitting"]["seeds"] = [42] * 5
    with pytest.raises(ValueError, match="Five unique seeds"):
        check_protocol(config)
