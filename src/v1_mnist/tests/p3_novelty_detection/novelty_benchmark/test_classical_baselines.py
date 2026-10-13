"""Tests for classical novelty benchmark scores."""

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.classical_baselines import (
    fit_classical_baselines,
)


def test_classical_baseline_scores() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(120, 84)).astype(np.float32)
    validation = rng.normal(size=(30, 84)).astype(np.float32)

    config = {
        "seed": 42,
        "isolation_forest": {
            "n_estimators": 20,
            "max_samples": 64,
            "contamination": 0.05,
        },
        "one_class_svm": {
            "kernel": "rbf",
            "nu": 0.05,
            "gamma": "scale",
            "max_training_samples": 100,
        },
    }

    scores = fit_classical_baselines(train, validation, config)

    assert set(scores) == {"isolation_forest", "one_class_svm"}
    for values in scores.values():
        assert values.shape == (30,)
        assert np.isfinite(values).all()
