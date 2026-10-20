"""Test the MNIST feature autoencoder."""

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_baseline import (
    fit_autoencoder,
)


def test_autoencoder_scores() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 84)).astype(np.float32)
    validation = rng.normal(size=(16, 84)).astype(np.float32)

    config = {
        "seed": 42,
        "autoencoder": {
            "input_dim": 84,
            "hidden_dim": 32,
            "latent_dim": 16,
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
        },
    }

    scores, losses = fit_autoencoder(train, validation, config)

    assert scores.shape == (16,)
    assert len(losses) == 2
    assert np.isfinite(scores).all()
    assert np.isfinite(losses).all()
