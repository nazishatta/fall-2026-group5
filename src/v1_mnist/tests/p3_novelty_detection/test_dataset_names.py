"""Phase 3 result names follow the in-distribution dataset; MNIST keeps its historical names."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.utils.in_distribution import InDistribution

FASHION = InDistribution("fashion_mnist", "Fashion-MNIST", "class")


def test_svhn_transform_is_grey_for_one_channel_and_colour_for_three() -> None:
    pytest.importorskip("torchvision")
    from torchvision import transforms

    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )

    def has_grey(transform: object) -> bool:
        return any(isinstance(step, transforms.Grayscale) for step in transform.transforms)

    assert has_grey(image_transform("svhn", (28, 28)))  # the MNIST default, unchanged
    assert has_grey(image_transform("svhn", (28, 28), 1))
    assert not has_grey(image_transform("svhn", (32, 32), 3))
    with pytest.raises(ValueError, match="1-channel"):
        image_transform("svhn", (28, 28), 2)


def test_norm_dependence_keys_follow_the_dataset() -> None:
    pytest.importorskip("scipy")
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.qe_norm_dependence import (
        analyze,
    )

    rng = np.random.default_rng(0)
    train, val, svhn = rng.normal(size=(60, 4)), rng.normal(size=(30, 4)), rng.normal(size=(40, 4)) * 2
    settings = {"target_tpr": 0.95, "bootstrap_iterations": 20, "seed": 1, "confidence": 0.95,
                "norm_matched_band": [5, 95]}
    args = (train, val, svhn, np.abs(val[:, 0]), np.abs(svhn[:, 0]))
    mnist = analyze(*args, qe_threshold=1.0, settings=settings)
    fashion = analyze(*args, qe_threshold=1.0, settings=settings, dataset=FASHION)
    assert set(mnist["norms"]) == {"mnist_train", "mnist_validation", "svhn_test"}
    assert set(fashion["norms"]) == {"fashion_mnist_train", "fashion_mnist_validation", "svhn_test"}
    assert "mnist_validation" in mnist["qe_norm_spearman"]
    assert mnist["qe_reference"]["threshold_source"] == "frozen MNIST training-QE 95th percentile"
    assert fashion["qe_reference"]["threshold_source"] == "frozen Fashion-MNIST training-QE 95th percentile"


def test_landing_profile_and_top_neurons_name_the_dataset_class() -> None:
    from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.svhn_som_overlay import (
        class_landing_profile,
        top_missed_neurons,
    )

    bmu, qe, dominant = np.array([0, 1, 1]), np.array([0.5, 2.0, 0.1]), np.array([0, 1])
    assert "dominant_mnist_class" in class_landing_profile(bmu, qe, 1.0, dominant, 2)[0]
    assert "dominant_fashion_mnist_class" in class_landing_profile(bmu, qe, 1.0, dominant, 2, FASHION)[0]
    summary = {"missed": np.array([1, 1]), "hits": np.array([1, 2]), "miss_rate": np.array([1.0, 0.5])}
    purity = np.array([1.0, 1.0])
    assert "dominant_mnist_class" in top_missed_neurons(summary, dominant, purity, 2)[0]
    assert "dominant_fashion_mnist_class" in top_missed_neurons(summary, dominant, purity, 2, FASHION)[0]


def test_unified_budget_key_follows_the_dataset() -> None:
    pytest.importorskip("torch")
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.unified_novelty_benchmark import (
        training_budget,
    )

    assert training_budget({"fitting": {"mnist_training_budget": 10000}}, InDistribution()) == 10000
    assert training_budget({"fitting": {"fashion_mnist_training_budget": 9000}}, FASHION) == 9000


def test_qe_u_matrix_summary_names_the_dataset() -> None:
    from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.run_qe_u_matrix import (
        evaluate_validation,
    )

    weights = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [5.0, 5.0]])
    topology = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]], dtype=float)
    bmu, qe = np.array([0, 1, 2, 3, 0]), np.array([0.1, 0.4, 0.2, 0.9, 0.3])
    assert evaluate_validation(weights, topology, bmu, qe)["dataset"] == "MNIST"
    fashion = evaluate_validation(weights, topology, bmu, qe, FASHION)
    assert fashion["dataset"] == "Fashion-MNIST"
    assert fashion["leakage_control"].startswith("Fashion-MNIST validation")
