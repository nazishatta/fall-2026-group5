"""Map measures, CKA and the paired comparison on small hand-made examples."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.cka import (
    cka_matrix,
    linear_cka,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.comparison import (
    decide,
    paired_summary,
    statements,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_measures import (
    class_scatter_index,
    hotspot_measures,
    low_purity_share,
    neuron_majority,
    weighted_purity,
)


def line_dist(n: int) -> np.ndarray:
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


def test_purity_counts_majority_images() -> None:
    bmu = np.array([0, 0, 0, 1, 1])
    labels = np.array([1, 1, 2, 3, 3])
    counts = neuron_majority(bmu, labels, 3, 4)
    assert weighted_purity(counts) == pytest.approx(4 / 5)
    assert low_purity_share(counts, 0.95) == pytest.approx(3 / 5)
    assert low_purity_share(counts, 0.5) == 0.0


def test_scatter_index_counts_separate_groups() -> None:
    # neurons 0,1 class 0 (neighbours); neuron 3 class 0 (far); neuron 2 class 1
    counts = np.zeros((4, 2), dtype=int)
    counts[[0, 1, 3], 0] = 5
    counts[2, 1] = 5
    assert class_scatter_index(counts, line_dist(4), 1.1) == pytest.approx((2 + 1) / 2)


def test_scatter_index_is_one_for_compact_classes() -> None:
    counts = np.zeros((4, 2), dtype=int)
    counts[[0, 1], 0] = 3
    counts[[2, 3], 1] = 3
    assert class_scatter_index(counts, line_dist(4), 1.1) == 1.0


def test_hotspot_measures_pick_the_worst_neuron() -> None:
    bmu = np.repeat(np.arange(5), 20)
    err = np.zeros(100)
    err[:10] = 1          # neuron 0 has 10 of 20 wrong
    err[40:42] = 1        # neuron 2 has 2 of 20 wrong
    out = hotspot_measures(bmu, err, 5, 0.05, 1)
    assert out["hotspots"] == [0]
    assert out["hotspot_error_rate"] == pytest.approx(0.5)
    assert out["hotspot_error_share"] == pytest.approx(10 / 12)
    assert out["overall_error"] == pytest.approx(0.12)


def test_cka_is_one_for_scaled_rotated_copy_and_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)
    assert linear_cka(x, rng.normal(size=(200, 6))) < 0.2
    names, mat = cka_matrix({"a": x, "b": x * 2})
    assert names == ["a", "b"] and mat[0, 1] == pytest.approx(1.0)


def test_paired_summary_uses_t_interval_and_direction() -> None:
    a = np.array([0.10, 0.12, 0.11, 0.13, 0.12])
    b = a + 0.05
    s = paired_summary(a, b, 0.95)
    assert s["mean_difference"] == pytest.approx(-0.05)
    assert s["share_a_smaller"] == 1.0 and s["excludes_zero"] is True
    assert decide("hotspot_error_rate", s) is True
    assert decide("weighted_purity", s) is False


def test_noisy_difference_is_not_decided() -> None:
    s = paired_summary(np.array([0.1, 0.3, 0.2, 0.4, 0.0]), np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 0.95)
    assert s["excludes_zero"] is False and decide("hotspot_error_rate", s) is False


def test_statements_follow_the_protocol_rules() -> None:
    good = {"weighted_purity": paired_summary(np.array([0.9, 0.91, 0.92, 0.9, 0.93]), np.full(5, 0.8), 0.95),
            "hotspot_error_rate": paired_summary(np.full(5, 0.1) + np.arange(5) * 0.001, np.full(5, 0.3), 0.95),
            "topographic_error_pct": paired_summary(np.array([1.0, 1.1, 0.9, 1.0, 1.2]), np.full(5, 1.0), 0.95)}
    out = statements({"a_vs_s": good, "a_vs_r": good}, 1.0, "s", "r", "a")
    assert out["R1_boundaries_sharpen_vs_standard"]["met"] and out["R4_guidance_shrinks_hotspots_vs_random"]["met"]
    assert out["R5_map_not_less_organised_vs_standard"]["met"] is True
    out2 = statements({"a_vs_s": good, "a_vs_r": good}, 0.01, "s", "r", "a")
    assert out2["R5_map_not_less_organised_vs_standard"]["met"] is False
