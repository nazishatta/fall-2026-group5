"""Hard example mining: the three difficulty rules and the three pools, on a tiny made-up table."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.difficulty import (
    difficulty_flags,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    class_counts,
    msp_pool,
    random_pool,
)

# Four neurons: 0 pure, 1 impure, 2 pure, 3 pure. Neuron 2 is a hotspot.
PURITY = np.array([1.0, 0.80, 0.99, 0.96])
BMU = np.array([0, 0, 1, 2, 3, 3, 0, 2])
DIST = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 10.0])


def test_flags_follow_the_three_rules() -> None:
    flags = difficulty_flags(BMU, DIST, PURITY, np.array([2]), purity_cutoff=0.95, qe_percentile=90)
    assert flags.in_hotspot.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
    assert flags.low_purity.tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert flags.qe_flag.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert flags.in_som_pool.tolist() == [0, 0, 1, 1, 0, 0, 0, 1]


def test_distance_cutoff_is_a_percentile_of_the_given_distances() -> None:
    flags = difficulty_flags(BMU, DIST, PURITY, np.array([2]), 0.95, 50)
    assert flags.qe_threshold == pytest.approx(float(np.percentile(DIST, 50)))
    assert flags.qe_flag.sum() == 4  # strictly above the median


def test_neuron_without_purity_is_refused() -> None:
    purity = PURITY.copy()
    purity[0] = np.nan
    with pytest.raises(ValueError):
        difficulty_flags(BMU, DIST, purity, np.array([2]), 0.95, 95)


def test_msp_pool_takes_least_confident_and_breaks_ties_by_sample_id() -> None:
    confidence = np.array([0.9, 0.5, 0.5, 1.0, 0.7])
    sample_id = np.array([40, 30, 10, 20, 50])
    mask = msp_pool(confidence, sample_id, size=2)
    assert np.flatnonzero(mask).tolist() == [1, 2]
    mask = msp_pool(confidence, sample_id, size=1)
    assert np.flatnonzero(mask).tolist() == [2]  # tie at 0.5 -> lower sample ID


def test_msp_pool_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        msp_pool(np.array([0.5, 0.6]), np.array([1, 2]), size=0)
    with pytest.raises(ValueError):
        msp_pool(np.array([0.5, 0.6]), np.array([1, 2]), size=3)


def test_random_pool_keeps_the_digit_mix_and_depends_on_the_seed() -> None:
    labels = np.repeat(np.arange(3), 50)
    per_class = np.array([5, 0, 12])
    first = random_pool(labels, per_class, seed=1)
    assert class_counts(labels, first, 3).tolist() == [5, 0, 12]
    assert np.array_equal(first, random_pool(labels, per_class, seed=1))
    assert not np.array_equal(first, random_pool(labels, per_class, seed=2))


def test_random_pool_refuses_more_images_than_exist() -> None:
    with pytest.raises(ValueError):
        random_pool(np.array([0, 0, 1]), np.array([3, 0]), seed=0)
