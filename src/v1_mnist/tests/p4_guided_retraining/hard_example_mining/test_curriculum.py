"""Curriculum: the schedule, the order hard images are let in, and the arms built from them."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum import (
    Schedule,
    curriculum_epoch_indices,
    hard_fraction,
    unlock_order,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum_arms import (
    make_curriculum_builder,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    resolve_pool,
)
from src.v1_mnist.component.utils.retraining_settings import ArmSpec
from src.v1_mnist.tests.p4_guided_retraining.synthetic_data import make_data

PROPOSAL = Schedule(initial_fraction=0.10, step_fraction=0.10, step_epochs=5)


def test_schedule_starts_at_ten_percent_and_adds_ten_percent_every_five_epochs() -> None:
    shares = [hard_fraction(PROPOSAL, e) for e in range(20)]
    assert shares[0] == shares[4] == pytest.approx(0.10)
    assert shares[5] == pytest.approx(0.20)
    assert shares[10] == pytest.approx(0.30)
    assert shares[19] == pytest.approx(0.40)


def test_share_never_exceeds_one() -> None:
    assert hard_fraction(Schedule(0.5, 0.5, 1), 10) == 1.0


def test_bad_schedule_is_refused() -> None:
    with pytest.raises(ValueError):
        Schedule(0.0, 0.1, 5)
    with pytest.raises(ValueError):
        Schedule(0.1, 0.1, 0)


def test_easy_images_are_always_in_and_hard_ones_grow() -> None:
    easy = np.arange(100)
    ordered_pool = np.arange(100, 200)
    indices = curriculum_epoch_indices(easy, ordered_pool, PROPOSAL)
    first, later = indices(0), indices(5)
    assert set(easy) <= set(first) and set(easy) <= set(later)
    assert len(first) == 100 + 10 and len(later) == 100 + 20
    assert set(first) < set(later)  # nothing is dropped once let in


def test_som_pool_goes_from_fewer_rules_and_closer_to_farther() -> None:
    rows = np.array([0, 1, 2, 3])
    order = unlock_order("som", rows, rules_satisfied=np.array([2, 1, 1, 3]),
                         bmu_distance=np.array([0.1, 0.9, 0.2, 0.0]), confidence=np.zeros(4),
                         sample_id=np.arange(4), seed=0)
    assert order.tolist() == [2, 1, 0, 3]  # 1 rule (closest first), then 2 rules, then 3


def test_msp_pool_goes_from_most_to_least_confident() -> None:
    rows = np.array([0, 1, 2])
    order = unlock_order("msp", rows, np.zeros(3, int), np.zeros(3), np.array([0.6, 0.9, 0.7]),
                         np.arange(3), seed=0)
    assert order.tolist() == [1, 2, 0]


def test_random_order_depends_on_the_seed_only() -> None:
    rows = np.arange(50)
    args = (rows, np.zeros(50, int), np.zeros(50), np.zeros(50), np.arange(50))
    assert np.array_equal(unlock_order("random", *args, seed=1), unlock_order("random", *args, seed=1))
    assert not np.array_equal(unlock_order("random", *args, seed=1), unlock_order("random", *args, seed=2))


class _Settings:
    """Just what the arm builder reads from the settings."""

    options = {"schedules": {"proposal": {"initial_fraction": 0.1, "step_fraction": 0.1, "step_epochs": 5}}}

    class train:
        epochs = 20


def test_curriculum_arms_use_equal_sized_pools_and_the_same_schedule() -> None:
    data = make_data()
    build = make_curriculum_builder(_Settings())  # type: ignore[arg-type]
    plans = {
        pool: build(ArmSpec(f"c_{pool}", "curriculum", pool, "proposal"), data, seed=3)
        for pool in ("som", "random", "msp")
    }
    sizes = {p.info["pool_images"] for p in plans.values()}
    assert sizes == {12}
    assert len({tuple(p.info["training_images_by_epoch"]) for p in plans.values()}) == 1
    assert plans["som"].info["easy_images"] == 120 - 12


def test_standard_arm_uses_every_image_every_epoch() -> None:
    data = make_data()
    plan = make_curriculum_builder(_Settings())(ArmSpec("standard", "standard"), data, seed=1)  # type: ignore[arg-type]
    assert len(plan.epoch_indices(0)) == len(plan.epoch_indices(19)) == 120


def test_random_pool_has_the_digit_mix_of_the_som_pool() -> None:
    data = make_data()
    pool = resolve_pool("random", data.som_pool, data.msp_pool, data.train_in_hotspot, data.train.labels, seed=7)
    assert np.bincount(data.train.labels[pool]).tolist() == np.bincount(data.train.labels[data.som_pool]).tolist()
