"""Targeted augmentation: the altered copies and the arms built from them."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.augmentation_arms import (
    make_augmentation_builder,
)
from src.v1_mnist.component.utils.image_transforms import TransformRanges, make_copies
from src.v1_mnist.component.utils.retraining_settings import ArmSpec
from src.v1_mnist.tests.p4_guided_retraining.synthetic_data import make_data

RANGES = TransformRanges(10.0, 2.0, 0.9, 1.1, 8.0)


def test_copies_have_the_right_count_shape_and_range() -> None:
    images = np.random.default_rng(0).random((5, 28, 28)).astype(np.float32)
    copies = make_copies(images, 3, seed=1, ranges=RANGES)
    assert copies.shape == (15, 28, 28) and copies.dtype == np.float32
    assert copies.min() >= 0.0 and copies.max() <= 1.0


def test_copies_are_reproducible_and_change_with_the_seed() -> None:
    images = np.random.default_rng(0).random((4, 28, 28)).astype(np.float32)
    assert np.array_equal(make_copies(images, 2, 5, RANGES), make_copies(images, 2, 5, RANGES))
    assert not np.array_equal(make_copies(images, 2, 5, RANGES), make_copies(images, 2, 6, RANGES))


def test_no_change_ranges_give_back_the_original() -> None:
    images = np.random.default_rng(0).random((3, 28, 28)).astype(np.float32)
    same = make_copies(images, 1, 0, TransformRanges(0.0, 0.0, 1.0, 1.0, 0.0))
    assert np.allclose(same, images, atol=1e-5)


def test_copies_are_not_the_original() -> None:
    images = np.random.default_rng(0).random((3, 28, 28)).astype(np.float32)
    assert not np.allclose(make_copies(images, 1, 0, RANGES), images)


class _Settings:
    options = {"copies_per_image": 3, "transforms": {
        "max_rotation_degrees": 10.0, "max_shift_pixels": 2.0, "min_zoom": 0.9,
        "max_zoom": 1.1, "max_shear_degrees": 8.0},
        "schedules": {"proposal": {"initial_fraction": 0.10, "step_fraction": 0.10, "step_epochs": 5}}}

    class train:
        num_classes = 3
        epochs = 20


def test_every_augmented_arm_adds_the_same_number_of_images() -> None:
    data = make_data()
    build = make_augmentation_builder(_Settings())  # type: ignore[arg-type]
    added = {
        pool: build(ArmSpec(f"a_{pool}", "augmentation", pool), data, seed=2).info["added_images"]
        for pool in ("som", "random", "msp")
    }
    assert set(added.values()) == {12 * 3}
    hotspot = {
        pool: build(ArmSpec(f"a_{pool}", "augmentation", pool), data, seed=2).info["added_images"]
        for pool in ("hotspot", "random_hotspot")
    }
    assert set(hotspot.values()) == {4 * 3}


def test_added_images_keep_their_true_labels_and_the_originals_stay() -> None:
    data = make_data()
    plan = make_augmentation_builder(_Settings())(ArmSpec("a", "augmentation", "som"), data, seed=1)  # type: ignore[arg-type]
    assert len(plan.train_y) == 120 + 36
    assert np.array_equal(plan.train_y[:120], data.train.labels)
    assert np.array_equal(plan.train_x[:120], data.train.images)
    added_per_digit = np.bincount(plan.train_y[120:], minlength=3)
    assert added_per_digit.tolist() == (np.bincount(data.train.labels[data.som_pool]) * 3).tolist()
    assert len(plan.epoch_indices(0)) == 156


def test_random_arm_matches_the_som_arm_digit_mix() -> None:
    data = make_data()
    build = make_augmentation_builder(_Settings())  # type: ignore[arg-type]
    som = build(ArmSpec("s", "augmentation", "som"), data, seed=4).info["added_per_digit"]
    rnd = build(ArmSpec("r", "augmentation", "random"), data, seed=4).info["added_per_digit"]
    assert som == rnd


def test_a_paced_arm_brings_the_copies_in_gradually_and_ends_with_the_same_images() -> None:
    data = make_data()
    build = make_augmentation_builder(_Settings())  # type: ignore[arg-type]
    paced = build(ArmSpec("p", "augmentation", "som", "proposal"), data, seed=2)
    plain = build(ArmSpec("u", "augmentation", "som"), data, seed=2)
    originals = len(data.train.labels)
    sizes = paced.info["training_images_by_epoch"]
    assert sizes[0] == originals + 3 * round(0.10 * 12)  # 10% of the pool, 3 copies each
    assert sizes == sorted(sizes) and sizes[-1] < originals + 3 * 12  # 40% by epoch 20, never all
    assert set(plain.info["training_images_by_epoch"]) == {originals + 3 * 12}
    assert paced.info["added_images"] == plain.info["added_images"]
    assert len(paced.train_x) == len(plain.train_x)


def test_a_paced_epoch_only_uses_rows_that_exist_and_always_keeps_the_originals() -> None:
    data = make_data()
    build = make_augmentation_builder(_Settings())  # type: ignore[arg-type]
    plan = build(ArmSpec("p", "augmentation", "random", "proposal"), data, seed=3)
    for epoch in range(20):
        rows = plan.epoch_indices(epoch)
        assert rows.max() < len(plan.train_y) and len(set(rows.tolist())) == len(rows)
        assert set(range(len(data.train.labels))) <= set(rows.tolist())
