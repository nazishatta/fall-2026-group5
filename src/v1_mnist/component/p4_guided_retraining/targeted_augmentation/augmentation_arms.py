"""The arms of the targeted augmentation comparison: standard training and seven augmented arms.

An augmented arm trains on all original training images plus ``copies_per_image`` altered copies of
every image in its pool. Arms differ only in which images form the pool, so all of them add the
same number of images (SOM, max-softmax and random: the SOM pool's size; hotspot and random
hotspot: the hotspot pool's size). Copies keep the true label.

An arm with a ``schedule`` (the ``*_paced`` arms) introduces the copies gradually, like the
curriculum: the copies of 10% of the pool at first, +10% every 5 epochs, easiest images first, so
the hard-example share does not jump to its full size in epoch 1. Arms without a schedule add all
copies from the first epoch.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum import (
    hard_fraction,
    unlock_order,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum_arms import (
    schedules_from_options,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    resolve_pool,
)
from src.v1_mnist.component.utils.image_transforms import TransformRanges, make_copies
from src.v1_mnist.component.utils.retraining_data import RetrainingData
from src.v1_mnist.component.utils.retraining_plans import (
    ArmPlan,
    BuildArm,
    standard_plan,
)
from src.v1_mnist.component.utils.retraining_settings import ArmSpec, ExperimentSettings


def ranges_from_options(options: dict[str, Any]) -> TransformRanges:
    """The transform ranges of the config."""
    t = options["transforms"]
    return TransformRanges(
        max_rotation_degrees=float(t["max_rotation_degrees"]),
        max_shift_pixels=float(t["max_shift_pixels"]),
        min_zoom=float(t["min_zoom"]),
        max_zoom=float(t["max_zoom"]),
        max_shear_degrees=float(t["max_shear_degrees"]),
    )


def make_augmentation_builder(settings: ExperimentSettings) -> BuildArm:
    """The ``build(arm, data, seed)`` function for this experiment's arms."""
    copies = int(settings.options["copies_per_image"])
    ranges = ranges_from_options(settings.options)
    schedules = schedules_from_options(settings.options)

    def build(arm: ArmSpec, data: RetrainingData, seed: int) -> ArmPlan:
        if arm.kind == "standard":
            return standard_plan(data)
        if arm.kind != "augmentation":
            raise ValueError(f"Arm {arm.name}: the augmentation comparison has no '{arm.kind}' arms.")
        pool = resolve_pool(arm.pool, data.som_pool, data.msp_pool, data.train_in_hotspot,
                            data.train.labels, seed)
        rows = np.flatnonzero(pool)
        if arm.schedule:  # paced: the copies of the easiest pool images enter first
            rows = unlock_order(arm.pool, rows, data.rules_satisfied, data.bmu_distance,
                                data.confidence, data.train.sample_ids, seed)
        extra_x = make_copies(data.train.images[rows], copies, seed, ranges)
        extra_y = np.tile(data.train.labels[rows], copies)
        train_x = np.concatenate([data.train.images, extra_x])
        train_y = np.concatenate([data.train.labels, extra_y])
        originals, pool_size = len(data.train.labels), len(rows)
        if arm.schedule:
            schedule = schedules[arm.schedule]

            def epoch_indices(epoch: int) -> np.ndarray:
                unlocked = int(round(hard_fraction(schedule, epoch) * pool_size))
                added = [originals + c * pool_size + np.arange(unlocked) for c in range(copies)]
                return np.concatenate([np.arange(originals), *added])
        else:
            every_row = np.arange(len(train_y))

            def epoch_indices(_epoch: int) -> np.ndarray:
                return every_row
        epochs = settings.train.epochs
        return ArmPlan(
            train_x, train_y, epoch_indices,
            {
                "pool": arm.pool,
                "pool_images": len(rows),
                "copies_per_image": copies,
                "schedule": arm.schedule or "none (all copies from epoch 1)",
                "added_images": len(extra_y),
                "training_images_by_epoch": [len(epoch_indices(e)) for e in range(epochs)],
                "added_per_digit": np.bincount(extra_y, minlength=settings.train.num_classes).tolist(),
                "added_images_sha256": hashlib.sha256(extra_x.tobytes()).hexdigest(),
            },
        )

    return build
