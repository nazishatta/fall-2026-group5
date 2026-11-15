"""The arms of the curriculum comparison: standard training and three curricula.

Every curriculum arm uses the same schedule and a hard pool of the same size. They differ in which
images form the pool: chosen with the SOM, chosen by low max-softmax confidence, or at random.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.curriculum import (
    Schedule,
    curriculum_epoch_indices,
    hard_fraction,
    unlock_order,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    resolve_pool,
)
from src.v1_mnist.component.utils.retraining_data import RetrainingData
from src.v1_mnist.component.utils.retraining_plans import (
    ArmPlan,
    BuildArm,
    standard_plan,
)
from src.v1_mnist.component.utils.retraining_settings import ArmSpec, ExperimentSettings


def schedules_from_options(options: dict[str, Any]) -> dict[str, Schedule]:
    """The named schedules of the config."""
    return {
        name: Schedule(float(s["initial_fraction"]), float(s["step_fraction"]), int(s["step_epochs"]))
        for name, s in options["schedules"].items()
    }


def make_curriculum_builder(settings: ExperimentSettings) -> BuildArm:
    """The ``build(arm, data, seed)`` function for this experiment's arms."""
    schedules = schedules_from_options(settings.options)

    def build(arm: ArmSpec, data: RetrainingData, seed: int) -> ArmPlan:
        if arm.kind == "standard":
            return standard_plan(data)
        if arm.kind != "curriculum":
            raise ValueError(f"Arm {arm.name}: the curriculum comparison has no '{arm.kind}' arms.")
        schedule = schedules[arm.schedule]
        pool = resolve_pool(arm.pool, data.som_pool, data.msp_pool, data.train_in_hotspot,
                            data.train.labels, seed)
        pool_rows = np.flatnonzero(pool)
        easy_rows = np.flatnonzero(~pool)
        ordered = unlock_order(arm.pool, pool_rows, data.rules_satisfied, data.bmu_distance,
                               data.confidence, data.train.sample_ids, seed)
        epoch_indices = curriculum_epoch_indices(easy_rows, ordered, schedule)
        epochs = settings.train.epochs
        return ArmPlan(
            data.train.images, data.train.labels, epoch_indices,
            {
                "pool": arm.pool,
                "pool_images": len(pool_rows),
                "easy_images": len(easy_rows),
                "schedule": arm.schedule,
                "hard_share_by_epoch": [round(hard_fraction(schedule, e), 4) for e in range(epochs)],
                "training_images_by_epoch": [len(epoch_indices(e)) for e in range(epochs)],
            },
        )

    return build
