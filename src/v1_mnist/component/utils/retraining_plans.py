"""What one arm trains on (torch-free, so the arm builders can be tested without PyTorch)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.v1_mnist.component.utils.retraining_data import RetrainingData
from src.v1_mnist.component.utils.retraining_settings import ArmSpec

EpochIndices = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class ArmPlan:
    """The image arrays, which rows each epoch uses, and facts to record in the run's metrics file."""

    train_x: np.ndarray
    train_y: np.ndarray
    epoch_indices: EpochIndices
    info: dict[str, Any] = field(default_factory=dict)


BuildArm = Callable[[ArmSpec, RetrainingData, int], ArmPlan]


def standard_plan(data: RetrainingData) -> ArmPlan:
    """Standard training: every original training image in every epoch."""
    rows = np.arange(len(data.train.labels))
    return ArmPlan(data.train.images, data.train.labels, lambda _epoch: rows,
                   {"training_images_per_epoch": len(rows)})
