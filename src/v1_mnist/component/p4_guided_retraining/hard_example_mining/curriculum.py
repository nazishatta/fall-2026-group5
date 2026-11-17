"""Curriculum schedule: easy images always, hard images let in gradually.

Easy images are everything outside the hard pool. The share of the pool that is in the training
set in epoch ``e`` (counting from 0) is
``initial_fraction + step_fraction * (e // step_epochs)``, at most 1 (``configs/retraining.yaml``).
Pool images are let in from the easiest to the hardest, so the hardest come last.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EpochIndices = Callable[[int], np.ndarray]


@dataclass(frozen=True)
class Schedule:
    """How fast the hard pool is let in."""

    initial_fraction: float
    step_fraction: float
    step_epochs: int

    def __post_init__(self) -> None:
        if not 0 < self.initial_fraction <= 1 or self.step_fraction < 0 or self.step_epochs < 1:
            raise ValueError("Schedule needs 0 < initial_fraction <= 1, step_fraction >= 0, step_epochs >= 1.")


def hard_fraction(schedule: Schedule, epoch: int) -> float:
    """Share of the hard pool used in this epoch (0-based)."""
    return min(1.0, schedule.initial_fraction + schedule.step_fraction * (epoch // schedule.step_epochs))


def unlock_order(
    pool_kind: str,
    pool_rows: np.ndarray,
    rules_satisfied: np.ndarray,
    bmu_distance: np.ndarray,
    confidence: np.ndarray,
    sample_id: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Pool rows from the easiest to the hardest.

    SOM and hotspot pools: fewer hard-image rules satisfied first, then closer to the neuron.
    Max-softmax pool: most confident first. Random pools: a random order (per seed).
    Ties are broken by sample ID, so the order is reproducible.
    """
    if pool_kind in ("som", "hotspot"):
        keys = (sample_id[pool_rows], bmu_distance[pool_rows], rules_satisfied[pool_rows])
    elif pool_kind == "msp":
        keys = (sample_id[pool_rows], -confidence[pool_rows])
    elif pool_kind in ("random", "random_hotspot"):
        keys = (np.random.default_rng(seed).permutation(len(pool_rows)),)
    else:
        raise ValueError(f"Unknown pool kind '{pool_kind}'.")
    return pool_rows[np.lexsort(keys)]


def curriculum_epoch_indices(
    easy_rows: np.ndarray, ordered_pool: np.ndarray, schedule: Schedule
) -> EpochIndices:
    """The function the trainer calls: rows (easy + unlocked hard) used in each epoch."""

    def indices(epoch: int) -> np.ndarray:
        unlocked = int(round(hard_fraction(schedule, epoch) * len(ordered_pool)))
        return np.concatenate([easy_rows, ordered_pool[:unlocked]])

    return indices
