"""Three ways the SOM marks a training image as hard (RETRAINING_PROTOCOL.md, section 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifficultyFlags:
    """One boolean per training image for each rule, plus the distance cut-off used."""

    in_hotspot: np.ndarray
    low_purity: np.ndarray
    qe_flag: np.ndarray
    qe_threshold: float

    @property
    def in_som_pool(self) -> np.ndarray:
        """Hard by at least one rule."""
        return self.in_hotspot | self.low_purity | self.qe_flag


def difficulty_flags(
    bmu: np.ndarray,
    bmu_distance: np.ndarray,
    purity_by_neuron: np.ndarray,
    hotspot_neurons: np.ndarray,
    purity_cutoff: float,
    qe_percentile: float,
) -> DifficultyFlags:
    """Apply the three rules. The distance cut-off is a percentile of the training distances only."""
    image_purity = purity_by_neuron[bmu]
    if np.isnan(image_purity).any():
        raise ValueError("Some training images sit on a neuron with no recorded purity.")
    threshold = float(np.percentile(bmu_distance, qe_percentile))
    return DifficultyFlags(
        in_hotspot=np.isin(bmu, hotspot_neurons),
        low_purity=image_purity < purity_cutoff,
        qe_flag=bmu_distance > threshold,
        qe_threshold=threshold,
    )
