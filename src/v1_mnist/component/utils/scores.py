"""Per-sample scores computed from the CNN's output that several analyses share."""

from __future__ import annotations

import numpy as np


def max_softmax_score(confidence: np.ndarray) -> np.ndarray:
    """1 - the CNN's top class probability: high means the CNN is unsure."""
    return 1.0 - np.asarray(confidence, dtype=float)
