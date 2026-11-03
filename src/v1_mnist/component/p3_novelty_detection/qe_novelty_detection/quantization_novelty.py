"""Training-calibrated SOM BMU-distance novelty scoring, with no test access."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_distances(path: Path, expected_count: int) -> np.ndarray:
    """Load finite, nonnegative, one-dimensional BMU distances safely."""
    data = np.load(path, allow_pickle=False)
    if data.shape != (expected_count,):
        raise ValueError(f"{path}: expected {(expected_count,)}, got {data.shape}")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"{path}: distances must be numeric")
    result = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{path}: distances must be finite and nonnegative")
    return result


def calculate_threshold(training_distances: np.ndarray, percentile: float) -> float:
    """Calibrate solely on training distances using NumPy's default percentile."""
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be strictly between 0 and 100")
    values = np.asarray(training_distances, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Training distances must be nonempty and one-dimensional")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Training distances must be finite and nonnegative")
    return float(np.percentile(values, percentile))


def summarize(
    train: np.ndarray, validation: np.ndarray, percentile: float
) -> dict[str, object]:
    """Report only development split novelty counts, not external OOD metrics."""
    threshold = calculate_threshold(train, percentile)
    val = np.asarray(validation, dtype=np.float64)
    if (
        val.ndim != 1
        or val.size == 0
        or not np.all(np.isfinite(val))
        or np.any(val < 0)
    ):
        raise ValueError(
            "Validation distances must be nonempty, finite, nonnegative 1D"
        )
    return {
        "method": "som_bmu_quantization_distance",
        "threshold_source": "train_only",
        "threshold_percentile": percentile,
        "threshold_rule": "distance > threshold",
        "threshold": threshold,
        "train_count": int(train.size),
        "train_flagged": int(np.count_nonzero(train > threshold)),
        "validation_count": int(val.size),
        "validation_flagged": int(np.count_nonzero(val > threshold)),
        "validation_flag_rate": float(np.mean(val > threshold)),
        "external_ood_evaluated": False,
        "final_test_used": False,
    }


def error_association(
    validation: np.ndarray, correct: np.ndarray, threshold: float
) -> dict[str, object]:
    """Describe association with classification errors, not external OOD accuracy."""
    if correct.shape != validation.shape or correct.dtype != np.dtype("bool"):
        raise ValueError("CNN correctness must be a matching boolean array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid threshold")
    flagged = validation > threshold
    errors = ~correct
    both = int(np.count_nonzero(flagged & errors))
    return {
        "validation_samples": int(validation.size),
        "flagged_samples": int(np.count_nonzero(flagged)),
        "cnn_errors": int(np.count_nonzero(errors)),
        "flagged_and_error": both,
        "flagged_and_correct": int(np.count_nonzero(flagged & correct)),
        "unflagged_and_error": int(np.count_nonzero(~flagged & errors)),
        "unflagged_and_correct": int(np.count_nonzero(~flagged & correct)),
        "final_test_used": False,
        "external_ood_evaluated": False,
    }
