"""Tests for the exploratory Phase 4 test breakdown (pure counting, no models)."""

from __future__ import annotations

import numpy as np

from src.v1_mnist.component.utils.retraining_breakdown import errors_fixed, per_class_error
from src.v1_mnist.component.utils.retraining_cli import STAGES


def test_errors_fixed_counts_both_directions() -> None:
    base = np.array([True, True, False, False, True])
    arm = np.array([False, True, True, False, False])
    everything = np.ones(5, dtype=bool)
    counts = errors_fixed(base, arm, everything)
    assert counts == {"baseline_errors": 3, "arm_errors": 2, "fixed": 2, "new": 1, "net_fixed": 1}
    # Net fixed always equals the drop in the number of errors.
    assert counts["net_fixed"] == counts["baseline_errors"] - counts["arm_errors"]


def test_errors_fixed_respects_the_mask() -> None:
    base = np.array([True, True, False, False])
    arm = np.array([False, False, True, False])
    first_two = np.array([True, True, False, False])
    counts = errors_fixed(base, arm, first_two)
    assert counts["fixed"] == 2 and counts["new"] == 0 and counts["baseline_errors"] == 2


def test_per_class_error() -> None:
    labels = np.array([0, 0, 1, 1, 1, 2])
    wrong = np.array([True, False, False, False, True, False])
    assert np.allclose(per_class_error(wrong, labels, 3), [0.5, 1 / 3, 0.0])


def test_breakdown_is_a_stage_after_the_test_pass() -> None:
    assert STAGES.index("breakdown") == STAGES.index("final_test") + 1
