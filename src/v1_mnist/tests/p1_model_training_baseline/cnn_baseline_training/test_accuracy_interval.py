"""CNN test accuracy is reported with a 95% Wilson interval (project rule: report uncertainty)."""

from __future__ import annotations

import math

from src.v1_mnist.component.utils.stats import wilson_interval, wilson_lower_bound


def test_interval_for_the_current_cnn_run() -> None:
    low, high = wilson_interval(10414, 10500)          # lenet5_best: 99.18% test accuracy
    assert 0.9898 < low < 0.9900 and 0.9933 < high < 0.9934
    assert low < 10414 / 10500 < high


def test_interval_matches_the_error_rate_bound() -> None:
    # The accuracy's upper limit is one minus the error rate's Wilson lower bound.
    _, high = wilson_interval(10414, 10500)
    assert math.isclose(1 - high, wilson_lower_bound(86, 10500), rel_tol=1e-12)


def test_edge_cases() -> None:
    assert math.isclose(wilson_interval(0, 50)[0], 0.0, abs_tol=1e-12)
    assert math.isclose(wilson_interval(50, 50)[1], 1.0, abs_tol=1e-12)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
