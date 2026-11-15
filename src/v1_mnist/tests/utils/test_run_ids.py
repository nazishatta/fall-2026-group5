"""Tests for the shared run-ID check."""

from __future__ import annotations

from typing import Any

import pytest

from src.v1_mnist.component.utils.run_ids import check_run_id


def test_safe_run_ids_are_returned() -> None:
    """Letters, digits, underscores and hyphens are accepted."""
    assert (
        check_run_id("mnist_svhn_qe_norm_dependence_s42_v1")
        == "mnist_svhn_qe_norm_dependence_s42_v1"
    )
    assert check_run_id("run-2") == "run-2"


@pytest.mark.parametrize(
    "value", ["", "../escape", "a/b", "a.b", "has space", None, 42]
)
def test_unsafe_run_ids_are_refused(value: Any) -> None:
    """Empty, non-string and path-like run IDs raise ValueError."""
    with pytest.raises(ValueError, match="Invalid run ID"):
        check_run_id(value)
