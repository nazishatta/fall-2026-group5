"""Week 4 (Cluster Analysis): the quantization-error model check."""

import math
from pathlib import Path

import pytest

from src.v1_mnist.component.p2_som_training_visualization.cluster_analysis.run_cluster_analysis import (
    check_quantization_error,
)
from src.v1_mnist.component.utils.run_settings import analysis_config

REPO_ROOT = Path(__file__).resolve().parents[5]
QE_RTOL = float(analysis_config(None, REPO_ROOT).qe_relative_tolerance)   # 1e-6 in configs/som.yaml


def test_rounding_difference_between_library_versions_passes() -> None:
    # Real values: Week 3 on EC2 vs recomputed on the laptop (15x15 training split).
    result = check_quantization_error("train", 1.038720629452448, 1.0387206275713383, QE_RTOL)
    assert result["relative_difference"] < 1e-8
    assert result["relative_tolerance"] == QE_RTOL == 1e-6


def test_a_different_model_or_data_fails() -> None:
    # The 10x10 model's QE (about 1.19) is far outside the tolerance of the 15x15 value.
    with pytest.raises(RuntimeError, match="model check failed"):
        check_quantization_error("train", 1.19, 1.0387206275713383, QE_RTOL)


def test_small_but_real_change_fails() -> None:
    with pytest.raises(RuntimeError):
        check_quantization_error("val", 1.0387206275713383 * (1 + 1e-4), 1.0387206275713383, QE_RTOL)


def test_nan_fails() -> None:
    with pytest.raises(RuntimeError):
        check_quantization_error("val", math.nan, 1.0, QE_RTOL)
