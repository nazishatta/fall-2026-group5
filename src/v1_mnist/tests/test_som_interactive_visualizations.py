"""Tests for SOM interactive artifact loading and figure generation."""

from pathlib import Path

import pytest

from src.v1_mnist.component.visualization.generate_som_qe_interactive import (
    build_qe_figure,
)
from src.v1_mnist.component.visualization.generate_som_quality_interactive import (
    build_quality_figure,
)
from src.v1_mnist.component.visualization.generate_som_radius_interactive import (
    build_radius_figure,
)
from src.v1_mnist.component.visualization.som_interactive import (
    load_interactive_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

METRICS_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/logs/"
    "som_15x15_seed42_final_metrics.json"
)

QE_HISTORY_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/som/tables/"
    "som_15x15_seed42_final_qe_history.csv"
)


def test_interactive_artifacts_load_expected_run() -> None:
    artifacts = load_interactive_artifacts(
        METRICS_PATH,
        QE_HISTORY_PATH,
    )

    assert artifacts.run_id == "som_15x15_seed42_final"
    assert artifacts.grid_height == 15
    assert artifacts.grid_width == 15
    assert artifacts.num_neurons == 225
    assert artifacts.random_seed == 42
    assert artifacts.epochs == 250
    assert len(artifacts.qe_history) == 251


def test_qe_figure_contains_expected_trace() -> None:
    figure = build_qe_figure()

    assert len(figure.data) >= 1
    assert figure.data[0].name == "Training QE"
    assert len(figure.data[0].x) == 251
    assert len(figure.data[0].y) == 251


def test_radius_figure_contains_expected_trace() -> None:
    figure = build_radius_figure()

    assert len(figure.data) == 1
    assert figure.data[0].name == "Neighborhood radius"
    assert len(figure.data[0].x) == 251
    assert len(figure.data[0].y) == 251


def test_quality_figure_uses_development_splits_only() -> None:
    figure = build_quality_figure()

    assert len(figure.data) == 4

    title = figure.layout.title.text

    assert "Train vs Validation" in title
    assert "development splits only" in title
    assert "Test" not in title


def test_loader_rejects_missing_metrics_file(
    tmp_path: Path,
) -> None:
    missing_metrics = tmp_path / "missing_metrics.json"

    with pytest.raises(FileNotFoundError):
        load_interactive_artifacts(
            missing_metrics,
            QE_HISTORY_PATH,
        )
