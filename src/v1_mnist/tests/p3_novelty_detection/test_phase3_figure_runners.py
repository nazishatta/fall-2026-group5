"""Safety tests for reproducible Phase 3 PDF figure runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_missed_svhn_figure,
    run_qe_umatrix_validation_figure,
    run_u_matrix_grid_figure,
)

RUNNERS = [
    run_missed_svhn_figure,
    run_qe_umatrix_validation_figure,
    run_u_matrix_grid_figure,
]


@pytest.mark.parametrize("runner", RUNNERS)
def test_figure_runner_rejects_existing_run(
    runner: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing results must remain untouched."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    destination = tmp_path / "outputs" / "existing_run"
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("original result", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "run_id": "existing_run",
            "output_root": "outputs",
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        runner.run(config_path)

    assert sentinel.read_text(encoding="utf-8") == "original result"


@pytest.mark.parametrize("runner", RUNNERS)
def test_figure_runner_rejects_path_escape(
    runner: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Figure configuration paths must not escape the repository."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="outside repository|escapes repository"):
        runner.repository_path("../outside")


def test_hexagon_geometry_has_six_vertices() -> None:
    """Each neuron is drawn as one six-sided cell."""
    centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    vertices = run_u_matrix_grid_figure.hexagon_vertices(
        centers, 0.56
    )

    assert vertices.shape == (2, 6, 2)
    np.testing.assert_allclose(vertices.mean(axis=1), centers, atol=1e-12)


def test_hexagon_geometry_rejects_invalid_radius() -> None:
    """Zero or negative radius cannot produce a valid grid."""
    centers = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="Invalid hexagon geometry"):
        run_u_matrix_grid_figure.hexagon_vertices(centers, 0.0)
