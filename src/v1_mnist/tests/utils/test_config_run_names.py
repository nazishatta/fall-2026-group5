"""Configs use the proposal's names; the old week-numbered run names are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.config import load_config


def write_config(folder: Path, run_name: str) -> Path:
    path = folder / "run.yaml"
    path.write_text(f"run:\n  name: {run_name}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("old, new", [("week_2", "cnn_baseline"), ("week_3", "som")])
def test_old_run_names_are_refused(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ValueError, match=f"Use '{new}'"):
        load_config(write_config(tmp_path, old))


@pytest.mark.parametrize(
    "name, folder",
    [
        ("cnn_baseline", "p1_model_training_baseline/cnn_baseline_training"),
        ("som", "p2_som_training_visualization/som_nnsom_training"),
    ],
)
def test_run_names_map_to_phase_folders(tmp_path: Path, name: str, folder: str) -> None:
    config = load_config(write_config(tmp_path, name))
    assert Path(config.paths.output_dir) == Path("./outputs/v1_mnist") / folder
