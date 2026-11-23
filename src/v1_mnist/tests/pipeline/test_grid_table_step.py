"""The SOM runner adds the grid-size table only when every SOM it compares is trained."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.v1_mnist.component.pipeline.p2_som_training_visualization.run_som_nnsom_training import (
    grid_table_ready,
)

RUN_IDS = ["som_10x10", "som_15x15", "som_20x20"]


def _config(logs: Path, with_table: bool = True) -> SimpleNamespace:
    config = SimpleNamespace(paths=SimpleNamespace(logs_dir=str(logs)))
    if with_table:
        config.grid_comparison = SimpleNamespace(run_ids=RUN_IDS)
    return config


def _trained(logs: Path, *run_ids: str) -> None:
    for run_id in run_ids:
        (logs / f"{run_id}_metrics.json").write_text("{}", encoding="utf-8")


def test_ready_when_every_som_is_trained(tmp_path: Path) -> None:
    _trained(tmp_path, *RUN_IDS)
    assert grid_table_ready(_config(tmp_path), "som_15x15", will_train=False) == (True, [])


def test_waits_for_the_soms_not_yet_trained(tmp_path: Path) -> None:
    _trained(tmp_path, "som_15x15")
    assert grid_table_ready(_config(tmp_path), "som_15x15", will_train=False) == (False, ["som_10x10", "som_20x20"])


def test_the_som_this_run_trains_counts_as_present(tmp_path: Path) -> None:
    _trained(tmp_path, "som_10x10", "som_20x20")
    assert grid_table_ready(_config(tmp_path), "som_15x15", will_train=True) == (True, [])


def test_a_config_without_grid_comparison_never_asks_for_the_table(tmp_path: Path) -> None:
    assert grid_table_ready(_config(tmp_path, with_table=False), "som_10x10", will_train=True) == (False, [])
