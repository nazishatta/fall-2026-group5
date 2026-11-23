"""The Fashion-MNIST settings: they load, name Fashion-MNIST, and never write into the MNIST results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.dataset_info import dataset_info
from src.v1_mnist.component.utils.image_datasets import DatasetSpec, dataset_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "v1_fashion_mnist" / "component" / "configs"
CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name != "base.yaml")


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


def test_there_are_configs() -> None:
    names = {p.name for p in CONFIGS}
    assert {"cnn_baseline.yaml", "som.yaml"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads_with_the_fashion_mnist_dataset(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw.get("dataset"), str):
        # Two Phase 3 configs name their dataset as a plain string, as their MNIST originals do;
        # the code reads that string (it replaces base.yaml's dataset section when merged).
        assert raw["dataset"] == "fashion_mnist"
        return
    config = load_config(str(path))
    assert dataset_spec(config) == DatasetSpec("fashion_mnist", 1, 28)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_result_goes_to_the_mnist_folders(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for text in _strings(raw):
        assert "outputs/v1_mnist" not in text, text
        assert "src/v1_mnist/component/configs" not in text, text
        if "outputs/" in text:
            assert "outputs/v1_fashion_mnist" in text, text


def test_loaded_paths_point_at_the_fashion_mnist_results() -> None:
    config = load_config(str(CONFIG_DIR / "cnn_baseline.yaml"))
    assert "outputs/v1_fashion_mnist/p1_model_training_baseline" in Path(config.paths.output_dir).as_posix()


def test_ten_named_classes_for_figure_titles() -> None:
    info = dataset_info(load_config(str(CONFIG_DIR / "som.yaml")))
    assert info.num_classes == 10
    assert info.class_name(0) == "T-shirt/top" and info.class_name(9) == "Ankle boot"


def test_the_som_run_ids_are_new() -> None:
    """Fashion-MNIST has its own runs: no MNIST '_v2' run IDs, and the CNN run is not MNIST's."""
    for path in CONFIGS:
        assert "_v2" not in path.read_text(encoding="utf-8"), path.name
    assert load_config(str(CONFIG_DIR / "cnn_baseline.yaml")).run.id != "lenet5_best"
