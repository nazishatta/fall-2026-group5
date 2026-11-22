"""The transform check blocks the train stage only when the config sets a limit (Fashion-MNIST)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytest.importorskip("torch")

from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.augmentation_check import (  # noqa: E402
    accuracy_drop_limit,
    check_path,
    require_passed_check,
)

ROOT = Path(__file__).resolve().parents[5]
MNIST = ROOT / "src/v1_mnist/component/configs/retraining.yaml"
FASHION = ROOT / "src/v1_fashion_mnist/component/configs/retraining.yaml"


def _settings(config: Path, tmp_path: Path) -> Any:
    """The two fields the check reads, as load_experiment_settings builds them (no data needed)."""
    block = yaml.safe_load(config.read_text(encoding="utf-8"))["augmentation"]
    options = {k: v for k, v in block.items() if k not in ("run", "paths", "arms")}
    return SimpleNamespace(options=options, output_dir=tmp_path)


def _write_check(settings: Any, drop: float, passed: bool) -> None:
    path = check_path(settings)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"accuracy_drop": drop, "max_accuracy_drop": 0.02, "passed": passed}))


def test_only_the_fashion_mnist_config_sets_a_limit(tmp_path: Path) -> None:
    assert accuracy_drop_limit(_settings(MNIST, tmp_path)) is None
    assert accuracy_drop_limit(_settings(FASHION, tmp_path)) == 0.02


def test_without_a_limit_nothing_is_required(tmp_path: Path) -> None:
    require_passed_check(_settings(MNIST, tmp_path))      # MNIST: no check file, no error


def test_with_a_limit_the_check_must_exist_and_pass(tmp_path: Path) -> None:
    settings = _settings(FASHION, tmp_path)
    with pytest.raises(RuntimeError, match="run --stage check"):
        require_passed_check(settings)
    _write_check(settings, 0.031, passed=False)
    with pytest.raises(RuntimeError, match="transform check failed"):
        require_passed_check(settings)


def test_a_passed_check_lets_training_start(tmp_path: Path) -> None:
    settings = _settings(FASHION, tmp_path)
    _write_check(settings, 0.012, passed=True)
    require_passed_check(settings)
