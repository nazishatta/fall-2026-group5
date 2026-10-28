"""Safety and protocol tests for the equal-pool baseline experiment."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_fair_budget_baselines as runner,
)


def test_rejects_unequal_svm_budget() -> None:
    """SVM must use the same training pool as other methods."""
    config = {
        "seed_list": [42, 43, 44, 45, 46],
        "training_sample_count": 10000,
        "one_class_svm": {"max_training_samples": 5000},
        "analysis_status": "post_hoc_exploratory",
    }
    with pytest.raises(ValueError, match="budget differs"):
        runner.validate_protocol(config)


def test_requires_five_distinct_seeds() -> None:
    """Repeated seeds do not count as independent fits."""
    config = {
        "seed_list": [42, 42, 43, 44, 45],
        "training_sample_count": 10000,
        "one_class_svm": {"max_training_samples": 10000},
        "analysis_status": "post_hoc_exploratory",
    }
    with pytest.raises(ValueError, match="five distinct"):
        runner.validate_protocol(config)


def test_rejects_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never overwrite previous experiment results."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / "outputs" / "previous_run"
    directory.mkdir(parents=True)
    sentinel = directory / "keep.txt"
    sentinel.write_text("historical", encoding="utf-8")

    config = {
        "run_id": "previous_run",
        "output_root": "outputs",
        "seed_list": [42, 43, 44, 45, 46],
        "training_sample_count": 10000,
        "one_class_svm": {"max_training_samples": 10000},
        "analysis_status": "post_hoc_exploratory",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.run(path)

    assert sentinel.read_text(encoding="utf-8") == "historical"
