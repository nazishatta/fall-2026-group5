"""Tests for the autoencoder scaling diagnostic (synthetic data; no torch, no test split)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark import (
    autoencoder_scaling as scaling,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark import (
    run_autoencoder_scaling as runner,
)

SEEDS = [1, 2, 3, 4, 5]


def fake_fitter(train: np.ndarray, joined: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, list[float]]:
    """Stand-in for the autoencoder: score = mean absolute value of the input."""
    assert train.dtype == np.float32 and joined.dtype == np.float32
    assert set(settings) == {"seed", "autoencoder"}
    return np.abs(joined).mean(axis=1).astype(np.float64), [1.0, 0.5]


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 5.0, (60, 6))
    validation = rng.normal(0.0, 5.0, (20, 6))
    svhn = rng.normal(0.0, 0.5, (30, 6))  # shorter embeddings, like SVHN
    return train, validation, svhn


def make_config(**changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "run_id": "unit_test_scaling", "output_root": "out", "reference_metrics": "reference.json",
        "seeds": SEEDS, "training_sizes": [30, 60], "scalings": ["raw", "standardized"], "target_tpr": 0.95,
        "scoring": {"batch_size": 8, "bootstrap_iterations": 10, "bootstrap_seed": 1},
        "autoencoder": {"hidden_dim": 4}, "isolation_forest": {"n_estimators": 10, "max_samples": 16, "contamination": 0.05},
        "expected": {"training_samples": 60},
    }
    config.update(changes)
    return config


def test_standardized_inputs_use_training_rows_only() -> None:
    train, validation, svhn = make_data()
    joined = np.concatenate((validation, svhn))
    train_in, _ = scaling.prepare_inputs(train, joined, "standardized")
    assert np.allclose(train_in.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(train_in.std(axis=0), 1.0, atol=1e-4)
    raw_train, raw_joined = scaling.prepare_inputs(train, joined, "raw")
    assert np.allclose(raw_train, train.astype(np.float32)) and np.allclose(raw_joined, joined.astype(np.float32))
    with pytest.raises(ValueError):
        scaling.prepare_inputs(train, joined, "minmax")


def test_length_control_sees_shorter_novel_embeddings() -> None:
    _, validation, svhn = make_data()
    config = make_config()
    control = runner.length_control(validation, svhn, config)
    assert control["auroc"] < 0.1
    assert control["mean_length_svhn"] < control["mean_length_mnist_validation"]
    assert control["auroc_interval_95"][0] <= control["auroc"] <= control["auroc_interval_95"][1] + 1e-12


def test_design_checks() -> None:
    for bad in (
        {"sizes": [], "scalings": ["raw"], "seeds": SEEDS},
        {"sizes": [61], "scalings": ["raw"], "seeds": SEEDS},
        {"sizes": [30, 30], "scalings": ["raw"], "seeds": SEEDS},
        {"sizes": [30], "scalings": ["log"], "seeds": SEEDS},
        {"sizes": [30], "scalings": ["raw"], "seeds": [1, 2, 3]},
        {"sizes": [30], "scalings": ["raw"], "seeds": [1, 1, 2, 3, 4]},
    ):
        with pytest.raises(ValueError):
            scaling.check_design(bad["sizes"], bad["scalings"], bad["seeds"], 60)
    scaling.check_design([30, 60], ["raw", "standardized"], SEEDS, 60)


def test_every_cell_is_evaluated_once_per_seed() -> None:
    train, validation, svhn = make_data()
    rows = scaling.evaluate_conditions(train, np.concatenate((validation, svhn)), len(validation), make_config(), fake_fitter)
    # Per seed: 2 forest rows (one per size) + 4 autoencoder rows (size x scaling).
    assert len(rows) == len(SEEDS) * 6
    summary = scaling.summarize_conditions(rows)
    assert len(summary) == 6
    assert {(s["method"], s["training_size"], s["scaling"]) for s in summary} >= {
        ("autoencoder", 30, "raw"), ("autoencoder", 60, "standardized"), ("isolation_forest", 30, "standardized"),
    }
    assert all(len(s["auroc_per_seed"]) == 5 for s in summary)
    # Same seed and size give the same training rows for every scaling.
    first = {(r["seed"], r["training_size"]): r["training_subset_sha256"] for r in rows}
    assert all(first[(r["seed"], r["training_size"])] == r["training_subset_sha256"] for r in rows)


def test_summary_needs_five_seeds() -> None:
    rows = [{"method": "autoencoder", "training_size": 1, "scaling": "raw", "seed": s, "auroc": 0.5} for s in range(4)]
    with pytest.raises(ValueError):
        scaling.summarize_conditions(rows)


def reference_file(root: Path, rows: list[dict[str, Any]]) -> None:
    per_seed = [
        {"seed": seed, "metrics": {m: {"auroc": next(r["auroc"] for r in rows if r["seed"] == seed and r["method"] == m
                                       and r["training_size"] == 30 and r["scaling"] == "standardized") + 0.01}
                                   for m in ("autoencoder", "isolation_forest")}}
        for seed in SEEDS
    ]
    (root / "reference.json").write_text(
        json.dumps({"fitting": {"common_training_rows_per_seed": 30}, "fitted_methods_per_seed": per_seed}), encoding="utf-8"
    )


def patch_heavy(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    train, validation, svhn = make_data()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "load_embeddings", lambda config: {
        "train": train, "validation": validation, "svhn": svhn, "hashes": {"train": "abc"}, "provenance": {"qe_threshold": 1.0},
    })
    monkeypatch.setattr(runner, "get_autoencoder_fitter", lambda: fake_fitter)


def write_config(root: Path, config: dict[str, Any]) -> Path:
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump({**config, "expected": {**config["expected"], "svhn_samples": 30, "embedding_dimensions": 6}}), encoding="utf-8")
    return path


def test_run_writes_outputs_and_reference_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_heavy(monkeypatch, tmp_path)
    train, validation, svhn = make_data()
    rows = scaling.evaluate_conditions(train, np.concatenate((validation, svhn)), len(validation), make_config(), fake_fitter)
    reference_file(tmp_path, rows)
    result = runner.run(write_config(tmp_path, make_config()))
    out = tmp_path / "out" / "unit_test_scaling"
    assert (out / "autoencoder_scaling_metrics.json").is_file()
    assert (out / "tables" / "autoencoder_scaling_by_condition.csv").is_file()
    saved = json.loads((out / "autoencoder_scaling_metrics.json").read_text(encoding="utf-8"))
    assert len(saved["condition_summary"]) == 6 and saved["study_type"] == "post_hoc_exploratory"
    assert result["reference_check"]["autoencoder"]["max_abs_auroc_difference"] == pytest.approx(0.01)
    with pytest.raises(FileExistsError):
        runner.run(write_config(tmp_path, make_config()))


def test_run_without_reference_file_reports_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_heavy(monkeypatch, tmp_path)
    assert runner.run(write_config(tmp_path, make_config()))["reference_check"] is None


def test_run_rejects_wrong_svhn_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_heavy(monkeypatch, tmp_path)
    config = make_config()
    config["expected"] = {"training_samples": 60}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**config, "expected": {"training_samples": 60, "svhn_samples": 31, "embedding_dimensions": 6}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        runner.run(path)
    assert not (tmp_path / "out" / "unit_test_scaling").exists()


def test_read_config_rejects_unsafe_run_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(make_config(run_id="../escape")), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.read_config(path)
