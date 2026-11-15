"""Tests for the QE threshold sweep (synthetic data only; no test split, no torch)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection import (
    run_qe_threshold_sweep as runner,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.qe_threshold_sweep import (
    check_percentiles,
    permutation_auroc_null,
    threshold_sweep,
)

TRAIN = np.arange(1.0, 101.0)


def test_cutoffs_come_from_training_only_and_rates_are_exact() -> None:
    ids = np.array([1.0, 2.0, 96.0, 3.0])
    oods = np.array([50.0, 97.0, 120.0, 200.0])
    rows = threshold_sweep(TRAIN, ids, oods, [90.0, 95.0])
    assert [row["percentile"] for row in rows] == [90.0, 95.0]
    assert rows[0]["threshold"] == pytest.approx(np.percentile(TRAIN, 90.0))
    assert rows[1]["threshold"] == pytest.approx(np.percentile(TRAIN, 95.0))
    # 95th percentile of 1..100 is 95.05: 96 is flagged, 50 is not.
    assert rows[1]["id_false_alarm_rate"] == 0.25
    assert rows[1]["ood_detection_rate"] == 0.75
    assert rows[1]["ood_missed"] == 1


def test_higher_percentile_never_flags_more() -> None:
    rng = np.random.default_rng(0)
    ids, oods = rng.gamma(2.0, 1.0, 500), rng.gamma(4.0, 1.0, 500)
    rows = threshold_sweep(rng.gamma(2.0, 1.0, 2000), ids, oods, [90.0, 95.0, 99.0])
    detections = [row["ood_detection_rate"] for row in rows]
    alarms = [row["id_false_alarm_rate"] for row in rows]
    assert detections == sorted(detections, reverse=True)
    assert alarms == sorted(alarms, reverse=True)


def test_wilson_interval_contains_the_rate() -> None:
    row = threshold_sweep(TRAIN, np.array([1.0, 99.0, 2.0, 3.0]), np.array([99.0, 98.0]), [95.0])[0]
    assert row["id_false_alarm_low"] <= row["id_false_alarm_rate"] <= row["id_false_alarm_high"]
    assert row["ood_detection_low"] <= row["ood_detection_rate"] <= row["ood_detection_high"]


def test_bad_percentiles_are_rejected() -> None:
    for bad in ([], [95.0, 95.0], [0.0], [100.0], [-1.0]):
        with pytest.raises(ValueError):
            check_percentiles(bad)


def test_bad_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        threshold_sweep(TRAIN, np.array([np.nan]), np.array([1.0]), [95.0])
    with pytest.raises(ValueError):
        threshold_sweep(TRAIN, np.array([-1.0]), np.array([1.0]), [95.0])


def test_permutation_null_sits_near_half_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    ids, oods = rng.normal(1.0, 0.3, 300).clip(0), rng.normal(2.0, 0.3, 300).clip(0)
    first = permutation_auroc_null(ids, oods, iterations=50, seed=7)
    second = permutation_auroc_null(ids, oods, iterations=50, seed=7)
    assert first == second
    assert first["observed_auroc"] > 0.95
    assert abs(first["shuffled_mean"] - 0.5) < 0.05
    assert first["p_value"] == pytest.approx(1 / 51)


def make_inputs(root: Path, threshold: float) -> dict[str, Any]:
    """Tiny arrays and a config that point at them."""
    rng = np.random.default_rng(2)
    train = rng.gamma(2.0, 1.0, 200)
    arrays = {"train_qe": train, "validation_qe": rng.gamma(2.0, 1.0, 50), "svhn_qe": rng.gamma(5.0, 1.0, 80)}
    inputs = {}
    for name, values in arrays.items():
        np.save(root / f"{name}.npy", values)
        inputs[name] = f"{name}.npy"
    (root / "qe_summary.json").write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    inputs["qe_summary"] = "qe_summary.json"
    return {
        "run_id": "unit_test_sweep", "output_root": "out", "percentiles": [90.0, 95.0, 99.0],
        "target_tpr": 0.95, "bootstrap": {"iterations": 20, "seed": 1, "confidence": 0.95},
        "permutation_null": {"iterations": 20, "seed": 1},
        "expected": {
            "training_samples": 200, "validation_samples": 50, "svhn_samples": 80,
            "frozen_percentile_95_threshold": threshold,
        },
        "inputs": inputs,
    }


def write_config(root: Path, config: dict[str, Any]) -> Path:
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_writes_outputs_and_refuses_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    rng = np.random.default_rng(2)
    threshold = float(np.percentile(rng.gamma(2.0, 1.0, 200), 95.0))
    config_path = write_config(tmp_path, make_inputs(tmp_path, threshold))
    result = runner.run(config_path)
    destination = tmp_path / "out" / "unit_test_sweep"
    assert (destination / "tables" / "qe_threshold_sweep.csv").is_file()
    saved = json.loads((destination / "qe_threshold_sweep.json").read_text(encoding="utf-8"))
    assert [row["percentile"] for row in saved["threshold_sweep"]] == [90.0, 95.0, 99.0]
    assert 0.5 < result["threshold_free_metrics"]["auroc"] <= 1.0
    with pytest.raises(FileExistsError):
        runner.run(config_path)


def test_run_stops_if_frozen_cutoff_does_not_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config_path = write_config(tmp_path, make_inputs(tmp_path, 123.456))
    with pytest.raises(RuntimeError):
        runner.run(config_path)
    assert not (tmp_path / "out" / "unit_test_sweep").exists()


def test_run_rejects_wrong_array_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config = make_inputs(tmp_path, 1.0)
    config["expected"]["svhn_samples"] = 81
    with pytest.raises(RuntimeError):
        runner.run(write_config(tmp_path, config))


def test_config_requires_95th_percentile_and_safe_run_id(tmp_path: Path) -> None:
    bad_id = tmp_path / "a.yaml"
    bad_id.write_text("run_id: ../x\npercentiles: [95.0]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.read_config(bad_id)
    no_95 = tmp_path / "b.yaml"
    no_95.write_text("run_id: ok_run\npercentiles: [90.0, 99.0]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.read_config(no_95)
