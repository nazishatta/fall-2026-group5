"""Tests for the SVHN-on-SOM overlay and miss profile (no torch needed)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis import (
    run_svhn_som_overlay as runner,
)
from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.svhn_som_overlay import (
    NO_CLASS,
    class_landing_profile,
    miss_typicality,
    neuron_dominant_class,
    save_svhn_overlay,
    svhn_neuron_summary,
    top_missed_neurons,
)

THRESHOLD = 1.0


def small_counts() -> np.ndarray:
    """Two classes, four neurons; neuron 3 is empty."""
    return np.array([[5, 1, 0, 0], [0, 3, 4, 0]], dtype=float)


def test_dominant_class_and_purity() -> None:
    dominant, purity = neuron_dominant_class(small_counts())
    assert dominant.tolist() == [0, 1, 1, NO_CLASS]
    assert np.allclose(purity[:3], [1.0, 0.75, 1.0])
    assert np.isnan(purity[3])


def test_dominant_class_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        neuron_dominant_class(np.array([[-1.0, 2.0]]))


def test_neuron_summary_counts_misses_at_or_below_cutoff() -> None:
    bmu = np.array([0, 0, 0, 2])
    qe = np.array([0.5, 1.0, 2.0, 0.2])
    summary = svhn_neuron_summary(bmu, qe, THRESHOLD, 4)
    assert summary["hits"].tolist() == [3, 0, 1, 0]
    assert summary["missed"].tolist() == [2, 0, 1, 0]
    assert summary["miss_rate"][0] == pytest.approx(2 / 3)
    assert np.isnan(summary["miss_rate"][[1, 3]]).all()


def test_summary_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        svhn_neuron_summary(np.array([4]), np.array([0.1]), THRESHOLD, 4)
    with pytest.raises(ValueError):
        svhn_neuron_summary(np.array([0]), np.array([np.nan]), THRESHOLD, 4)
    with pytest.raises(ValueError):
        svhn_neuron_summary(np.array([0]), np.array([0.1]), -1.0, 4)


def test_class_profile_adds_up_to_totals() -> None:
    dominant, _ = neuron_dominant_class(small_counts())
    bmu = np.array([0, 0, 1, 2, 3, 3])
    qe = np.array([0.5, 2.0, 0.5, 0.5, 0.5, 2.0])
    rows = class_landing_profile(bmu, qe, THRESHOLD, dominant, 2)
    assert sum(r["svhn_images"] for r in rows) == 6
    assert sum(r["missed"] for r in rows) == 4
    assert [r["dominant_mnist_class"] for r in rows] == [0, 1, "none"]
    assert sum(r["share_of_all_missed"] for r in rows) == pytest.approx(1.0)


def test_typicality_compares_to_cell_median() -> None:
    val_bmu = np.array([0, 0, 0, 1])
    val_qe = np.array([1.0, 1.0, 1.0, 5.0])
    bmu = np.array([0, 0, 1, 2])
    qe = np.array([0.5, 2.0, 0.5, 0.5])
    result = miss_typicality(bmu, qe, THRESHOLD, val_bmu, val_qe, 4, 3)
    # Missed: images 0, 2, 3. Neuron 1 has one digit and neuron 2 none, so only image 0 is compared.
    assert result["missed_images"] == 3
    assert result["compared_images"] == 1
    assert result["median_qe_ratio"] == pytest.approx(0.5)
    assert result["share_at_or_below_cell_median"] == 1.0


def test_top_missed_neurons_skips_zero_and_uses_none_for_nan() -> None:
    dominant, purity = neuron_dominant_class(small_counts())
    summary = svhn_neuron_summary(np.array([1, 1, 3]), np.array([0.1, 0.2, 0.3]), THRESHOLD, 4)
    rows = top_missed_neurons(summary, dominant, purity, 4)
    assert [r["neuron"] for r in rows] == [1, 3]
    assert rows[1]["dominant_mnist_class"] is None
    assert rows[1]["class_purity"] is None


def hex_centers(count: int) -> np.ndarray:
    return np.array([[float(i % 2) + 0.5 * (i // 2 % 2), float(i // 2) * 0.87] for i in range(count)])


def test_overlay_writes_pdf_and_refuses_overwrite(tmp_path: Path) -> None:
    dominant, _ = neuron_dominant_class(small_counts())
    summary = svhn_neuron_summary(np.array([0, 1, 1]), np.array([0.1, 2.0, 0.3]), THRESHOLD, 4)
    target = tmp_path / "overlay.pdf"
    saved = save_svhn_overlay(hex_centers(4), summary, dominant, 2, THRESHOLD, 0.56, target)
    assert saved.read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        save_svhn_overlay(hex_centers(4), summary, dominant, 2, THRESHOLD, 0.56, target)


def test_overlay_rejects_non_pdf(tmp_path: Path) -> None:
    dominant, _ = neuron_dominant_class(small_counts())
    summary = svhn_neuron_summary(np.array([0]), np.array([0.1]), THRESHOLD, 4)
    with pytest.raises(ValueError):
        save_svhn_overlay(hex_centers(4), summary, dominant, 2, THRESHOLD, 0.56, tmp_path / "overlay.png")


def synthetic_run(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Build a tiny config, reference and scores that the real code path can write."""
    inputs = {}
    for name in ("train_class_neuron", "validation_bmu", "validation_qe", "historical_svhn_qe"):
        path = tmp_path / f"{name}.dat"
        path.write_bytes(name.encode())
        inputs[name] = path.name
    config = {
        "run_id": "unit_test_run", "class_count": 2, "hex_radius": 0.56,
        "min_validation_hits": 1, "top_missed_neurons": 3, "max_abs_qe_difference": 1e-4,
        "expected": {"svhn_samples": 6, "som_units": 4},
        "inputs": inputs,
    }
    reference = {
        "class_counts": small_counts(),
        "validation_bmu": np.array([0, 0, 1, 2]),
        "validation_qe": np.array([1.0, 1.2, 1.1, 0.9]),
    }
    scores = {
        "bmu": np.array([0, 0, 1, 2, 3, 3]),
        "qe": np.array([0.5, 2.0, 0.5, 0.5, 0.5, 2.0]),
        "centers": hex_centers(4), "threshold": THRESHOLD, "hashes": {"som_model": "abc"},
    }
    return config, reference, scores, tmp_path


def test_write_run_creates_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, reference, scores, root = synthetic_run(tmp_path)
    monkeypatch.setattr(runner, "ROOT", root)
    destination = tmp_path / "out" / "unit_test_run"
    result = runner.write_run(destination, config, "cfg-hash", scores, reference, 0.0)
    assert result["missed"] == 4 and result["detected"] == 2
    for name in (
        "assignments/svhn_bmu.npy", "assignments/svhn_qe.npy", "tables/svhn_neuron_summary.csv",
        "tables/svhn_class_landing_profile.csv", "pdf/svhn_som_overlay.pdf", "analysis.json",
    ):
        assert (destination / name).is_file(), name
    saved = json.loads((destination / "analysis.json").read_text(encoding="utf-8"))
    assert saved["missed"] == 4
    assert np.load(destination / "assignments" / "svhn_bmu.npy").tolist() == [0, 0, 1, 2, 3, 3]


def test_write_run_refuses_existing_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, reference, scores, root = synthetic_run(tmp_path)
    monkeypatch.setattr(runner, "ROOT", root)
    destination = tmp_path / "out" / "unit_test_run"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.write_run(destination, config, "cfg-hash", scores, reference, 0.0)


def test_write_run_rejects_wrong_sample_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, reference, scores, root = synthetic_run(tmp_path)
    monkeypatch.setattr(runner, "ROOT", root)
    config["expected"]["svhn_samples"] = 7
    with pytest.raises(RuntimeError):
        runner.write_run(tmp_path / "out" / "x", config, "h", scores, reference, 0.0)
    assert not (tmp_path / "out" / "x").exists()


def test_history_check_flags_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    np.save(tmp_path / "history.npy", np.array([1.0, 2.0]))
    config = {"inputs": {"historical_svhn_qe": "history.npy"}, "max_abs_qe_difference": 1e-4}
    assert runner.check_against_history(config, np.array([1.0, 2.00005])) == pytest.approx(5e-5)
    with pytest.raises(RuntimeError):
        runner.check_against_history(config, np.array([1.0, 2.1]))


def test_read_config_rejects_unsafe_run_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: ../escape\nmin_validation_hits: 1\ntop_missed_neurons: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.read_config(path)
