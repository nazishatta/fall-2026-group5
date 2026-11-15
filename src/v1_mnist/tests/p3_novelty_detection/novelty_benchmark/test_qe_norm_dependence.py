"""Tests for the QE-norm dependence check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark import (
    run_qe_norm_dependence as runner,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.qe_norm_dependence import (
    analyze,
    embedding_norms,
    norm_matched_qe,
    spearman,
)

SETTINGS: dict[str, Any] = {
    "target_tpr": 0.95,
    "bootstrap_iterations": 20,
    "seed": 42,
    "confidence": 0.95,
    "norm_matched_band": [5, 95],
}


def test_norms_and_spearman() -> None:
    """Norms are row lengths; Spearman needs aligned vectors."""
    assert embedding_norms(np.array([[3.0, 4.0], [0.0, 1.0]])).tolist() == [5.0, 1.0]
    with pytest.raises(ValueError):
        embedding_norms(np.array([1.0, 2.0]))
    assert spearman(np.arange(5.0), np.arange(5.0) * 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman(np.arange(3.0), np.arange(4.0))


def test_analysis_reports_qe_and_the_norm_link() -> None:
    """QE reference with an interval, three correlations and the band check."""
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 4)) * 3
    validation = rng.normal(size=(200, 4)) * 3
    svhn = rng.normal(size=(300, 4))
    result = analyze(
        train,
        validation,
        svhn,
        rng.random(200),
        rng.random(300) + 0.5,
        qe_threshold=0.9,
        settings=SETTINGS,
    )
    low, high = result["qe_reference"]["auroc_ci"]
    assert low <= result["qe_reference"]["auroc"] <= high
    assert set(result["qe_norm_spearman"]) == {
        "mnist_validation",
        "svhn_test",
        "pooled",
    }
    assert result["norms"]["mnist_train"]["count"] == 400
    assert result["norm_matched_qe"]["band_percentiles"] == [5.0, 95.0]
    with pytest.raises(ValueError, match="aligned"):
        analyze(
            train,
            validation,
            svhn,
            rng.random(10),
            rng.random(300),
            qe_threshold=0.9,
            settings=SETTINGS,
        )


def test_norm_matched_band_keeps_only_ordinary_norms() -> None:
    """Only images inside the training-norm band are scored."""
    train = np.linspace(1.0, 10.0, 100)
    id_norms, ood_norms = np.array([0.5, 5.0, 6.0]), np.array([5.5, 20.0, 7.0])
    result = norm_matched_qe(
        train,
        id_norms,
        ood_norms,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.9, 0.8, 0.7]),
        band=[5, 95],
        qe_threshold=0.5,
        settings=SETTINGS,
    )
    assert (result["id_count"], result["ood_count"]) == (2, 2)
    assert result["qe"]["auroc"] == 1.0
    tiny = norm_matched_qe(
        train,
        np.array([0.1]),
        np.array([0.2]),
        np.array([0.1]),
        np.array([0.2]),
        band=[5, 95],
        qe_threshold=0.5,
        settings=SETTINGS,
    )
    assert tiny["qe"] is None


def test_config_is_fixed_before_the_run() -> None:
    """New run ID, training-derived band and the recorded QE reference."""
    config = runner.read_config(runner.CONFIG_PATH)
    assert config["run_id"] == "mnist_svhn_qe_norm_dependence_s42_v1"
    assert config["norm_matched_band"] == [5, 95]
    assert config["expected"]["qe_reference_auroc"] == pytest.approx(0.9811068135920626)


def test_bad_run_id_and_existing_run_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe run IDs fail, and an existing run folder is never overwritten."""
    config = yaml.safe_load(runner.CONFIG_PATH.read_text(encoding="utf-8"))
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump(dict(config, run_id="../escape")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid run ID"):
        runner.read_config(tmp_path / "bad.yaml")
    (tmp_path / "good.yaml").write_text(
        yaml.safe_dump(dict(config, output_root="out")), encoding="utf-8"
    )
    (tmp_path / "out" / config["run_id"]).mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(FileExistsError, match="Existing run"):
        runner.run(tmp_path / "good.yaml")


def test_qe_reference_check_stops_on_a_changed_auroc() -> None:
    """The run stops if QE no longer reproduces the recorded headline AUROC."""
    config = {"expected": {"qe_reference_auroc": 0.98, "qe_auroc_tolerance": 1e-6}}
    runner.check_qe_reference({"qe_reference": {"auroc": 0.98}}, config)
    with pytest.raises(RuntimeError, match="differs"):
        runner.check_qe_reference({"qe_reference": {"auroc": 0.97}}, config)
