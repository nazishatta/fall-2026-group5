"""Both datasets' configs follow the pre-registration, and every stage runs end to end on toy data."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison import (
    analysis,
    figure,
    provenance,
    stages,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    Contrast,
    GridSettings,
    load_grid_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
MNIST = "src/v1_mnist/component/configs/grid_comparison.yaml"
FASHION = "src/v1_fashion_mnist/component/configs/grid_comparison.yaml"


def test_mnist_config_is_the_preregistered_design() -> None:
    s = load_grid_settings(MNIST, REPO_ROOT)
    assert s.grids == (10, 15, 20)
    assert {g: s.candidates(g) for g in s.grids} == {10: (2, 5, 7), 15: (3, 7, 11), 20: (4, 9, 15)}
    assert (s.epochs, s.tune_seed, s.bootstrap_maps, s.resample_seed_base) == (250, 42, 10, 1000)
    assert (s.reps, s.split_seed_base, s.margin, s.low_confidence_quantile) == (200, 5000, 0.02, 0.05)
    assert s.primary == Contrast(20, 15, "auroc_err")
    assert [(c.big, c.small) for c in s.secondary] == [(20, 15)] * 3 + [(15, 10)] * 2
    assert s.exploratory == (Contrast(20, 10, "auroc_err"), Contrast(20, 10, "auroc_lowconf"))
    assert s.expected["validation_errors"] == 92 and s.reproduction is not None
    assert s.protocol_path.is_file()


def test_fashion_config_differs_only_in_dataset_values() -> None:
    m, f = load_grid_settings(MNIST, REPO_ROOT), load_grid_settings(FASHION, REPO_ROOT)
    same = ("grids", "candidate_fractions", "epochs", "tune_seed", "bootstrap_maps", "resample_seed_base", "reps",
            "split_seed_base", "margin", "low_confidence_quantile", "primary", "secondary", "exploratory",
            "protocol_path", "run_id")
    assert all(getattr(m, k) == getattr(f, k) for k in same)
    assert f.reproduction is None and f.expected["validation_errors"] == 876
    assert "v1_fashion_mnist" in str(f.output_dir) and "v1_fashion_mnist" in str(f.embeddings_dir)


def test_names_carry_no_week_numbers() -> None:
    s = load_grid_settings(MNIST, REPO_ROOT)
    assert "week" not in s.run_id and "week" not in str(s.output_dir)
    code = REPO_ROOT / "src/v1_mnist/component/p2_som_training_visualization/som_nnsom_training/grid_comparison"
    assert not [p.name for p in code.rglob("*") if "week" in p.name.lower()]


def _toy_settings(tmp_path: Path) -> GridSettings:
    rng = np.random.default_rng(0)
    emb = tmp_path / "embeddings"
    emb.mkdir()
    np.save(emb / "train_features.npy", rng.normal(size=(150, 5)))
    np.save(emb / "val_features.npy", rng.normal(size=(80, 5)))
    np.save(emb / "val_labels.npy", rng.integers(0, 3, 80))
    correct = np.ones(80, dtype=bool)
    correct[:12] = False
    np.save(emb / "val_correct.npy", correct)
    np.save(emb / "val_confidence.npy", rng.permutation(np.linspace(0.3, 1.0, 80)))
    for name in ("config.yaml", "protocol.md", "som.yaml", "cnn.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    return GridSettings(
        config_path=tmp_path / "config.yaml", run_id="toy", output_dir=tmp_path / "out" / "toy",
        protocol_path=tmp_path / "protocol.md",
        input_configs={"som_config": tmp_path / "som.yaml", "cnn_config": tmp_path / "cnn.yaml"},
        embeddings_dir=emb, num_classes=3, grids=(3, 4), candidate_fractions=(0.34, 0.67), epochs=3, tune_seed=42,
        bootstrap_maps=2, resample_seed_base=1000, reps=4, split_seed_base=5000, margin=0.02,
        low_confidence_quantile=0.1, primary=Contrast(4, 3, "auroc_err"),
        secondary=(Contrast(4, 3, "ap_err"),), exploratory=(Contrast(4, 3, "auroc_lowconf"),),
        expected={"train_samples": 150, "validation_samples": 80, "validation_errors": 12,
                  "low_confidence_images": 8})


def test_every_stage_runs_and_nothing_is_overwritten(tmp_path: Path) -> None:
    s = _toy_settings(tmp_path)
    provenance.start_run(s)
    data = provenance.load_data(s)
    stages.run_tune(s, data)
    assert len(list(s.records_dir.glob("tune_*.json"))) == 4
    choice = stages.run_choose(s)
    stages.run_boot(s, data)
    stages.run_boot(s, data)                       # finished maps are skipped, not retrained
    assert len(list(s.models_dir.glob("boot_*.npz"))) == 4
    result = analysis.run_analyze(s, data, choice)
    assert result["n_errors"] == 12 and result["n_lowconf"] == 8 and len(result["exploratory"]) == 1
    assert json.loads(s.results_path.read_text())["primary"]["metric"] == "auroc_err"
    assert figure.run_figure(s).suffix == ".pdf"
    with pytest.raises(FileExistsError):
        analysis.run_analyze(s, data, choice)
    with pytest.raises(FileExistsError):
        figure.run_figure(s)


def test_a_changed_input_stops_the_run(tmp_path: Path) -> None:
    s = _toy_settings(tmp_path)
    provenance.start_run(s)
    (tmp_path / "som.yaml").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Inputs changed"):
        provenance.verify_run(s)


def test_wrong_counts_stop_before_any_training(tmp_path: Path) -> None:
    s = dataclasses.replace(_toy_settings(tmp_path), expected={"train_samples": 150, "validation_samples": 80,
                                                                "validation_errors": 11, "low_confidence_images": 8})
    with pytest.raises(RuntimeError, match="expected counts"):
        provenance.load_data(s)


def test_a_run_can_start_after_the_saved_map_check(tmp_path: Path) -> None:
    s = _toy_settings(tmp_path)
    (s.output_dir / "reproduction").mkdir(parents=True)
    (s.output_dir / "reproduction" / "saved_maps_check.json").write_text("{}", encoding="utf-8")
    provenance.start_run(s)
    assert s.inputs_path.is_file()


def test_a_folder_with_other_results_and_no_record_stops_the_run(tmp_path: Path) -> None:
    s = _toy_settings(tmp_path)
    (s.output_dir / "models").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without a provenance record"):
        provenance.start_run(s)
