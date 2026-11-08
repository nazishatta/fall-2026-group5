"""A tiny SOM stored and measured end to end (no PyTorch), and the real settings file."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    cell_name,
    load_reanalysis_settings,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_cell import (
    store_cell,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_measures import (
    all_measures,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_tiny_som_is_stored_and_measured(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=5.0, size=(3, 6))
    labels = np.repeat(np.arange(3), 40)
    x = centers[labels] + rng.normal(scale=0.3, size=(120, 6))
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x)
    som, _ = train_som_with_history(x, 4, 4, 3, 15, 15, scaler.transform, seed=1)

    stub = SimpleNamespace(folder=lambda kind: tmp_path / kind)
    store_cell(stub, "tiny", som, x, x, {})  # type: ignore[arg-type]
    npz = np.load(tmp_path / "assignments" / "tiny.npz")
    assert npz["val_bmu"].shape == (120,) and npz["pos"].shape == (2, 16)
    assert (tmp_path / "metrics" / "tiny.json").is_file()

    out = all_measures(npz["val_bmu"], labels, labels, npz["neuron_dist"], 3, 0.95, 1.1, 0.05, 2)
    assert out["weighted_purity"] > 0.95            # three clean blobs
    assert out["hotspot_error_rate"] == 0.0 and out["overall_error"] == 0.0
    assert 1.0 <= out["class_scatter_index"] <= 3.0


def test_real_settings_match_the_protocol() -> None:
    try:
        settings = load_reanalysis_settings(None, REPO_ROOT)
    except FileNotFoundError:
        pytest.skip("Phase 2 records are not in this checkout.")
    assert settings.arms == ("standard", "augmented_som", "augmented_random")
    assert settings.seeds == (1, 2, 3, 4, 5)
    assert settings.recipe.grid_height == 15 and settings.recipe.init_neighborhood == 11
    assert settings.recipe.epochs == 250 and settings.recipe.seed == 42
    assert settings.purity_cutoff == 0.95
    assert settings.pairs == (("augmented_som", "standard"), ("augmented_som", "augmented_random"))
    assert cell_name("standard", 3) in settings.cells and len(settings.cells) == 15
    assert "week" not in settings.run_id.lower()
