"""Shared pieces of the Optuna SOM tuning: config, frozen inputs and one scored SOM run.

Used by optuna_grid_tuning.py (the search) and optuna_confirmation.py (the seed check).
The test split is never passed on, and labels are never loaded into a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.quality_metrics import (
    evaluate_som_quality,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.run_records import (
    deterministic_subset,
    sha256_file,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.dataset_info import dataset_info
from src.v1_mnist.component.utils.som_data import load_som_data, resolve_embeddings_dir

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = "src/v1_mnist/component/configs/som_optuna.yaml"
PROTOCOL = (REPO_ROOT / "src/v1_mnist/docs/p2_som_training_visualization/som_nnsom_training"
            / "OPTUNA_TUNING_PROTOCOL.md")


def resolve(path: str | Path) -> Path:
    """Absolute path; relative paths start at the code root (the folder holding src/)."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_tuning_config(path: str | Path) -> dict[str, Any]:
    """Read som_optuna.yaml as a plain dictionary."""
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def new_run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    """Create the run folder; refuse if it exists (runs are never overwritten)."""
    out_dir = resolve(cfg["paths"]["output_dir"]) / run_id
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists. Runs are never overwritten: change run.revision.")
    out_dir.mkdir(parents=True)
    return out_dir


@dataclass
class FrozenInputs:
    """Training and validation embeddings, the fitted scaler and the input fingerprints."""

    x_train: np.ndarray
    x_val: np.ndarray
    scaler: Any
    hashes: dict[str, str]


def load_frozen_inputs(cfg: dict[str, Any], smoke: bool) -> FrozenInputs:
    """Load the embeddings named in the headline SOM config and check their SHA-256."""
    som_cfg = load_config(str(resolve(cfg["inputs"]["som_config"])))
    emb_dir = resolve_embeddings_dir(som_cfg.paths.embeddings_dir, REPO_ROOT)
    hashes = {
        "train_features_sha256": sha256_file(emb_dir / "train_features.npy"),
        "val_features_sha256": sha256_file(emb_dir / "val_features.npy"),
    }
    for name, observed in hashes.items():
        expected = getattr(som_cfg.expected_inputs, name)
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {observed}. Not the frozen embeddings.")
    data = load_som_data(embeddings_dir=emb_dir, expected_feature_dim=dataset_info(som_cfg).feature_dim)
    x_train, x_val = data.train.features, data.val.features  # data.test is never used here
    if smoke:
        st = cfg["smoke_test"]
        x_train = deterministic_subset(x_train, st["train_samples"], 0)
        x_val = deterministic_subset(x_val, st["val_samples"], 1)
    return FrozenInputs(x_train, x_val, data.scaler, hashes)


def neighbourhood_from_fraction(fraction: float, side: int) -> int:
    """Starting neighbourhood = fraction of the side length, rounded half up, at least 1."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    return max(1, int(np.floor(fraction * side + 0.5)))


def train_and_score(inputs: FrozenInputs, grid: tuple[int, int], init_neighborhood: int,
                    epochs: int, seed: int) -> dict[str, float]:
    """Train one SOM (project trainer, steps = epochs) and score it on validation features."""
    start = time.perf_counter()
    som, _history = train_som_with_history(
        x_train=inputs.x_train, grid_height=grid[0], grid_width=grid[1],
        init_neighborhood=init_neighborhood, epochs=epochs, steps=epochs,
        norm_func=inputs.scaler.transform, seed=seed, history_every=epochs,
    )
    seconds = time.perf_counter() - start
    val = evaluate_som_quality(som, inputs.x_val)
    return {
        "val_qe": float(val.quantization_error),
        "val_te1_pct": float(val.topological_error_1st_order_pct),
        "val_te12_pct": float(val.topological_error_1st_2nd_order_pct),
        "val_occupancy": float(val.occupancy_rate),
        "train_seconds": round(seconds, 1),
    }
