"""Stage 2: train a SOM on one cell's embeddings (Phase 2 recipe) and store the assignments.

Also builds the reference cell: the Phase 2 SOM, loaded and checked, on the Phase 1 embeddings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.quality_metrics import (
    evaluate_som_quality,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    embeddings_folder,
)
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.run_settings import verify_selected_model
from src.v1_mnist.component.utils.som_io import (
    load_selected_som,
    load_split,
    reconstruct_assignments,
    sha256_file,
    som_coordinates,
)

log = logging.getLogger("v1_mnist.som_reanalysis")


def assign(som: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neuron and distance for every row of ``x`` (the SOM scales the rows itself)."""
    clusters, distances, _max_dist, _sizes = som.cluster_data(x)
    return reconstruct_assignments(clusters, distances, x.shape[0])


def store_cell(settings: ReanalysisSettings, cell: str, som: Any, train_x: np.ndarray,
               val_x: np.ndarray, extra: dict[str, Any]) -> None:
    """Write assignments (NPZ) and the small metrics record of one cell. The record is written last."""
    train_bmu, train_dist = assign(som, train_x)
    val_bmu, val_dist = assign(som, val_x)
    quality = evaluate_som_quality(som, val_x)
    folder = settings.folder("assignments")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{cell}.npz", train_bmu=train_bmu, train_dist=train_dist, val_bmu=val_bmu,
        val_dist=val_dist, neuron_dist=np.asarray(som.neuron_dist, dtype=np.float64),
        pos=np.asarray(som_coordinates(som), dtype=np.float64),
    )
    write_json(settings.folder("metrics") / f"{cell}.json", {
        "cell": cell, "validation_quality": quality.to_dict(), **extra,
    })


def train_cell(settings: ReanalysisSettings, cell: str) -> None:
    """Train and store the SOM of one retrained cell, or skip it if its record exists."""
    record = settings.folder("metrics") / f"{cell}.json"
    if record.is_file():
        log.info("%s: SOM already trained, skipped.", cell)
        return
    emb = embeddings_folder(settings, cell)
    train_x = np.load(emb / "train_features.npy")
    val_x = np.load(emb / "val_features.npy")
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(train_x)    # training embeddings only
    r = settings.recipe
    som, _history = train_som_with_history(
        x_train=train_x, grid_height=r.grid_height, grid_width=r.grid_width,
        init_neighborhood=r.init_neighborhood, epochs=r.epochs, steps=r.steps,
        norm_func=scaler.transform, seed=r.seed, history_every=r.history_every,
    )
    model_dir = settings.folder("som_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    som.save_pickle(cell, str(model_dir) + os.sep)
    store_cell(settings, cell, som, train_x, val_x, {
        "som_seed": r.seed, "model_sha256": sha256_file(model_dir / cell),
        "train_features_sha256": sha256_file(emb / "train_features.npy"),
        "val_features_sha256": sha256_file(emb / "val_features.npy"),
    })
    log.info("%s: SOM trained and stored.", cell)


def reference_cell(settings: ReanalysisSettings) -> None:
    """Store the assignments of the Phase 2 SOM on the Phase 1 embeddings (loaded, not retrained)."""
    cell = settings.reference
    if (settings.folder("metrics") / f"{cell}.json").is_file():
        log.info("%s: already stored, skipped.", cell)
        return
    ana = settings.retraining.analysis
    model_hash = verify_selected_model(ana)
    som = load_selected_som(ana.model_path, ana.grid_height, ana.grid_width, ana.feature_dim)
    train_x = load_split("train", ana.embeddings_dir, ana.feature_dim)["features"]
    val_x = load_split("val", ana.embeddings_dir, ana.feature_dim)["features"]
    store_cell(settings, cell, som, train_x, val_x, {"som_run_id": ana.selected_model, "model_sha256": model_hash})
    log.info("%s: Phase 2 SOM stored as the reference.", cell)


def run_train_som(settings: ReanalysisSettings) -> None:
    reference_cell(settings)
    for cell in settings.cells:
        train_cell(settings, cell)


def model_path(settings: ReanalysisSettings, cell: str) -> Path:
    return settings.folder("som_models") / cell
