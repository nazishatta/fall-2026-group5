"""Stage 2: one SOM per layer and SOM seed (Phase 2 recipe), scored with A1 to A3. Needs NNSOM.

A finished cell (its metrics file exists) is skipped, so the stage can be re-run after an interruption.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.measures import (
    map_measures,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.provenance import (
    verify_run,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
    AblationSettings,
    cell_name,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.quality_metrics import (
    evaluate_som_quality,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_cell import assign
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.feature_layer_ablation")


def _checked(settings: AblationSettings, layer: str, split: str, record: dict) -> np.ndarray:
    path = settings.embedding_path(layer, split)
    if sha256_file(path) != record[f"{layer}_{split}_sha256"]:
        raise RuntimeError(f"{path} differs from the extraction record.")
    return np.load(path)


def train_cell(settings: AblationSettings, layer: str, seed: int, record: dict) -> None:
    """Train, assign and score one map; the metrics file is written last."""
    out = settings.metrics_path(layer, seed)
    if out.is_file():
        log.info("%s: already done, skipped.", cell_name(layer, seed))
        return
    x_train, x_val, x_svhn = (_checked(settings, layer, s, record) for s in ("train", "val", "svhn"))
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x_train)    # training embeddings only
    r = settings.recipe
    som, _history = train_som_with_history(
        x_train=x_train, grid_height=r.grid_height, grid_width=r.grid_width,
        init_neighborhood=r.init_neighborhood, epochs=r.epochs, steps=r.steps,
        norm_func=scaler.transform, seed=seed, history_every=r.epochs,
    )
    val_bmu, val_qe = assign(som, x_val)
    _svhn_bmu, svhn_qe = assign(som, x_svhn)
    labels = np.load(settings.embeddings_dir / "val_labels.npy")
    preds = np.load(settings.embeddings_dir / "val_preds.npy")
    measures = map_measures(val_bmu, labels, preds, val_qe, svhn_qe, r.num_neurons, settings.num_classes,
                            settings.hotspot_fraction, settings.min_hotspots, settings.detection_repeats,
                            settings.split_seed_base)
    quality = evaluate_som_quality(som, x_val)
    write_json(out, {"cell": cell_name(layer, seed), "layer": layer, "som_seed": seed,
                     "validation_quality": quality.to_dict(), **measures})
    log.info("%s: A1 %.3f  A2 %.4f  A3 %.4f", cell_name(layer, seed), measures["A1_error_cell_auroc"]["mean"],
             measures["A2_weighted_purity"], measures["A3_qe_novelty_auroc"])


def run_soms(settings: AblationSettings, only_layer: str | None = None) -> None:
    """Every layer and SOM seed, in config order."""
    verify_run(settings)
    if not settings.extraction_path.is_file():
        raise FileNotFoundError("Run the 'extract' stage first.")
    record = json.loads(settings.extraction_path.read_text(encoding="utf-8"))
    for layer in settings.layers:
        if only_layer and layer != only_layer:
            continue
        for seed in settings.som_seeds:
            train_cell(settings, layer, seed, record)
