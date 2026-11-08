"""Stage 1: embeddings of the training and validation images from each retrained CNN.

Needs PyTorch. The model is rebuilt from the saved checkpoint (its SHA-256 was checked by
``run_state``), the ``fc2`` layer is read like in Phase 1, and the predictions found on the way must
match the validation predictions saved after the retraining.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
    FeatureExtractor,
)
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    cell_name,
    embeddings_folder,
)
from src.v1_mnist.component.utils.cnn_retraining import load_model, resolve_device
from src.v1_mnist.component.utils.frozen_split import SplitImages, load_train_val_images
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.som_reanalysis")


def _extract(model: torch.nn.Module, split: SplitImages, settings: ReanalysisSettings,
             device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    data = TensorDataset(torch.as_tensor(split.images).unsqueeze(1), torch.as_tensor(split.labels))
    loader = DataLoader(data, batch_size=settings.batch_size, shuffle=False)
    features, _labels, predictions = FeatureExtractor(model, settings.layer).extract(loader, device)
    return features.astype(np.float32), predictions


def extract_cell(settings: ReanalysisSettings, arm: str, seed: int, train: SplitImages, val: SplitImages,
                 device: torch.device) -> None:
    """Write the embeddings of one cell, or skip it if it is already complete."""
    retr = settings.retraining
    cell = cell_name(arm, seed)
    folder = embeddings_folder(settings, cell)
    record_path = folder / "extraction.json"
    if record_path.is_file():
        log.info("%s: embeddings already exist, skipped.", cell)
        return

    checkpoint = retr.checkpoint_path(arm, seed)
    state = torch.load(checkpoint, map_location="cpu")
    model = load_model(state, retr.train.num_classes, device)
    train_x, _ = _extract(model, train, settings, device)
    val_x, val_pred = _extract(model, val, settings, device)

    saved = np.load(retr.prediction_path(arm, seed, "val"))["pred"]
    agreement = float(np.mean(saved == val_pred))
    if agreement < settings.min_agreement:
        raise RuntimeError(
            f"{cell}: predictions found during the extraction agree with the saved validation "
            f"predictions for only {agreement:.4f} of the images (needed {settings.min_agreement})."
        )
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "train_features.npy", train_x)
    np.save(folder / "val_features.npy", val_x)
    np.save(folder / "val_preds.npy", val_pred.astype(np.int64))
    write_json(record_path, {
        "cell": cell, "layer": settings.layer, "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_agreement": agreement, "train_shape": list(train_x.shape), "val_shape": list(val_x.shape),
        "train_features_sha256": sha256_file(folder / "train_features.npy"),
        "val_features_sha256": sha256_file(folder / "val_features.npy"),
    })
    log.info("%s: embeddings written (agreement %.4f).", cell, agreement)


def run_embeddings(settings: ReanalysisSettings, repo_root: Path) -> None:
    """Extract the embeddings of every cell. The test split is not loaded."""
    retr = settings.retraining
    train, val = load_train_val_images(retr.data_root, retr.embeddings_dir)
    device = resolve_device(settings.device)
    for arm in settings.arms:
        for seed in settings.seeds:
            extract_cell(settings, arm, seed, train, val, device)
