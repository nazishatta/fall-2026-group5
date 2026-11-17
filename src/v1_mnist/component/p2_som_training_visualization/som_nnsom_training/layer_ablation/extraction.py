"""Stage 1: fc1 and fc2 embeddings of the Phase 1 LeNet-5 for training, validation and SVHN test images.

Needs PyTorch. Both layers are read in one pass (output of the linear layer before the ReLU, as in Phase 1).
The fc2 embeddings must equal the frozen Phase 1 embeddings and give the same validation predictions; this
checks that the images and the network are exactly those of Phase 1. The MNIST test split is not loaded.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.provenance import (
    check_frozen_files,
    start_run,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
    LAYER_DIMS,
    REFERENCE_LAYER,
    AblationSettings,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import svhn_test_loader
from src.v1_mnist.component.utils.cnn_retraining import load_model, resolve_device
from src.v1_mnist.component.utils.frozen_split import load_train_val_images
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.feature_layer_ablation")


def extract_layers(model: torch.nn.Module, loader: Any, layers: tuple[str, ...],
                   device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Outputs of the named modules and the predictions, for every image of ``loader`` in order."""
    modules = dict(model.named_modules())
    store: dict[str, list[np.ndarray]] = {name: [] for name in layers}
    handles = [modules[name].register_forward_hook(
        lambda _m, _i, out, name=name: store[name].append(out.detach().cpu().numpy())) for name in layers]
    preds: list[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch[0].to(device)).argmax(1).cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    feats = {name: np.concatenate(store[name]).astype(np.float32) for name in layers}
    for name, x in feats.items():
        if x.shape[1] != LAYER_DIMS[name]:
            raise RuntimeError(f"{name}: expected {LAYER_DIMS[name]} numbers per image, got {x.shape[1]}.")
    return feats, np.concatenate(preds).astype(np.int64)


def check_reference(fc2: np.ndarray, preds: np.ndarray, frozen: np.ndarray, frozen_preds: np.ndarray,
                    tolerance: float, split: str) -> float:
    """Largest |fc2 - Phase 1 fc2|; stops if it is above ``tolerance`` or a prediction differs."""
    if fc2.shape != frozen.shape:
        raise RuntimeError(f"{split}: fc2 shape {fc2.shape} differs from Phase 1 {frozen.shape}.")
    diff = float(np.max(np.abs(fc2.astype(np.float64) - frozen.astype(np.float64))))
    if diff > tolerance:
        raise RuntimeError(f"{split}: fc2 differs from Phase 1 by {diff:.2e} (allowed {tolerance:.0e}).")
    if not np.array_equal(preds, frozen_preds):
        raise RuntimeError(f"{split}: predictions differ from Phase 1.")
    return diff


def run_extraction(settings: AblationSettings) -> None:
    """Write the embeddings of both layers, or skip if the extraction record exists."""
    start_run(settings)
    if settings.extraction_path.is_file():
        log.info("Embeddings already extracted, skipped.")
        return
    check_frozen_files(settings)
    device = resolve_device(settings.device)
    state = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
    model = load_model(state, settings.num_classes, device)
    train, val = load_train_val_images(settings.data_root, settings.embeddings_dir, settings.dataset)

    record: dict[str, Any] = {"layers": list(settings.layers), "fc2_max_abs_difference": {}}
    sources = {
        "train": DataLoader(TensorDataset(torch.as_tensor(train.images).unsqueeze(1)), batch_size=settings.batch_size),
        "val": DataLoader(TensorDataset(torch.as_tensor(val.images).unsqueeze(1)), batch_size=settings.batch_size),
        "svhn": svhn_test_loader(settings.svhn_file, settings.image_size, settings.batch_size),
    }
    for split, loader in sources.items():
        feats, preds = extract_layers(model, loader, settings.layers, device)
        if split in ("train", "val"):
            frozen = np.load(settings.embeddings_dir / f"{split}_features.npy")
            frozen_preds = np.load(settings.embeddings_dir / f"{split}_preds.npy")
            record["fc2_max_abs_difference"][split] = check_reference(
                feats[REFERENCE_LAYER], preds, frozen, frozen_preds, settings.fc2_tolerance, split)
        for layer, x in feats.items():
            path = settings.embedding_path(layer, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, x)
            record[f"{layer}_{split}_sha256"] = sha256_file(path)
            record[f"{layer}_{split}_shape"] = list(x.shape)
        log.info("%s: %d images extracted.", split, len(preds))
    record["checkpoint_sha256"] = sha256_file(settings.checkpoint)
    write_json(settings.extraction_path, record)   # written last: marks the stage as done
    log.info("fc2 matches Phase 1 (largest difference %s).", record["fc2_max_abs_difference"])
