"""Look at the altered images before trusting the transform ranges (validation images only).

Two checks, both written down before the arms are compared:
1. a contact sheet: original image next to its altered copies, for a few pool images;
2. the Phase 1 CNN's accuracy on altered validation images against its accuracy on the originals.
   If the altered images are much harder than the originals, the ranges are too strong.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.v1_mnist.component.p4_guided_retraining.targeted_augmentation.augmentation_arms import (
    ranges_from_options,
)
from src.v1_mnist.component.utils.cnn_retraining import (
    load_model,
    predict,
    resolve_device,
)
from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.image_transforms import make_copies
from src.v1_mnist.component.utils.retraining_data import RetrainingData
from src.v1_mnist.component.utils.retraining_settings import ExperimentSettings
from src.v1_mnist.component.utils.run_provenance import ensure_not_done, write_json
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger("v1_mnist.augmentation_check")
SHEET_IMAGES = 10


def contact_sheet(data: RetrainingData, settings: ExperimentSettings, path: Path) -> Path:
    """Original (left) and its altered copies (right) for the first pool images by sample ID."""
    copies = int(settings.options["copies_per_image"])
    rows = np.flatnonzero(data.som_pool)
    rows = rows[np.argsort(data.train.sample_ids[rows])][:SHEET_IMAGES]
    altered = make_copies(data.train.images[rows], copies, settings.seeds[0], ranges_from_options(settings.options))
    fig, axes = plt.subplots(SHEET_IMAGES, copies + 1, figsize=(1.1 * (copies + 1), 1.1 * SHEET_IMAGES))
    for i, row in enumerate(rows):
        axes[i, 0].imshow(data.train.images[row], cmap="gray_r")
        for j in range(copies):
            axes[i, j + 1].imshow(altered[j * len(rows) + i], cmap="gray_r")
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].set_title("original", fontsize=7)
    for j in range(copies):
        axes[0, j + 1].set_title(f"copy {j + 1}", fontsize=7)
    return save_pdf(fig, path)


def run_check(settings: ExperimentSettings, data: RetrainingData) -> dict[str, Any]:
    """Write the contact sheet and augmentation_check.json (once)."""
    out = settings.output_dir / "check"
    result_path = out / "augmentation_check.json"
    ensure_not_done(result_path)
    sheet = contact_sheet(data, settings, settings.output_dir / "figures" / "augmentation_contact_sheet")

    device = resolve_device(settings.device)
    state = torch.load(settings.cnn_checkpoint, map_location="cpu")
    model = load_model(state, settings.train.num_classes, device)
    ranges = ranges_from_options(settings.options)
    altered = make_copies(data.val.images, 1, settings.seeds[0], ranges)
    original_pred, _ = predict(model, data.val.images, device)
    altered_pred, _ = predict(model, altered, device)
    hot = data.val_in_hotspot
    result = {
        "run_id": settings.run_id,
        "checkpoint": settings.cnn_checkpoint.name,
        "checkpoint_sha256": sha256_file(settings.cnn_checkpoint),
        "validation_images": len(data.val.labels),
        "accuracy_original": float((original_pred == data.val.labels).mean()),
        "accuracy_altered": float((altered_pred == data.val.labels).mean()),
        "accuracy_original_hotspot": float((original_pred == data.val.labels)[hot].mean()),
        "accuracy_altered_hotspot": float((altered_pred == data.val.labels)[hot].mean()),
        "transforms": settings.options["transforms"],
        "contact_sheet": sheet.name,
    }
    write_json(result_path, result)
    LOGGER.info("Accuracy on validation images: original %.4f, altered %.4f",
                result["accuracy_original"], result["accuracy_altered"])
    return result
