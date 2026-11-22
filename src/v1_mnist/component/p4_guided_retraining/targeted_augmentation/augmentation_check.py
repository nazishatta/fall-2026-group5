"""Look at the altered images before trusting the transform ranges (validation images only).

Two checks, both written down before the arms are compared:
1. a contact sheet: original image next to its altered copies, for a few pool images;
2. the Phase 1 CNN's accuracy on altered validation images against its accuracy on the originals.
   If the altered images are much harder than the originals, the ranges are too strong.

When the config sets ``augmentation.transform_check.max_accuracy_drop`` (Fashion-MNIST), the check
records whether it passed and the train stage refuses to start until it has. Without that setting
(MNIST, whose check was judged by hand before this existed) nothing is enforced.
"""

from __future__ import annotations

import json
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


def check_path(settings: ExperimentSettings) -> Path:
    """Where the transform check writes its result."""
    return settings.output_dir / "check" / "augmentation_check.json"


def accuracy_drop_limit(settings: ExperimentSettings) -> float | None:
    """The largest allowed accuracy drop on altered images, or None when the config sets none."""
    limit = (settings.options.get("transform_check") or {}).get("max_accuracy_drop")
    return None if limit is None else float(limit)


def require_passed_check(settings: ExperimentSettings) -> None:
    """Stop the train stage unless the transform check ran and passed (only when a limit is set)."""
    if accuracy_drop_limit(settings) is None:
        return
    path = check_path(settings)
    if not path.is_file():
        raise RuntimeError(f"{path} is missing: run --stage check before --stage train.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed"):
        raise RuntimeError(
            f"The transform check failed (accuracy drop {result.get('accuracy_drop')}, allowed "
            f"{result.get('max_accuracy_drop')}): make the ranges smaller under a new revision "
            "and record it in DEVIATIONS.md before any arm runs.")


def run_check(settings: ExperimentSettings, data: RetrainingData) -> dict[str, Any]:
    """Write the contact sheet and augmentation_check.json (once)."""
    result_path = check_path(settings)
    ensure_not_done(result_path)
    sheet = contact_sheet(data, settings, settings.output_dir / "figures" / "augmentation_contact_sheet")

    device = resolve_device(settings.device)
    state = torch.load(settings.cnn_checkpoint, map_location="cpu")
    model = load_model(state, settings.train.num_classes, device,
                       settings.train.in_channels, settings.train.image_size)
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
    limit = accuracy_drop_limit(settings)
    if limit is not None:
        result["accuracy_drop"] = result["accuracy_original"] - result["accuracy_altered"]
        result["max_accuracy_drop"] = limit
        result["passed"] = bool(result["accuracy_drop"] <= limit)
    write_json(result_path, result)
    LOGGER.info("Accuracy on validation images: original %.4f, altered %.4f",
                result["accuracy_original"], result["accuracy_altered"])
    if limit is not None:
        LOGGER.info("Transform check %s: accuracy drop %.4f, allowed %.4f. Look at the contact sheet too.",
                    "PASSED" if result["passed"] else "FAILED", result["accuracy_drop"], limit)
    return result
