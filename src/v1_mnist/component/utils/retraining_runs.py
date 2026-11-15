"""Run one arm with one seed, save it, and loop over arms and seeds.

An arm is built by a function supplied by the experiment (curriculum or augmentation):
``build(arm, data, seed)`` returns an ``ArmPlan``. This module does the training, the validation
metrics and the saving, so both experiments share them.

Files per run (names from ``ExperimentSettings``):
  metrics/<arm>_seed<k>.json     small record, written LAST, so its presence means "run finished"
  checkpoints/<arm>_seed<k>.pth  best-epoch weights (DVC)
  predictions/<arm>_seed<k>_val.npz  validation predictions and confidence (DVC)
A finished run is skipped, never overwritten.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from src.v1_mnist.component.utils.cnn_retraining import (
    load_model,
    predict,
    resolve_device,
    train_cnn,
)
from src.v1_mnist.component.utils.retraining_data import RetrainingData
from src.v1_mnist.component.utils.retraining_metrics import (
    confusion,
    evaluation_metrics,
)
from src.v1_mnist.component.utils.retraining_plans import ArmPlan, BuildArm
from src.v1_mnist.component.utils.retraining_settings import ArmSpec, ExperimentSettings
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = logging.getLogger("v1_mnist.retraining_runs")


def run_arm_seed(
    settings: ExperimentSettings,
    arm: ArmSpec,
    seed: int,
    data: RetrainingData,
    plan: ArmPlan,
    device: torch.device,
) -> dict[str, Any]:
    """Train, evaluate on validation images, and save one (arm, seed) run."""
    started = time.time()
    result = train_cnn(plan.train_x, plan.train_y, data.val.images, data.val.labels,
                       data.val_in_hotspot, plan.epoch_indices, settings.train, seed, device)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = settings.checkpoint_path(arm.name, seed)
    torch.save(result.best_state, checkpoint)

    model = load_model(result.best_state, settings.train.num_classes, device)
    val_pred, val_conf = predict(model, data.val.images, device)
    np.savez_compressed(settings.prediction_path(arm.name, seed, "val"), pred=val_pred, confidence=val_conf)

    metrics = evaluation_metrics(data.val.labels, val_pred, data.val_in_hotspot, settings.train.num_classes)
    record = {
        "run_id": settings.run_id,
        "arm": arm.name,
        "kind": arm.kind,
        "pool": arm.pool,
        "seed": seed,
        "split_evaluated": "validation",
        "best_epoch": result.best_epoch,
        "best_val_accuracy": result.best_val_accuracy,
        "validation": metrics,
        "validation_confusion": confusion(data.val.labels, val_pred, settings.train.num_classes).tolist(),
        "history": result.history,
        "plan": plan.info,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_sha256": data.sha256,
        "device": str(device),
        "seconds": round(time.time() - started, 1),
    }
    write_json(settings.metrics_path(arm.name, seed), record)
    LOGGER.info("%s seed %d: val error %.4f, hotspot error %.4f (%d s)", arm.name, seed,
                metrics["overall"]["error_rate"], metrics["hotspot"]["error_rate"], record["seconds"])
    return record


def run_arms(
    settings: ExperimentSettings,
    data: RetrainingData,
    build: BuildArm,
    only_arms: list[str] | None = None,
    only_seeds: list[int] | None = None,
) -> int:
    """Run every (arm, seed) that is not finished yet. Returns the number of runs done now."""
    known = {a.name for a in settings.arms}
    unknown = set(only_arms or []) - known
    if unknown:
        raise ValueError(f"Unknown arm(s) {sorted(unknown)}; the config has {sorted(known)}.")
    device = resolve_device(settings.device)
    LOGGER.info("Run %s on %s", settings.run_id, device)
    done = 0
    for arm in settings.arms:
        if only_arms and arm.name not in only_arms:
            continue
        for seed in settings.seeds:
            if only_seeds and seed not in only_seeds:
                continue
            if settings.metrics_path(arm.name, seed).exists():
                LOGGER.info("%s seed %d is finished already: skipped (never overwritten).", arm.name, seed)
                continue
            run_arm_seed(settings, arm, seed, data, build(arm, data, seed), device)
            done += 1
    return done
