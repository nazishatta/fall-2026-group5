"""Train LeNet-5 from scratch with the Phase 1 recipe, on any list of images, without the test split.

Same recipe as ``train_baseline.py`` (Adam, StepLR, 20 epochs, batch 64, best epoch chosen on
validation accuracy), with two differences that Phase 4 needs:
- the images come from arrays plus an ``epoch_indices`` function, so a curriculum can change which
  images are in the training set from epoch to epoch;
- only the training and validation images are ever used. The test split is not loaded here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
    LeNet5,
)
from src.v1_mnist.component.utils.config import set_seed
from src.v1_mnist.component.utils.retraining_plans import EpochIndices
from src.v1_mnist.component.utils.retraining_settings import TrainSettings

LOGGER = logging.getLogger("v1_mnist.cnn_retraining")


@dataclass(frozen=True)
class TrainResult:
    """The best model (by validation accuracy) and the per-epoch record."""

    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_val_accuracy: float
    history: list[dict[str, Any]]


def resolve_device(name: str) -> torch.device:
    """'auto' uses the GPU when there is one."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict(
    model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted digit and max-softmax confidence for each image."""
    model.eval()
    preds, conf = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(images[start:start + batch_size]).unsqueeze(1).to(device)
            probs = torch.softmax(model(batch), dim=1)
            top, index = probs.max(dim=1)
            preds.append(index.cpu().numpy())
            conf.append(top.cpu().numpy())
    return np.concatenate(preds).astype(np.int64), np.concatenate(conf).astype(np.float32)


def train_cnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_in_hotspot: np.ndarray,
    epoch_indices: EpochIndices,
    settings: TrainSettings,
    seed: int,
    device: torch.device,
) -> TrainResult:
    """Train one LeNet-5. ``epoch_indices(e)`` lists the training rows used in epoch ``e`` (0-based).

    Every epoch records the validation accuracy overall, in the hotspot images and on all other
    images, so instability or forgetting on the easy images shows up during training."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = LeNet5(num_classes=settings.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings.step_size, gamma=settings.gamma)
    criterion = nn.CrossEntropyLoss()

    x_all = torch.as_tensor(train_x).unsqueeze(1).to(device)
    y_all = torch.as_tensor(train_y, dtype=torch.long).to(device)

    best_state: dict[str, torch.Tensor] = {}
    best_acc, best_epoch = -1.0, -1
    history: list[dict[str, Any]] = []
    for epoch in range(settings.epochs):
        rows = np.asarray(epoch_indices(epoch), dtype=np.int64)
        order = torch.as_tensor(rng.permutation(rows)).to(device)
        model.train()
        loss_sum, correct = 0.0, 0
        for start in range(0, len(order), settings.batch_size):
            batch = order[start:start + settings.batch_size]
            inputs, targets = x_all[batch], y_all[batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
        scheduler.step()

        val_pred, _ = predict(model, val_x, device)
        val_ok = val_pred == val_y
        val_acc = float(val_ok.mean())
        history.append({
            "epoch": epoch + 1,
            "training_images": len(rows),
            "train_loss": loss_sum / len(rows),
            "train_accuracy": correct / len(rows),
            "val_accuracy": val_acc,
            "val_hotspot_accuracy": float(val_ok[val_in_hotspot].mean()),
            "val_non_hotspot_accuracy": float(val_ok[~val_in_hotspot].mean()),
        })
        LOGGER.info("epoch %d/%d: %d images, train acc %.4f, val acc %.4f",
                    epoch + 1, settings.epochs, len(rows), correct / len(rows), val_acc)
        if val_acc > best_acc:  # strict: the first best epoch wins, as in Phase 1
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return TrainResult(best_state, best_epoch, best_acc, history)


def load_model(state: dict[str, torch.Tensor], num_classes: int, device: torch.device) -> LeNet5:
    """A LeNet-5 holding saved weights, ready for evaluation."""
    model = LeNet5(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    return model.eval()
