"""The retraining trainer and the arm runner, on tiny made-up images (skipped if PyTorch is missing)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.v1_mnist.component.utils.cnn_retraining import predict, train_cnn  # noqa: E402
from src.v1_mnist.component.utils.retraining_plans import ArmPlan, standard_plan  # noqa: E402
from src.v1_mnist.component.utils.retraining_runs import run_arm_seed  # noqa: E402
from src.v1_mnist.component.utils.retraining_settings import ArmSpec, TrainSettings  # noqa: E402
from src.v1_mnist.tests.p4_guided_retraining.synthetic_data import (  # noqa: E402
    NUM_CLASSES,
    make_data,
)

CPU = torch.device("cpu")
SETTINGS = TrainSettings(epochs=3, lr=0.001, weight_decay=0.0001, step_size=10, gamma=0.1,
                         batch_size=16, num_classes=NUM_CLASSES)


def _digits(per_class: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Images whose class is easy to see: a bright block at a class-specific place."""
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(NUM_CLASSES), per_class)
    images = rng.random((len(labels), 28, 28)).astype(np.float32) * 0.1
    for i, c in enumerate(labels):
        images[i, 5 + 8 * c:12 + 8 * c, 5:20] += 1.0
    return images.clip(0, 1), labels


def test_trainer_uses_the_rows_of_each_epoch_and_learns() -> None:
    x, y = _digits()
    rows = {0: np.arange(30), 1: np.arange(60), 2: np.arange(90)}
    hot = np.arange(len(y)) < 10
    result = train_cnn(x, y, x, y, hot, lambda e: rows[e], SETTINGS, seed=1, device=CPU)
    assert [h["training_images"] for h in result.history] == [30, 60, 90]
    assert {"val_hotspot_accuracy", "val_non_hotspot_accuracy"} <= set(result.history[0])
    assert 1 <= result.best_epoch <= 3
    assert result.best_val_accuracy > 0.5


def test_same_seed_gives_the_same_training() -> None:
    x, y = _digits()
    every = np.arange(len(y))
    hot = np.arange(len(y)) < 10
    first = train_cnn(x, y, x, y, hot, lambda _e: every, SETTINGS, seed=5, device=CPU)
    second = train_cnn(x, y, x, y, hot, lambda _e: every, SETTINGS, seed=5, device=CPU)
    assert [h["val_accuracy"] for h in first.history] == [h["val_accuracy"] for h in second.history]


def test_predict_returns_a_digit_and_a_confidence_per_image() -> None:
    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )

    x, _ = _digits(5)
    pred, conf = predict(LeNet5(num_classes=NUM_CLASSES), x, CPU)
    assert pred.shape == conf.shape == (len(x),)
    assert pred.min() >= 0 and pred.max() < NUM_CLASSES and (conf > 0).all() and (conf <= 1).all()


def test_arm_run_saves_its_files_and_the_record_is_written_last(tmp_path: Path) -> None:
    data = make_data()
    settings = SimpleNamespace(
        run_id="test_run", train=SETTINGS, checkpoints_dir=tmp_path / "checkpoints",
        predictions_dir=tmp_path / "predictions",
        checkpoint_path=lambda arm, seed: tmp_path / "checkpoints" / f"{arm}_seed{seed}.pth",
        prediction_path=lambda arm, seed, split: tmp_path / "predictions" / f"{arm}_seed{seed}_{split}.npz",
        metrics_path=lambda arm, seed: tmp_path / "metrics" / f"{arm}_seed{seed}.json",
    )
    plan = standard_plan(data)
    assert isinstance(plan, ArmPlan)
    record = run_arm_seed(settings, ArmSpec("standard", "standard"), 1, data, plan, CPU)  # type: ignore[arg-type]
    saved = json.loads((tmp_path / "metrics" / "standard_seed1.json").read_text(encoding="utf-8"))
    assert saved["arm"] == "standard" and saved["split_evaluated"] == "validation"
    assert saved["checkpoint_sha256"] == record["checkpoint_sha256"]
    assert (tmp_path / "checkpoints" / "standard_seed1.pth").is_file()
    assert (tmp_path / "predictions" / "standard_seed1_val.npz").is_file()
    assert len(saved["history"]) == SETTINGS.epochs
