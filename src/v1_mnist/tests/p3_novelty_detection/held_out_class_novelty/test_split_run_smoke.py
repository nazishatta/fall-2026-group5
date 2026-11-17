"""One split end to end on tiny made-up images (skipped without PyTorch or NNSOM)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("NNSOM")

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import (  # noqa: E402
    SomRecipe,
    load_held_out_settings,
)
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.split_run import run_split  # noqa: E402
from src.v1_mnist.component.utils.frozen_split import SplitImages  # noqa: E402
from src.v1_mnist.component.utils.retraining_settings import TrainSettings  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[5]
CONFIG = CODE_ROOT / "src/v1_mnist/component/configs/held_out_class_novelty.yaml"


def _digits(per_class: int, start_id: int, seed: int) -> SplitImages:
    """A bright block at a digit-specific place, so even 2 epochs can learn the known digits."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), per_class)
    images = rng.random((len(labels), 28, 28)).astype(np.float32) * 0.1
    for i, d in enumerate(labels):
        r, c = 2 + 12 * (d // 5), 1 + 5 * (d % 5)
        images[i, r:r + 10, c:c + 5] += 1.0
    return SplitImages(images=images.clip(0, 1), labels=labels.astype(np.int64),
                       sample_ids=np.arange(start_id, start_id + len(labels)))


def test_one_split_writes_its_records_last(tmp_path: Path) -> None:
    # NNSOM's initialisation needs more training images than embedding dimensions (84).
    base = load_held_out_settings(CONFIG, CODE_ROOT)
    settings = dataclasses.replace(
        base, output_dir=tmp_path / "run", device="cpu", batch_size=64,
        train=TrainSettings(epochs=2, lr=0.001, weight_decay=0.0001, step_size=10, gamma=0.1, batch_size=16,
                            num_classes=6),
        som=SomRecipe(grid_height=3, grid_width=3, init_neighborhood=1, epochs=2, steps=2, seed=42),
        bootstrap_resamples=20, null_iterations=20, knn_neighbors=3,
    )
    run_split(settings, 1, _digits(20, 0, 1), _digits(6, 1000, 2), torch.device("cpu"))
    folder = settings.split_dir(1)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["unknown_digits"] == [0, 1, 5, 6] and metrics["n_train"] == 120
    assert metrics["n_normal"] == 36 and metrics["n_novel"] == 24 and metrics["mnist_test_read"] is False
    assert {"qe", "msp", "energy", "mahalanobis", "knn", "length"} <= set(metrics["scores"])
    assert set(metrics["per_unknown_digit_qe_auroc"]) == {"0", "1", "5", "6"}
    for name in ("checkpoint.pth", "som", "scores.npz", "train_features.npy", "novel_features.npy"):
        assert (folder / name).is_file()
    run_split(settings, 1, _digits(20, 0, 1), _digits(6, 1000, 2), torch.device("cpu"))  # done: skipped
