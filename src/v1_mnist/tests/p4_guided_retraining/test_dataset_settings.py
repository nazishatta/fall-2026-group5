"""Phase 4 builds its network and finds its protocol from the dataset's config, MNIST by default."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.v1_mnist.component.utils.cnn_retraining import (  # noqa: E402
    load_model,
    network_input,
)
from src.v1_mnist.component.utils.retraining_settings import (  # noqa: E402
    TrainSettings,
    load_experiment_settings,
)

ROOT = Path(__file__).resolve().parents[4]
PIPELINE = ROOT / "src/v1_mnist/component/pipeline/p4_guided_retraining"


def test_network_input_adds_a_channel_to_grey_images_only() -> None:
    grey = np.zeros((5, 28, 28), dtype=np.float32)
    colour = np.zeros((5, 3, 32, 32), dtype=np.float32)
    assert tuple(network_input(grey).shape) == (5, 1, 28, 28)
    assert tuple(network_input(colour).shape) == (5, 3, 32, 32)
    # For grey images this is exactly the old torch.as_tensor(x).unsqueeze(1).
    assert torch.equal(network_input(grey), torch.as_tensor(grey).unsqueeze(1))


def test_load_model_defaults_to_grey_28_and_accepts_other_shapes() -> None:
    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import (
        LeNet5,
    )

    device = torch.device("cpu")
    grey = load_model(LeNet5(num_classes=10).state_dict(), 10, device)
    assert tuple(grey(torch.zeros(2, 1, 28, 28)).shape) == (2, 10)
    colour_state = LeNet5(num_classes=10, in_channels=3, image_size=32).state_dict()
    colour = load_model(colour_state, 10, device, in_channels=3, image_size=32)
    assert tuple(colour(torch.zeros(2, 3, 32, 32)).shape) == (2, 10)


def test_train_settings_default_to_the_mnist_input() -> None:
    settings = TrainSettings(epochs=1, lr=0.001, weight_decay=0.0, step_size=1, gamma=0.1,
                             batch_size=8, num_classes=10)
    assert (settings.in_channels, settings.image_size) == (1, 28)


def test_mnist_retraining_config_gives_grey_28_and_the_digit_word() -> None:
    embeddings = ROOT / "outputs/v1_mnist/p1_model_training_baseline/cnn_baseline_training/embeddings"
    if not (embeddings / "train_features.npy").is_file():
        pytest.skip("needs the MNIST embeddings (dvc pull)")
    settings = load_experiment_settings("curriculum", None, ROOT)
    assert (settings.train.in_channels, settings.train.image_size) == (1, 28)
    assert settings.dataset.name == "mnist"
    assert settings.dataset.class_label == "digit"


def _import(name: str) -> object:
    sys.path.insert(0, str(PIPELINE))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(PIPELINE))


def test_protocol_stamp_uses_the_config_it_is_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stamp = _import("run_protocol_stamp")
    seen: list[object] = []
    monkeypatch.setattr(stamp, "load_experiment_settings", lambda section, config, root: seen.append(config) or "s")
    monkeypatch.setattr(stamp, "create_stamp", lambda settings: tmp_path / "stamp.txt")
    stamp.main(["--config", "src/v1_fashion_mnist/component/configs/retraining.yaml"])
    stamp.main([])
    assert seen == ["src/v1_fashion_mnist/component/configs/retraining.yaml", None]


def test_augmentation_check_reads_the_config_option() -> None:
    augmentation = _import("run_targeted_augmentation")
    assert augmentation.check_config(["--stage", "check"]) is None
    assert augmentation.check_config(["--stage", "check", "--config", "x/retraining.yaml"]) == "x/retraining.yaml"
