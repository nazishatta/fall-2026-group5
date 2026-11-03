"""Shared Phase 4 support code: frozen split, metrics, paired statistics, protocol stamp, arm pairs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.v1_mnist.component.utils.frozen_split import (
    IMAGE_MAGIC,
    LABEL_MAGIC,
    load_test_images,
    load_train_val_images,
)
from src.v1_mnist.component.utils.paired_stats import (
    paired_bootstrap,
    seed_average_error,
    seed_spread,
)
from src.v1_mnist.component.utils.protocol_stamp import create_stamp, verify_stamp
from src.v1_mnist.component.utils.retraining_metrics import (
    confusion,
    evaluation_metrics,
)
from src.v1_mnist.component.utils.retraining_settings import ArmSpec
from src.v1_mnist.component.utils.retraining_stability import (
    largest_drop,
    run_stability,
)
from src.v1_mnist.component.utils.retraining_summary import comparison_pairs


def _write_idx(path: Path, magic: int, array: np.ndarray) -> None:
    header = magic.to_bytes(4, "big") + len(array).to_bytes(4, "big")
    if magic == IMAGE_MAGIC:
        header += array.shape[1].to_bytes(4, "big") + array.shape[2].to_bytes(4, "big")
    path.write_bytes(header + array.astype(np.uint8).tobytes())


@pytest.fixture()
def tiny_mnist(tmp_path: Path) -> tuple[Path, Path]:
    """Ten made-up 'MNIST' images and an embeddings folder with a saved split of them."""
    raw = tmp_path / "data" / "MNIST" / "raw"
    raw.mkdir(parents=True)
    images = np.random.default_rng(0).integers(0, 256, size=(10, 28, 28))
    labels = np.arange(10)
    _write_idx(raw / "train-images-idx3-ubyte", IMAGE_MAGIC, images[:6])
    _write_idx(raw / "train-labels-idx1-ubyte", LABEL_MAGIC, labels[:6])
    _write_idx(raw / "t10k-images-idx3-ubyte", IMAGE_MAGIC, images[6:])
    _write_idx(raw / "t10k-labels-idx1-ubyte", LABEL_MAGIC, labels[6:])
    emb = tmp_path / "emb"
    emb.mkdir()
    for split, ids in {"train": [7, 0, 3, 9], "val": [1, 5], "test": [2, 8, 4, 6]}.items():
        np.save(emb / f"{split}_sample_ids.npy", np.array(ids))
        np.save(emb / f"{split}_labels.npy", labels[ids])
    return tmp_path / "data", emb


def test_split_is_looked_up_by_saved_sample_ids(tiny_mnist: tuple[Path, Path]) -> None:
    data_root, emb = tiny_mnist
    train, val = load_train_val_images(data_root, emb)
    assert train.sample_ids.tolist() == [7, 0, 3, 9] and train.labels.tolist() == [7, 0, 3, 9]
    assert train.images.shape == (4, 28, 28) and train.images.max() <= 1.0
    assert val.sample_ids.tolist() == [1, 5]
    assert load_test_images(data_root, emb).sample_ids.tolist() == [2, 8, 4, 6]


def test_wrong_data_folder_is_caught_by_the_label_check(tiny_mnist: tuple[Path, Path]) -> None:
    data_root, emb = tiny_mnist
    np.save(emb / "train_labels.npy", np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError, match="differ from the saved labels"):
        load_train_val_images(data_root, emb)


def test_confusion_and_metrics_on_a_known_example() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    assert confusion(y_true, y_pred, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    hot = np.array([True, True, False, False, False, False])
    m = evaluation_metrics(y_true, y_pred, hot, 3)
    assert m["overall"]["errors"] == 2 and m["hotspot"]["errors"] == 1 and m["hotspot"]["images"] == 2
    assert m["non_hotspot"]["errors"] == 1
    assert m["per_class_error"] == pytest.approx([0.5, 0.0, 0.5])
    low, high = m["hotspot"]["error_wilson_95"]
    assert 0.0 <= low <= 0.5 <= high <= 1.0


def test_paired_bootstrap_finds_a_clear_difference_and_not_a_null_one() -> None:
    a = np.zeros(400)
    b = np.zeros(400)
    b[:100] = 1.0  # b makes 100 more mistakes
    mask = np.ones(400, dtype=bool)
    clear = paired_bootstrap(a, b, mask, resamples=500, seed=1)
    assert clear["mean_difference"] == pytest.approx(-0.25) and clear["excludes_zero"]
    null = paired_bootstrap(b, b, mask, resamples=500, seed=1)
    assert null["mean_difference"] == 0.0 and not null["excludes_zero"]
    assert paired_bootstrap(a, b, mask, 500, 1) == clear  # same seed, same answer


def test_seed_helpers() -> None:
    errors = np.array([[1, 0, 0], [1, 1, 0]])
    assert seed_average_error(errors).tolist() == [1.0, 0.5, 0.0]
    assert seed_spread(np.array([1.0, 3.0]))["std"] == pytest.approx(1.4142135)
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool), 10, 0)


def test_comparison_pairs_pit_the_som_against_its_controls() -> None:
    arms = (
        ArmSpec("standard", "standard"),
        ArmSpec("a_som", "augmentation", "som"),
        ArmSpec("a_random", "augmentation", "random"),
        ArmSpec("a_msp", "augmentation", "msp"),
        ArmSpec("a_hot", "augmentation", "hotspot"),
        ArmSpec("a_hot_random", "augmentation", "random_hotspot"),
    )
    pairs = comparison_pairs(arms)
    assert ("a_som", "a_random") in pairs and ("a_som", "a_msp") in pairs
    assert ("a_hot", "a_hot_random") in pairs and ("a_som", "standard") in pairs
    assert all(g != c for g, c in pairs)


def test_paced_arms_are_compared_with_paced_controls_and_with_their_unpaced_twin() -> None:
    arms = (
        ArmSpec("standard", "standard"),
        ArmSpec("a_som", "augmentation", "som"),
        ArmSpec("a_random", "augmentation", "random"),
        ArmSpec("p_som", "augmentation", "som", "proposal"),
        ArmSpec("p_random", "augmentation", "random", "proposal"),
    )
    pairs = comparison_pairs(arms)
    assert ("p_som", "p_random") in pairs and ("a_som", "a_random") in pairs
    assert ("p_som", "a_som") in pairs and ("p_random", "a_random") in pairs
    assert ("p_som", "a_random") not in pairs and ("a_som", "p_random") not in pairs


def test_stability_numbers_pick_out_falls_in_validation_accuracy() -> None:
    assert largest_drop(np.array([0.90, 0.95, 0.93, 0.96])) == pytest.approx(0.02)
    assert largest_drop(np.array([0.9, 0.95, 0.96])) == 0.0
    history = [
        {"val_accuracy": a, "val_non_hotspot_accuracy": e}
        for a, e in ((0.90, 0.95), (0.96, 0.99), (0.94, 0.97), (0.95, 0.98))
    ]
    stability = run_stability(history)
    assert stability["largest_drop_overall"] == pytest.approx(0.02)
    assert stability["largest_drop_non_hotspot"] == pytest.approx(0.02)
    assert stability["last_epoch_below_best"] == pytest.approx(0.01)
    assert stability["non_hotspot_last_below_peak"] == pytest.approx(0.01)


def _stamp_settings(tmp_path: Path, protocol_text: str, guard: float | None) -> SimpleNamespace:
    protocol = tmp_path / "protocol.md"
    protocol.write_text(protocol_text, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("a: 1\n", encoding="utf-8")
    return SimpleNamespace(protocol_path=protocol, config_path=config, stamp_path=tmp_path / "stamp.txt",
                           macro_f1_guard=guard)


def test_stamp_refuses_open_items_and_an_empty_guard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="open items"):
        create_stamp(_stamp_settings(tmp_path, "Guard: [TBD]\n", 0.01))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="macro_f1_guard"):
        create_stamp(_stamp_settings(tmp_path, "all fixed\n", None))  # type: ignore[arg-type]


def test_test_split_stays_closed_until_stamped_and_after_any_edit(tmp_path: Path) -> None:
    settings = _stamp_settings(tmp_path, "all fixed\n", 0.01)
    with pytest.raises(PermissionError, match="not frozen"):
        verify_stamp(settings)  # type: ignore[arg-type]
    create_stamp(settings)  # type: ignore[arg-type]
    verify_stamp(settings)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        create_stamp(settings)  # type: ignore[arg-type]
    settings.protocol_path.write_text("all fixed, then changed\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="changed after it was frozen"):
        verify_stamp(settings)  # type: ignore[arg-type]
    settings.protocol_path.write_text("all fixed\n", encoding="utf-8")
    settings.config_path.write_text("a: 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="config"):
        verify_stamp(settings)  # type: ignore[arg-type]


def test_the_fixed_evaluation_settings_are_in_the_config() -> None:
    import yaml

    config = Path(__file__).resolve().parents[2] / "component" / "configs" / "retraining.yaml"
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert raw["training"]["seeds"] == [1, 2, 3, 4, 5]
    assert raw["evaluation"]["non_hotspot_damage_tolerance"] == 0.002
    assert raw["evaluation"]["bootstrap"] == {"resamples": 2000, "seed": 2026}
    assert raw["curriculum"]["schedules"]["proposal"] == {
        "initial_fraction": 0.10, "step_fraction": 0.10, "step_epochs": 5}
    assert raw["augmentation"]["copies_per_image"] == 3
