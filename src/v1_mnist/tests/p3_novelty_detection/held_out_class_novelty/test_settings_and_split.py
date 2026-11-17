"""Config, class splits and image selection of the held-out-class novelty step (NumPy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.class_split import (
    cells_covering,
    known_digits,
    make_split,
    renumber,
)
from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import (
    draw_splits,
    load_held_out_settings,
)
from src.v1_mnist.component.utils.frozen_split import SplitImages

CODE_ROOT = Path(__file__).resolve().parents[5]
CONFIG = CODE_ROOT / "src/v1_mnist/component/configs/held_out_class_novelty.yaml"
PROTOCOL_SPLITS = ((0, 1, 5, 6), (2, 3, 5, 8), (1, 5, 6, 7), (2, 5, 6, 8), (0, 4, 5, 8))


def test_config_matches_the_protocol() -> None:
    settings = load_held_out_settings(CONFIG, CODE_ROOT)
    assert settings.unknown_digits == PROTOCOL_SPLITS
    assert settings.train.num_classes == 6 and settings.num_classes == 10
    assert settings.training_seed == 42 and settings.layer == "fc2"
    assert (settings.som.grid_height, settings.som.grid_width, settings.som.init_neighborhood) == (15, 15, 11)
    assert settings.auroc_margin == 0.01 and settings.bootstrap_seed == 42
    assert settings.run_id == "held_out_class_novelty_lenet5_som15x15_nb11_ep250_s42"
    assert settings.split_ids == (1, 2, 3, 4, 5)


def test_the_drawing_rule_reproduces_the_splits() -> None:
    assert draw_splits(2026, 10, 4, 5) == PROTOCOL_SPLITS


def test_a_changed_split_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["splits"]["unknown_digits"][0] = [0, 1, 2, 3]
    edited = tmp_path / "held_out_class_novelty.yaml"
    edited.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        load_held_out_settings(edited, CODE_ROOT)


def test_known_digits_and_renumbering() -> None:
    known = known_digits((0, 1, 5, 6), 10)
    assert known == (2, 3, 4, 7, 8, 9)
    assert renumber(np.array([2, 9, 4, 7]), known).tolist() == [0, 5, 2, 3]
    with pytest.raises(ValueError):
        renumber(np.array([2, 5]), known)
    with pytest.raises(ValueError):
        known_digits((0, 0, 1, 2), 10)


def _images(labels: list[int], start_id: int) -> SplitImages:
    y = np.array(labels, dtype=np.int64)
    x = np.stack([np.full((28, 28), d / 10, dtype=np.float32) for d in y])
    return SplitImages(images=x, labels=y, sample_ids=np.arange(start_id, start_id + len(y)))


def test_unknown_digits_never_reach_training() -> None:
    train = _images([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3, 0)
    val = _images([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2, 100)
    cs = make_split((0, 1, 5, 6), 10, train, val)
    assert len(cs.train_y) == 18 and set(cs.train_y.tolist()) == set(range(6))
    assert not np.isin(train.labels[np.isin(train.sample_ids, cs.train_ids)], (0, 1, 5, 6)).any()
    assert len(cs.normal_y) == 12 and len(cs.novel_digits) == 8
    assert set(cs.novel_digits.tolist()) == {0, 1, 5, 6}
    assert not set(cs.normal_ids.tolist()) & set(cs.novel_ids.tolist())


def test_cells_covering_counts_the_fullest_cells_first() -> None:
    bmu = np.array([0] * 5 + [1] * 3 + [2] * 2)
    assert cells_covering(bmu, 0.9) == 3
    assert cells_covering(bmu, 0.8) == 2
    assert cells_covering(bmu, 0.5) == 1
