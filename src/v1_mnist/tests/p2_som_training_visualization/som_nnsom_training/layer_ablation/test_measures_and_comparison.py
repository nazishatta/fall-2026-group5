"""Feature-layer ablation: measures A1 to A3 and the fc1 - fc2 verdicts on small made-up data."""

from __future__ import annotations

import numpy as np
import pytest

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.comparison import (
    compare,
    verdict,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.measures import (
    error_cell_auroc,
    map_measures,
    qe_novelty_auroc,
)

SPLIT_SEED_BASE = 7000


def test_qe_auroc_is_one_when_every_novel_image_is_farther() -> None:
    assert qe_novelty_auroc(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.9])) == 1.0
    assert qe_novelty_auroc(np.array([0.5, 0.9]), np.array([0.1, 0.2])) == 0.0


def test_error_cells_score_high_when_errors_sit_in_one_cell() -> None:
    rng = np.random.default_rng(SPLIT_SEED_BASE)
    bmu = rng.integers(0, 9, 900)
    errors = np.zeros(900, dtype=bool)
    errors[bmu == 4] = rng.random((bmu == 4).sum()) < 0.6       # errors only in cell 4
    concentrated = error_cell_auroc(bmu, errors, 9, repeats=10, split_seed_base=SPLIT_SEED_BASE)
    scattered = error_cell_auroc(rng.integers(0, 9, 900), errors, 9, repeats=10, split_seed_base=SPLIT_SEED_BASE)
    assert concentrated["mean"] > 0.9
    assert scattered["mean"] < 0.65
    assert concentrated["low"] <= concentrated["mean"] <= concentrated["high"]


def test_map_measures_has_every_protocol_measure() -> None:
    rng = np.random.default_rng(SPLIT_SEED_BASE)
    labels = rng.integers(0, 3, 600)
    bmu = labels * 3 + rng.integers(0, 3, 600)                   # 9 cells, mostly pure
    preds = labels.copy()
    preds[:30] = (labels[:30] + 1) % 3
    out = map_measures(bmu, labels, preds, rng.random(600), rng.random(200) + 0.5, 9, 3, 0.05, 2, 5,
                       SPLIT_SEED_BASE)
    for key in ("A1_error_cell_auroc", "A2_weighted_purity", "A3_qe_novelty_auroc", "hotspot_error_share",
                "hotspot_error_rate", "occupied_neurons"):
        assert key in out
    assert 0.9 <= out["A2_weighted_purity"] <= 1.0
    assert out["validation_errors"] == 30


@pytest.mark.parametrize("interval,expected", [([0.01, 0.05], "fc1 better"), ([-0.05, -0.01], "fc2 better"),
                                               ([-0.01, 0.02], "inconclusive")])
def test_three_way_verdict(interval: list[float], expected: str) -> None:
    assert verdict(interval, "fc1", "fc2") == expected


def _cell(layer: str, seed: int, a1: float, a2: float, a3: float) -> dict:
    return {"cell": f"{layer}_seed{seed}", "som_seed": seed, "A1_error_cell_auroc": {"mean": a1},
            "A2_weighted_purity": a2, "A3_qe_novelty_auroc": a3, "hotspot_error_share": 0.3,
            "hotspot_error_rate": 0.2, "occupied_neurons": 225, "val_qe_mean": 1.0,
            "validation_quality": {"topological_error_1st_order_pct": 5.0}}


def test_compare_pairs_seeds_and_reads_the_statements() -> None:
    seeds = (1, 2, 3, 4, 5)
    per_layer = {
        "fc1": [_cell("fc1", s, 0.75 + 0.001 * s, 0.98, 0.95 + 0.01 * (s % 2)) for s in seeds],
        "fc2": [_cell("fc2", s, 0.68 + 0.001 * s, 0.98, 0.96) for s in seeds],
    }
    out = compare(per_layer, ("fc1", "fc2"), 0.95)
    assert out["statements"]["L1_error_cell_auroc"]["verdict"] == "fc1 better"
    assert out["statements"]["L1_error_cell_auroc"]["primary"] is True
    assert out["statements"]["L3_qe_novelty_auroc"]["verdict"] == "inconclusive"
    assert out["means"]["A1_error_cell_auroc"]["fc2"] == pytest.approx(0.683)
