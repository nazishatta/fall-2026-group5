from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.hotspot_comparison import (
    analyse_partition,
    check_reproduction,
    hotspot_shares,
    summarize_seeds,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.error_pattern_robustness import (
    RobustnessSettings,
)

CFG = RobustnessSettings(n_splits=5, n_permutations=5, min_errors=5, split_seed_base=7000,
                         perm_seed_base=9000, hotspot_fraction=0.2, min_hotspots=2)


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 10, 1000)
    labels = rng.integers(0, 3, 1000)
    errors = (cells < 2) & (rng.random(1000) < 0.4)
    return cells, errors, labels


def test_hotspots_catch_more_errors_than_their_share_of_images() -> None:
    cells, errors, _ = _toy()
    shares = hotspot_shares(cells, errors, 10, CFG)
    assert shares["mean_error_share_in_hotspots"] > 2 * shares["mean_image_share_in_hotspots"]


def test_analyse_partition_uses_the_project_statistics() -> None:
    cells, errors, labels = _toy()
    result = analyse_partition(cells, errors, labels, 10, 3, CFG)
    assert result["hotspot_concentration_held_out"]["median_ratio"] > 1.5
    assert {"overlap_regions", "hotspot_shares", "settings"} <= set(result)


def test_reproduction_check_detects_a_mismatch(tmp_path: Path) -> None:
    cells, errors, labels = _toy()
    result = analyse_partition(cells, errors, labels, 10, 3, CFG)
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps(result))
    assert check_reproduction(result, reference)["reproduced_exactly"]
    result["hotspot_concentration_held_out"]["median_ratio"] += 0.5
    assert not check_reproduction(result, reference)["reproduced_exactly"]


def test_summarize_seeds_gives_median_and_range() -> None:
    def seed_result(ratio: float) -> dict:
        return {"hotspot_concentration_held_out": {"median_ratio": ratio, "chance_median_ratio": 1.0, "p_value_vs_chance": 0.005},
                "hotspot_shares": {"mean_image_share_in_hotspots": 0.04, "mean_error_share_in_hotspots": 0.2}}
    summary = summarize_seeds({0: seed_result(4.0), 1: seed_result(6.0), 2: seed_result(5.0)})
    assert summary["median_ratio_over_seeds"] == 5.0
    assert (summary["min_ratio"], summary["max_ratio"]) == (4.0, 6.0)
