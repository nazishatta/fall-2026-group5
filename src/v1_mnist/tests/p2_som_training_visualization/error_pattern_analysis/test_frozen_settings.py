"""The analysis settings fixed in advance keep their values in every SOM config.

They live in the configs (project rule: no hard-coded hyperparameters), but the
hotspot rule and the robustness check were fixed in ERROR_PATTERN_ROBUSTNESS_CRITERIA.md
and HOTSPOT_RULE.md before the results were seen, so they must not change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[3] / "component" / "configs"
SOM_CONFIGS = ["som.yaml", "som_10x10_robustness.yaml", "som_20x20_robustness.yaml"]


@pytest.mark.parametrize("name", SOM_CONFIGS)
def test_fixed_settings_unchanged(name: str) -> None:
    analysis = load_config(str(CONFIGS / name)).analysis
    assert (analysis.hotspot_fraction, analysis.min_hotspots) == (0.05, 5)
    rob = analysis.robustness
    assert (rob.split_repeats, rob.shuffles, rob.min_errors) == (100, 200, 20)
    assert (rob.split_seed_base, rob.shuffle_seed_base, rob.alpha) == (7000, 9000, 0.05)
    assert analysis.qe_relative_tolerance == 1e-6
