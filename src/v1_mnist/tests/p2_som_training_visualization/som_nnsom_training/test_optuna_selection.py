"""Tests for the Optuna tuning rules and config (no SOM training)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_common import (
    neighbourhood_from_fraction,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.optuna_selection import (
    TrialResult,
    average_ranks,
    build_run_id,
    metric_verdict,
    overall_verdict,
    paired_difference,
    pareto_front,
    parse_grid,
    select_trial,
)

CODE_ROOT = Path(__file__).resolve().parents[5]
CONFIG = CODE_ROOT / "src/v1_mnist/component/configs/som_optuna.yaml"


def _t(number: int, nb: int, epochs: int, qe: float, te1: float, te12: float) -> TrialResult:
    return TrialResult(number, nb, epochs, qe, te1, te12)


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_dominated_trial_is_dropped() -> None:
    good = _t(0, 5, 200, 1.0, 5.0, 3.0)
    worse = _t(1, 6, 200, 1.1, 6.0, 2.0)  # worse on QE and TE1
    other = _t(2, 3, 100, 0.9, 7.0, 4.0)  # better QE, worse TE1: not dominated
    assert pareto_front([good, worse, other]) == [good, other]


def test_rank_sum_picks_the_balanced_trial() -> None:
    low_qe = _t(0, 2, 100, 0.90, 9.0, 6.0)
    balanced = _t(1, 7, 250, 0.95, 5.0, 3.0)
    low_te = _t(2, 12, 400, 1.20, 4.0, 7.0)
    assert select_trial([low_qe, balanced, low_te]) == balanced


def test_ties_go_to_smaller_neighbourhood_then_fewer_epochs() -> None:
    a = _t(0, 9, 300, 1.0, 5.0, 3.0)
    b = _t(1, 7, 300, 1.0, 5.0, 3.0)
    c = _t(2, 7, 200, 1.0, 5.0, 3.0)
    assert select_trial([a, b, c]) == c


def test_tied_values_share_their_average_rank() -> None:
    assert average_ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]


def test_no_trials_is_an_error() -> None:
    with pytest.raises(ValueError):
        select_trial([])


def test_fraction_gives_the_three_point_settings() -> None:
    # The 3-point search used 0.20 / 0.47 / 0.73 of the side length.
    assert [neighbourhood_from_fraction(f, 15) for f in (0.20, 0.47, 0.73)] == [3, 7, 11]
    assert [neighbourhood_from_fraction(f, 20) for f in (0.20, 0.47, 0.73)] == [4, 9, 15]
    assert neighbourhood_from_fraction(0.01, 10) == 1
    with pytest.raises(ValueError):
        neighbourhood_from_fraction(1.5, 10)


def test_paired_difference_interval() -> None:
    res = paired_difference([-1.0, -1.0, -1.0])
    assert res["mean"] == -1.0 and res["ci_low"] == res["ci_high"] == -1.0
    wide = paired_difference([-2.0, 1.0, 0.5, -0.5])
    assert wide["ci_low"] < 0 < wide["ci_high"]
    with pytest.raises(ValueError):
        paired_difference([1.0])


def test_verdicts() -> None:
    assert metric_verdict(-0.3, -0.1) == "optuna better"
    assert metric_verdict(0.1, 0.3) == "optuna worse"
    assert metric_verdict(-0.1, 0.1) == "no clear difference"
    assert overall_verdict(["no clear difference"] * 3).startswith("3-point search confirmed")
    assert overall_verdict(["optuna better", "no clear difference", "no clear difference"]) == "Optuna setting better"
    assert overall_verdict(["optuna better", "optuna worse", "no clear difference"]).startswith("trade-off")


def test_grid_and_run_id() -> None:
    assert parse_grid("15x15") == (15, 15)
    with pytest.raises(ValueError):
        parse_grid("15-15")
    run_id = build_run_id("15x15", 20, 42, "v1")
    assert run_id == "som_optuna_15x15_t20_s42_v1"
    assert "week" not in run_id
    assert build_run_id("5x5", 3, 42, "v1", smoke=True).endswith("_smoke")


def test_config_is_fair_and_matches_the_three_point_study() -> None:
    cfg = _cfg()
    opt = cfg["optuna"]
    assert opt["grids"] == ["10x10", "15x15", "20x20"]
    assert isinstance(opt["n_trials"], int) and opt["n_trials"] > opt["n_startup_trials"]
    space = opt["search_space"]
    assert 0 < space["nb_fraction_min"] <= 0.20 and space["nb_fraction_max"] == 1.0
    assert space["epochs_min"] <= 250 <= space["epochs_max"]
    assert (250 - space["epochs_min"]) % space["epochs_step"] == 0
    assert cfg["reference_settings"] == {
        "10x10": {"init_neighborhood": 7, "epochs": 250},
        "15x15": {"init_neighborhood": 11, "epochs": 250},
        "20x20": {"init_neighborhood": 15, "epochs": 250},
    }
    seeds = cfg["confirmation"]["seeds"]
    assert len(seeds) >= 5 and opt["search_seed"] not in seeds
