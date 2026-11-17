"""Statements N1 to N4 and the config of the novelty-after-retraining step."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.comparison import (
    INCONCLUSIVE,
    OPPOSITE,
    SHOWN,
    detection_verdict,
    margin_verdict,
    statements,
)

CODE_ROOT = Path(__file__).resolve().parents[5]
CONFIG = CODE_ROOT / "src/v1_mnist/component/configs/novelty_after_retraining.yaml"


def test_margin_verdict_has_three_outcomes() -> None:
    assert margin_verdict([-0.005, 0.02], 0.01) == SHOWN
    assert margin_verdict([-0.08, -0.02], 0.01) == OPPOSITE
    assert margin_verdict([-0.03, 0.01], 0.01) == INCONCLUSIVE


def test_detection_verdict_lists_failing_cells() -> None:
    assert detection_verdict({"a": [0.9, 0.95], "b": [0.7, 0.8]}) == (SHOWN, [])
    assert detection_verdict({"a": [0.9, 0.95], "b": [0.45, 0.6]}) == (OPPOSITE, ["b"])


def _cell(qe: float, msp: float) -> dict:
    return {"scores": {"qe": {"auroc": qe, "auroc_interval": [qe - 0.01, qe + 0.01]}, "msp": {"auroc": msp}}}


def test_statements_on_synthetic_cells() -> None:
    seeds = (1, 2, 3, 4, 5)
    metrics = {}
    for s in seeds:
        metrics[f"standard_seed{s}"] = _cell(0.980 + 0.001 * s, 0.998)
        metrics[f"augmented_som_seed{s}"] = _cell(0.981 + 0.001 * s, 0.998)
        metrics[f"augmented_random_seed{s}"] = _cell(0.979 + 0.001 * s, 0.998)
    settings = SimpleNamespace(seeds=seeds, auroc_margin=0.01, confidence=0.95)
    out = statements(metrics, settings)
    assert out["N1_som_still_detects_novelty"]["verdict"] == SHOWN
    assert out["N2_guided_retraining_does_not_harm"]["verdict"] == SHOWN
    assert out["N3_guidance_does_not_harm"]["verdict"] == SHOWN
    assert out["N4_gap_to_max_softmax_does_not_widen"]["verdict"] == SHOWN


def test_config_matches_the_protocol() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["cells"]["arms"] == ["standard", "augmented_som", "augmented_random"]
    assert cfg["cells"]["seeds"] == [1, 2, 3, 4, 5]
    assert cfg["ood"]["dataset"] == "svhn" and cfg["ood"]["image_size"] == [28, 28]
    assert cfg["scores"]["energy_temperature"] == 1.0
    assert cfg["statistics"]["auroc_margin"] == 0.01 and cfg["statistics"]["bootstrap_seed"] == 42
    assert ["augmented_som", "standard"] in cfg["comparison"]["pairs"]
    assert ["augmented_som", "augmented_random"] in cfg["comparison"]["pairs"]
