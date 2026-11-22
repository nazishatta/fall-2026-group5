"""The Fashion-MNIST Phase 4 configs match the protocol (docs/p4_guided_retraining/RETRAINING_PROTOCOL.md).

Every MNIST setting is copied unchanged except the dataset's own paths and the few values the protocol
names: the macro-F1 guard (measured by its rule), the protocol and stamp files, the first run's revision
and the transform-check limit. The guard is either still empty or exactly what its rule gives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = REPO_ROOT / "src" / "v1_fashion_mnist" / "component" / "configs"
MNIST_CONFIGS = REPO_ROOT / "src" / "v1_mnist" / "component" / "configs"
PROTOCOL = REPO_ROOT / "src" / "v1_fashion_mnist" / "docs" / "p4_guided_retraining" / "RETRAINING_PROTOCOL.md"
PHASE4 = ("retraining.yaml", "som_reanalysis.yaml", "som_reanalysis_msp_arm.yaml", "novelty_after_retraining.yaml")
P4_OUT = REPO_ROOT / "outputs" / "v1_fashion_mnist" / "p4_guided_retraining"
# The validation summaries the guard is measured from (protocol section 2: the larger of the two).
SUMMARIES = (
    P4_OUT / "hard_example_mining" / "curriculum_comparison_som_15x15_nb11_ep250_s42" / "tables" / "validation_summary.json",
    P4_OUT / "targeted_augmentation" / "augmentation_comparison_som_15x15_nb11_ep250_s42" / "tables" / "validation_summary.json",
)
# Values the protocol sets for Fashion-MNIST itself (retraining.yaml).
OWN_VALUES = {
    ("evaluation", "macro_f1_guard"), ("protocol", "document"), ("protocol", "stamp"),
    ("augmentation", "run", "revision"), ("augmentation", "transform_check"),
}


def _load(folder: Path, name: str) -> dict[str, Any]:
    return yaml.safe_load((folder / name).read_text(encoding="utf-8"))


def _as_mnist(value: Any) -> Any:
    """The Fashion-MNIST value with its own folder names turned back into MNIST's."""
    if isinstance(value, dict):
        return {k: _as_mnist(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_mnist(v) for v in value]
    return value.replace("v1_fashion_mnist", "v1_mnist") if isinstance(value, str) else value


def _drop(config: dict[str, Any], keys: set[tuple[str, ...]]) -> dict[str, Any]:
    for key in keys:
        parent = config
        for part in key[:-1]:
            parent = parent.get(part, {})
        parent.pop(key[-1], None)
    return config


@pytest.mark.parametrize("name", PHASE4)
def test_only_the_dataset_and_the_protocols_own_values_differ_from_mnist(name: str) -> None:
    own = OWN_VALUES if name == "retraining.yaml" else set()
    fashion = _drop(_as_mnist(_load(CONFIGS, name)), own)
    mnist = _drop(_load(MNIST_CONFIGS, name), own)
    assert fashion == mnist


@pytest.mark.parametrize("name", PHASE4)
def test_no_path_points_at_mnist_results_or_configs(name: str) -> None:
    def strings(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return [value] if isinstance(value, str) else []

    for value in strings(_load(CONFIGS, name)):
        assert "outputs/v1_mnist" not in value and "v1_mnist/component/configs" not in value, value


def test_the_protocol_and_its_stamp_are_fashion_mnist_files() -> None:
    protocol = _load(CONFIGS, "retraining.yaml")["protocol"]
    assert REPO_ROOT / protocol["document"] == PROTOCOL
    assert protocol["stamp"].startswith("src/v1_fashion_mnist/docs/p4_guided_retraining/")


def test_the_first_run_uses_mnists_final_ranges_and_the_protocols_check_limit() -> None:
    fashion = _load(CONFIGS, "retraining.yaml")["augmentation"]
    assert fashion["transforms"] == _load(MNIST_CONFIGS, "retraining.yaml")["augmentation"]["transforms"]
    assert fashion["transforms"]["max_shift_pixels"] == 1.0
    assert fashion["run"]["revision"] == ""
    assert fashion["transform_check"] == {"max_accuracy_drop": 0.02}      # 2 percentage points


def test_the_macro_f1_guard_is_empty_or_follows_its_rule() -> None:
    guard = _load(CONFIGS, "retraining.yaml")["evaluation"]["macro_f1_guard"]
    protocol = PROTOCOL.read_text(encoding="utf-8")
    if guard is None:
        assert "[TBD: measured by the rule" in protocol
        return
    suggestions = [json.loads(p.read_text(encoding="utf-8"))["macro_f1_guard_suggestion"] for p in SUMMARIES]
    assert guard == pytest.approx(max(suggestions), rel=1e-12)
    assert "[TBD: measured by the rule" not in protocol
