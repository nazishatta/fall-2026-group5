"""The Fashion-MNIST Phase 3 configs match the protocol (docs/p3_novelty_detection/NOVELTY_DETECTION_PROTOCOL.md).

These tests pin what the protocol fixes: which steps the study has, the frozen inputs, the settings
of statements O1 to O4, and the held-out class splits. A config change that breaks one of them
needs a written amendment first (docs/DEVIATIONS.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_novelty_benchmark,
    run_qe_novelty_detection,
    run_u_matrix_analysis,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "src" / "v1_fashion_mnist" / "component" / "configs"
RUNNER_STEPS = [*run_qe_novelty_detection.STEPS, *run_novelty_benchmark.STEPS, *run_u_matrix_analysis.STEPS]

# Protocol section 3: MNIST steps that only re-created or repaired MNIST's own earlier runs. None of them
# has a Fashion-MNIST config. The two diagnostics that run only if MNIST's problem appears are not listed:
# it appeared, so they have configs (docs/DEVIATIONS.md, 2026-11-21).
NOT_REPEATED = {
    "qe_score_distribution_figure",      # re-ran qe_novelty because MNIST's first run lost its figure
    "historical_baseline_verification", "historical_svhn_inspection", "consolidate_evidence",
}
# Values that only an earlier Fashion-MNIST step can give (filled in stages; see the Phase 3 README).
MAY_WAIT = {
    "qe_threshold_sweep.yaml", "novelty_benchmark.yaml", "svhn_paired_comparison.yaml", "svhn_score_export.yaml",
    "qe_energy_knn_paired.yaml", "qe_u_matrix_random_ood.yaml", "fair_budget_baselines.yaml",
    "unified_novelty_benchmark.yaml", "qe_norm_dependence.yaml", "missed_svhn_figure.yaml",
    "qe_umatrix_validation_figure.yaml", "svhn_som_overlay.yaml",
}

MNIST_CONFIG_DIR = REPO_ROOT / "src" / "v1_mnist" / "component" / "configs"
# Settings a Fashion-MNIST config has on purpose and its MNIST original does not.
ADDED_ON_PURPOSE = {
    "base.yaml": {"dataset.class_names"},
    "missed_svhn_figure.yaml": {"scores_subfolder"},
    "qe_norm_dependence.yaml": {"analysis_status"},
    "unified_novelty_benchmark.yaml": {"study_type"},
}

def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _values(v)]
    if isinstance(value, list):
        return [x for v in value for x in _values(v)]
    return [value]


def test_the_configs_are_exactly_the_protocols_phase3_steps() -> None:
    present = {step.name for step in RUNNER_STEPS if (CONFIG_DIR / step.config).is_file()}
    assert present == {step.name for step in RUNNER_STEPS} - NOT_REPEATED
    for extra in ("novelty_statements.yaml", "held_out_class_novelty.yaml", "feature_layer_ablation.yaml",
                  "phase3_shared_provenance.yaml"):
        assert (CONFIG_DIR / extra).is_file(), extra


def test_every_config_uses_the_frozen_fashion_mnist_inputs() -> None:
    manifest = _load("phase3_shared_provenance.yaml")
    frozen = manifest["expected_sha256"]
    names = {"checkpoint": "checkpoint", "train_features": "train_features", "train": "train_features",
             "validation_features": "validation_features", "validation": "validation_features",
             "svhn_file": "svhn_file"}
    checked = 0
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in (raw.get("expected_sha256") or {}).items():
            if key in names:
                assert value == frozen[names[key]], (path.name, key)
                checked += 1
        if "cnn_sha256" in raw:
            assert raw["cnn_sha256"] == frozen["checkpoint"], path.name
    assert checked >= 20
    assert manifest["svhn_file"] == "data/ood_svhn/test_32x32.mat"
    assert frozen["svhn_file"].startswith("cdce80df")          # the same SVHN file as MNIST's


def test_run_ids_are_unique_and_new() -> None:
    run_ids = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        run_ids += [raw["run_id"]] if "run_id" in raw else []
        run_ids += [entry["run_id"] for entry in raw.get("ood_datasets", [])]
    assert len(run_ids) == len(set(run_ids))
    assert all(run_id.startswith("fashion_mnist_") for run_id in run_ids if "svhn" in run_id or "qe" in run_id)


def test_tbd_values_appear_only_where_an_earlier_step_gives_them() -> None:
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        waiting = [v for v in _values(yaml.safe_load(path.read_text(encoding="utf-8")))
                   if isinstance(v, str) and v.startswith("[TBD")]
        if path.name not in MAY_WAIT:
            assert not waiting, (path.name, waiting)


def test_statements_use_the_protocols_numbers() -> None:
    evaluation = _load("novelty_statements.yaml")["evaluation"]
    assert evaluation == {
        "energy_temperature": 1.0, "knn_neighbors": 5, "target_tpr": 0.95,
        "bootstrap_iterations": 2000, "bootstrap_seed": 42,
        "paired_iterations": 2000, "paired_seed": 42,
        "shuffle_iterations": 1000, "shuffle_seed": 42,
        "competitive_margin": 0.01, "protocol_statements": True,
    }


def test_held_out_splits_are_the_protocols() -> None:
    splits = _load("held_out_class_novelty.yaml")["splits"]
    assert splits["generation_seed"] == 2026 and splits["unknown_per_split"] == 4
    assert splits["unknown_digits"] == [[0, 1, 5, 6], [2, 3, 5, 8], [1, 5, 6, 7], [2, 5, 6, 8], [0, 4, 5, 8]]


def test_the_outside_set_is_svhn_test_only() -> None:
    ood = _load("qe_ood_evaluation.yaml")["ood_datasets"]
    assert [(entry["name"], entry["split"]) for entry in ood] == [("svhn", "test")]
    assert _load("dataset/fashion_mnist.yaml")["dataset"]["ood_dataset"] == "svhn"


def _keys(value: Any, prefix: str = "") -> set[str]:
    """Every setting name in a config, nested names joined with dots."""
    if not isinstance(value, dict):
        return set()
    names: set[str] = set()
    for key, item in value.items():
        name = f"{prefix}{key}"
        names |= {name} | _keys(item, f"{name}.")
    return names


def test_no_setting_of_the_mnist_original_was_dropped() -> None:
    """Each Fashion-MNIST copy keeps every MNIST setting; only MNIST names become Fashion-MNIST names."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        original = MNIST_CONFIG_DIR / path.name
        if not original.exists():
            continue
        expected = {name.replace("mnist", "fashion_mnist") for name in _keys(yaml.safe_load(original.read_text(encoding="utf-8")))}
        actual = _keys(_load(path.name))
        assert expected - actual == set(), f"{path.name} lost settings"
        assert actual - expected <= ADDED_ON_PURPOSE.get(path.name, set()), f"{path.name} has unexpected settings"
