"""Validate non-destructive Phase 3 evidence consolidation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.consolidate_phase3_evidence import (
    CONFIG_PATH,
    consolidate,
    settings_from_config,
)

ROOT = Path(__file__).resolve().parents[5]
SETTINGS = settings_from_config(yaml.safe_load((ROOT / CONFIG_PATH).read_text(encoding="utf-8")))
HISTORICAL = SETTINGS["historical"]
ADDITIONAL = SETTINGS["additional"]
PAIRED = SETTINGS["paired_source"]


def test_consolidation_detects_verified_values(tmp_path: Path) -> None:
    """Read separate measured sources without fabricating or averaging runs."""
    fixtures = {
        HISTORICAL: {
            'sample_counts': {'mnist_validation': 10500, 'svhn_test': 26032},
            'methods': {'qe': {'auroc': 0.981, 'aupr': 0.97, 'fpr_at_95_tpr': 0.05},
                        'u_matrix': {'auroc': 0.969, 'aupr': 0.98, 'fpr_at_95_tpr': 0.07}},
        },
        ADDITIONAL: {
            'id_samples': 10500, 'ood_samples': 26032,
            'methods': {'max_softmax': {'auroc': 0.998, 'aupr': 0.99, 'fpr_at_95_tpr': 0.008}},
        },
        PAIRED: {'n_id': 10500, 'n_ood': 26032, 'qe_auroc': 0.981,
                 'msp_auroc': 0.998, 'qe_minus_msp_auroc': -0.017,
                 'difference_ci_95': [-0.019, -0.015]},
    }
    for name, obj in fixtures.items():
        path = tmp_path / SETTINGS["source_root"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')
    rows, metadata = consolidate(tmp_path, **SETTINGS)
    assert {row['method'] for row in rows} == {'qe', 'u_matrix', 'max_softmax'}
    assert metadata['headline_baseline_source'] == ADDITIONAL
    assert metadata['qe_minus_msp'] == pytest.approx(-0.017)

def test_consolidation_accepts_alternative_source_paths(
    tmp_path: Path,
) -> None:
    """Alternative source files are selected without using historical paths."""
    source_root = Path("custom_evidence")

    fixtures = {
        "historical.json": {
            "sample_counts": {"mnist_validation": 3, "svhn_test": 2},
            "methods": {
                "qe": {"auroc": 0.8, "aupr": 0.7, "fpr_at_95_tpr": 0.1},
            },
        },
        "additional.json": {
            "id_samples": 3,
            "ood_samples": 2,
            "methods": {
                "max_softmax": {
                    "auroc": 0.9,
                    "aupr": 0.8,
                    "fpr_at_95_tpr": 0.05,
                },
            },
        },
        "paired.json": {
            "n_id": 3,
            "n_ood": 2,
            "qe_auroc": 0.8,
            "msp_auroc": 0.9,
            "qe_minus_msp_auroc": -0.1,
            "difference_ci_95": [-0.2, 0.0],
        },
    }

    directory = tmp_path / source_root
    directory.mkdir()

    for filename, payload in fixtures.items():
        (directory / filename).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    rows, metadata = consolidate(
        tmp_path,
        source_root=source_root,
        historical="historical.json",
        additional="additional.json",
        paired_source="paired.json",
        expected_id_count=3,
        expected_ood_count=2,
        tolerance=1e-10,
    )

    assert {row["method"] for row in rows} == {"qe", "max_softmax"}
    assert metadata["headline_baseline_source"] == "additional.json"
    assert metadata["id_count"] == 3
    assert metadata["ood_count"] == 2


def test_config_matches_the_recorded_consolidation_run() -> None:
    """The config holds the same sources the existing consolidated run recorded."""
    config = yaml.safe_load((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    recorded = json.loads(
        (ROOT / config["output_root"] / config["run_id"] / "phase3_consolidation_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert recorded["historical_u_matrix_source"] == HISTORICAL
    assert recorded["headline_baseline_source"] == ADDITIONAL
    assert recorded["paired_qe_msp_source"] == PAIRED
    assert (recorded["id_count"], recorded["ood_count"]) == (
        SETTINGS["expected_id_count"],
        SETTINGS["expected_ood_count"],
    )
