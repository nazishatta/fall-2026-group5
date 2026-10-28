"""Validate non-destructive Phase 3 evidence consolidation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v1_mnist.component.pipeline.p3_novelty_detection.consolidate_phase3_evidence import (
    ADDITIONAL,
    BASE,
    HISTORICAL,
    PAIRED,
    consolidate,
)


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
        path = tmp_path / BASE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')
    rows, metadata = consolidate(tmp_path)
    assert {row['method'] for row in rows} == {'qe', 'u_matrix', 'max_softmax'}
    assert metadata['headline_baseline_source'] == ADDITIONAL
    assert metadata['qe_minus_msp'] == pytest.approx(-0.017)
