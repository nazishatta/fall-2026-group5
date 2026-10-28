"""Build a traceable Phase 3 comparison from immutable EC2 result JSONs."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.v1_mnist.component.utils.logging import setup_logger

LOGGER = setup_logger(__name__, level=logging.INFO)
BASE = Path('outputs/v1_mnist/p3_novelty_detection/novelty_benchmark')
HISTORICAL = 'mnist_svhn_five_method_comparison_s42_initial/comparison_metrics.json'
ADDITIONAL = 'mnist_svhn_required_baselines_knn_s42_v1/required_baseline_metrics.json'
PAIRED = 'mnist_svhn_qe_msp_reproduction_s42_v1/qe_msp_paired_metrics.json'


def load_json(path: Path) -> dict[str, Any]:
    """Read a metric JSON without changing its original file."""
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f'Expected JSON object: {path}')
    return value


def consolidate(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Join historical U-Matrix and required baselines without refitting models."""
    historical = load_json(root / BASE / HISTORICAL)
    additional = load_json(root / BASE / ADDITIONAL)
    paired = load_json(root / BASE / PAIRED)
    counts = historical['sample_counts']
    if (counts['mnist_validation'], counts['svhn_test']) != (10500, 26032):
        raise ValueError('Unexpected historical evaluation counts')
    if (additional['id_samples'], additional['ood_samples']) != (10500, 26032):
        raise ValueError('Unexpected additional evaluation counts')
    if (paired['n_id'], paired['n_ood']) != (10500, 26032):
        raise ValueError('Unexpected paired evaluation counts')
    if abs(historical['methods']['qe']['auroc'] - paired['qe_auroc']) > 1e-10:
        raise ValueError('QE reference AUROC differs between experiments')
    if abs(additional['methods']['max_softmax']['auroc'] - paired['msp_auroc']) > 1e-10:
        raise ValueError('MSP reference AUROC differs between experiments')
    rows = []
    for source, collection in ((HISTORICAL, historical['methods']), (ADDITIONAL, additional['methods'])):
        for method, values in collection.items():
            rows.append({
                'method': method,
                'auroc': f"{values['auroc']:.9f}",
                'aupr': f"{values['aupr']:.9f}",
                'fpr_at_95_tpr': f"{values['fpr_at_95_tpr']:.9f}",
                'source_run_id': source.split('/')[0],
                'evaluation': 'MNIST validation (10500) versus SVHN test (26032)',
                'comparison_type': 'historical' if source == HISTORICAL else 'additional frozen CNN',
            })
    metadata = {
        'scope': 'descriptive side-by-side consolidation, not a newly run or equal-budget experiment',
        'headline_baseline_source': ADDITIONAL,
        'historical_u_matrix_source': HISTORICAL,
        'paired_qe_msp_source': PAIRED,
        'qe_minus_msp': paired['qe_minus_msp_auroc'],
        'qe_minus_msp_ci_95': paired['difference_ci_95'],
        'id_count': 10500,
        'ood_count': 26032,
        'limitations': [
            'Historical method fitting budgets differ; this table does not imply matched fitting protocols.',
            'Historical QE score arrays lack per-image IDs; paired analysis assumes deterministic order.',
            'SVHN was already examined; new diagnostics are exploratory.',
            'No MNIST final-test set was used or evaluated by this consolidation.',
        ],
    }
    return rows, metadata


def main() -> None:
    """Write small metrics and documentation into a new non-overwriting output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    rows, metadata = consolidate(args.root)
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite output: {destination}')
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'phase3_consolidated_metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / 'phase3_consolidation_provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
    LOGGER.info('Wrote consolidated evidence to %s', destination)


if __name__ == '__main__':
    main()
