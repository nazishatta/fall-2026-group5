"""Build a traceable Phase 3 comparison from immutable EC2 result JSONs."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.v1_mnist.component.utils.file_hash import sha256_file
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.utils.run_ids import check_run_id

LOGGER = setup_logger(__name__, level=logging.INFO)
CONFIG_PATH = Path('src/v1_mnist/component/configs/phase3_evidence_consolidation.yaml')


def load_json(path: Path) -> dict[str, Any]:
    """Read a metric JSON without changing its original file."""
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f'Expected JSON object: {path}')
    return value


def settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Turn the consolidation config into the keyword arguments of ``consolidate``."""
    return {
        'source_root': Path(config['source_root']),
        'historical': str(config['sources']['historical']),
        'additional': str(config['sources']['additional']),
        'paired_source': str(config['sources']['paired']),
        'expected_id_count': int(config['expected']['id_count']),
        'expected_ood_count': int(config['expected']['ood_count']),
        'tolerance': float(config['tolerance']),
    }


def consolidate(
    root: Path,
    *,
    source_root: Path,
    historical: str,
    additional: str,
    paired_source: str,
    expected_id_count: int,
    expected_ood_count: int,
    tolerance: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Join historical U-Matrix and required baselines without refitting models."""
    historical_data = load_json(root / source_root / historical)
    additional_data = load_json(root / source_root / additional)
    paired_data = load_json(root / source_root / paired_source)
    counts = historical_data["sample_counts"]
    expected = (expected_id_count, expected_ood_count)

    if (counts["mnist_validation"], counts["svhn_test"]) != expected:
        raise ValueError("Unexpected historical evaluation counts")

    if (
        additional_data["id_samples"],
        additional_data["ood_samples"],
    ) != expected:
        raise ValueError("Unexpected additional evaluation counts")

    if (paired_data["n_id"], paired_data["n_ood"]) != expected:
        raise ValueError("Unexpected paired evaluation counts")

    if abs(
        historical_data["methods"]["qe"]["auroc"]
        - paired_data["qe_auroc"]
    ) > tolerance:
        raise ValueError("QE reference AUROC differs between experiments")

    if abs(
        additional_data["methods"]["max_softmax"]["auroc"]
        - paired_data["msp_auroc"]
    ) > tolerance:
        raise ValueError("MSP reference AUROC differs between experiments")

    rows = []
    for source, collection in ((historical, historical_data['methods']), (additional, additional_data['methods'])):
        for method, values in collection.items():
            rows.append({
                'method': method,
                'auroc': f"{values['auroc']:.9f}",
                'aupr': f"{values['aupr']:.9f}",
                'fpr_at_95_tpr': f"{values['fpr_at_95_tpr']:.9f}",
                'source_run_id': source.split('/')[0],
                'evaluation': 'MNIST validation (10500) versus SVHN test (26032)',
                'comparison_type': 'historical' if source == historical else 'additional frozen CNN',
            })
    metadata = {
        'scope': 'descriptive side-by-side consolidation, not a newly run or equal-budget experiment',
        'headline_baseline_source': additional,
        'historical_u_matrix_source': historical,
        'paired_qe_msp_source': paired_source,
        'qe_minus_msp': paired_data['qe_minus_msp_auroc'],
        'qe_minus_msp_ci_95': paired_data['difference_ci_95'],
        'id_count': expected_id_count,
        'ood_count': expected_ood_count,
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
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    with args.config.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    run_id = check_run_id(config['run_id'])

    rows, metadata = consolidate(args.root, **settings_from_config(config))
    metadata['run_id'] = run_id
    metadata['config_sha256'] = sha256_file(args.config)
    destination = (args.root / config['output_root'] / run_id).resolve()
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
