"""Run Phase 3 QE baseline against frozen, hash-verified Phase 2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.novelty_figures import (
    save_qe_distribution,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.quantization_novelty import (
    error_association,
    load_distances,
    summarize,
)
from src.v1_mnist.component.utils.run_settings import (
    cluster_output_dir,
    load_analysis_settings,
    verify_selected_model,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[5]


def sha256(path: Path) -> str:
    """Hash a local artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    """Fail closed for missing or changed upstream artifacts."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path} (expected {expected}; got {observed})")
    return observed


def run(config_path: Path, save: bool = False) -> dict[str, object]:
    """Analyze training and validation only; optionally save without overwrite."""
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    settings = load_analysis_settings(str(cfg['som_config']), ROOT)
    verified_model = verify_selected_model(settings)
    cluster_dir = cluster_output_dir(str(cfg['som_config']), ROOT)
    summary_file = cluster_dir / 'cluster_summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    if summary.get('test_used') is not False or summary.get('development_only') is not True:
        raise RuntimeError('Upstream cluster summary does not establish development-only use')
    if summary.get('som') != settings.selected_model or summary.get('model_sha256') != verified_model:
        raise RuntimeError('SOM model identity differs from cluster-analysis source')

    hashes = json.loads((cluster_dir / 'metadata/artifact_hashes.json').read_text(encoding='utf-8'))
    input_hashes = json.loads((cluster_dir / 'metadata/input_hashes.json').read_text(encoding='utf-8'))
    assignments = cluster_dir / 'assignments'
    train_file = assignments / 'train_bmu_distance.npy'
    val_file = assignments / 'val_bmu_distance.npy'
    correct_file = settings.embeddings_dir / 'val_correct.npy'
    artifacts = {
        'train_bmu_distance': require_hash(train_file, hashes['assignments/train_bmu_distance.npy']),
        'val_bmu_distance': require_hash(val_file, hashes['assignments/val_bmu_distance.npy']),
        'val_correct': require_hash(correct_file, input_hashes['val_correct']),
    }
    train = load_distances(train_file, int(summary['splits']['train']['samples']))
    val = load_distances(val_file, int(summary['splits']['val']['samples']))
    correct = np.load(correct_file, allow_pickle=False)
    if correct.shape != val.shape or correct.dtype != np.dtype('bool'):
        raise ValueError('Validation correctness alignment or dtype mismatch')

    result = summarize(train, val, float(cfg['threshold_percentile']))
    association = error_association(val, correct, float(result['threshold']))
    result.update({
        'som_run': settings.selected_model,
        'model_sha256': verified_model,
        'input_sha256': artifacts,
        'error_association': association,
        'run_id': str(cfg['run_id']),
    })
    if save:
        output = ROOT / str(cfg['output_dir']) / str(cfg['run_id'])
        # Preflight: stop if anything exists; no partial overwrite allowed.
        if output.exists():
            raise FileExistsError(f'Run already exists: {output}')
        output.mkdir(parents=True, exist_ok=False)
        destination = output / 'qe_summary.json'
        with destination.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
        save_qe_distribution(train, val, float(result['threshold']), output / 'qe_score_distribution.pdf')
        LOGGER.info('Saved %s', destination)
    return result


def main() -> None:
    """Execute configured QE analysis after required artifact checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('src/v1_mnist/component/configs/qe_novelty.yaml'))
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info('QE result: %s', json.dumps(run(args.config, args.save), indent=2))


if __name__ == '__main__':
    main()
