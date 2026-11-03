"""Frozen-score paired OOD comparison for Phase 3."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)


def evaluate_paired_qe(
    config: Any,
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    run_id: str,
    hashes: dict[str, str],
    config_path: Path,
    input_path: Callable[[str], Path],
    fingerprint: Callable[[Path], str],
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Preserve the historical QE-versus-MSP paired comparison."""
    paired_result: dict[str, Any] | None = None
    if hasattr(config, "paired_qe"):
        qe_settings = config.paired_qe
        qe_id_path = input_path(str(qe_settings.id_scores))
        qe_ood_path = input_path(str(qe_settings.ood_scores))

        for label, path, expected_hash in (
            ("MNIST QE", qe_id_path, str(qe_settings.id_sha256)),
            ("SVHN QE", qe_ood_path, str(qe_settings.ood_sha256)),
        ):
            if fingerprint(path) != expected_hash:
                raise RuntimeError(f"STOP: {label} fingerprint mismatch")

        qe_id = np.load(qe_id_path, allow_pickle=False)
        qe_ood = np.load(qe_ood_path, allow_pickle=False)
        msp_id, msp_ood = scores["max_softmax"]

        paired_result = compare_auroc_paired(
            qe_id,
            qe_ood,
            msp_id,
            msp_ood,
            iterations=int(config.evaluation.bootstrap_iterations),
            seed=int(config.evaluation.bootstrap_seed),
        )

        for method, actual, expected in (
            ("QE", paired_result["qe_auroc"], qe_settings.expected_qe_auroc),
            ("MSP", paired_result["msp_auroc"], qe_settings.expected_msp_auroc),
        ):
            if not np.isclose(
                float(actual),
                float(expected),
                rtol=0,
                atol=float(qe_settings.auroc_tolerance),
            ):
                raise RuntimeError(f"STOP: {method} AUROC mismatch")

        paired_result.update({
            "run_id": run_id,
            "id_dataset": "mnist_validation",
            "ood_dataset": "svhn_test",
            "qe_id_sha256": fingerprint(qe_id_path),
            "qe_ood_sha256": fingerprint(qe_ood_path),
            "checkpoint_sha256": hashes["checkpoint"],
            "svhn_sha256": hashes["svhn_file"],
            "source_config_sha256": fingerprint(config_path),
            "leakage_control": (
                "Frozen CNN and historical QE; no OOD fitting, tuning "
                "or final MNIST test-set use."
            ),
            "alignment_evidence": (
                "Same frozen checkpoint, torchvision SVHN test ordering, "
                "SVHN preprocessing and unshuffled inference. Historical "
                "QE arrays lack per-image identifiers; alignment is "
                "supported but cannot be independently ID-verified."
            ),
        })
        logger.info(
            "QE minus MSP AUROC=%.9f; paired CI=%s",
            paired_result["qe_minus_msp_auroc"],
            paired_result["difference_ci_95"],
        )

    return paired_result
