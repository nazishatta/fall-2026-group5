"""Evaluate frozen detectors on user-provided held-out ID and OOD scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    analyze_misses,
    evaluate_scores,
    save_new_json,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = get_logger("v1_mnist.p3.ood_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-scores", type=Path, required=True)
    parser.add_argument("--ood-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--frozen-threshold", type=float)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/v1_mnist/component/configs/qe_ood_evaluation.yaml"),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists: do not overwrite an existing run")
    if not args.run_id.strip():
        parser.error("A non-empty new run-id is required")
    normal = np.load(args.id_scores, allow_pickle=False)
    novel = np.load(args.ood_scores, allow_pickle=False)
    settings = load_config(args.config)
    metrics = evaluate_scores(
        normal,
        novel,
        seed=int(settings.bootstrap.seed),
        bootstrap_repeats=int(settings.bootstrap.iterations),
        target_tpr=float(settings.target_tpr),
    )
    payload: dict[str, object] = {
        "run_id": args.run_id,
        "detector": args.detector,
        "id_scores_sha256": sha256_file(args.id_scores),
        "ood_scores_sha256": sha256_file(args.ood_scores),
        "evaluation": metrics,
        "leakage_control": (
            "Scores must come from frozen models trained without held-out ID/OOD evaluation examples; "
            "this runner does not fit or tune models"
        ),
    }
    if args.frozen_threshold is not None:
        payload["missed_ood"] = analyze_misses(
            novel, threshold=args.frozen_threshold
        )
    save_new_json(args.output, payload)
    LOGGER.info("Created novelty evaluation: %s", args.output)


if __name__ == "__main__":
    main()
