"""Evaluate frozen detectors on user-provided held-out ID and OOD scores."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.ood_comparison import (
    analyze_misses,
    evaluate_scores,
    save_new_json,
)

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-scores", type=Path, required=True)
    parser.add_argument("--ood-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--frozen-threshold", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.output.exists():
        parser.error("Output exists: do not overwrite an existing run")
    if not args.run_id.strip():
        parser.error("A non-empty new run-id is required")
    normal = np.load(args.id_scores, allow_pickle=False)
    novel = np.load(args.ood_scores, allow_pickle=False)
    metrics = evaluate_scores(normal, novel, seed=args.seed,
                              bootstrap_repeats=args.bootstrap_repeats)
    payload: dict[str, object] = {
        "run_id": args.run_id,
        "detector": args.detector,
        "id_scores_sha256": _sha256(args.id_scores),
        "ood_scores_sha256": _sha256(args.ood_scores),
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
    LOGGER.info("Created Week 8 evaluation: %s", args.output)


if __name__ == "__main__":
    main()
