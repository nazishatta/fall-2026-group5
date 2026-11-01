"""Inspect historical SVHN novelty-score separation without refitting or altering it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.v1_mnist.component.utils.logging import get_logger

LOGGER = get_logger(__name__)


def inspect(folder: Path) -> dict[str, object]:
    """Compare existing saved scores against immutable historical metrics."""
    historical = json.loads((folder / "comparison_metrics.json").read_text(encoding="utf-8"))
    output: dict[str, object] = {"historical_run_id": historical["run_id"], "methods": {}}
    for method in ("isolation_forest", "autoencoder", "one_class_svm"):
        id_file = folder / "score_arrays" / f"{method}_mnist_val_scores.npy"
        ood_file = folder / "score_arrays" / f"{method}_svhn_test_scores.npy"
        if not id_file.is_file() or not ood_file.is_file():
            output["methods"][method] = {"status": "missing score arrays; DVC pull may be required"}
            continue
        id_scores = np.load(id_file, allow_pickle=False).ravel()
        ood_scores = np.load(ood_file, allow_pickle=False).ravel()
        if not np.isfinite(id_scores).all() or not np.isfinite(ood_scores).all():
            raise ValueError(f"Nonfinite scores: {method}")
        y = np.r_[np.zeros(len(id_scores), dtype=int), np.ones(len(ood_scores), dtype=int)]
        scores = np.r_[id_scores, ood_scores]
        auc = float(roc_auc_score(y, scores))
        saved = float(historical["methods"][method]["auroc"])
        output["methods"][method] = {
            "status": "reproduced" if abs(auc - saved) < 1e-10 else "MISMATCH",
            "mnist_val_n": len(id_scores),
            "svhn_test_n": len(ood_scores),
            "mnist_val_score_mean": float(id_scores.mean()),
            "svhn_test_score_mean": float(ood_scores.mean()),
            "recomputed_auroc": auc,
            "recorded_auroc": saved,
            "higher_means_more_novel": True,
            "interpretation": "Descriptive only; do not flip score direction based on OOD outcome",
        }
    return output


def main() -> None:
    """Print diagnosis to stdout; make no filesystem writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from src.v1_mnist.component.utils.config import load_config

    settings = load_config(args.config)
    folder = Path(settings.historical_run_directory)
    result = inspect(folder)
    LOGGER.info("Historical SVHN separation diagnostic (no refits)")
    LOGGER.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
