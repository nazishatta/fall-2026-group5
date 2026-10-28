"""Reproduce Phase 3 QE/U-Matrix paired and random-score OOD analyses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = setup_logger(__name__, level=logging.INFO)
ROOT = Path(__file__).resolve().parents[5]




def within_root(relative: str) -> Path:
    """Resolve and validate a repository-relative path."""
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def run(config_path: Path) -> Path:
    """Compute both controls and save to a new, exclusive run directory."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_id = config["run_id"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("Invalid run ID")

    destination = within_root(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"STOP: Run already exists: {destination}")

    source_root = within_root(config["source_root"])
    scores = {}
    hashes = {}

    for name, expected in config["expected_sha256"].items():
        path = source_root / name
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"STOP: SHA-256 mismatch: {name}")

        values = np.load(path, allow_pickle=False).astype(np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid score array: {name}")

        scores[name] = values
        hashes[name] = observed

    def get(method: str, split: str) -> np.ndarray:
        return scores[f"{method}_{split}_scores.npy"]

    qe_id = get("qe", "mnist_val")
    qe_ood = get("qe", "svhn_test")
    um_id = get("u_matrix", "mnist_val")
    um_ood = get("u_matrix", "svhn_test")

    n_id = int(config["expected_id_count"])
    n_ood = int(config["expected_ood_count"])
    if not (
        len(qe_id) == len(um_id) == n_id
        and len(qe_ood) == len(um_ood) == n_ood
    ):
        raise RuntimeError("STOP: Sample counts differ from frozen inputs")

    seed = int(config["seed"])
    paired = compare_auroc_paired(
        qe_id,
        qe_ood,
        um_id,
        um_ood,
        iterations=int(config["bootstrap_iterations"]),
        seed=seed,
    )

    simulations = int(config["random_simulations"])
    if simulations < 2:
        raise ValueError("At least two random simulations required")

    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)]
    random_aucs = np.empty(simulations, dtype=np.float64)

    for index in range(simulations):
        random_aucs[index] = roc_auc_score(
            labels, rng.random(len(labels))
        )

    result = {
        "run_id": run_id,
        "dataset": "MNIST validation vs SVHN test",
        "paired_qe_vs_u_matrix": {
            "qe_auroc": paired["qe_auroc"],
            "u_matrix_auroc": paired["msp_auroc"],
            "qe_minus_u_matrix": paired["qe_minus_msp_auroc"],
            "difference_ci_95": paired["difference_ci_95"],
            "bootstrap_iterations": paired["bootstrap_iterations"],
        },
        "random_score_null": {
            "mean_auroc": float(np.mean(random_aucs)),
            "simulation_interval_95": [
                float(value)
                for value in np.quantile(random_aucs, [0.025, 0.975])
            ],
            "simulations": simulations,
            "interval_type": "Monte Carlo simulation interval",
        },
        "seed": seed,
        "n_id": n_id,
        "n_ood": n_ood,
        "input_sha256": hashes,
        "config_sha256": sha256_file(config_path),
        "leakage_control": (
            "Historical frozen scores only; no model fitting, "
            "no threshold tuning, no official MNIST final test."
        ),
        "limitation": (
            "Paired inference assumes identical sample order within "
            "the historical score arrays; both methods share one run."
        ),
    }

    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "qe_umatrix_random_metrics.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")

    LOGGER.info("Saved %s", output)
    return output


def main() -> None:
    """Run the configured reproducibility analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
