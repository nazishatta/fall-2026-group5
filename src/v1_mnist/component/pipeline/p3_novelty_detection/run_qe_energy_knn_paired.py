"""Paired AUROC comparisons of SOM QE against energy and kNN."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.paired_bootstrap import (
    compare_auroc_paired,
)
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.utils.som_io import sha256_file

LOGGER = setup_logger(__name__, level=logging.INFO)
ROOT = Path(__file__).resolve().parents[5]


def resolve_path(value: str) -> Path:
    """Allow only repository-contained configured paths."""
    candidate = (ROOT / value).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {value}")
    return candidate


def run(config_path: Path) -> Path:
    """Verify frozen score arrays and save new paired comparisons."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_id = str(config["run_id"])
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\\\" in run_id:
        raise ValueError("Invalid run ID")

    destination = resolve_path(config["output_root"]) / run_id
    if destination.exists():
        raise FileExistsError(f"STOP: Existing run: {destination}")

    counts = config["expected_counts"]
    expected = {
        "mnist_val": int(counts["mnist_validation"]),
        "svhn_test": int(counts["svhn_test"]),
    }
    arrays = {}
    hashes = {}

    for method, settings in config["scores"].items():
        arrays[method] = {}
        hashes[method] = {}
        folder = resolve_path(settings["directory"])
        for split, count in expected.items():
            source = folder / f"{method}_{split}_scores.npy"
            actual_hash = sha256_file(source)
            if actual_hash != settings["sha256"][split]:
                raise RuntimeError(f"STOP: SHA-256 mismatch: {source}")

            values = np.load(source, allow_pickle=False)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid score array: {source}")
            arrays[method][split] = values
            hashes[method][split] = actual_hash

    results = {}
    for method in ("energy", "knn"):
        comparison = compare_auroc_paired(
            arrays["qe"]["mnist_val"],
            arrays["qe"]["svhn_test"],
            arrays[method]["mnist_val"],
            arrays[method]["svhn_test"],
            iterations=int(config["bootstrap_iterations"]),
            seed=int(config["seed"]),
        )
        results[method] = {
            "qe_auroc": comparison["qe_auroc"],
            "baseline_auroc": comparison["msp_auroc"],
            "qe_minus_baseline_auroc": comparison["qe_minus_msp_auroc"],
            "paired_difference_ci_95": comparison["difference_ci_95"],
        }

    result = {
        "run_id": run_id,
        "comparisons": results,
        "seed": int(config["seed"]),
        "bootstrap_iterations": int(config["bootstrap_iterations"]),
        "sample_counts": expected,
        "input_sha256": hashes,
        "config_sha256": sha256_file(config_path),
        "leakage_control": (
            "Frozen historical novelty scores; no fitting, tuning, "
            "or official MNIST final-test access."
        ),
        "alignment_limitation": (
            "Historical score arrays lack independent per-image identifiers. "
            "Paired analysis assumes the established identical sample order."
        ),
    }

    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "paired_comparison_metrics.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write(chr(10))

    LOGGER.info("Saved %s", output)
    return output


def main() -> None:
    """Run configured Phase 3 paired comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
