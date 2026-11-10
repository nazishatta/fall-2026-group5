"""Report the QE novelty score at several percentile cutoffs, as a new run.

Uses only saved score arrays (no model, no SVHN images), so it needs no PyTorch.
It refuses to reuse a run ID and never edits earlier outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_metrics import (
    ood_metrics,
    stratified_auroc_interval,
)
from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.qe_threshold_sweep import (
    check_percentiles,
    permutation_auroc_null,
    threshold_sweep,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.repo_paths import resolve_inside
from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/qe_threshold_sweep.yaml"
LOGGER = get_logger(__name__)
HEADLINE_PERCENTILE = 95.0


def repo_path(value: str) -> Path:
    """Resolve only safe, repository-relative paths."""
    return resolve_inside(ROOT, value)


def read_config(config_path: Path) -> dict[str, Any]:
    """Load the config and reject unsafe run IDs or percentile lists."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ValueError("Unsafe run ID")
    percentiles = check_percentiles(config["percentiles"])
    if HEADLINE_PERCENTILE not in percentiles:
        raise ValueError("The 95th percentile must stay in the list so the frozen cutoff is checked")
    return config


def load_scores(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load the saved QE arrays and check their sizes."""
    expected = config["expected"]
    paths = {key: repo_path(value) for key, value in config["inputs"].items() if key != "qe_summary"}
    scores = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
    sizes = {
        "train_qe": int(expected["training_samples"]),
        "validation_qe": int(expected["validation_samples"]),
        "svhn_qe": int(expected["svhn_samples"]),
    }
    for key, size in sizes.items():
        if scores[key].shape != (size,):
            raise RuntimeError(f"{key} has shape {scores[key].shape}, expected ({size},)")
    return scores


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the sweep table as CSV. Never overwrites."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    """Compute the sweep under a new run ID; refuse to reuse an existing one."""
    config = read_config(config_path)
    destination = repo_path(config["output_root"]) / config["run_id"]
    if destination.exists():
        raise FileExistsError(f"STOP: Existing run: {destination}")
    scores = load_scores(config)
    rows = threshold_sweep(scores["train_qe"], scores["validation_qe"], scores["svhn_qe"], config["percentiles"])
    headline = next(row for row in rows if row["percentile"] == HEADLINE_PERCENTILE)
    frozen = float(config["expected"]["frozen_percentile_95_threshold"])
    summary_path = repo_path(config["inputs"]["qe_summary"])
    saved = float(json.loads(summary_path.read_text(encoding="utf-8"))["threshold"])
    if headline["threshold"] != frozen or saved != frozen:
        raise RuntimeError("The 95th-percentile cutoff does not match the frozen value")
    point = ood_metrics(scores["validation_qe"], scores["svhn_qe"], frozen, float(config["target_tpr"]))
    boot = config["bootstrap"]
    interval = stratified_auroc_interval(
        scores["validation_qe"], scores["svhn_qe"], iterations=int(boot["iterations"]),
        seed=int(boot["seed"]), confidence=float(boot["confidence"]),
    )
    null = permutation_auroc_null(
        scores["validation_qe"], scores["svhn_qe"],
        iterations=int(config["permutation_null"]["iterations"]), seed=int(config["permutation_null"]["seed"]),
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "tables").mkdir()
    write_table(destination / "tables" / "qe_threshold_sweep.csv", rows)
    result = {
        "run_id": config["run_id"],
        "analysis": "qe_threshold_sweep",
        "headline_percentile": HEADLINE_PERCENTILE,
        "threshold_sweep": rows,
        "threshold_free_metrics": {
            "auroc": point["auroc"],
            "auroc_bootstrap_interval": list(interval),
            "bootstrap_confidence": float(boot["confidence"]),
            "aupr_ood": point["aupr_ood"],
            "fpr_at_target_tpr": point["fpr_at_target_tpr"],
            "target_tpr": point["target_tpr"],
        },
        "label_permutation_null": null,
        "note": (
            "AUROC and AUPR do not depend on the percentile cutoff; only the rates in "
            "threshold_sweep do. The 95th percentile remains the headline cutoff, and "
            "no percentile is chosen from SVHN results."
        ),
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "input_sha256": {key: sha256_file(repo_path(path)) for key, path in config["inputs"].items()},
        },
        "leakage_control": (
            "Cutoffs come from MNIST training QE only. MNIST validation is the in-distribution "
            "set and SVHN test is the novel set, both used for evaluation only. SVHN was "
            "examined in earlier runs, so this is a post-hoc sensitivity analysis. "
            "The final MNIST test split is never opened."
        ),
    }
    with (destination / "qe_threshold_sweep.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    LOGGER.info("AUROC %.4f; wrote %s", point["auroc"], destination)
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
