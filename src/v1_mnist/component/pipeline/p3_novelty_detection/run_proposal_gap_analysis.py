"""Run a new Phase 3 *descriptive* analysis without modifying historical outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.novelty_overlay import (
    save_novelty_overlay,
    summarize_qe_misses,
)

ROOT = Path(__file__).resolve().parents[5]


def resolve_repo_path(value: str) -> Path:
    """Resolve a repository-relative path and reject parent traversal."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes project root: {value}")
    return path


def sha256_file(path: Path) -> str:
    """Read-only provenance hash for this standalone diagnostic."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path) -> dict[str, Any]:
    """Create fresh output with no test-set access, refits, or overwrites."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = str(config["run_id"])
    if not run_id or run_id in {".", ".."} or any(c in run_id for c in "/\\"):
        raise ValueError("Invalid run ID")
    inputs = {key: resolve_repo_path(value) for key, value in config["inputs"].items()}
    destination = resolve_repo_path(str(config["output_root"])) / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run: {destination}")
    summary = json.loads(inputs["qe_summary"].read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    bmu = np.load(inputs["validation_bmu"], allow_pickle=False)
    val_qe = np.load(inputs["validation_qe"], allow_pickle=False)
    svhn_qe = np.load(inputs["svhn_qe"], allow_pickle=False)
    grid = config["som_grid"]
    if bmu.size != int(config["expected_validation_count"]) or svhn_qe.size != int(config["expected_svhn_count"]):
        raise ValueError("Unexpected sample counts")
    if (
        "input_sha256" in summary
        and "val_bmu_distance" in summary["input_sha256"]
        and sha256_file(inputs["validation_qe"]) != summary["input_sha256"]["val_bmu_distance"]
    ):
        raise ValueError("QE array does not match frozen QE summary")
    # Preflight all outputs before creating the run directory.
    root = destination / "pdf"
    if root.exists():
        raise FileExistsError(root)
    destination.mkdir(parents=True, exist_ok=False)
    overlay = save_novelty_overlay(bmu, val_qe, int(grid["height"]), int(grid["width"]), destination / "pdf" / "mnist_validation_qe_som_overlay.pdf")
    failures = summarize_qe_misses(svhn_qe, threshold)
    result = {
        "run_id": run_id,
        "analysis": "proposal_gap_descriptive_review",
        "overlay": overlay,
        "svhn_qe_failures": failures,
        "qe_threshold": threshold,
        "provenance": {"config_sha256": sha256_file(config_path), "input_sha256": {key: sha256_file(value) for key, value in inputs.items()}},
        "leakage_control": "MNIST validation used only for visualization; SVHN scores used for descriptive failure analysis only. No hyperparameter tuning, threshold changes, or final MNIST test use.",
        "limitation": "The saved SVHN sample-to-BMU assignments were not provided. This analysis cannot establish why individual images were missed or produce an SVHN miss-rate SOM overlay.",
    }
    with (destination / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    """CLI for the new standalone Phase 3 proposal-gap analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
