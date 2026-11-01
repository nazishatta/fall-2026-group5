"""Reproduce MNIST validation U-Matrix statistics using the frozen SOM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.u_matrix_novelty import (
    neuron_u_matrix,
    sample_u_matrix,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file
from src.v1_mnist.component.utils.stats import finite_spearman

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("v1_mnist.p3.qe_u_matrix")




def resolve_input(path_string: str) -> Path:
    """Resolve a repository-relative input path."""
    path = (ROOT / path_string).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {path_string}")
    return path


def evaluate_validation(
    weights: np.ndarray,
    topology: np.ndarray,
    bmu: np.ndarray,
    qe: np.ndarray,
) -> dict[str, Any]:
    """Calculate original Phase 3 U-Matrix validation statistics."""
    bmu_values = np.asarray(bmu)
    qe_values = np.asarray(qe, dtype=np.float64)

    if qe_values.ndim != 1 or qe_values.size == 0:
        raise ValueError("Validation QE must be nonempty 1D")
    if not np.isfinite(qe_values).all() or (qe_values < 0).any():
        raise ValueError("Validation QE contains invalid distances")
    if bmu_values.shape != qe_values.shape:
        raise ValueError("BMU and QE arrays must be aligned")

    neuron_scores = neuron_u_matrix(weights, topology)
    sample_scores = sample_u_matrix(neuron_scores, bmu_values)
    correlation = finite_spearman(qe_values, sample_scores)

    if correlation is None:
        raise ValueError("Spearman correlation is undefined")

    return {
        "analysis": "qe_u_matrix_validation_comparison",
        "dataset": "MNIST",
        "split": "validation",
        "validation_samples": int(qe_values.size),
        "som_neurons": len(neuron_scores),
        "qe_mean": float(np.mean(qe_values)),
        "u_matrix_mean": float(np.mean(sample_scores)),
        "spearman_correlation": correlation,
        "ood_evaluation": "not_performed",
        "qe_reconstruction_difference": "not_measured",
        "leakage_control": (
            "MNIST validation assignments and distances only; "
            "no final test or external OOD inputs."
        ),
    }


def run(config_path: Path, *, save: bool = False) -> dict[str, Any]:
    """Check fingerprints, load the frozen SOM, and compute the metrics."""
    config = load_config(config_path)
    run_id = str(config.run_id)

    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run ID")

    output_root = resolve_input(str(config.output_root))
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Existing run must not be overwritten: {destination}")

    model_path = resolve_input(str(config.som_model))
    bmu_path = resolve_input(str(config.validation_bmu))
    qe_path = resolve_input(str(config.validation_qe))
    summary_path = resolve_input(str(config.qe_summary))

    hashes = {
        "som_model": sha256_file(model_path),
        "validation_bmu": sha256_file(bmu_path),
        "validation_qe": sha256_file(qe_path),
        "qe_summary": sha256_file(summary_path),
    }

    if hashes["som_model"] != str(config.expected_sha256.som_model):
        raise RuntimeError("Selected SOM model SHA-256 mismatch")
    if hashes["validation_qe"] != str(config.expected_sha256.validation_qe):
        raise RuntimeError("Frozen validation QE SHA-256 mismatch")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if hashes["validation_qe"] != summary["input_sha256"]["val_bmu_distance"]:
        raise RuntimeError("QE input disagrees with frozen QE summary")

    bmu = np.load(bmu_path, allow_pickle=False)
    qe = np.load(qe_path, allow_pickle=False)

    if bmu.shape != (int(config.validation_count),):
        raise ValueError("Unexpected validation BMU count")
    if qe.shape != bmu.shape:
        raise ValueError("Validation BMU/QE count mismatch")
    if len(qe) != int(summary["validation_count"]):
        raise RuntimeError("Validation count disagrees with QE summary")

    # Import only when loading the actual model; NumPy-only tests need no NNSOM.
    try:
        from src.v1_mnist.component.utils.som_io import load_selected_som
    except ModuleNotFoundError as exc:
        if exc.name == "NNSOM":
            raise RuntimeError(
                "NNSOM is required to load the frozen SOM; "
                "run the full reproduction in the pinned project environment."
            ) from exc
        raise

    som = load_selected_som(
        model_path,
        int(config.grid_height),
        int(config.grid_width),
        int(config.feature_dim),
    )

    result = evaluate_validation(
        np.asarray(som.w),
        np.asarray(som.neuron_dist),
        bmu,
        qe,
    )

    if result["som_neurons"] != int(config.grid_height) * int(config.grid_width):
        raise RuntimeError("Unexpected SOM neuron count")

    result.update(
        {
            "run_id": run_id,
            "som_run": model_path.name,
            "qe_threshold": float(summary["threshold"]),
            "source_summary_run_id": str(summary["run_id"]),
            "input_sha256": hashes,
            "config_sha256": sha256_file(config_path),
            "historical_result_overwritten": False,
        }
    )

    if save:
        destination.mkdir(parents=True, exist_ok=False)
        output_path = destination / "qe_u_matrix_metrics.json"
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        LOGGER.info("Saved %s", output_path)

    return result


def main() -> None:
    """Compute read-only by default; --save creates a distinct run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "src/v1_mnist/component/configs/qe_u_matrix_reproduction.yaml"
        ),
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    LOGGER.info(
        "U-Matrix validation: %s",
        json.dumps(run(args.config, save=args.save), indent=2),
    )


if __name__ == "__main__":
    main()
