"""Recompute validation-only QE enrichment against uniform random selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import hypergeom

from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file
from src.v1_mnist.component.utils.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[5]
LOGGER = get_logger("v1_mnist.p3.qe_random_control")




def source_path(value: str) -> Path:
    """Resolve a configured relative path inside the repository."""
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def compute_control(
    distances: np.ndarray,
    correct: np.ndarray,
    threshold: float,
    seeds: list[int],
) -> dict[str, Any]:
    """Recompute enrichment and seeded random controls without test data."""
    qe = np.asarray(distances, dtype=np.float64)
    correctness = np.asarray(correct)

    if qe.ndim != 1 or qe.size == 0 or not np.isfinite(qe).all():
        raise ValueError("QE must be a finite nonempty 1D array")
    if correctness.shape != qe.shape or correctness.dtype != np.dtype("bool"):
        raise ValueError("CNN correctness must be aligned boolean data")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid frozen threshold")
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("At least five unique random seeds are required")

    flagged = qe > threshold
    errors = ~correctness

    n = int(qe.size)
    selected = int(np.count_nonzero(flagged))
    total_errors = int(np.count_nonzero(errors))
    captured = int(np.count_nonzero(flagged & errors))

    if selected == 0 or total_errors == 0:
        raise ValueError("Cannot evaluate enrichment without flags and errors")

    random_counts: list[int] = []
    for seed in seeds:
        indices = np.random.default_rng(seed).choice(
            n, size=selected, replace=False
        )
        random_counts.append(int(np.count_nonzero(errors[indices])))

    low, high = wilson_interval(captured, total_errors)

    return {
        "evaluation": "cnn_error_enrichment_vs_uniform_random_selection",
        "validation_samples": n,
        "qe_flagged_samples": selected,
        "cnn_errors": total_errors,
        "qe_captured_errors": captured,
        "qe_error_capture_rate": captured / total_errors,
        "wilson_95_interval": [low, high],
        "random_expected_errors": selected * total_errors / n,
        "random_exact_one_sided_p_value": float(
            hypergeom.sf(captured - 1, n, total_errors, selected)
        ),
        "random_seeded_control": {
            "seeds": seeds,
            "captured_errors": random_counts,
            "mean": float(np.mean(random_counts)),
            "sampling": "numpy.default_rng(seed).choice without replacement",
            "historical_counts_not_assumed_reproduced": True,
        },
        "leakage_control": {
            "threshold_source": "train_only",
            "final_test_used": False,
            "external_ood_evaluated": False,
        },
        "limitations": [
            "Uniform random selection is a control, not an OOD detector.",
            "The selected sample count is held equal to QE's flag count.",
            "No final MNIST test samples or external OOD samples were used.",
        ],
    }


def run(config_path: Path, *, save: bool = False) -> dict[str, Any]:
    """Verify frozen inputs and optionally create a new non-overwriting run."""
    config = load_config(config_path)
    run_id = str(config.run_id)

    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run ID")

    destination = source_path(str(config.output_root)) / run_id
    if destination.exists():
        raise FileExistsError(f"Run already exists: {destination}")

    summary_path = source_path(str(config.source_summary))
    distance_path = source_path(str(config.validation_distances))
    correct_path = source_path(str(config.validation_correct))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_hashes = summary["input_sha256"]

    hashes = {
        "val_bmu_distance": sha256_file(distance_path),
        "val_correct": sha256_file(correct_path),
    }
    for key, observed in hashes.items():
        if observed != expected_hashes[key]:
            raise RuntimeError(f"Frozen input SHA-256 mismatch: {key}")

    qe = np.load(distance_path, allow_pickle=False)
    correct = np.load(correct_path, allow_pickle=False)

    result = compute_control(
        qe,
        correct,
        float(summary["threshold"]),
        [int(seed) for seed in config.random_control.seeds],
    )

    association = summary["error_association"]
    checks = {
        "validation_samples": int(summary["validation_count"]),
        "qe_flagged_samples": int(association["flagged_samples"]),
        "cnn_errors": int(association["cnn_errors"]),
        "qe_captured_errors": int(association["flagged_and_error"]),
    }
    for key, expected in checks.items():
        if result[key] != expected:
            raise RuntimeError(f"Frozen summary disagreement: {key}")

    result.update({
        "run_id": run_id,
        "source_summary_run_id": str(summary["run_id"]),
        "input_sha256": hashes,
        "source_summary_sha256": sha256_file(summary_path),
        "config_sha256": sha256_file(config_path),
    })

    if save:
        destination.mkdir(parents=True, exist_ok=False)
        path = destination / "qe_random_control_metrics.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        LOGGER.info("Created %s", path)

    return result


def main() -> None:
    """Run a read-only check by default; save only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "src/v1_mnist/component/configs/qe_random_control.yaml"
        ),
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    result = run(args.config, save=args.save)
    LOGGER.info("Random control: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
