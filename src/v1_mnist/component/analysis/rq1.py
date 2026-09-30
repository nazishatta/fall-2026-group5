from __future__ import annotations

import json
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional
import numpy as np

from src.v1_mnist.component.analysis.io import (
    FEATURE_DIM,
    GRID_HEIGHT,
    GRID_WIDTH,
    NUM_NEURONS,
    SPLIT_FILES,
    json_float,
    load_selected_som,
    load_split,
    reconstruct_assignments,
    sha256_file,
    som_coordinates,
    write_rows,
    write_sample_assignments,
)
from src.v1_mnist.component.analysis.metrics import (
    CLASS_COUNT,
    finite_spearman,
    normalized_entropy,
    safe_mean,
    safe_median,
    wilson_lower_bound,
)
from src.v1_mnist.component.analysis.plotting import (
    matrix_from_neuron_values,
    save_heatmap,
)
from src.v1_mnist.component.som.data import resolve_embeddings_dir
from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.analysis.rq1")


def build_neuron_rows(
    split: str,
    arrays: dict[str, np.ndarray],
    bmu: np.ndarray,
    distance: np.ndarray,
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    """Build RQ1 and validation RQ2 statistics per neuron.

    Supports arbitrary numbers of neurons based on coordinates.

    Args:
        split: 'train' or 'val'.
        arrays: Dictionary of embeddings metadata.
        bmu: Per-sample BMU assignments.
        distance: Per-sample BMU distances.
        coordinates: Shape (2, num_neurons) neuron positions.

    Returns:
        List of dictionaries with per-neuron metrics.
    """
    rows: list[dict[str, Any]] = []
    num_neurons = coordinates.shape[1]

    for neuron_id in range(num_neurons):
        idx = np.flatnonzero(bmu == neuron_id)
        support = int(idx.size)

        counts = np.bincount(
            arrays["labels"][idx],
            minlength=CLASS_COUNT,
        ).astype(np.int64)

        if support > 0:
            dominant_class: Optional[int] = int(np.argmax(counts))
            purity = float(counts[dominant_class] / support)
            entropy = normalized_entropy(counts)
            mean_distance = safe_mean(distance[idx])
        else:
            dominant_class = None
            purity = float("nan")
            entropy = float("nan")
            mean_distance = float("nan")

        row: dict[str, Any] = {
            "split": split,
            "neuron_id": neuron_id,
            "pos_0": float(coordinates[0, neuron_id]),
            "pos_1": float(coordinates[1, neuron_id]),
            "support": support,
            "dominant_class": dominant_class,
            "purity": purity,
            "normalized_entropy": entropy,
            "mean_bmu_distance": mean_distance,
        }

        for class_id in range(CLASS_COUNT):
            row[f"class_{class_id}_count"] = int(counts[class_id])

        if split == "val":
            correct = arrays["correct"][idx]
            error_mask = ~correct
            correct_count = int(correct.sum())
            error_count = support - correct_count
            error_rate = error_count / support if support > 0 else float("nan")

            row.update(
                {
                    "correct_count": correct_count,
                    "error_count": error_count,
                    "error_rate": error_rate,
                    "error_rate_wilson_lower_95": wilson_lower_bound(
                        error_count,
                        support,
                    ),
                    "mean_confidence": safe_mean(arrays["confidence"][idx]),
                    "mean_confidence_correct": safe_mean(
                        arrays["confidence"][idx][correct]
                    ),
                    "mean_confidence_error": safe_mean(
                        arrays["confidence"][idx][error_mask]
                    ),
                    "mean_bmu_distance_correct": safe_mean(
                        distance[idx][correct]
                    ),
                    "mean_bmu_distance_error": safe_mean(
                        distance[idx][error_mask]
                    ),
                }
            )

        rows.append(row)

    return rows


def summarize_rq1(
    rows: list[dict[str, Any]],
    num_neurons: Optional[int] = None,
) -> dict[str, Any]:
    """Return sample-weighted representation purity and entropy summaries.

    Args:
        rows: List of per-neuron rows from build_neuron_rows.
        num_neurons: Optional total neuron count (defaults to len(rows)).

    Returns:
        Dictionary summarizing occupied neurons, occupancy rate, mean purity, and mean entropy.
    """
    total_neurons = num_neurons if num_neurons is not None else len(rows)
    occupied = [row for row in rows if row["support"] > 0]

    if not occupied:
        return {
            "occupied_neurons": 0,
            "total_neurons": total_neurons,
            "occupancy_rate": 0.0,
            "sample_weighted_mean_purity": float("nan"),
            "sample_weighted_mean_normalized_entropy": float("nan"),
        }

    support = np.asarray([row["support"] for row in occupied], dtype=np.float64)
    purity = np.asarray([row["purity"] for row in occupied], dtype=np.float64)
    entropy = np.asarray(
        [row["normalized_entropy"] for row in occupied], dtype=np.float64
    )

    return {
        "occupied_neurons": len(occupied),
        "total_neurons": total_neurons,
        "occupancy_rate": len(occupied) / total_neurons,
        "sample_weighted_mean_purity": float(
            np.average(purity, weights=support)
        ),
        "sample_weighted_mean_normalized_entropy": float(
            np.average(entropy, weights=support)
        ),
    }


def run_rq1_analysis(
    output_dir: Optional[Path | str] = None,
    overwrite: bool = False,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute the full frozen development-stage RQ1/RQ2 analysis pipeline.

    Args:
        output_dir: Destination directory for analysis output.
        overwrite: Whether to overwrite existing analysis results.
        repo_root: Optional project repository root path.

    Returns:
        Dictionary summary of the RQ1/RQ2 results.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    run_id = "som_15x15_seed42_rq1_rq2_v1"
    embeddings_dir = resolve_embeddings_dir(
        "outputs/v1_mnist/cnn_baseline/embeddings/mnist",
        repo_root,
    )
    model_path = (
        repo_root
        / "outputs/v1_mnist/som/som_models"
        / "som_15x15_seed42_final"
    )
    frozen_metrics_path = (
        repo_root
        / "outputs/v1_mnist/som/logs"
        / "som_15x15_seed42_final_metrics.json"
    )

    if output_dir is None:
        output_base = repo_root / "outputs/v1_mnist/som/analysis"
        target_output_dir = output_base / run_id
    else:
        target_output_dir = Path(output_dir)
        output_base = target_output_dir.parent

    temp_dir = output_base / f".{target_output_dir.name}.tmp"

    if target_output_dir.exists():
        if overwrite:
            logger.info("Overwriting existing directory: %s", target_output_dir)
            shutil.rmtree(target_output_dir)
        else:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {target_output_dir}. "
                f"Specify overwrite=True or use a different output directory."
            )

    if temp_dir.exists():
        logger.warning("Cleaning up stale temp directory: %s", temp_dir)
        shutil.rmtree(temp_dir)

    output_base.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    assignments_dir = temp_dir / "assignments"
    tables_dir = temp_dir / "tables"
    figures_dir = temp_dir / "figures"
    metadata_dir = temp_dir / "metadata"

    for d in (assignments_dir, tables_dir, figures_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading SOM model from %s", model_path)
        model_hash = sha256_file(model_path)
        som = load_selected_som(model_path)
        coordinates = som_coordinates(som)

        frozen = json.loads(frozen_metrics_path.read_text(encoding="utf-8"))
        if frozen.get("test_evaluated") is not False:
            raise RuntimeError(
                "Frozen selected metrics indicate test evaluation was performed."
            )

        input_hashes: dict[str, str] = {
            "selected_model": model_hash,
            "frozen_metrics": sha256_file(frozen_metrics_path),
        }

        manifest_path = embeddings_dir / "embedding_manifest.json"
        if manifest_path.is_file():
            input_hashes["embedding_manifest"] = sha256_file(manifest_path)

        split_results: dict[str, Any] = {}

        for split in ("train", "val"):
            logger.info("Analyzing %s split...", split)
            arrays = load_split(split, embeddings_dir=embeddings_dir)

            for suffix in sorted(SPLIT_FILES):
                input_hashes[f"{split}_{suffix}"] = sha256_file(
                    embeddings_dir / f"{split}_{suffix}.npy"
                )

            (
                clusters,
                cluster_distances,
                _max_distances,
                cluster_sizes,
            ) = som.cluster_data(arrays["features"])

            sizes = np.asarray(cluster_sizes, dtype=np.int64)
            if int(sizes.sum()) != len(arrays["features"]):
                raise RuntimeError(
                    f"{split}: cluster-size sum ({int(sizes.sum())}) != sample count ({len(arrays['features'])})."
                )

            bmu, distance = reconstruct_assignments(
                clusters,
                cluster_distances,
                len(arrays["features"]),
            )

            qe = float(som.quantization_error(cluster_distances))
            occupied = int(np.count_nonzero(sizes))

            frozen_metrics = frozen[
                "train_metrics" if split == "train" else "validation_metrics"
            ]
            expected_qe = float(frozen_metrics["quantization_error"])
            expected_occupied = int(frozen_metrics["occupied_neurons"])

            if not np.isclose(qe, expected_qe, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{split}: QE regression mismatch. Expected {expected_qe}, observed {qe}"
                )

            if occupied != expected_occupied:
                raise RuntimeError(
                    f"{split}: occupancy regression mismatch. Expected {expected_occupied}, observed {occupied}"
                )

            write_sample_assignments(
                assignments_dir / f"{split}_bmu_assignments.csv",
                split,
                arrays,
                bmu,
                distance,
                coordinates,
            )

            neuron_rows = build_neuron_rows(
                split,
                arrays,
                bmu,
                distance,
                coordinates,
            )

            write_rows(
                tables_dir / f"{split}_neurons.csv",
                neuron_rows,
            )

            split_results[split] = {
                "arrays": arrays,
                "bmu": bmu,
                "distance": distance,
                "neuron_rows": neuron_rows,
                "qe": qe,
                "occupied": occupied,
            }

            logger.info("PASS: %s assignments = %d", split, len(bmu))
            logger.info("PASS: %s QE regression = %.12f", split, qe)
            logger.info("PASS: %s occupied neurons = %d", split, occupied)

        train_rows = split_results["train"]["neuron_rows"]
        val_rows = split_results["val"]["neuron_rows"]

        # RQ1 figures (saved as SVG and PDF)
        train_support = np.asarray(
            [r["support"] for r in train_rows], dtype=np.float64
        )
        train_dominant = np.asarray(
            [
                np.nan if r["dominant_class"] is None else r["dominant_class"]
                for r in train_rows
            ],
            dtype=np.float64,
        )
        val_purity = np.asarray(
            [r["purity"] for r in val_rows], dtype=np.float64
        )
        val_entropy = np.asarray(
            [r["normalized_entropy"] for r in val_rows], dtype=np.float64
        )

        save_heatmap(
            figures_dir / "rq1_train_hit_count",
            matrix_from_neuron_values(coordinates, np.log1p(train_support)),
            "RQ1 — Training SOM Hit Map (log1p count)",
            "log1p(sample hits)",
        )
        save_heatmap(
            figures_dir / "rq1_train_dominant_class",
            matrix_from_neuron_values(coordinates, train_dominant),
            "RQ1 — Dominant Ground-Truth Class per Training Neuron",
            "MNIST class",
            categorical=True,
        )
        save_heatmap(
            figures_dir / "rq1_validation_purity",
            matrix_from_neuron_values(coordinates, val_purity),
            "RQ1 — Validation Class Purity",
            "purity",
        )
        save_heatmap(
            figures_dir / "rq1_validation_entropy",
            matrix_from_neuron_values(coordinates, val_entropy),
            "RQ1 — Validation Normalized Class Entropy",
            "normalized entropy",
        )

        # RQ2 tables + figures
        occupied_val_rows = [r for r in val_rows if r["support"] > 0]
        hotspot_rows = sorted(
            occupied_val_rows,
            key=lambda r: (
                -r["error_rate_wilson_lower_95"],
                -r["error_count"],
                -r["support"],
                r["neuron_id"],
            ),
        )

        top_hotspots = hotspot_rows[:20]
        write_rows(tables_dir / "validation_hotspot_top20.csv", top_hotspots)

        error_count = np.asarray(
            [r["error_count"] for r in val_rows], dtype=np.float64
        )
        error_rate = np.asarray(
            [r["error_rate"] for r in val_rows], dtype=np.float64
        )
        wilson_lower = np.asarray(
            [r["error_rate_wilson_lower_95"] for r in val_rows], dtype=np.float64
        )
        mean_confidence = np.asarray(
            [r["mean_confidence"] for r in val_rows], dtype=np.float64
        )

        save_heatmap(
            figures_dir / "rq2_validation_error_count",
            matrix_from_neuron_values(coordinates, error_count),
            "RQ2 — Validation Error Count by SOM Neuron",
            "error count",
        )
        save_heatmap(
            figures_dir / "rq2_validation_error_rate",
            matrix_from_neuron_values(coordinates, error_rate),
            "RQ2 — Validation Error Rate by SOM Neuron",
            "error rate",
        )
        save_heatmap(
            figures_dir / "rq2_validation_wilson_lower95",
            matrix_from_neuron_values(coordinates, wilson_lower),
            "RQ2 — Support-Aware Validation Error Priority",
            "Wilson 95% lower bound of error rate",
        )
        save_heatmap(
            figures_dir / "rq2_validation_mean_confidence",
            matrix_from_neuron_values(coordinates, mean_confidence),
            "RQ2 — Mean Baseline Confidence by Validation Neuron",
            "mean confidence",
        )

        # Overall summaries
        val_arrays = split_results["val"]["arrays"]
        val_distance = split_results["val"]["distance"]
        val_correct = val_arrays["correct"]

        total_errors = int((~val_correct).sum())
        overall_error_rate = total_errors / len(val_correct)

        by_error_count = sorted(
            occupied_val_rows,
            key=lambda r: (
                -r["error_count"],
                -r["error_rate"],
                -r["support"],
                r["neuron_id"],
            ),
        )

        top10_error_count = sum(
            int(r["error_count"]) for r in by_error_count[:10]
        )
        top10_error_fraction = (
            top10_error_count / total_errors if total_errors > 0 else 0.0
        )

        val_error_rates = np.asarray(
            [r["error_rate"] for r in occupied_val_rows], dtype=np.float64
        )
        val_entropies = np.asarray(
            [r["normalized_entropy"] for r in occupied_val_rows],
            dtype=np.float64,
        )
        val_purities = np.asarray(
            [r["purity"] for r in occupied_val_rows], dtype=np.float64
        )
        val_mean_distances = np.asarray(
            [r["mean_bmu_distance"] for r in occupied_val_rows],
            dtype=np.float64,
        )

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
            ).strip()
        except Exception:
            git_commit = "unknown"

        summary = {
            "run_id": run_id,
            "stage": "development",
            "som_grid": [GRID_HEIGHT, GRID_WIDTH],
            "seed": 42,
            "selected_model": str(model_path.relative_to(repo_root)),
            "selected_model_sha256": model_hash,
            "git_commit": git_commit,
            "nnsom_version": version("NNSOM"),
            "test_evaluated": False,
            "inputs": input_hashes,
            "regression_checks": {
                "train_quantization_error": split_results["train"]["qe"],
                "validation_quantization_error": split_results["val"]["qe"],
                "train_occupied_neurons": split_results["train"]["occupied"],
                "validation_occupied_neurons": split_results["val"]["occupied"],
                "status": "PASS",
            },
            "rq1": {
                "train": summarize_rq1(train_rows),
                "validation": summarize_rq1(val_rows),
            },
            "rq2": {
                "validation_samples": int(len(val_correct)),
                "validation_errors": total_errors,
                "overall_validation_error_rate": float(overall_error_rate),
                "top10_error_count": top10_error_count,
                "fraction_of_validation_errors_in_top10_error_count_neurons": float(
                    top10_error_fraction
                ),
                "mean_bmu_distance_correct": json_float(
                    safe_mean(val_distance[val_correct])
                ),
                "mean_bmu_distance_error": json_float(
                    safe_mean(val_distance[~val_correct])
                ),
                "median_bmu_distance_correct": json_float(
                    safe_median(val_distance[val_correct])
                ),
                "median_bmu_distance_error": json_float(
                    safe_median(val_distance[~val_correct])
                ),
                "mean_confidence_correct": json_float(
                    safe_mean(val_arrays["confidence"][val_correct])
                ),
                "mean_confidence_error": json_float(
                    safe_mean(val_arrays["confidence"][~val_correct])
                ),
                "exploratory_spearman": {
                    "error_rate_vs_normalized_entropy": finite_spearman(
                        val_error_rates,
                        val_entropies,
                    ),
                    "error_rate_vs_purity": finite_spearman(
                        val_error_rates,
                        val_purities,
                    ),
                    "error_rate_vs_mean_bmu_distance": finite_spearman(
                        val_error_rates,
                        val_mean_distances,
                    ),
                },
                "hotspot_ranking": (
                    "Wilson 95% lower bound descending; "
                    "error count descending; "
                    "support descending; "
                    "neuron index ascending."
                ),
            },
        }

        (metadata_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        (metadata_dir / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifact_hashes: dict[str, str] = {}
        for p in sorted(temp_dir.rglob("*")):
            if p.is_file():
                artifact_hashes[str(p.relative_to(temp_dir))] = sha256_file(p)

        (metadata_dir / "artifact_hashes.json").write_text(
            json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Atomic promotion
        temp_dir.rename(target_output_dir)

        logger.info("=" * 72)
        logger.info("RQ1/RQ2 ANALYSIS COMPLETE")
        logger.info("=" * 72)
        logger.info("Output: %s", target_output_dir)
        logger.info("Test evaluated: NO")
        logger.info("Regression checks: PASS")
        logger.info(
            "Validation errors: %d/%d (%.6f)",
            total_errors,
            len(val_correct),
            overall_error_rate,
        )
        logger.info(
            "Top-10 error-count neurons contain: %d/%d validation errors (%.4f)",
            top10_error_count,
            total_errors,
            top10_error_fraction,
        )

        return summary

    except Exception as exc:
        logger.error("ANALYSIS FAILED: %s", exc)
        logger.info("Temporary artifacts preserved for inspection: %s", temp_dir)
        raise
