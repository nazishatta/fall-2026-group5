"""Week 5: Error Pattern Analysis.

Uses the Week 4 BMU assignments of the validation images (run
run_cluster_analysis.py with the same config first) and the CNN's predictions
to find where on the map the CNN makes its mistakes:
  - per neuron: error count, error rate, 95% Wilson lower bound of the error
    rate, and mean CNN confidence (all images, correct ones, wrong ones);
  - the error hotspots, by the project's one hotspot rule (utils/hotspots.py;
    docs/p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md):
    the top k = max(5, 5% of neurons) occupied neurons by Wilson lower bound;
  - true -> predicted confusion pairs, overall and inside the hotspots;
  - the list of misclassified validation images (read by hotspot_samples.py).

The hotspot counts here are descriptive: the same images are used to choose
the hotspots and to count their errors, which overstates the concentration.
The tested, held-out result is written by error_pattern_robustness.py.
Distances to the BMU (wrong vs correct images) belong to Week 6 (novelty
detection) and are not compared here.

Validation split only; the test split is never used.

Output: outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/<selected_model>/
    error_summary.json
    tables/val_error_geography.csv, val_hotspots.csv, val_confusion_pairs.csv,
           val_misclassified_samples.csv
    metadata/input_hashes.json, metadata/artifact_hashes.json
Heatmaps: error_heatmaps.py. Error maps: error_figures.py. Hotspot images: hotspot_samples.py.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/error_patterns.py \\
      [--config CONFIG]
Results are never overwritten: if the output folder exists the script stops;
move or delete that folder to redo the analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.som_io import load_split, sha256_file, write_rows  # noqa: E402
from src.v1_mnist.component.utils.heatmaps import read_neuron_table  # noqa: E402
from src.v1_mnist.component.utils.hotspots import (  # noqa: E402
    n_hotspots,
    top_hotspots,
)
from src.v1_mnist.component.utils.stats import safe_mean, wilson_lower_bound_array  # noqa: E402
from src.v1_mnist.component.utils.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    analysis_config,
    cluster_output_dir,
    error_output_dir,
    load_analysis_settings,
    verify_selected_model,
)
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402

logger = get_logger("v1_mnist.analysis.error_patterns.error_patterns")

HOTSPOT_RULE_DOC = (
    "src/v1_mnist/docs/p2_som_training_visualization/error_pattern_analysis/HOTSPOT_RULE.md"
)


def build_error_rows(
    correct: np.ndarray,
    confidence: np.ndarray,
    bmu: np.ndarray,
    coordinates: np.ndarray,
    num_neurons: int,
    hotspot_rank: dict[int, int],
) -> list[dict]:
    """Error-geography statistics for every SOM neuron, in neuron order."""
    error = ~correct
    support = np.bincount(bmu, minlength=num_neurons)
    errors = np.bincount(bmu, weights=error.astype(float), minlength=num_neurons)
    wilson = wilson_lower_bound_array(errors, support)
    overall_error_rate = float(np.mean(error))

    rows = []
    for neuron_id in range(num_neurons):
        idx = np.flatnonzero(bmu == neuron_id)
        n = int(support[neuron_id])
        local_correct = correct[idx]
        error_count = int(round(errors[neuron_id]))
        error_rate = float(error_count / n) if n else float("nan")
        rows.append(
            {
                "neuron_id": neuron_id,
                "pos_0": float(coordinates[0, neuron_id]),
                "pos_1": float(coordinates[1, neuron_id]),
                "support": n,
                "correct_count": n - error_count,
                "error_count": error_count,
                "error_rate": error_rate,
                "wilson_lower_95": float(wilson[neuron_id]) if n else float("nan"),
                "error_rate_ratio_vs_overall": (
                    error_rate / overall_error_rate
                    if n and overall_error_rate > 0
                    else float("nan")
                ),
                "mean_confidence": safe_mean(confidence[idx]),
                "mean_confidence_correct": safe_mean(confidence[idx][local_correct]),
                "mean_confidence_error": safe_mean(confidence[idx][~local_correct]),
                "hotspot": neuron_id in hotspot_rank,
                "hotspot_rank": hotspot_rank.get(neuron_id, ""),
            }
        )
    return rows


def confusion_pairs(
    labels: np.ndarray,
    preds: np.ndarray,
    correct: np.ndarray,
    bmu: np.ndarray,
    hotspot_neurons: set[int],
) -> list[dict]:
    """Summarize true→predicted confusion pairs, overall and inside hotspot neurons."""
    error_idx = np.flatnonzero(~correct)
    counts: dict[tuple[int, int], int] = {}
    hotspot_counts: dict[tuple[int, int], int] = {}
    for i in error_idx:
        pair = (int(labels[i]), int(preds[i]))
        counts[pair] = counts.get(pair, 0) + 1
        if int(bmu[i]) in hotspot_neurons:
            hotspot_counts[pair] = hotspot_counts.get(pair, 0) + 1
    rows = []
    for pair, count in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        rows.append(
            {
                "true_class": pair[0],
                "predicted_class": pair[1],
                "validation_error_count": count,
                "hotspot_error_count": hotspot_counts.get(pair, 0),
            }
        )
    return rows


def write_metadata(folder: Path, input_hashes: dict[str, str]) -> None:
    """metadata/input_hashes.json and metadata/artifact_hashes.json (every other file in folder)."""
    metadata_dir = folder / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    artifact_hashes = {
        p.relative_to(folder).as_posix(): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    (metadata_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 SOM error pattern analysis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the grid and selected model.")
    args = parser.parse_args()

    settings = load_analysis_settings(args.config, REPO_ROOT)
    model_hash = verify_selected_model(settings)

    cluster_dir = cluster_output_dir(settings.config_path, REPO_ROOT)  # Week 4 inputs (read only)
    output_dir = error_output_dir(settings.config_path, REPO_ROOT)     # Week 5 outputs
    analysis = analysis_config(settings.config_path, REPO_ROOT)
    hotspot_fraction, min_hotspots = float(analysis.hotspot_fraction), int(analysis.min_hotspots)

    # Week 4 must exist, come from the same model, and its files must be unchanged.
    summary_path = cluster_dir / "cluster_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{summary_path} not found: run run_cluster_analysis.py with the same config first."
        )
    cluster_info = json.loads(summary_path.read_text(encoding="utf-8"))
    if cluster_info.get("model_sha256") != model_hash:
        raise RuntimeError(
            f"Week 4 outputs in {cluster_dir} were made from a different model "
            f"({cluster_info.get('model_sha256')}) than {settings.selected_model} ({model_hash})."
        )
    cluster_hashes = json.loads((cluster_dir / "metadata" / "artifact_hashes.json").read_text(encoding="utf-8"))
    cluster_inputs = ["assignments/val_bmu.npy", "assignments/val_bmu_distance.npy", "tables/val_cluster_metrics.csv"]
    input_hashes: dict[str, str] = {"selected_model": model_hash, "cluster_summary": sha256_file(summary_path)}
    for rel in cluster_inputs:
        observed = sha256_file(cluster_dir / rel)
        if cluster_hashes.get(rel) != observed:
            raise RuntimeError(f"Week 4 file {rel} changed since Week 4 wrote it.")
        input_hashes[f"cluster_analysis/{rel}"] = observed

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Results are never overwritten; "
            "move or delete that folder to redo the analysis."
        )
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if temp_dir.exists():
        logger.warning("Removing stale temporary folder %s", temp_dir)
        shutil.rmtree(temp_dir)
    tables_dir = temp_dir / "tables"
    tables_dir.mkdir(parents=True)

    try:
        arrays = load_split("val", embeddings_dir=settings.embeddings_dir, feature_dim=settings.feature_dim)
        for name in ("labels", "preds", "correct", "confidence", "sample_ids"):
            input_hashes[f"val_{name}"] = sha256_file(settings.embeddings_dir / f"val_{name}.npy")
        bmu = np.load(cluster_dir / "assignments" / "val_bmu.npy")
        distance = np.load(cluster_dir / "assignments" / "val_bmu_distance.npy")
        coordinates, _ = read_neuron_table(cluster_dir / "tables" / "val_cluster_metrics.csv")

        n = len(arrays["labels"])
        num_neurons = settings.num_neurons
        if bmu.shape != (n,) or distance.shape != (n,) or coordinates.shape != (2, num_neurons):
            raise RuntimeError("Week 4 validation files do not match the validation data or the grid.")

        correct = arrays["correct"].astype(bool)
        error = ~correct

        # The one hotspot rule (same code the robustness test uses).
        k = n_hotspots(num_neurons, hotspot_fraction, min_hotspots)
        hotspots = [int(x) for x in top_hotspots(bmu, error.astype(float), num_neurons, k)]
        hotspot_rank = {neuron: rank for rank, neuron in enumerate(hotspots, start=1)}

        rows = build_error_rows(correct, arrays["confidence"], bmu, coordinates, num_neurons, hotspot_rank)
        write_rows(tables_dir / "val_error_geography.csv", rows)
        write_rows(tables_dir / "val_hotspots.csv", [rows[neuron] for neuron in hotspots])

        pair_rows = confusion_pairs(arrays["labels"], arrays["preds"], correct, bmu, set(hotspots))
        if pair_rows:
            write_rows(tables_dir / "val_confusion_pairs.csv", pair_rows)

        # Individual misclassified validation images, for hotspot_samples.py.
        sample_rows = [
            {
                "split_index": int(i),
                "sample_id": int(arrays["sample_ids"][i]),
                "true_class": int(arrays["labels"][i]),
                "predicted_class": int(arrays["preds"][i]),
                "confidence": float(arrays["confidence"][i]),
                "bmu": int(bmu[i]),
                "bmu_distance": float(distance[i]),
                "hotspot": int(bmu[i]) in hotspot_rank,
                "hotspot_rank": hotspot_rank.get(int(bmu[i]), ""),
            }
            for i in np.flatnonzero(error)
        ]
        if sample_rows:
            write_rows(tables_dir / "val_misclassified_samples.csv", sample_rows)

        total_errors = int(error.sum())
        in_hot = np.isin(bmu, hotspots)
        hot_errors = int(error[in_hot].sum())
        hot_images = int(in_hot.sum())
        summary = {
            "analysis": "error_pattern_analysis",
            "som": settings.selected_model,
            "model_sha256": model_hash,
            "som_grid": [settings.grid_height, settings.grid_width],
            "config": settings.config_path.name,
            "development_split": "val",
            "test_used": False,
            "validation_samples": n,
            "validation_errors": total_errors,
            "validation_error_rate": float(total_errors / n),
            "mean_confidence_correct": safe_mean(arrays["confidence"][correct]),
            "mean_confidence_error": safe_mean(arrays["confidence"][error]),
            "hotspot_rule": {
                "rule": "top k occupied neurons by 95% Wilson lower bound of the error rate; "
                        "ties: more errors, then more images, then lower neuron index",
                "k": k,
                "hotspot_fraction": hotspot_fraction,
                "min_hotspots": min_hotspots,
                "document": HOTSPOT_RULE_DOC,
            },
            "hotspot_neurons_in_rank_order": hotspots,
            "hotspots_descriptive": {
                "note": "Same validation images used to choose the hotspots and to count "
                        "their errors, so this overstates the concentration. The tested, "
                        "held-out result is hotspot_concentration_held_out in "
                        "robustness/<model>_error_pattern_robustness.json.",
                "images_in_hotspots": hot_images,
                "errors_in_hotspots": hot_errors,
                "share_of_images": float(hot_images / n),
                "share_of_errors": float(hot_errors / total_errors) if total_errors else float("nan"),
            },
        }
        (temp_dir / "error_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_metadata(temp_dir, input_hashes)

        temp_dir.rename(output_dir)

    except Exception as exc:
        logger.error("Week 5 error pattern analysis FAILED: %s", exc)
        logger.info("Temporary files kept for inspection: %s", temp_dir)
        raise

    logger.info("Week 5 error pattern analysis COMPLETE")
    logger.info("Validation samples: %d, errors: %d", n, total_errors)
    logger.info("Hotspots (k=%d, rank order): %s", k, hotspots)
    logger.info("Errors inside hotspots (descriptive): %d of %d", hot_errors, total_errors)
    logger.info("Results: %s", output_dir)
    logger.info("Held-out test split was NOT used.")


if __name__ == "__main__":
    main()
