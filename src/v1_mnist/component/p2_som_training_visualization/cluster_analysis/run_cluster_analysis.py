"""Week 4: Cluster Analysis.

Assigns every train and validation image to its best-matching neuron (BMU) on
the selected SOM and measures the class structure of the map:
  - per neuron: class counts (NNSOM cal_class_cluster_intersect), dominant
    class, purity, normalized class entropy, and the neuron's position on the map;
  - the class-by-neuron intersection table;
  - per split: neurons used, sample-weighted purity and entropy;
  - validation only: a label-shuffle comparison for the weighted purity (labels
    shuffled, each image keeps its BMU; number of shuffles and seed from the
    config's analysis section). Added 2026-10-10, see docs/DEVIATIONS.md.
Each image's BMU and its distance to the BMU are saved for Week 5 (error
patterns) and Week 6 (novelty detection).

Safety checks, all before anything is written:
  - the model's SHA-256 must match run_manifest.csv (utils/run_settings.py);
  - the embeddings must match the hashes in the config (expected_inputs);
  - the model's metrics file must say the test split was never evaluated;
  - model check: the SOM must reproduce its Week 3 number of used neurons
    exactly, and its Week 3 quantization error to within one part in a million
    (analysis.qe_relative_tolerance in the config), which allows for rounding
    differences between library versions.
    The quantization error itself is a Week 3 result; only the check is
    reported here (both values and their relative difference).
Results are built in a temporary folder and moved into place only when every
step succeeded. The test split is never used.

Output: outputs/v1_mnist/p2_som_training_visualization/cluster_analysis/<selected_model>/
    cluster_summary.json
    tables/{train,val}_cluster_metrics.csv, tables/{train,val}_class_neuron_intersection.csv
    assignments/{train,val}_bmu.npy, {train,val}_bmu_distance.npy, {train,val}_bmu_assignments.csv
    metadata/input_hashes.json, metadata/artifact_hashes.json
Heatmaps: cluster_heatmaps.py. Class maps: cluster_figures.py.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/run_cluster_analysis.py \\
      [--config CONFIG]
Results are never overwritten: if the output folder exists the script stops;
move or delete that folder to redo the analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.som_io import (  # noqa: E402
    SPLIT_FILES,
    load_selected_som,
    load_split,
    reconstruct_assignments,
    sha256_file,
    som_coordinates,
    write_sample_assignments,
)
from src.v1_mnist.component.p2_som_training_visualization.cluster_analysis.cluster_metrics import (  # noqa: E402
    ClusterMetrics,
    compute_cluster_metrics,
    purity_label_shuffle,
    weighted_cluster_purity,
    weighted_mean_entropy,
)
from src.v1_mnist.component.utils.run_settings import (  # noqa: E402
    DEFAULT_CONFIG,
    AnalysisSettings,
    analysis_config,
    cluster_output_dir,
    load_analysis_settings,
    verify_inputs,
    verify_selected_model,
)
from src.v1_mnist.component.utils.config import load_config  # noqa: E402
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402


logger = get_logger("v1_mnist.analysis.cluster_analysis.run_cluster_analysis")

# Model check tolerance for the quantization error (config: analysis.qe_relative_tolerance,
# one part in a million). Different library versions (e.g. scikit-learn 1.9.0 on EC2 vs
# 1.7.2 on the laptop) round the same arithmetic differently: the 15x15 training QE
# differs by about 2 parts in a billion. A different model or different data would change
# the QE by far more; the model and embedding fingerprints are checked separately anyway.
# (Until 2026-10-10 this was an absolute 1e-12, which only EC2 could meet.)


def check_quantization_error(split: str, recomputed: float, training_run: float, rtol: float) -> dict:
    """Model check: the recomputed QE must match Week 3's to within rtol (relative).

    Returns the numbers for the summary; raises RuntimeError if they differ by more.
    """
    relative_difference = abs(recomputed - training_run) / abs(training_run) if training_run else abs(recomputed)
    if not relative_difference <= rtol:          # also fails for NaN
        raise RuntimeError(
            f"{split}: model check failed: quantization error {recomputed} differs from the "
            f"value recorded by the SOM training run (Week 3), {training_run}, by a relative {relative_difference:.3g} "
            f"(allowed: {rtol:g})."
        )
    return {
        "training_run": training_run,
        "recomputed": recomputed,
        "relative_difference": relative_difference,
        "relative_tolerance": rtol,
    }


def write_cluster_table(
    path: Path, metrics: list[ClusterMetrics], coordinates: np.ndarray, num_classes: int,
) -> None:
    """One row per neuron, in neuron order, with the neuron's position on the map."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "neuron_id",
        "pos_0",
        "pos_1",
        "support",
        "dominant_class",
        "dominant_count",
        "purity",
        "normalized_entropy",
    ] + [f"class_{i}_count" for i in range(num_classes)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for metric in metrics:
            row = {
                "neuron_id": metric.neuron_id,
                "pos_0": float(coordinates[0, metric.neuron_id]),
                "pos_1": float(coordinates[1, metric.neuron_id]),
                "support": metric.support,
                "dominant_class": metric.dominant_class,
                "dominant_count": metric.dominant_count,
                "purity": metric.purity,
                "normalized_entropy": metric.normalized_entropy,
            }

            for class_id, count in enumerate(metric.class_counts):
                row[f"class_{class_id}_count"] = count

            writer.writerow(row)


def purity_null_settings(config_path: Path) -> tuple[int, int]:
    """Number of label shuffles and their seed (config: analysis.purity_null_*)."""
    analysis = load_config(str(config_path)).analysis
    try:
        return int(analysis.purity_null_shuffles), int(analysis.purity_null_seed)
    except AttributeError as exc:
        raise KeyError(
            f"{config_path.name}: set analysis.purity_null_shuffles and analysis.purity_null_seed."
        ) from exc


def analyze_split(
    split: str,
    som: SOMModel,
    settings: AnalysisSettings,
    training_metrics: dict,
    coordinates: np.ndarray,
    output_dir: Path,
    input_hashes: dict[str, str],
) -> dict:
    logger.info("Analyzing %s...", split)

    arrays = load_split(split, embeddings_dir=settings.embeddings_dir, feature_dim=settings.feature_dim)
    for suffix in sorted(SPLIT_FILES):
        input_hashes[f"{split}_{suffix}"] = sha256_file(
            settings.embeddings_dir / f"{split}_{suffix}.npy"
        )
    verify_inputs(settings, input_hashes)

    clusters, cluster_distances, _, cluster_sizes = som.cluster_data(
        arrays["features"]
    )

    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    num_samples = len(arrays["features"])
    if int(sizes.sum()) != num_samples:
        raise RuntimeError(
            f"{split}: cluster-size sum ({int(sizes.sum())}) != sample count ({num_samples})."
        )

    bmu, distances = reconstruct_assignments(
        clusters,
        cluster_distances,
        num_samples,
    )

    # Model check: the same SOM on the same embeddings must give its Week 3 numbers.
    qe = float(som.quantization_error(cluster_distances))
    occupied = int(np.count_nonzero(sizes))
    expected = training_metrics["train_metrics" if split == "train" else "validation_metrics"]
    rtol = float(analysis_config(settings.config_path, REPO_ROOT).qe_relative_tolerance)
    qe_check = check_quantization_error(split, qe, float(expected["quantization_error"]), rtol)
    if occupied != int(expected["occupied_neurons"]):
        raise RuntimeError(
            f"{split}: model check failed: {occupied} neurons used, Week 3 recorded "
            f"{int(expected['occupied_neurons'])}."
        )

    metrics = compute_cluster_metrics(
        arrays["labels"],
        bmu,
        num_neurons=settings.num_neurons,
        num_classes=settings.num_classes,
    )
    intersection = np.asarray([m.class_counts for m in metrics], dtype=np.int64).T

    write_cluster_table(
        output_dir / "tables" / f"{split}_cluster_metrics.csv",
        metrics,
        coordinates,
        settings.num_classes,
    )
    np.savetxt(
        output_dir / "tables" / f"{split}_class_neuron_intersection.csv",
        intersection,
        delimiter=",",
        fmt="%d",
    )

    assignments_dir = output_dir / "assignments"
    np.save(assignments_dir / f"{split}_bmu.npy", bmu)
    np.save(assignments_dir / f"{split}_bmu_distance.npy", distances)
    write_sample_assignments(
        assignments_dir / f"{split}_bmu_assignments.csv",
        split,
        arrays,
        bmu,
        distances,
        coordinates,
    )

    summary = {
        "samples": num_samples,
        "occupied_neurons": occupied,
        "total_neurons": settings.num_neurons,
        "occupancy_rate": float(occupied / settings.num_neurons),
        "sample_weighted_purity": weighted_cluster_purity(metrics),
        "sample_weighted_mean_normalized_entropy": weighted_mean_entropy(metrics),
        "model_check": {
            "quantization_error": qe_check,
            "occupied_neurons": {"training_run": int(expected["occupied_neurons"]), "recomputed": occupied},
        },
    }

    logger.info("Samples:              %d", summary["samples"])
    logger.info("Occupied neurons:     %d/%d", occupied, settings.num_neurons)
    logger.info("Weighted purity:      %.6f", summary["sample_weighted_purity"])
    logger.info("Weighted mean entropy: %.6f", summary["sample_weighted_mean_normalized_entropy"])

    if split == "val":
        shuffles, seed = purity_null_settings(settings.config_path)
        null = purity_label_shuffle(
            arrays["labels"], bmu, settings.num_neurons, settings.num_classes,
            shuffles=shuffles, seed=seed,
        )
        if not np.isclose(null["observed_purity"], summary["sample_weighted_purity"], rtol=0, atol=1e-12):
            raise RuntimeError("Label-shuffle purity does not match the NNSOM-based purity.")
        summary["purity_label_shuffle"] = null
        logger.info(
            "Purity vs label shuffle: observed %.4f, shuffled mean %.4f (95%% range %.4f-%.4f), p = %.4g",
            null["observed_purity"], null["null_mean"], *null["null_95_range"], null["p_value"],
        )

    return summary


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


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
    parser = argparse.ArgumentParser(description="Week 4 SOM cluster analysis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the grid and selected model.")
    args = parser.parse_args()

    settings = load_analysis_settings(args.config, REPO_ROOT)
    # Check the model before touching any output folder.
    model_hash = verify_selected_model(settings)
    logger.info("Model integrity check passed: %s (SHA-256 %s)", settings.selected_model, model_hash)

    training_metrics = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
    if training_metrics.get("test_evaluated") is not False:
        raise RuntimeError("The model's metrics file says the test split was evaluated.")

    output_dir = cluster_output_dir(settings.config_path, REPO_ROOT)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Results are never overwritten; "
            "move or delete that folder to redo the analysis."
        )

    temp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if temp_dir.exists():
        logger.warning("Removing stale temporary folder %s", temp_dir)
        shutil.rmtree(temp_dir)
    for sub in ("tables", "assignments", "metadata"):
        (temp_dir / sub).mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading frozen SOM...")
        som = load_selected_som(
            settings.model_path, settings.grid_height, settings.grid_width, settings.feature_dim,
        )
        coordinates = som_coordinates(som)

        input_hashes: dict[str, str] = {
            "selected_model": model_hash,
            "som_training_metrics": sha256_file(settings.metrics_path),
        }
        embedding_manifest = settings.embeddings_dir / "embedding_manifest.json"
        if embedding_manifest.is_file():
            input_hashes["embedding_manifest"] = sha256_file(embedding_manifest)

        splits = {
            split: analyze_split(split, som, settings, training_metrics, coordinates, temp_dir, input_hashes)
            for split in ("train", "val")
        }

        qe_rtol = float(analysis_config(settings.config_path, REPO_ROOT).qe_relative_tolerance)
        summary = {
            "analysis": "cluster_analysis",
            "som": settings.selected_model,
            "model_sha256": model_hash,
            "som_grid": [settings.grid_height, settings.grid_width],
            "config": settings.config_path.name,
            "development_only": True,
            "test_used": False,
            "nnsom_version": version("NNSOM"),
            "library_versions": {name: version(name) for name in ("NNSOM", "numpy", "scikit-learn")},
            "git_commit": git_commit(REPO_ROOT),
            "class_counts_from": "NNSOM.utils.cal_class_cluster_intersect",
            "model_check": {
                "embeddings_match_config": "PASS",
                "quantization_error_matches_training_run": f"PASS (relative tolerance {qe_rtol:g}; values per split)",
                "occupied_neurons_match_training_run": "PASS",
            },
            "splits": splits,
        }
        (temp_dir / "cluster_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        write_metadata(temp_dir, input_hashes)

        temp_dir.rename(output_dir)

    except Exception as exc:
        logger.error("Week 4 cluster analysis FAILED: %s", exc)
        logger.info("Temporary files kept for inspection: %s", temp_dir)
        raise

    logger.info("Week 4 cluster analysis COMPLETE.")
    logger.info("Results: %s", output_dir)
    logger.info("Held-out test split was NOT used.")


if __name__ == "__main__":
    main()
