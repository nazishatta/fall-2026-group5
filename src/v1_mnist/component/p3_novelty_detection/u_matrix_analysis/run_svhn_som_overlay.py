"""Save SVHN-to-SOM assignments and describe where missed novel images land.

New run only: it never touches earlier outputs and refuses to reuse a run ID.
The heavy part (CNN + SOM scoring of SVHN) needs PyTorch, NNSOM and the SVHN
test files, so it is imported inside ``compute_svhn_assignments``. The rest is
plain numpy and is covered by unit tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from src.v1_mnist.component.p3_novelty_detection.u_matrix_analysis.svhn_som_overlay import (
    class_landing_profile,
    miss_typicality,
    neuron_dominant_class,
    save_svhn_overlay,
    svhn_neuron_summary,
    top_missed_neurons,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/svhn_som_overlay.yaml"
LOGGER = get_logger(__name__)


def repo_path(value: str) -> Path:
    """Resolve only safe, repository-relative paths."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative path: {value}")
    return ROOT / relative


def read_config(config_path: Path) -> dict[str, Any]:
    """Load the config and reject unsafe run IDs."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ValueError("Unsafe run ID")
    if int(config["min_validation_hits"]) < 1 or int(config["top_missed_neurons"]) < 1:
        raise ValueError("Counts in the config must be positive")
    return config


def load_reference(config: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """Load the MNIST reference arrays and check they have the expected shape."""
    expected = config["expected"]
    paths = {key: repo_path(value) for key, value in config["inputs"].items()}
    counts = np.loadtxt(paths["train_class_neuron"], delimiter=",")
    shape = (int(config["class_count"]), int(expected["som_units"]))
    if counts.shape != shape:
        raise RuntimeError(f"Class-neuron table has shape {counts.shape}, expected {shape}")
    val_bmu = np.load(paths["validation_bmu"], allow_pickle=False)
    val_qe = np.load(paths["validation_qe"], allow_pickle=False)
    if len(val_bmu) != int(expected["validation_samples"]) or val_qe.shape != val_bmu.shape:
        raise RuntimeError("MNIST validation assignments have an unexpected size")
    return {"class_counts": counts, "validation_bmu": val_bmu, "validation_qe": val_qe}


def compute_svhn_assignments(config: dict[str, Any]) -> dict[str, Any]:
    """Score SVHN test with the frozen CNN and SOM (needs torch, nnsom and the SVHN files)."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import SVHN

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.run_svhn_benchmark import (
        score_som,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.run_qe_ood import (
        prepare,
        read_settings,
    )
    from src.v1_mnist.component.utils.shared_provenance import (
        load_manifest,
        verify_shared_inputs,
    )
    from src.v1_mnist.component.utils.som_io import som_coordinates

    expected = config["expected"]
    manifest = load_manifest()
    shared = dict(manifest["inputs"])
    shared["svhn_file"] = manifest["svhn_file"]
    verify_shared_inputs(shared, required_keys=tuple(shared))
    settings = read_settings(repo_path(config["qe_config"]))
    model, som, _validation_qe, provenance = prepare(settings)
    if tuple(som.w.shape) != (int(expected["som_units"]), int(expected["embedding_dimensions"])):
        raise RuntimeError("Frozen SOM shape mismatch")
    if provenance["qe_threshold"] != float(expected["frozen_training_qe_threshold"]):
        raise RuntimeError("Frozen QE threshold mismatch")
    svhn_path = repo_path(manifest["svhn_file"])
    dataset = SVHN(
        root=str(svhn_path.parent), split="test", download=False,
        transform=image_transform("svhn", tuple(settings["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False)
    torch.set_num_threads(1)
    features, _, _ = FeatureExtractor(model, "fc2").extract(loader, torch.device("cpu"))
    if features.shape != (int(expected["svhn_samples"]), int(expected["embedding_dimensions"])):
        raise RuntimeError("SVHN embedding shape mismatch")
    bmu, qe = score_som(som, features)
    return {
        "bmu": np.asarray(bmu, dtype=np.int64),
        "qe": np.asarray(qe, dtype=np.float64),
        "centers": np.asarray(som_coordinates(som).T, dtype=np.float64),
        "threshold": float(provenance["qe_threshold"]),
        "hashes": {
            **{key: sha256_file(repo_path(path)) for key, path in shared.items()},
            "som_model": provenance["som_sha256"],
        },
    }


def check_against_history(config: dict[str, Any], qe: NDArray[np.floating]) -> float:
    """Return the largest QE gap to the saved historical SVHN scores; fail if too large."""
    historical = np.load(repo_path(config["inputs"]["historical_svhn_qe"]), allow_pickle=False)
    if historical.shape != qe.shape:
        raise RuntimeError("Historical SVHN QE has a different length")
    gap = float(np.max(np.abs(historical - qe)))
    if gap > float(config["max_abs_qe_difference"]):
        raise RuntimeError(f"Reconstructed SVHN QE differs from history by {gap}")
    return gap


def analyze(
    config: dict[str, Any],
    bmu: NDArray[np.integer],
    qe: NDArray[np.floating],
    threshold: float,
    reference: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    """Compute all descriptive tables from the SVHN assignments."""
    units = int(config["expected"]["som_units"])
    classes = int(config["class_count"])
    dominant, purity = neuron_dominant_class(reference["class_counts"])
    summary = svhn_neuron_summary(bmu, qe, threshold, units)
    return {
        "dominant": dominant,
        "purity": purity,
        "summary": summary,
        "profile": class_landing_profile(bmu, qe, threshold, dominant, classes),
        "typicality": miss_typicality(
            bmu, qe, threshold, reference["validation_bmu"], reference["validation_qe"],
            units, int(config["min_validation_hits"]),
        ),
        "top_neurons": top_missed_neurons(summary, dominant, purity, int(config["top_missed_neurons"])),
    }


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write a small CSV; NaN and None become empty cells. Never overwrites."""
    def cell(value: Any) -> Any:
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else value

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows([[cell(value) for value in row] for row in rows])


def write_run(
    destination: Path,
    config: dict[str, Any],
    config_sha256: str,
    scores: dict[str, Any],
    reference: dict[str, NDArray[Any]],
    history_gap: float,
) -> dict[str, Any]:
    """Write every output of one run into a brand-new folder."""
    bmu, qe, threshold = scores["bmu"], scores["qe"], float(scores["threshold"])
    if len(bmu) != int(config["expected"]["svhn_samples"]):
        raise RuntimeError("Unexpected number of SVHN assignments")
    analysis = analyze(config, bmu, qe, threshold, reference)
    summary = analysis["summary"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("assignments", "tables", "pdf"):
        (destination / name).mkdir()
    np.save(destination / "assignments" / "svhn_bmu.npy", bmu, allow_pickle=False)
    np.save(destination / "assignments" / "svhn_qe.npy", qe, allow_pickle=False)
    write_csv(
        destination / "tables" / "svhn_neuron_summary.csv",
        ["neuron", "svhn_images", "missed", "miss_rate", "dominant_mnist_class", "class_purity"],
        [
            [n, int(summary["hits"][n]), int(summary["missed"][n]), float(summary["miss_rate"][n]),
             int(analysis["dominant"][n]), float(analysis["purity"][n])]
            for n in range(len(summary["hits"]))
        ],
    )
    profile = analysis["profile"]
    write_csv(
        destination / "tables" / "svhn_class_landing_profile.csv",
        list(profile[0]),
        [list(row.values()) for row in profile],
    )
    figure = save_svhn_overlay(
        scores["centers"], summary, analysis["dominant"], int(config["class_count"]),
        threshold, float(config["hex_radius"]), destination / "pdf" / "svhn_som_overlay.pdf",
    )
    missed = int(summary["missed"].sum())
    result = {
        "run_id": config["run_id"],
        "analysis": "svhn_som_overlay_and_miss_profile",
        "qe_threshold": threshold,
        "svhn_samples": len(bmu),
        "detected": int(len(bmu) - missed),
        "missed": missed,
        "miss_rate": missed / len(bmu),
        "svhn_occupied_neurons": int((summary["hits"] > 0).sum()),
        "neurons_holding_misses": int((summary["missed"] > 0).sum()),
        "class_landing_profile": profile,
        "missed_typicality": analysis["typicality"],
        "top_missed_neurons": analysis["top_neurons"],
        "figure": str(figure.relative_to(destination)),
        "history_check": {"max_abs_qe_difference": history_gap, "limit": float(config["max_abs_qe_difference"])},
        "provenance": {
            "config_sha256": config_sha256,
            "input_sha256": {key: sha256_file(repo_path(path)) for key, path in config["inputs"].items()},
            "frozen_input_sha256": scores["hashes"],
        },
        "leakage_control": (
            "SVHN test is the novel set and is used only for description. "
            "MNIST validation and the training class table are references. "
            "The QE cutoff comes from MNIST training and is not changed. "
            "Nothing is tuned and the final MNIST test split is never opened."
        ),
        "limitation": (
            "Descriptive only. Where a missed image lands and how typical its QE is "
            "does not prove why the SOM missed it."
        ),
    }
    with (destination / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def run(config_path: Path) -> dict[str, Any]:
    """Run the full analysis under a new run ID; refuse to reuse an existing one."""
    config = read_config(config_path)
    destination = repo_path(config["output_root"]) / config["run_id"]
    if destination.exists():
        raise FileExistsError(f"STOP: Existing run: {destination}")
    reference = load_reference(config)
    scores = compute_svhn_assignments(config)
    gap = check_against_history(config, scores["qe"])
    LOGGER.info("SVHN QE matches history within %.2e", gap)
    result = write_run(destination, config, sha256_file(config_path), scores, reference, gap)
    LOGGER.info("Missed %d of %d SVHN images", result["missed"], result["svhn_samples"])
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
