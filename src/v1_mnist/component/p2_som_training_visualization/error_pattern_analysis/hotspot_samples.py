"""Representative images for the SOM error hotspots (error pattern analysis).

The images are found by their stable sample IDs (the original training images first, then the
original test images) in the dataset named in the config, through the same loader as Phases 2 to 4
(``utils/frozen_split.py``). Titles use the class names in the config (dataset.class_names), or
the class numbers if it names none.

Development validation samples only.

Reads the error pattern analysis table for the SOM named in a config (default som.yaml) and
writes images to <output_root>/p2_som_training_visualization/error_pattern_analysis/<selected_model>/
representative_samples/.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/hotspot_samples.py \
      [--config CONFIG] [--data-root PATH]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.dataset_info import DatasetInfo, dataset_info
from src.v1_mnist.component.utils.frozen_split import load_combined_images
from src.v1_mnist.component.utils.image_datasets import dataset_spec
from src.v1_mnist.component.utils.run_settings import (
    DEFAULT_CONFIG,
    error_output_dir,
)
from src.v1_mnist.component.utils.figures import save_pdf
from src.v1_mnist.component.utils.logging import get_logger


# Set in main() from the config; module-level so the helpers below keep working.
TABLE_PATH: Path = Path()
OUTPUT_DIR: Path = Path()
DATA_ROOT: Path = Path()
INFO: DatasetInfo | None = None

MAX_PER_HOTSPOT = 6

logger = get_logger("v1_mnist.analysis.error_patterns.hotspot_samples")


def load_rows() -> list[dict]:
    with TABLE_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["true_class"] = int(row["true_class"])
        row["predicted_class"] = int(row["predicted_class"])
        row["bmu"] = int(row["bmu"])
        row["confidence"] = float(row["confidence"])
        row["bmu_distance"] = float(row["bmu_distance"])
        row["hotspot"] = row["hotspot"].lower() == "true"

    return rows


def get_image(sample_id: int, images: np.ndarray) -> np.ndarray:
    """The image with this sample ID, as height x width (x 3 for colour)."""
    if not 0 <= sample_id < len(images):
        raise ValueError(f"Invalid sample ID: {sample_id} (the dataset has {len(images)} images).")
    image = images[sample_id]
    return image if image.ndim == 2 else np.transpose(image, (1, 2, 0))


def verify_label(row: dict, labels: np.ndarray) -> None:
    sid = row["sample_id"]
    source_label = int(labels[sid])
    if source_label != row["true_class"]:
        raise RuntimeError(
            f"Sample-ID reconstruction mismatch for {sid}: "
            f"source label={source_label}, "
            f"stored label={row['true_class']}"
        )


def name(class_id: int) -> str:
    return INFO.class_name(class_id) if INFO is not None else str(class_id)


def plot_hotspot(neuron: int, rows: list[dict], images: np.ndarray, labels: np.ndarray) -> None:
    rows = sorted(
        rows,
        key=lambda r: r["bmu_distance"],
        reverse=True,
    )

    selected = rows[:MAX_PER_HOTSPOT]

    n = len(selected)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(3.3 * cols, 3.5 * rows_n),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    # The grid pads its last row with empty panels, so there can be more axes than samples.
    for ax, row in zip(axes, selected, strict=False):
        verify_label(row, labels)

        image = get_image(row["sample_id"], images)

        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"ID {row['sample_id']}\n"
            f"True {name(row['true_class'])} → Pred {name(row['predicted_class'])}\n"
            f"Conf {row['confidence']:.3f} | "
            f"BMU dist {row['bmu_distance']:.3f}",
            fontsize=9,
        )

    fig.suptitle(
        f"SOM Error Hotspot — Neuron {neuron}",
        fontsize=14,
    )

    fig.tight_layout()

    stem = OUTPUT_DIR / f"hotspot_neuron_{neuron:03d}"

    save_pdf(fig, stem.with_suffix(".pdf"))


def plot_overall(rows: list[dict], images: np.ndarray, labels: np.ndarray) -> None:
    """Plot one representative error from each detected hotspot."""

    by_neuron: dict[int, list[dict]] = {}

    for row in rows:
        if row["hotspot"]:
            by_neuron.setdefault(row["bmu"], []).append(row)

    representatives = []

    for _neuron, neuron_rows in sorted(by_neuron.items()):
        # Select highest BMU-distance error as a representative difficult case.
        representative = max(
            neuron_rows,
            key=lambda r: r["bmu_distance"],
        )
        representatives.append(representative)

    n = len(representatives)

    if n == 0:
        return

    cols = 3
    rows_n = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows_n,
        cols,
        figsize=(10, 3.4 * rows_n),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    # The grid pads its last row with empty panels, so there can be more axes than samples.
    for ax, row in zip(axes, representatives, strict=False):
        verify_label(row, labels)

        image = get_image(row["sample_id"], images)

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        ax.set_title(
            f"Neuron {row['bmu']}\n"
            f"True {name(row['true_class'])} → Pred {name(row['predicted_class'])}\n"
            f"Conf {row['confidence']:.3f}",
            fontsize=10,
        )

    fig.suptitle(
        "Representative Misclassification from Each SOM Error Hotspot",
        fontsize=14,
    )

    fig.tight_layout()

    stem = OUTPUT_DIR / "hotspot_representatives"

    save_pdf(fig, stem.with_suffix(".pdf"))


def main() -> None:
    global TABLE_PATH, OUTPUT_DIR, DATA_ROOT, INFO

    parser = argparse.ArgumentParser(description="Representative images for the error hotspots.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    parser.add_argument("--data-root", default=None,
                        help="Folder holding the torchvision download of the dataset "
                             "(default: paths.data_root from the config).")
    args = parser.parse_args()

    run_dir = error_output_dir(args.config, REPO_ROOT)
    TABLE_PATH = run_dir / "tables" / "val_misclassified_samples.csv"
    OUTPUT_DIR = run_dir / "representative_samples"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(str(config_path))
    INFO = dataset_info(config)
    # Same folder baseline training downloaded the dataset into (base.yaml paths.data_root).
    DATA_ROOT = Path(args.data_root) if args.data_root is not None else Path(config.paths.data_root)
    if not DATA_ROOT.is_absolute():
        DATA_ROOT = (REPO_ROOT / DATA_ROOT).resolve()
    if not TABLE_PATH.is_file():
        raise FileNotFoundError(f"{TABLE_PATH} not found: run error_patterns.py with the same config first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    hotspot_rows = [row for row in rows if row["hotspot"]]

    if not hotspot_rows:
        raise RuntimeError("No hotspot validation errors were found.")

    logger.info("Loading %s for sample-ID reconstruction from %s", INFO.display_name, DATA_ROOT)
    images, labels = load_combined_images(DATA_ROOT, dataset_spec(config))

    # Verify every hotspot error before producing any figure.
    for row in hotspot_rows:
        verify_label(row, labels)

    neurons = sorted({row["bmu"] for row in hotspot_rows})

    logger.info("Verified hotspot error samples: %d", len(hotspot_rows))
    logger.info("Hotspot neurons: %s", neurons)

    for neuron in neurons:
        neuron_rows = [
            row
            for row in hotspot_rows
            if row["bmu"] == neuron
        ]

        plot_hotspot(
            neuron,
            neuron_rows,
            images,
            labels,
        )

    plot_overall(hotspot_rows, images, labels)

    logger.info("Representative-image generation COMPLETE.")
    logger.info("Hotspot neurons represented: %d", len(neurons))
    logger.info("Hotspot error samples available: %d", len(hotspot_rows))
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
