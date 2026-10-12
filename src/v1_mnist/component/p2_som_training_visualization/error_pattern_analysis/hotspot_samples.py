"""Generate representative MNIST images for Week 5 SOM error hotspots.

This step is MNIST-specific on purpose: it fetches the original MNIST images
(torchvision) by their global sample IDs. Another dataset needs its own image
loader here; every other Week 3-5 step reads the dataset facts from the config.

Uses stable global MNIST sample IDs:
    0..59999     -> original MNIST training set
    60000..69999 -> original MNIST test set

Development validation samples only.

Reads the Week 5 table for the SOM named in a config (default som.yaml) and
writes images to outputs/v1_mnist/p2_som_training_visualization/error_pattern_analysis/<selected_model>/
representative_samples/.

Usage (from the code root):
  python src/v1_mnist/component/p2_som_training_visualization/error_pattern_analysis/hotspot_samples.py \
      [--config CONFIG] [--mnist-root PATH_TO_MNIST_DOWNLOAD]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.config import load_config
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


def load_original_mnist() -> tuple[datasets.MNIST, datasets.MNIST]:
    train = datasets.MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=False,
    )

    test = datasets.MNIST(
        root=str(DATA_ROOT),
        train=False,
        download=False,
    )

    return train, test


def get_image(sample_id: int, train: datasets.MNIST, test: datasets.MNIST) -> np.ndarray:
    if 0 <= sample_id < 60000:
        return train.data[sample_id].numpy()

    if 60000 <= sample_id < 70000:
        return test.data[sample_id - 60000].numpy()

    raise ValueError(f"Invalid global MNIST sample ID: {sample_id}")


def verify_label(row: dict, train: datasets.MNIST, test: datasets.MNIST) -> None:
    sid = row["sample_id"]

    if sid < 60000:
        source_label = int(train.targets[sid])
    else:
        source_label = int(test.targets[sid - 60000])

    if source_label != row["true_class"]:
        raise RuntimeError(
            f"Sample-ID reconstruction mismatch for {sid}: "
            f"source label={source_label}, "
            f"stored label={row['true_class']}"
        )


def plot_hotspot(neuron: int, rows: list[dict], train: datasets.MNIST, test: datasets.MNIST) -> None:
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

    for ax, row in zip(axes, selected):
        verify_label(row, train, test)

        image = get_image(row["sample_id"], train, test)

        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"ID {row['sample_id']}\n"
            f"True {row['true_class']} → Pred {row['predicted_class']}\n"
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


def plot_overall(rows: list[dict], train: datasets.MNIST, test: datasets.MNIST) -> None:
    """Plot one representative error from each detected hotspot."""

    by_neuron: dict[int, list[dict]] = {}

    for row in rows:
        if row["hotspot"]:
            by_neuron.setdefault(row["bmu"], []).append(row)

    representatives = []

    for neuron, neuron_rows in sorted(by_neuron.items()):
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

    for ax, row in zip(axes, representatives):
        verify_label(row, train, test)

        image = get_image(row["sample_id"], train, test)

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        ax.set_title(
            f"Neuron {row['bmu']}\n"
            f"True {row['true_class']} → Pred {row['predicted_class']}\n"
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
    global TABLE_PATH, OUTPUT_DIR, DATA_ROOT

    parser = argparse.ArgumentParser(description="Representative images for Week 5 hotspots.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="SOM config naming the selected model.")
    parser.add_argument("--mnist-root", default=None,
                        help="Folder holding the torchvision MNIST download "
                             "(default: paths.data_root from the config).")
    args = parser.parse_args()

    run_dir = error_output_dir(args.config, REPO_ROOT)
    TABLE_PATH = run_dir / "tables" / "val_misclassified_samples.csv"
    OUTPUT_DIR = run_dir / "representative_samples"
    if args.mnist_root is not None:
        DATA_ROOT = Path(args.mnist_root)
    else:
        # Same folder baseline training downloaded MNIST into (base.yaml paths.data_root).
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        DATA_ROOT = Path(load_config(str(config_path)).paths.data_root)
        if not DATA_ROOT.is_absolute():
            DATA_ROOT = (REPO_ROOT / DATA_ROOT).resolve()
    if not TABLE_PATH.is_file():
        raise FileNotFoundError(f"{TABLE_PATH} not found: run error_patterns.py with the same config first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    hotspot_rows = [row for row in rows if row["hotspot"]]

    if not hotspot_rows:
        raise RuntimeError("No hotspot validation errors were found.")

    logger.info("Loading original MNIST for sample-ID reconstruction from %s", DATA_ROOT)
    train, test = load_original_mnist()

    # Verify every hotspot error before producing any figure.
    for row in hotspot_rows:
        verify_label(row, train, test)

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
            train,
            test,
        )

    plot_overall(hotspot_rows, train, test)

    logger.info("Representative-image generation COMPLETE.")
    logger.info("Hotspot neurons represented: %d", len(neurons))
    logger.info("Hotspot error samples available: %d", len(hotspot_rows))
    logger.info("Output: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
