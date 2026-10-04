"""Generate representative MNIST images for Week 5 SOM error hotspots.

Uses stable global MNIST sample IDs:
    0..59999     -> original MNIST training set
    60000..69999 -> original MNIST test set

Development validation samples only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from src.v1_mnist.component.utils.logging import get_logger


REPO_ROOT = Path(__file__).resolve().parents[4]

TABLE_PATH = (
    REPO_ROOT
    / "outputs/v1_mnist/cluster_analysis/tables"
    / "val_misclassified_samples.csv"
)

OUTPUT_DIR = (
    Path.home()
    / "Downloads/V1_Minist/week5_error_patterns/representative_samples"
)

DATA_ROOT = Path.home() / "All_Data"

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


def load_original_mnist():
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


def get_image(sample_id: int, train, test) -> np.ndarray:
    if 0 <= sample_id < 60000:
        return train.data[sample_id].numpy()

    if 60000 <= sample_id < 70000:
        return test.data[sample_id - 60000].numpy()

    raise ValueError(f"Invalid global MNIST sample ID: {sample_id}")


def verify_label(row: dict, train, test) -> None:
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


def plot_hotspot(neuron: int, rows: list[dict], train, test) -> None:
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

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_overall(rows: list[dict], train, test) -> None:
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

    stem = OUTPUT_DIR / "week5_hotspot_representatives"

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    hotspot_rows = [row for row in rows if row["hotspot"]]

    if not hotspot_rows:
        raise RuntimeError("No hotspot validation errors were found.")

    logger.info("Loading original MNIST for sample-ID reconstruction...")
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
