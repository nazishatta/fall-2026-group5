"""Where the CNN's most frequent confusions land on the SOM map.

The proposal asks for confusion shown on the SOM, to see where class confusion
occurs. For the most frequent (true class -> predicted class) pairs among the
validation errors, this draws one map per pair: every neuron is placed at its
position on the map and shaded by its dominant class; a neuron that holds
images of the pair is marked with the number of those images.

Reads only files that the cluster analysis has already written (nothing is
trained or recomputed, and no SOM model or torch is needed):
  <cluster_dir>/<model>/assignments/val_bmu_assignments.csv  (label, prediction, neuron)
  <cluster_dir>/<model>/tables/val_cluster_metrics.csv       (positions, dominant class, hits)
Validation split only; the test split is never used.

Writes, under <cluster_dir>/<model>/:
  figures/pdf/validation_confusion_overlay.pdf
  tables/val_confusion_overlay_counts.csv  (pair, neuron, number of images)
Stops if either file exists already (results are never overwritten).

Usage (from the code root, the folder that holds src/ and outputs/):
  python src/v1_mnist/component/p2_som_training_visualization/cluster_analysis/confusion_overlay.py [--config CONFIG]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.config import load_config  # noqa: E402
from src.v1_mnist.component.utils.dataset_info import DatasetInfo, dataset_info  # noqa: E402
from src.v1_mnist.component.utils.figures import pdf_figure_path, save_pdf  # noqa: E402
from src.v1_mnist.component.utils.heatmaps import (  # noqa: E402
    class_colormap_name,
    column,
    read_neuron_table,
)
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402
from src.v1_mnist.component.utils.run_settings import DEFAULT_CONFIG  # noqa: E402

logger = get_logger("v1_mnist.analysis.cluster_analysis.confusion_overlay")

FIGURE_NAME = "validation_confusion_overlay"
TABLE_NAME = "val_confusion_overlay_counts.csv"


@dataclass(frozen=True)
class Assignments:
    """Validation images with their true class, predicted class and neuron."""

    labels: np.ndarray
    predictions: np.ndarray
    neurons: np.ndarray


def read_assignments(path: Path) -> Assignments:
    """Read label, prediction and neuron of every validation image."""
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} has no rows.")
    return Assignments(
        labels=np.asarray([int(r["label"]) for r in rows], dtype=np.int64),
        predictions=np.asarray([int(r["prediction"]) for r in rows], dtype=np.int64),
        neurons=np.asarray([int(r["bmu_index"]) for r in rows], dtype=np.int64),
    )


def top_confusion_pairs(
    labels: np.ndarray, predictions: np.ndarray, count: int,
) -> list[tuple[int, int, int]]:
    """The ``count`` most frequent (true, predicted, errors) pairs among the errors.

    Ordered by number of errors, largest first; ties go to the smaller true class,
    then the smaller predicted class, so the choice never depends on chance.
    """
    wrong = labels != predictions
    tally: dict[tuple[int, int], int] = {}
    for true_class, predicted in zip(labels[wrong].tolist(), predictions[wrong].tolist()):
        tally[(true_class, predicted)] = tally.get((true_class, predicted), 0) + 1
    ranked = sorted(tally.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [(pair[0], pair[1], errors) for pair, errors in ranked[:count]]


def pair_neuron_counts(
    assignments: Assignments, true_class: int, predicted: int, num_neurons: int,
) -> np.ndarray:
    """Number of validation images of one confusion pair on each neuron."""
    selected = (assignments.labels == true_class) & (assignments.predictions == predicted)
    return np.bincount(assignments.neurons[selected], minlength=num_neurons)


def check_against_table(assignments: Assignments, support: np.ndarray) -> None:
    """The images per neuron must equal the hit counts in the cluster tables."""
    if assignments.neurons.min() < 0 or assignments.neurons.max() >= len(support):
        raise RuntimeError("An image is assigned to a neuron outside the map.")
    observed = np.bincount(assignments.neurons, minlength=len(support))
    if not np.array_equal(observed, support.astype(np.int64)):
        raise RuntimeError(
            "The images per neuron in val_bmu_assignments.csv differ from the 'support' "
            "column of val_cluster_metrics.csv; the two files are not from the same run."
        )


def overlay_rows(
    pairs: list[tuple[int, int, int]], counts: list[np.ndarray],
) -> list[dict[str, int]]:
    """One table row for every (pair, neuron) that holds at least one image."""
    return [
        {"true_class": true_class, "predicted_class": predicted, "pair_errors": errors,
         "neuron_id": int(neuron), "count": int(per_neuron[neuron])}
        for (true_class, predicted, errors), per_neuron in zip(pairs, counts)
        for neuron in np.flatnonzero(per_neuron)
    ]


def draw_overlay(
    coordinates: np.ndarray,
    dominant_class: np.ndarray,
    pairs: list[tuple[int, int, int]],
    counts: list[np.ndarray],
    info: DatasetInfo,
    columns: int,
) -> Figure:
    """One map per confusion pair, neurons shaded by dominant class, counts printed."""
    panels = len(pairs)
    columns = max(1, min(columns, panels))
    rows = math.ceil(panels / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 5.2 * rows + 0.6),
                             squeeze=False)
    cmap = plt.get_cmap(class_colormap_name(info.num_classes), info.num_classes)
    occupied = np.isfinite(dominant_class)

    for index, ax in enumerate(axes.ravel()):
        if index >= panels:
            ax.axis("off")
            continue
        true_class, predicted, errors = pairs[index]
        ax.scatter(coordinates[0], coordinates[1], s=95, marker="o", facecolors="none",
                   edgecolors="0.82", linewidths=0.45)
        ax.scatter(coordinates[0, occupied], coordinates[1, occupied], s=80, marker="o",
                   c=dominant_class[occupied], cmap=cmap, vmin=-0.5,
                   vmax=info.num_classes - 0.5, alpha=0.30, linewidths=0.0)
        marked = np.flatnonzero(counts[index])
        ax.scatter(coordinates[0, marked], coordinates[1, marked], s=150, marker="o",
                   facecolors="none", edgecolors="black", linewidths=1.2)
        for neuron in marked:
            ax.text(coordinates[0, neuron], coordinates[1, neuron], str(int(counts[index][neuron])),
                    ha="center", va="center", fontsize=7, fontweight="bold")
        ax.set_title(f"True {info.class_label} {true_class} → predicted {predicted}\n"
                     f"{errors} validation errors on {len(marked)} neurons", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [Patch(facecolor=cmap(c), alpha=0.5, label=str(c)) for c in range(info.num_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=info.num_classes, frameon=False,
               title=f"Dominant true {info.class_label} of the neuron (shading)", fontsize=8,
               title_fontsize=9)
    fig.suptitle(f"Where the most frequent confusions fall on the SOM ({info.display_name}, "
                 "validation split)\nCircled number = images of the pair on that neuron", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return fig


def write_counts(path: Path, rows: list[dict[str, int]]) -> None:
    fields = ["true_class", "predicted_class", "pair_errors", "neuron_id", "count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlay_settings(config: Any) -> tuple[int, int]:
    """Number of pairs and panels per row, both from analysis.confusion_overlay in the config."""
    section = getattr(getattr(config, "analysis", None), "confusion_overlay", None)
    if section is None:
        raise KeyError("The config has no analysis.confusion_overlay section (see configs/som.yaml).")
    num_pairs, columns = int(section.num_pairs), int(section.panel_columns)
    if num_pairs < 1 or columns < 1:
        raise ValueError("analysis.confusion_overlay.num_pairs and panel_columns must be at least 1.")
    return num_pairs, columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion overlay on the SOM map (validation).")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="SOM config naming the selected model.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else REPO_ROOT / config_path
    config = load_config(str(config_path))
    info = dataset_info(config)
    num_pairs, columns = overlay_settings(config)
    model = str(config.visualization.selected_model)
    cluster_dir = Path(str(config.paths.cluster_dir))
    cluster_dir = cluster_dir if cluster_dir.is_absolute() else (REPO_ROOT / cluster_dir).resolve()
    base = cluster_dir / model

    summary_path = base / "cluster_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} is missing: run run_cluster_analysis.py first.")
    if json.loads(summary_path.read_text(encoding="utf-8")).get("test_used") is not False:
        raise RuntimeError(f"{summary_path} does not record test_used: false.")

    figure_path = pdf_figure_path(base / "figures" / FIGURE_NAME)
    table_path = base / "tables" / TABLE_NAME
    existing = [p for p in (figure_path, table_path) if p.exists()]
    if existing:
        raise FileExistsError(
            "Results are never overwritten; these exist already: "
            + ", ".join(str(p) for p in existing)
        )

    assignments = read_assignments(base / "assignments" / "val_bmu_assignments.csv")
    coordinates, table_rows = read_neuron_table(base / "tables" / "val_cluster_metrics.csv")
    check_against_table(assignments, column(table_rows, "support"))
    if assignments.labels.max() >= info.num_classes or assignments.predictions.max() >= info.num_classes:
        raise RuntimeError(f"A class is outside 0..{info.num_classes - 1}.")

    pairs = top_confusion_pairs(assignments.labels, assignments.predictions, num_pairs)
    if not pairs:
        raise RuntimeError("There are no validation errors, so there is no confusion to draw.")
    counts = [pair_neuron_counts(assignments, t, p, len(table_rows)) for t, p, _ in pairs]
    for (true_class, predicted, errors), per_neuron in zip(pairs, counts):
        if int(per_neuron.sum()) != errors:
            raise RuntimeError(f"Pair {true_class}->{predicted}: neuron counts do not add up to {errors}.")

    dominant = column(table_rows, "dominant_class")
    dominant[column(table_rows, "support") == 0] = np.nan
    fig = draw_overlay(coordinates, dominant, pairs, counts, info, columns)
    saved = save_pdf(fig, base / "figures" / FIGURE_NAME)
    write_counts(table_path, overlay_rows(pairs, counts))
    logger.info("Confusion overlay of %d pairs written to %s and %s", len(pairs), saved, table_path)


if __name__ == "__main__":
    main()
