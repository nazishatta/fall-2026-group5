"""Tables and records written by the hard example mining step."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.difficulty import (
    DifficultyFlags,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.inputs import (
    TrainInputs,
)
from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.pool_selection import (
    class_counts,
)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_difficulty_table(
    path: Path,
    data: TrainInputs,
    purity: np.ndarray,
    flags: DifficultyFlags,
    msp: np.ndarray,
) -> None:
    """One row per training image (about 4 MB, so DVC-tracked, not committed to git)."""
    header = [
        "sample_id", "label", "correct", "confidence", "bmu_index", "bmu_purity", "bmu_distance",
        "in_hotspot", "low_purity", "qe_flag", "in_som_pool", "in_msp_pool",
    ]
    som = flags.in_som_pool
    rows = [
        [int(data.sample_id[i]), int(data.label[i]), int(data.correct[i]), float(data.confidence[i]),
         int(data.bmu[i]), float(purity[i]), float(data.bmu_distance[i]), int(flags.in_hotspot[i]),
         int(flags.low_purity[i]), int(flags.qe_flag[i]), int(som[i]), int(msp[i])]
        for i in range(len(data.sample_id))
    ]
    _write_csv(path, header, rows)


def pool_summary_rows(data: TrainInputs, flags: DifficultyFlags, msp: np.ndarray) -> list[list[Any]]:
    """Size of each pool and how many training mistakes it contains."""
    total = len(data.sample_id)
    errors = ~data.correct
    pools = {
        "in_hotspot": flags.in_hotspot,
        "low_purity": flags.low_purity,
        "qe_flag": flags.qe_flag,
        "som_pool (any of the three)": flags.in_som_pool,
        "msp_pool (same size)": msp,
        "som_and_msp_overlap": flags.in_som_pool & msp,
    }
    return [
        [name, int(mask.sum()), round(float(mask.sum()) / total, 6), int((mask & errors).sum())]
        for name, mask in pools.items()
    ]


def write_pool_tables(
    output_dir: Path, data: TrainInputs, flags: DifficultyFlags, msp: np.ndarray, num_classes: int
) -> None:
    """pool_summary.csv and pool_by_class.csv (small, kept in git)."""
    tables = output_dir / "tables"
    _write_csv(
        tables / "pool_summary.csv",
        ["pool", "images", "share_of_training", "training_errors_inside"],
        pool_summary_rows(data, flags, msp),
    )
    everything = np.ones(len(data.label), dtype=bool)
    columns = {
        "all_training": class_counts(data.label, everything, num_classes),
        "som_pool": class_counts(data.label, flags.in_som_pool, num_classes),
        "msp_pool": class_counts(data.label, msp, num_classes),
    }
    rows = [[digit] + [int(col[digit]) for col in columns.values()] for digit in range(num_classes)]
    _write_csv(tables / "pool_by_class.csv", ["digit", *columns.keys()], rows)
