"""Utilities for interactive SOM visualizations.

This module intentionally consumes already-versioned SOM artifacts.
It does not retrain the SOM, recompute BMUs, or access the final test split.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SomInteractiveArtifacts:
    """Paths and metadata required for interactive SOM visualizations."""

    run_id: str
    grid_height: int
    grid_width: int
    num_neurons: int
    random_seed: int
    epochs: int
    train_qe: float
    validation_qe: float
    train_topological_error_pct: float
    validation_topological_error_pct: float
    qe_history: tuple[dict[str, float], ...]


def _require_file(path: Path) -> None:
    """Raise a clear error when an expected artifact is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required SOM artifact not found: {path}")


def load_interactive_artifacts(
    metrics_path: Path,
    qe_history_path: Path,
) -> SomInteractiveArtifacts:
    """Load and validate frozen SOM metrics and QE history."""

    _require_file(metrics_path)
    _require_file(qe_history_path)

    metrics: dict[str, Any] = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    if metrics.get("test_evaluated") is not False:
        raise ValueError(
            "Interactive development visualizations require a SOM run "
            "with final test evaluation disabled."
        )

    som = metrics["som"]
    train_metrics = metrics["train_metrics"]
    validation_metrics = metrics["validation_metrics"]

    grid_height = int(som["grid_height"])
    grid_width = int(som["grid_width"])
    num_neurons = int(som["num_neurons"])

    if grid_height * grid_width != num_neurons:
        raise ValueError(
            "SOM grid dimensions are inconsistent with num_neurons."
        )

    history: list[dict[str, float]] = []

    with qe_history_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        expected_columns = {
            "epoch",
            "quantization_error",
            "neighborhood_radius",
        }

        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(
                "Unexpected QE-history columns: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            history.append(
                {
                    "epoch": float(row["epoch"]),
                    "quantization_error": float(
                        row["quantization_error"]
                    ),
                    "neighborhood_radius": float(
                        row["neighborhood_radius"]
                    ),
                }
            )

    if not history:
        raise ValueError("QE history is empty.")

    return SomInteractiveArtifacts(
        run_id=str(metrics["experiment"]["run_id"]),
        grid_height=grid_height,
        grid_width=grid_width,
        num_neurons=num_neurons,
        random_seed=int(som["random_seed"]),
        epochs=int(som["epochs"]),
        train_qe=float(train_metrics["quantization_error"]),
        validation_qe=float(
            validation_metrics["quantization_error"]
        ),
        train_topological_error_pct=float(
            train_metrics["topological_error_1st_order_pct"]
        ),
        validation_topological_error_pct=float(
            validation_metrics[
                "topological_error_1st_order_pct"
            ]
        ),
        qe_history=tuple(history),
    )
