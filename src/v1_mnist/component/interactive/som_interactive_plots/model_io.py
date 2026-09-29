"""Load a saved NNSOM model (same logic as pipeline/visualize_som.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from NNSOM.plots import SOMPlots


def load_model_and_grid(model_name: str, models_dir: Path, logs_dir: Path):
    """Load a saved SOM, inferring grid dimensions from its metrics record."""

    metrics_path = logs_dir / f"{model_name}_metrics.json"

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Could not infer SOM grid because the metrics file is missing: {metrics_path}"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    grid_height = int(metrics["som"]["grid_height"])
    grid_width = int(metrics["som"]["grid_width"])

    model_path = models_dir / model_name

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    som = SOMPlots(dimensions=(grid_height, grid_width))

    loaded = som.load_pickle(model_path.name, str(model_path.parent) + os.sep)

    if loaded is not None and hasattr(loaded, "cluster_data"):
        som = loaded

    return som, metrics
