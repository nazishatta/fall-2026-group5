"""Inputs and provenance of a grid-comparison run.

The first stage writes ``metadata/inputs.json`` (SHA-256 of the rules, the configs and every embedding
file); every later stage checks that nothing changed, so one run cannot mix inputs. The test split is
never loaded.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.settings import (
    EMBEDDING_FILES,
    GridSettings,
)
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.grid_comparison")


def current_inputs(settings: GridSettings) -> dict[str, Any]:
    """Hashes of every file the run depends on."""
    return {
        "run_id": settings.run_id,
        "protocol_sha256": sha256_file(settings.protocol_path),
        "config_sha256": sha256_file(settings.config_path),
        "input_configs_sha256": {k: sha256_file(p) for k, p in sorted(settings.input_configs.items())},
        "embeddings_sha256": {n: sha256_file(settings.embeddings_dir / n) for n in EMBEDDING_FILES},
    }


def start_run(settings: GridSettings) -> None:
    """Create the run record, or continue a run whose inputs are unchanged."""
    if settings.inputs_path.is_file():
        verify_run(settings)
        return
    # logs/ and the saved-map check (reproduction/) may be written before the first training stage.
    if settings.output_dir.exists() and any(p.name not in ("logs", "reproduction")
                                            for p in settings.output_dir.iterdir()):
        raise RuntimeError(f"{settings.output_dir} exists without a provenance record: set a new run.revision.")
    write_json(settings.inputs_path, current_inputs(settings))
    log.info("Started run %s.", settings.run_id)


def verify_run(settings: GridSettings) -> None:
    """Stop if any input differs from the record made when the run started."""
    if not settings.inputs_path.is_file():
        raise FileNotFoundError(f"{settings.inputs_path} is missing: run the 'tune' stage first.")
    recorded = json.loads(settings.inputs_path.read_text(encoding="utf-8"))
    observed = current_inputs(settings)
    changed = [k for k in observed if observed[k] != recorded.get(k)]
    if changed:
        raise RuntimeError(f"Inputs changed since the run started: {changed}. Set a new run.revision.")


def load_data(settings: GridSettings) -> dict[str, np.ndarray]:
    """Training features and the validation features, labels, CNN errors and confidence."""
    d = settings.embeddings_dir
    data = {
        "x_train": np.load(d / "train_features.npy"),
        "x_val": np.load(d / "val_features.npy"),
        "y_val": np.load(d / "val_labels.npy"),
        "err": ~np.load(d / "val_correct.npy").astype(bool),
        "conf": np.load(d / "val_confidence.npy"),
    }
    expected = settings.expected
    observed = {"train_samples": len(data["x_train"]), "validation_samples": len(data["x_val"]),
                "validation_errors": int(data["err"].sum())}
    wrong = {k: (v, expected[k]) for k, v in observed.items() if v != expected[k]}
    if wrong:
        raise RuntimeError(f"Inputs differ from the config's expected counts (observed, expected): {wrong}")
    return data


def low_confidence(conf: np.ndarray, quantile: float, expected_count: int) -> tuple[np.ndarray, float]:
    """The secondary target: validation images in the lowest ``quantile`` of CNN confidence."""
    threshold = float(np.quantile(conf, quantile))
    target = conf <= threshold
    if int(target.sum()) != expected_count:
        raise RuntimeError(f"{int(target.sum())} low-confidence images, expected {expected_count}.")
    return target, threshold
