"""Provenance of a feature-layer ablation run: SHA-256 of the protocol, the configs and the frozen inputs.

The first call writes ``metadata/inputs.json``; every later call checks that nothing changed, so one run
cannot mix inputs. Results are never overwritten: a finished cell is skipped, a changed input stops the run.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
    AblationSettings,
)
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.feature_layer_ablation")
SPLIT_FILES = ("train_sample_ids.npy", "train_labels.npy", "val_sample_ids.npy", "val_labels.npy",
               "train_features.npy", "val_features.npy", "val_preds.npy")


def current_inputs(settings: AblationSettings) -> dict[str, Any]:
    """Hashes of every file the run depends on (the test split's files are not read)."""
    return {
        "run_id": settings.run_id,
        "protocol_sha256": sha256_file(settings.protocol_path),
        "config_sha256": sha256_file(settings.config_path),
        "input_configs_sha256": {k: sha256_file(p) for k, p in sorted(settings.input_configs.items())},
        "frozen_split_sha256": {n: sha256_file(settings.embeddings_dir / n) for n in SPLIT_FILES},
        "layers": list(settings.layers),
        "som_seeds": list(settings.som_seeds),
    }


def check_frozen_files(settings: AblationSettings) -> None:
    """The checkpoint and SVHN must be the ones recorded in the Phase 3 shared provenance."""
    for name, path, expected in (("checkpoint", settings.checkpoint, settings.checkpoint_sha256),
                                 ("SVHN file", settings.svhn_file, settings.svhn_sha256)):
        seen = sha256_file(path)
        if seen != expected:
            raise RuntimeError(f"{name} {path}: SHA-256 {seen} differs from the shared provenance ({expected}).")


def start_run(settings: AblationSettings) -> None:
    """Create the run record, or continue a run whose inputs are unchanged."""
    if settings.inputs_path.is_file():
        verify_run(settings)
        log.info("Continuing run %s (inputs unchanged).", settings.run_id)
        return
    if settings.output_dir.exists() and any(p.name != "logs" for p in settings.output_dir.iterdir()):
        raise RuntimeError(f"{settings.output_dir} exists without a provenance record: set a new run.revision.")
    write_json(settings.inputs_path, current_inputs(settings))
    log.info("Started run %s.", settings.run_id)


def verify_run(settings: AblationSettings) -> None:
    """Stop if any input differs from the record made when the run started."""
    if not settings.inputs_path.is_file():
        raise FileNotFoundError(f"{settings.inputs_path} is missing: run the 'extract' stage first.")
    recorded = json.loads(settings.inputs_path.read_text(encoding="utf-8"))
    observed = current_inputs(settings)
    changed = [k for k in observed if observed[k] != recorded.get(k)]
    if changed:
        raise RuntimeError(f"Inputs changed since the run started: {changed}. Set a new run.revision.")
