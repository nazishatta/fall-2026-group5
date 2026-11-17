"""Provenance of a held-out-class run: SHA-256 of the protocol, the configs and the frozen split's files.

The first call writes ``metadata/inputs.json``; every later call checks that nothing changed, so one run
cannot mix inputs. A finished split also keeps the hashes of its own checkpoint, map and embeddings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import HeldOutSettings
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.held_out_class_novelty")
SPLIT_FILES = ("train_sample_ids.npy", "train_labels.npy", "val_sample_ids.npy", "val_labels.npy")


def current_inputs(settings: HeldOutSettings) -> dict[str, Any]:
    """Hashes of every file the run depends on (the test split's files are not read)."""
    return {
        "run_id": settings.run_id,
        "protocol_sha256": sha256_file(settings.protocol_path),
        "config_sha256": sha256_file(settings.config_path),
        "cnn_config_sha256": sha256_file(settings.cnn_config_path),
        "som_config_sha256": sha256_file(settings.som_config_path),
        "frozen_split_sha256": {name: sha256_file(settings.embeddings_dir / name) for name in SPLIT_FILES},
        "unknown_digits": [list(s) for s in settings.unknown_digits],
    }


def start_run(settings: HeldOutSettings) -> None:
    """Create the run record, or continue a run whose inputs are unchanged. Never overwrites results."""
    if settings.inputs_path.is_file():
        verify_run(settings)
        log.info("Continuing run %s (inputs unchanged).", settings.run_id)
        return
    if settings.output_dir.exists() and any(p.name != "logs" for p in settings.output_dir.iterdir()):
        raise RuntimeError(f"{settings.output_dir} exists without a provenance record: set a new run.revision.")
    write_json(settings.inputs_path, current_inputs(settings))
    log.info("Started run %s.", settings.run_id)


def verify_run(settings: HeldOutSettings) -> None:
    """Stop if any input differs from the record made when the run started."""
    if not settings.inputs_path.is_file():
        raise FileNotFoundError(f"{settings.inputs_path} is missing: run the 'run' stage first.")
    recorded = json.loads(settings.inputs_path.read_text(encoding="utf-8"))
    observed = current_inputs(settings)
    changed = [k for k in observed if observed[k] != recorded.get(k)]
    if changed:
        raise RuntimeError(f"Inputs changed since the run started: {changed}. Set a new run.revision.")
