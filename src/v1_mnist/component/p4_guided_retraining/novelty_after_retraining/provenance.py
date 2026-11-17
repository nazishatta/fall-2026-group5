"""Provenance of a novelty-after-retraining run: SHA-256 of the protocol, configs and every input.

The scoring stage writes ``metadata/inputs.json`` before the first cell; every later call checks that
nothing changed, so a run cannot mix inputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.settings import NoveltySettings
from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.som_cell import model_path
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.novelty_after_retraining")


def current_inputs(settings: NoveltySettings) -> dict[str, Any]:
    """Hashes of every file the run depends on."""
    re, retr = settings.reanalysis, settings.reanalysis.retraining
    return {
        "run_id": settings.run_id,
        "protocol_sha256": sha256_file(settings.protocol_path),
        "config_sha256": sha256_file(settings.config_path),
        "reanalysis_config_sha256": sha256_file(re.config_path),
        "svhn_sha256": sha256_file(settings.svhn_file),
        "phase1_checkpoint_sha256": sha256_file(retr.cnn_checkpoint),
        "checkpoint_sha256": {c: sha256_file(retr.checkpoint_path(*arm_and_seed(c))) for c in settings.cells},
        "som_model_sha256": {c: sha256_file(model_path(re, c)) for c in settings.cells},
    }


def arm_and_seed(cell: str) -> tuple[str, int]:
    arm, seed = cell.rsplit("_seed", 1)
    return arm, int(seed)


def start_run(settings: NoveltySettings) -> None:
    """Create the run record, or continue a run whose inputs are unchanged. Never overwrites results."""
    observed = current_inputs(settings)
    if observed["svhn_sha256"] != settings.svhn_sha256:
        raise RuntimeError("SVHN file differs from the Phase 3 shared provenance record.")
    if settings.inputs_path.is_file():
        verify_run(settings)
        log.info("Continuing run %s (inputs unchanged).", settings.run_id)
        return
    if settings.output_dir.exists() and any(p.name != "logs" for p in settings.output_dir.iterdir()):
        raise RuntimeError(f"{settings.output_dir} exists without a provenance record: set a new run.revision.")
    write_json(settings.inputs_path, observed)
    log.info("Started run %s.", settings.run_id)


def verify_run(settings: NoveltySettings) -> None:
    """Stop if any input differs from the record made when the run started."""
    if not settings.inputs_path.is_file():
        raise FileNotFoundError(f"{settings.inputs_path} is missing: run the score stage first.")
    recorded = json.loads(settings.inputs_path.read_text(encoding="utf-8"))
    observed = current_inputs(settings)
    changed = [k for k in observed if observed[k] != recorded.get(k)]
    if changed:
        raise RuntimeError(f"Inputs changed since the run started: {changed}. Set a new run.revision.")
