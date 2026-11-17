"""Provenance of a re-analysis run: SHA-256 of the protocol, the configs and the checkpoints.

The first stage writes ``metadata/inputs.json``; every later stage checks that nothing changed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
    ReanalysisSettings,
    cell_name,
)
from src.v1_mnist.component.utils.run_provenance import write_json
from src.v1_mnist.component.utils.som_io import sha256_file

log = logging.getLogger("v1_mnist.som_reanalysis")


def current_inputs(settings: ReanalysisSettings) -> dict[str, Any]:
    """Hashes of every input file the run depends on."""
    retr = settings.retraining
    checkpoints = {
        cell_name(arm, seed): sha256_file(retr.checkpoint_path(arm, seed))
        for arm in settings.arms
        for seed in settings.seeds
    }
    return {
        "run_id": settings.run_id,
        "protocol_sha256": sha256_file(settings.protocol_path),
        "config_sha256": sha256_file(settings.config_path),
        "retraining_config_sha256": sha256_file(retr.config_path),
        "som_config_sha256": sha256_file(retr.analysis.config_path),
        "checkpoint_sha256": checkpoints,
    }


def start_run(settings: ReanalysisSettings) -> None:
    """Create the run folder, or continue a run whose inputs are unchanged. Never overwrites results."""
    observed = current_inputs(settings)
    if settings.inputs_path.is_file():
        verify_run(settings)
        log.info("Continuing run %s (inputs unchanged).", settings.run_id)
        return
    started = [p for p in settings.output_dir.iterdir() if p.name != "logs"] if settings.output_dir.exists() else []
    if started:
        raise RuntimeError(
            f"{settings.output_dir} exists without a provenance record. Set a new run.revision "
            "in configs/som_reanalysis.yaml."
        )
    write_json(settings.inputs_path, observed)
    log.info("Started run %s.", settings.run_id)


def verify_run(settings: ReanalysisSettings) -> None:
    """Stop if any input differs from the record made when the run started."""
    if not settings.inputs_path.is_file():
        raise FileNotFoundError(f"{settings.inputs_path} is missing: run the embeddings stage first.")
    recorded = json.loads(settings.inputs_path.read_text(encoding="utf-8"))
    observed = current_inputs(settings)
    changed = [k for k in observed if observed[k] != recorded.get(k)]
    if changed:
        raise RuntimeError(
            f"Inputs changed since the run started: {changed}. Set a new run.revision and start again."
        )
