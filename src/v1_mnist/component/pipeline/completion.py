"""Completion records for the Phase 3 runners: a step's run folder counts as finished only with one.

A Phase 3 script makes its run folder when it starts, so an existing folder does not mean a finished
step. After a step's script exits successfully, the runner writes ``run_complete.json`` into the
folder. A folder without the record is an interrupted (or unrecorded) run: the runner stops and names
it instead of skipping it or starting it again.

Runs made before the records existed are recorded once from the dataset's ``run_manifest.csv``, whose
rows are written only for finished runs:

  python src/v1_mnist/component/pipeline/completion.py --manifest outputs/<dataset>/p3_novelty_detection/run_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: put the code root on the import path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.v1_mnist.component.utils.som_io import sha256_file  # noqa: E402

log = logging.getLogger("v1_mnist.completion")
RECORD = "run_complete.json"


def record_path(folder: Path) -> Path:
    """Where a run folder's completion record lives."""
    return folder / RECORD


def is_complete(folder: Path) -> bool:
    """True when the run folder holds its completion record."""
    return record_path(folder).is_file()


def _write(folder: Path, content: dict[str, Any]) -> Path:
    path = record_path(folder)
    if path.exists():
        raise FileExistsError(f"{path} exists: a completion record is never overwritten.")
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_record(folder: Path, step: str, script: Path, config: Path, repo_root: Path) -> Path:
    """Record that ``step`` finished successfully in ``folder`` (called by the runner after the script)."""
    return _write(folder, {
        "step": step,
        "script": script.relative_to(repo_root).as_posix(),
        "config": config.relative_to(repo_root).as_posix() if config.is_relative_to(repo_root) else str(config),
        "config_sha256": sha256_file(config),
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "runner",
    })


def backfill_from_manifest(manifest: Path) -> list[Path]:
    """Record every run folder listed in a run manifest that has no record yet; returns the new records.

    Only folders with a manifest row are recorded: the row is the evidence the run finished.
    """
    written = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            folder = manifest.parent / row["step_folder"] / row["run_folder"]
            if not folder.is_dir():
                raise FileNotFoundError(f"{folder} is listed in {manifest.name} but does not exist.")
            if is_complete(folder):
                continue
            written.append(_write(folder, {
                "step": row["step"], "script": row["script"], "config": row["config"], "role": row["role"],
                "source": f"backfilled from {manifest.name} (the run finished before completion records existed)",
            }))
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="A dataset's Phase 3 run_manifest.csv.")
    written = backfill_from_manifest(parser.parse_args().manifest)
    log.info("Wrote %d completion record(s).", len(written))


if __name__ == "__main__":
    main()
