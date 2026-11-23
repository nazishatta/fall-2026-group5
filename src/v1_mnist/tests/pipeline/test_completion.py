"""Completion records: written once after a step succeeds, and backfilled from a run manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.v1_mnist.component.pipeline.completion import (
    backfill_from_manifest,
    is_complete,
    record_path,
    write_record,
)

ROOT = Path(__file__).resolve().parents[4]
MANIFEST_COLUMNS = ["step_folder", "run_folder", "role", "last_commit", "step", "script", "config",
                    "config_check", "used_in"]


def test_a_record_marks_a_folder_complete_and_is_never_overwritten(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    folder.mkdir()
    config = tmp_path / "c.yaml"
    config.write_text("run_id: run\n")
    assert not is_complete(folder)
    write_record(folder, "qe_novelty", Path(__file__), config, ROOT)
    record = json.loads(record_path(folder).read_text())
    assert is_complete(folder) and record["step"] == "qe_novelty" and record["source"] == "runner"
    assert len(record["config_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_record(folder, "qe_novelty", Path(__file__), config, ROOT)


def _manifest(tmp_path: Path, runs: list[str]) -> Path:
    path = tmp_path / "run_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for run in runs:
            writer.writerow({"step_folder": "step", "run_folder": run, "role": "result", "last_commit": "-",
                             "step": "s", "script": "x.py", "config": "c.yaml", "config_check": "-", "used_in": "-"})
    return path


def test_backfill_records_each_listed_folder_once(tmp_path: Path) -> None:
    for run in ("a", "b"):
        (tmp_path / "step" / run).mkdir(parents=True)
    manifest = _manifest(tmp_path, ["a", "b"])
    assert len(backfill_from_manifest(manifest)) == 2
    assert backfill_from_manifest(manifest) == []                          # already recorded
    assert "backfilled" in json.loads(record_path(tmp_path / "step" / "a").read_text())["source"]


def test_backfill_refuses_a_listed_folder_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backfill_from_manifest(_manifest(tmp_path, ["missing"]))
