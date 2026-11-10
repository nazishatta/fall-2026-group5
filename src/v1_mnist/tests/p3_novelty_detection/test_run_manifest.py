"""Check that every Phase 3 run folder can be traced to the code and config that wrote it."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "outputs/v1_mnist/p3_novelty_detection"
MANIFEST = OUTPUTS / "run_manifest.csv"
COMPONENT = ROOT / "src/v1_mnist/component"
CONFIGS = COMPONENT / "configs"
STEP_FOLDERS = ("qe_novelty_detection", "u_matrix_analysis", "novelty_benchmark")
ROLES = {"result", "reproduction", "historical"}
HASH_MATCH = "recorded config SHA-256 equals this file"


def manifest_rows() -> list[dict[str, str]]:
    """Read the manifest rows."""
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_folders_on_disk() -> set[tuple[str, str]]:
    """Return (step folder, run folder) for every Phase 3 run folder."""
    return {
        (step, path.name)
        for step in STEP_FOLDERS
        for path in (OUTPUTS / step).iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    }


def recorded_config_hashes(value: Any) -> Iterator[str]:
    """Yield every config_sha256 value stored in a result file."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "config_sha256" and isinstance(item, str):
                yield item
            else:
                yield from recorded_config_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from recorded_config_hashes(item)


def test_every_run_folder_has_exactly_one_row() -> None:
    """No folder is missing from the manifest and no row points to a missing folder."""
    keys = [(row["step_folder"], row["run_folder"]) for row in manifest_rows()]
    assert len(keys) == len(set(keys)), "a run folder is listed twice"
    assert set(keys) == run_folders_on_disk()


def test_roles_are_known_and_reproductions_are_marked() -> None:
    """Repeat runs made during the code reorganisation are never cited as results."""
    for row in manifest_rows():
        assert row["role"] in ROLES, row["run_folder"]
        if row["run_folder"].startswith("reorg_"):
            assert row["role"] == "reproduction", row["run_folder"]
            assert row["used_in"].startswith("reproduces "), row["run_folder"]


@pytest.mark.parametrize("column, base", [("script", COMPONENT), ("config", CONFIGS)])
def test_named_scripts_and_configs_exist(column: str, base: Path) -> None:
    """Every script and config the manifest names is in the repository."""
    missing = [
        row[column]
        for row in manifest_rows()
        if row[column] != "-" and not (base / row[column]).is_file()
    ]
    assert not missing


def test_recorded_config_hashes_still_match() -> None:
    """Where a result stores its config fingerprint, today's config file still has it."""
    checked = 0
    for row in manifest_rows():
        if row["config_check"] != HASH_MATCH:
            continue
        expected = hashlib.sha256((CONFIGS / row["config"]).read_bytes()).hexdigest()
        folder = OUTPUTS / row["step_folder"] / row["run_folder"]
        recorded: set[str] = set()
        for result_file in folder.glob("*.json"):
            recorded.update(
                recorded_config_hashes(
                    json.loads(result_file.read_text(encoding="utf-8"))
                )
            )
        assert expected in recorded, row["run_folder"]
        checked += 1
    assert checked > 0


def test_every_kept_reproduction_config_is_used() -> None:
    """Configs kept under phase3_reorganization/ each wrote a listed run folder."""
    used = {row["config"] for row in manifest_rows()}
    kept = {
        path.relative_to(CONFIGS).as_posix()
        for path in (CONFIGS / "phase3_reorganization").rglob("*.yaml")
    }
    assert kept <= used
