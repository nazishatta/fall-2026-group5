"""Repository rules that .gitignore alone cannot enforce (project rules, Large files / Figures).

- Under outputs/, git keeps only small metrics and records (.json, .csv), DVC
  pointer files (.dvc), READMEs, .gitkeep and .gitignore files, each under 1 MB.
  Everything else there is generated (figures) or tracked with DVC.
- No SVG, PNG or JPG is kept in git anywhere; the one exception is a draw.io
  diagram's exported .drawio.svg, committed next to its .drawio source.
- No script offers an option to overwrite results (project rule: never
  overwrite a run; a new or re-tuned run gets a new run ID).
- No file or folder name under src/v1_mnist starts with a digit (project rule:
  importable names; order lives in a runner or a list, not in file names).
- Component code writes no random seed as a number; seeds come from the
  configs (project rule: set and record random seeds, no hard-coded settings).
- Every Phase 3 run folder has a row in run_manifest.csv naming the script and
  config that wrote it; where a result stores its config's SHA-256, today's config
  still has it (every output traces to its code).
- Every figure in the Phase 3 report is the PDF its run wrote, checked against the
  SHA-256 the run recorded (configs/phase3_report_figures.yaml lists them).
"""

from __future__ import annotations

from collections.abc import Iterator
import csv
import hashlib
import json
from typing import Any

from pathlib import Path
import shutil
import subprocess

import re

import pytest
import yaml

from src.v1_mnist.component.utils.report_figures import copy_figures, load_figures

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_OUTPUT_FILE_BYTES = 1_000_000
ALLOWED_OUTPUT_SUFFIXES = {".json", ".csv", ".dvc", ".md", ".gitkeep", ".gitignore"}


def tracked_files() -> list[str]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not inside a git checkout")
    result = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                            capture_output=True, check=True)
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def suffix(path: str) -> str:
    name = Path(path).name
    return name if name in (".gitkeep", ".gitignore") else Path(path).suffix.lower()


def test_outputs_keep_only_small_metrics_and_pointers() -> None:
    wrong_type, too_big = [], []
    for path in tracked_files():
        if not path.startswith("outputs/"):
            continue
        if suffix(path) not in ALLOWED_OUTPUT_SUFFIXES:
            wrong_type.append(path)
        elif (REPO_ROOT / path).is_file() and (REPO_ROOT / path).stat().st_size > MAX_OUTPUT_FILE_BYTES:
            too_big.append(path)
    assert not wrong_type, f"Generated files tracked under outputs/ (keep them out of git): {wrong_type}"
    assert not too_big, f"Tracked outputs/ files over 1 MB (track them with DVC instead): {too_big}"


def test_no_raster_or_svg_figures_in_git() -> None:
    found = [p for p in tracked_files()
             if Path(p).suffix.lower() in {".svg", ".png", ".jpg", ".jpeg"}
             and not p.lower().endswith(".drawio.svg")]
    assert not found, f"Figures must be PDF only; found: {found}"


def test_drawio_exports_have_their_source() -> None:
    files = set(tracked_files())
    missing = [p for p in files if p.lower().endswith(".drawio.svg") and p[: -len(".svg")] not in files]
    assert not missing, f".drawio.svg without its .drawio source: {missing}"


def test_no_option_to_overwrite_results() -> None:
    component = REPO_ROOT / "src" / "v1_mnist" / "component"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in component.rglob("*.py")
        if any(f'add_argument("--{name}"' in path.read_text(encoding="utf-8") for name in ("overwrite", "force"))
    ]
    assert not offenders, f"Scripts must not offer --overwrite/--force: {offenders}"


SEED_IN_CODE = re.compile(
    r"default_rng\(\s*\d"                      # np.random.default_rng(0)
    r"|[\"']seed[\"']\s*,\s*\d"                 # getattr(cfg, "seed", 42)
    r"|random_state\s*=\s*\d"                   # random_state=42
    r"|(?:manual_seed|random\.seed)\(\s*\d"      # torch.manual_seed(1), np.random.seed(1)
)


def test_no_seeds_written_in_code() -> None:
    component = REPO_ROOT / "src" / "v1_mnist" / "component"
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}"
        for path in sorted(component.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SEED_IN_CODE.search(line.split("#", 1)[0])
    ]
    assert not offenders, f"Seeds must come from the config, not the code: {offenders}"


def test_no_order_numbers_in_file_names() -> None:
    code = REPO_ROOT / "src" / "v1_mnist"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in code.rglob("*")
        if "__pycache__" not in path.parts and path.name[:1].isdigit()
    ]
    assert not offenders, f"Names must not start with a digit: {offenders}"


# ---- Phase 3: every output traces to its code -------------------------------

P3_OUTPUTS = REPO_ROOT / "outputs/v1_mnist/p3_novelty_detection"
P3_MANIFEST = P3_OUTPUTS / "run_manifest.csv"
COMPONENT = REPO_ROOT / "src/v1_mnist/component"
CONFIGS = COMPONENT / "configs"
STEP_FOLDERS = ("qe_novelty_detection", "u_matrix_analysis", "novelty_benchmark")
ROLES = {"result", "reproduction", "historical"}
HASH_MATCH = "recorded config SHA-256 equals this file"


def manifest_rows() -> list[dict[str, str]]:
    """Read the manifest rows."""
    with P3_MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_folders_on_disk() -> set[tuple[str, str]]:
    """Return (step folder, run folder) for every Phase 3 run folder."""
    return {
        (step, path.name)
        for step in STEP_FOLDERS
        for path in (P3_OUTPUTS / step).iterdir()
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
        folder = P3_OUTPUTS / row["step_folder"] / row["run_folder"]
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


# ---- Phase 3: every report figure is the PDF its run wrote ---------------------

REPORT_FIGURES = CONFIGS / "phase3_report_figures.yaml"
FIGURE_LIST = yaml.safe_load(REPORT_FIGURES.read_text(encoding="utf-8"))
REPORT_DIR = REPO_ROOT / FIGURE_LIST["report_dir"]
NOT_RUN_OUTPUTS = {
    "fig/gwu_logo.pdf",
    "fig/novelty_framework/phase3_novelty_framework.pdf",
}


def test_every_report_figure_is_listed() -> None:
    """Each figure the report includes is either listed with its run or is the logo/diagram."""
    tex = (REPORT_DIR / FIGURE_LIST["report_tex"]).read_text(encoding="utf-8")
    included = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    listed = {entry["report_file"] for entry in FIGURE_LIST["figures"]}
    assert included - NOT_RUN_OUTPUTS == listed


def test_every_source_run_folder_is_in_the_repository() -> None:
    """Each figure points at a run folder that exists and has a manifest row."""
    manifest = P3_MANIFEST.read_text(encoding="utf-8")
    for figure in load_figures(REPORT_FIGURES):
        assert figure["run_folder"].is_dir(), figure["run_folder"]
        assert f",{figure['run_folder'].name}," in manifest


def test_report_copies_match_their_runs() -> None:
    """Report copies exist and, where the run recorded a fingerprint, match it."""
    assert copy_figures(load_figures(REPORT_FIGURES), check_only=True) == []
