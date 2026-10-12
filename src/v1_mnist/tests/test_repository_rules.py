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
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import re

import pytest

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
