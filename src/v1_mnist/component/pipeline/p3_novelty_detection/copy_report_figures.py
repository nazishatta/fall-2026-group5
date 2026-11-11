"""Copy each Phase 3 report figure from the run folder that produced it.

Every figure in reports/Latex_report/v1_mnist/p3_novelty_detection/fig/ is the PDF a
saved run wrote (listed in configs/phase3_report_figures.yaml). Where the run recorded
the figure's SHA-256, the copy is refused unless the run folder's file still has it,
so the report shows exactly what the run produced. Figures are never regenerated here.

Run on the machine that holds the run folders' PDFs (they are not in git):
  python src/v1_mnist/component/pipeline/p3_novelty_detection/copy_report_figures.py [--check]
--check only compares the report copies with the recorded fingerprints.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

CODE_ROOT = Path(__file__).resolve().parents[5]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.v1_mnist.component.utils.file_hash import sha256_file  # noqa: E402
from src.v1_mnist.component.utils.logging import get_logger  # noqa: E402
from src.v1_mnist.component.utils.repo_paths import resolve_inside  # noqa: E402

CONFIG_PATH = CODE_ROOT / "src/v1_mnist/component/configs/phase3_report_figures.yaml"
LOGGER = get_logger(__name__)


def load_figures(
    config_path: Path = CONFIG_PATH, root: Path = CODE_ROOT
) -> list[dict[str, Any]]:
    """Read the figure list and resolve every path inside the repository."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    report_dir = resolve_inside(root, config["report_dir"])
    figures = []
    for entry in config["figures"]:
        run_folder = resolve_inside(root, entry["run_folder"])
        figures.append(
            {
                "report_file": resolve_inside(report_dir, entry["report_file"]),
                "run_folder": run_folder,
                "run_file": resolve_inside(run_folder, entry["run_file"]),
                "hash_record": None
                if entry["hash_record"] is None
                else resolve_inside(run_folder, entry["hash_record"]),
            }
        )
    return figures


def recorded_sha256(hash_record: Path) -> str:
    """Return the figure SHA-256 a run stored in its metrics file."""
    record = json.loads(hash_record.read_text(encoding="utf-8"))
    value = record.get("figure_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"No figure_sha256 in {hash_record}")
    return value


def copy_figures(figures: list[dict[str, Any]], check_only: bool = False) -> list[str]:
    """Copy (or only check) every figure; return a list of problems, empty when all is well."""
    problems: list[str] = []
    for figure in figures:
        expected = (
            recorded_sha256(figure["hash_record"]) if figure["hash_record"] else None
        )
        if not check_only:
            source = figure["run_file"]
            if not source.is_file():
                problems.append(f"missing run figure: {source}")
                continue
            if expected and sha256_file(source) != expected:
                problems.append(
                    f"run figure differs from the fingerprint its run recorded: {source}"
                )
                continue
            figure["report_file"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, figure["report_file"])
            LOGGER.info("Copied %s", figure["report_file"].name)
        if not figure["report_file"].is_file():
            problems.append(f"missing report figure: {figure['report_file']}")
        elif expected and sha256_file(figure["report_file"]) != expected:
            problems.append(
                f"report figure differs from its run's fingerprint: {figure['report_file']}"
            )
    return problems


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--check", action="store_true", help="only compare report copies with the runs"
    )
    args = parser.parse_args()
    problems = copy_figures(load_figures(args.config), check_only=args.check)
    for problem in problems:
        LOGGER.error(problem)
    if problems:
        sys.exit(1)
    LOGGER.info("All report figures match their runs.")


if __name__ == "__main__":
    main()
