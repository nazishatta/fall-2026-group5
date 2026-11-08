"""Shared pieces of the Phase 3 step runners.

Each Phase 3 runner in this folder is named after the component folder whose
scripts it calls (qe_novelty_detection, u_matrix_analysis, novelty_benchmark) and
hands a list of steps to ``run_phase3``. Unlike the Phase 1 and 2 runners, every
Phase 3 script has its own config (its own run ID), so each step names its config.

Options every Phase 3 runner has:
  --dry-run          print the steps and exit without running anything
  --only STEP        run just one step (stops if its results already exist)
  --config CONFIG    config for the --only step (a new run needs a new run ID)
  --steps STEP ...   run these steps in the runner's order, skipping any whose results
                     exist (used by run_all.py)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import (  # noqa: E402
    COMPONENT, REPO_ROOT, Step, log, resolve, run_steps,
)

P3 = COMPONENT / "p3_novelty_detection"
CONFIG_DIR = "src/v1_mnist/component/configs"


@dataclass(frozen=True)
class P3Step:
    """One Phase 3 script: its short name, script (under p3_novelty_detection/), config and fixed flags."""

    name: str
    label: str
    script: str
    config: str
    flags: tuple[str, ...] = ()


def results_folder(config_path: str | Path) -> Path | None:
    """Folder a config writes to (its output folder plus run ID), or None if the config does not name one."""
    raw = yaml.safe_load(resolve(config_path).read_text(encoding="utf-8")) or {}
    run_id = raw.get("run_id")
    if run_id is None:
        datasets = raw.get("ood_datasets") or []
        run_id = datasets[0].get("run_id") if datasets else None
    output = raw.get("output_root") or raw.get("output_dir")
    if not run_id or not output:
        return None
    return resolve(output) / str(run_id)


def show(path: Path) -> str:
    """Path relative to the code root when it is inside it, otherwise as given."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_phase3(title: str, description: str, steps: Sequence[P3Step]) -> None:
    """Parse the common options, skip steps whose results exist, and run the rest in order."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--only", choices=[s.name for s in steps], default=None,
                        help="Run just this step.")
    parser.add_argument("--steps", nargs="+", choices=[s.name for s in steps], default=None, metavar="STEP",
                        help="Run these steps in order, skipping those whose results already exist.")
    parser.add_argument("--config", default=None,
                        help="Config for the --only step (default: the step's own config in configs/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the steps without running them.")
    args = parser.parse_args()
    if args.config and not args.only:
        parser.error("--config needs --only, because every step has its own config")
    if args.only and args.steps:
        parser.error("use --only or --steps, not both")

    plan = []
    for step in steps:
        if args.only not in (None, step.name):
            continue
        if args.steps is not None and step.name not in args.steps:
            continue
        config = args.config or f"{CONFIG_DIR}/{step.config}"
        folder = results_folder(config)
        label = step.label
        if folder is not None and folder.exists():
            if args.only:
                message = (f"These results already exist:\n  - {show(folder)}\n"
                           "Results are never overwritten (project rule). "
                           "To repeat this step, use a new run_id in a new config.")
                if not args.dry_run:
                    log.error(message)
                    sys.exit(1)
                log.info(f"NOTE (dry run): this run would stop here.\n{message}\n")
            elif args.dry_run:
                label += "  [results exist: a real run skips this step]"
            else:
                log.info(f"Skipping {step.name}: results already exist in {show(folder)}")
                continue
        plan.append(Step(label, P3 / step.script, ["--config", config, *step.flags]))

    if not plan:
        log.info(f"{title}: every step already has results, nothing to run.")
        return
    # The Phase 3 scripts import the project as src.v1_mnist...; make that work for each child process.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    run_steps(title, plan, args.dry_run)
