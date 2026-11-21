"""Shared helpers for the step runners in pipeline/.

Each runner (for example p2_som_training_visualization/run_cluster_analysis.py)
builds a list of steps and hands it to ``run_steps``. A step is either one of the
scripts in the component folders, run as its own process from the code root
(the folder that holds src/ and outputs/), or a small Python function.

<output_root> in the runners' notes is project.output_root in the base.yaml next to the
config (outputs/v1_mnist for MNIST, outputs/v1_fashion_mnist for Fashion-MNIST).

Common options for every Phase 1 and 2 runner:
  --config    config file (relative to the code root or absolute)
  --dry-run   print the steps and exit without running anything

The Phase 3 runners (p3_novelty_detection/run_*.py) build a list of ``P3Step``
and hand it to ``run_phase3``. Unlike Phases 1 and 2, every Phase 3 script has
its own config (its own run ID), so each step names its config. Their options:
  --dry-run          print the steps and exit without running anything
  --only STEP        run just one step (stops if its results already exist)
  --config CONFIG    config for the --only step (a new run needs a new run ID)
  --steps STEP ...   run these steps in the runner's order, skipping any whose results
                     exist (used by run_all.py)
  --config-dir DIR   folder of the steps' configs (default: the MNIST configs folder)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]          # code root (holds src/ and outputs/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.in_distribution import in_distribution  # noqa: E402
from src.v1_mnist.component.utils.logging import get_console_logger  # noqa: E402

log = get_console_logger()

COMPONENT = REPO_ROOT / "src" / "v1_mnist" / "component"
P1 = COMPONENT / "p1_model_training_baseline"
P2 = COMPONENT / "p2_som_training_visualization"
P3 = COMPONENT / "p3_novelty_detection"
PIPELINE = COMPONENT / "pipeline"

CNN_CONFIG = "src/v1_mnist/component/configs/cnn_baseline.yaml"
SOM_CONFIG = "src/v1_mnist/component/configs/som.yaml"
CONFIG_DIR = "src/v1_mnist/component/configs"

# The step runners, in proposal order (used by run_all.py). Each is named after
# its component folder; the order of this dict is the order they run in.
STEP_RUNNERS = {
    "data_pipeline": PIPELINE / "p1_model_training_baseline" / "run_data_pipeline.py",
    "cnn_baseline_training": PIPELINE / "p1_model_training_baseline" / "run_cnn_baseline_training.py",
    "som_nnsom_training": PIPELINE / "p2_som_training_visualization" / "run_som_nnsom_training.py",
    "cluster_analysis": PIPELINE / "p2_som_training_visualization" / "run_cluster_analysis.py",
    "error_pattern_analysis": PIPELINE / "p2_som_training_visualization" / "run_error_pattern_analysis.py",
}
# Steps that use the CNN config; every later step uses the SOM config.
CNN_STEPS = ("data_pipeline", "cnn_baseline_training")
REPORT_RUNNER = PIPELINE / "p2_som_training_visualization" / "run_interactive_report.py"


@dataclass
class Step:
    """One step of a runner: a script with arguments, or a Python function."""

    label: str
    script: Optional[Path] = None
    args: Sequence[str] = field(default_factory=list)
    func: Optional[Callable[[], None]] = None
    listed_only: bool = False   # print the command but do not run it (a dry run of a script with no --dry-run)

    def describe(self) -> str:
        if self.func is not None:
            return f"(built-in) {self.func.__name__}()"
        rel = self.script.relative_to(REPO_ROOT).as_posix()
        return "python " + " ".join(shlex.quote(a) for a in [rel, *self.args])


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the code root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load(config_path: str | Path) -> SimpleNamespace:
    """Load a config with the project's own loader (base.yaml merged in)."""
    from src.v1_mnist.component.utils.config import load_config
    return load_config(str(resolve(config_path)))


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config,
                        help=f"Config file (default: {default_config}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the steps without running them.")
    return parser


def check_not_done(existing: Sequence[Path], dry_run: bool, hint: str = "") -> None:
    """Stop if any of this step's key results already exist (results are never overwritten)."""
    found = [p for p in existing if p.exists()]
    if not found:
        return
    lines = "\n".join(f"  - {p.relative_to(REPO_ROOT).as_posix()}" for p in found)
    msg = (f"These results already exist:\n{lines}\n"
           "Results are never overwritten (project rule). To redo this step, move or delete them first." + (f"\n{hint}" if hint else ""))
    if dry_run:
        log.info(f"NOTE (dry run): this run would stop here.\n{msg}\n")
        return
    log.error(msg)
    sys.exit(1)


def run_steps(title: str, steps: Sequence[Step], dry_run: bool) -> None:
    """Print and run the steps in order, from the code root. Stops at the first failure."""
    log.info("=" * 72)
    log.info(title + ("   [dry run: nothing is executed]" if dry_run else ""))
    log.info("=" * 72)
    for i, step in enumerate(steps, start=1):
        log.info(f"\n[{i}/{len(steps)}] {step.label}\n    $ {step.describe()}")
        if dry_run or step.listed_only:
            continue
        if step.func is not None:
            step.func()
            continue
        result = subprocess.run([sys.executable, str(step.script), *step.args], cwd=REPO_ROOT)
        if result.returncode != 0:
            log.error(f"\nStopped: step {i} ({step.label}) failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
    log.info(f"\nDone: {title}" if not dry_run else "\nDry run finished.")


# ---- Phase 3 runners -------------------------------------------------------


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
    parser.add_argument("--config-dir", default=CONFIG_DIR,
                        help=f"Folder holding the steps' configs, one folder per dataset (default: {CONFIG_DIR}).")
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
        config = args.config or f"{args.config_dir}/{step.config}"
        if not resolve(config).is_file():
            # A dataset's config folder lists the steps of its study; a step without a config is not one.
            message = f"{step.name}: not part of this dataset's study (no {show(resolve(config))})"
            if args.only:
                log.error(message)
                sys.exit(1)
            log.info(f"Skipping {message}")
            continue
        folder = results_folder(config)
        # Labels are written for MNIST; show the dataset of this config folder instead.
        label = in_distribution(resolve(config)).wording(step.label)
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
        log.info(f"{title}: nothing to run (every step already has results or is not part of this dataset's study).")
        return
    # The Phase 3 scripts import the project as src.v1_mnist...; make that work for each child process.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    run_steps(title, plan, args.dry_run)
