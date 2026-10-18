"""Shared helpers for the step runners in pipeline/.

Each runner (for example p2_som_training_visualization/run_cluster_analysis.py)
builds a list of steps and hands it to ``run_steps``. A step is either one of the
scripts in the component folders, run as its own process from the code root
(the folder that holds src/ and outputs/), or a small Python function.

Common options for every runner:
  --config    config file (relative to the code root or absolute)
  --dry-run   print the steps and exit without running anything
"""

from __future__ import annotations

from types import SimpleNamespace

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]          # code root (holds src/ and outputs/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.logging import get_console_logger  # noqa: E402

log = get_console_logger()

COMPONENT = REPO_ROOT / "src" / "v1_mnist" / "component"
P1 = COMPONENT / "p1_model_training_baseline"
P2 = COMPONENT / "p2_som_training_visualization"
PIPELINE = COMPONENT / "pipeline"

CNN_CONFIG = "src/v1_mnist/component/configs/cnn_baseline.yaml"
SOM_CONFIG = "src/v1_mnist/component/configs/som.yaml"

# The step runners, in proposal order (used by run_all.py). Each is named after
# its component folder; the order of this dict is the order they run in.
STEP_RUNNERS = {
    "data_pipeline": PIPELINE / "p1_model_training_baseline" / "run_data_pipeline.py",
    "cnn_baseline_training": PIPELINE / "p1_model_training_baseline" / "run_cnn_baseline_training.py",
    "som_nnsom_training": PIPELINE / "p2_som_training_visualization" / "run_som_nnsom_training.py",
    "cluster_analysis": PIPELINE / "p2_som_training_visualization" / "run_cluster_analysis.py",
    "error_pattern_analysis": PIPELINE / "p2_som_training_visualization" / "run_error_pattern_analysis.py",
    # Phase 3: separate adapter; Phase 1–2 entries above are unchanged.
    "qe_novelty": PIPELINE / "p3_novelty_detection" / "run_phase3_stage.py",
    "qe_ood": PIPELINE / "p3_novelty_detection" / "run_phase3_stage.py",
    "svhn_benchmark": PIPELINE / "p3_novelty_detection" / "run_phase3_stage.py",
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
        if dry_run:
            continue
        if step.func is not None:
            step.func()
            continue
        result = subprocess.run([sys.executable, str(step.script), *step.args], cwd=REPO_ROOT)
        if result.returncode != 0:
            log.error(f"\nStopped: step {i} ({step.label}) failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
    log.info(f"\nDone: {title}" if not dry_run else "\nDry run finished.")
