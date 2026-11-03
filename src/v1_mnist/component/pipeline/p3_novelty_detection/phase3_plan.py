"""How ``run_all.py`` runs Phase 3: the order that respects which step reads which results.

Phase 3 has three runners, but their steps depend on each other across runners, so
they cannot simply run one runner after another:

  1. qe_novelty               writes the QE summary that nearly everything reads.
  2. the whole novelty_benchmark runner
                              svhn_benchmark writes the SVHN score arrays that the
                              threshold sweep, the figures and the gap analysis read.
  3. qe_threshold_sweep (and the reproduction steps, if asked for).
  4. the whole u_matrix_analysis runner.

Three steps only re-create results that already exist under the historical run
folders (and that the ``reorg_...`` reproduction runs already repeated). They are
left out unless ``--with-reproductions`` is given: qe_ood, qe_random_control and
qe_u_matrix. A step whose results exist is skipped by the runner either way
(results are never overwritten).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.v1_mnist.component.pipeline.common import PIPELINE, Step  # noqa: E402
from src.v1_mnist.component.pipeline.p3_novelty_detection import (  # noqa: E402
    run_novelty_benchmark, run_qe_novelty_detection, run_u_matrix_analysis,
)
from src.v1_mnist.component.pipeline.p3_novelty_detection.phase3_steps import P3Step  # noqa: E402

PHASE3_STEP = "novelty_detection"
P3_PIPELINE = PIPELINE / "p3_novelty_detection"
REPRODUCTION_STEPS = frozenset({"qe_ood", "qe_random_control", "qe_u_matrix"})

RUNNER_STEPS: dict[str, list[P3Step]] = {
    "run_qe_novelty_detection.py": run_qe_novelty_detection.STEPS,
    "run_novelty_benchmark.py": run_novelty_benchmark.STEPS,
    "run_u_matrix_analysis.py": run_u_matrix_analysis.STEPS,
}
# (runner, steps to run from it; None means all of its steps), in dependency order.
ORDER: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("run_qe_novelty_detection.py", ("qe_novelty",)),
    ("run_novelty_benchmark.py", None),
    ("run_qe_novelty_detection.py", ("qe_ood", "qe_random_control", "qe_threshold_sweep")),
    ("run_u_matrix_analysis.py", None),
)


def groups(include_reproductions: bool) -> list[tuple[str, list[str]]]:
    """(runner, step names) groups in the order ``run_all.py`` runs them; empty groups are dropped."""
    result = []
    for runner, names in ORDER:
        chosen = list(names) if names is not None else [step.name for step in RUNNER_STEPS[runner]]
        chosen = [name for name in chosen if include_reproductions or name not in REPRODUCTION_STEPS]
        if chosen:
            result.append((runner, chosen))
    return result


def phase3_steps(dry_run: bool, include_reproductions: bool = False) -> list[Step]:
    """One ``Step`` per group, each run with ``--steps`` (a runner skips steps that already have results)."""
    return [
        Step(f"Phase 3 {Path(runner).stem.removeprefix('run_')}: {', '.join(names)}",
             P3_PIPELINE / runner, ["--steps", *names, *(["--dry-run"] if dry_run else [])])
        for runner, names in groups(include_reproductions)
    ]
