"""Run IDs for the CNN baseline: where each run's files go, and the overwrite check.

Every CNN training run has a run ID (``run.id`` in ``configs/cnn_baseline.yaml``).
Its checkpoint, test metrics, run record, figures and logs carry that ID in their
file names, and the scripts refuse to start if any of them already exist. A new
or re-tuned run therefore needs a new run ID; there is no overwrite option.

The run every SOM was built from is ``lenet5_best``. It was made before run IDs
existed, so only its checkpoint (``checkpoints/lenet5_best.pth``) has the ID in
its name; its ``metrics/test_evaluation.json``, figures and logs keep their old
names and are left as they are.

Embeddings: ``paths.embeddings_dir`` holds the embeddings of exactly one run. A
new run must point ``paths.embeddings_dir`` at a new folder; extraction refuses
to write into a folder that already holds embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

RUN_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class CnnRunPaths:
    """Files written by one CNN training run."""

    run_id: str
    checkpoint: Path
    test_metrics: Path
    run_record: Path
    training_curves: Path
    confusion_matrix: Path
    train_log: Path

    def outputs(self) -> list[Path]:
        """Every file that training writes for this run."""
        return [self.checkpoint, self.test_metrics, self.run_record,
                self.training_curves, self.confusion_matrix, self.train_log]


def cnn_run_id(config: Any) -> str:
    """Return ``run.id`` from the config; stop if it is missing or not a valid name."""
    run_id = getattr(getattr(config, "run", None), "id", None)
    if not run_id:
        raise ValueError("Set run.id in the CNN config (e.g. configs/cnn_baseline.yaml).")
    run_id = str(run_id)
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"run.id {run_id!r} must start with a letter and use only letters, digits, '_' or '-'."
        )
    return run_id


def cnn_run_paths(config: Any) -> CnnRunPaths:
    """Where the files of the configured CNN run go."""
    run_id = cnn_run_id(config)
    paths = config.paths
    output_dir = Path(paths.output_dir)
    figures_dir = Path(getattr(paths, "figures_dir", output_dir / "figures"))
    metrics_dir = Path(getattr(paths, "metrics_dir", output_dir / "metrics"))
    logs_dir = Path(getattr(paths, "logs_dir", output_dir / "logs"))
    checkpoints_dir = Path(getattr(paths, "checkpoints_dir", output_dir / "checkpoints"))
    return CnnRunPaths(
        run_id=run_id,
        checkpoint=checkpoints_dir / f"{run_id}.pth",
        test_metrics=metrics_dir / f"{run_id}_test_evaluation.json",
        run_record=metrics_dir / f"{run_id}_run.json",
        training_curves=figures_dir / f"{run_id}_training_curves",
        confusion_matrix=figures_dir / f"{run_id}_test_confusion_matrix",
        train_log=logs_dir / f"{run_id}_train_baseline.log",
    )


def figure_exists(stem: Path) -> bool:
    """True if a figure with this stem exists in any format folder (pdf/ or svg/) or next to it."""
    candidates = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    for fmt in ("pdf", "svg"):
        candidates.append(stem.parent / fmt / f"{stem.name}.{fmt}")
    return any(path.exists() for path in candidates)


def check_new_training_run(run_paths: CnnRunPaths) -> None:
    """Stop if this run ID already has any output (never overwrite a run)."""
    found = [p for p in run_paths.outputs()
             if (figure_exists(p) if p in (run_paths.training_curves, run_paths.confusion_matrix)
                 else p.exists())]
    if found:
        listed = "\n".join(f"  - {p}" for p in found)
        raise FileExistsError(
            f"CNN run {run_paths.run_id!r} already has results:\n{listed}\n"
            "Each new or re-tuned run needs a new run.id; existing runs are never overwritten."
        )


def check_new_embeddings_dir(embeddings_dir: Path) -> None:
    """Stop if the embeddings folder already holds embeddings from some run."""
    if not embeddings_dir.exists():
        return
    existing = sorted(p.name for p in embeddings_dir.iterdir()
                      if p.name == "embedding_manifest.json" or p.suffix == ".npy")
    if existing:
        shown = ", ".join(existing[:4]) + (" ..." if len(existing) > 4 else "")
        raise FileExistsError(
            f"{embeddings_dir} already holds embeddings ({shown}). Point paths.embeddings_dir "
            "at a new folder for this run; existing embeddings are never overwritten."
        )
