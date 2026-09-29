from __future__ import annotations

"""MNIST v1 Pipeline Runner.

Expresses the strict sequential ordering of experimental stages:
  1. prepare_data:       Download and verify MNIST, generate splits and distribution plots.
  2. train_baseline:     Train LeNet-5 baseline CNN with deterministic seeds.
  3. extract_embeddings: Extract frozen penultimate (fc2) representations.
  4. train_som:          Train and evaluate 2D Self-Organizing Map representations.
  5. visualize_som:      Generate U-Matrix, hit histograms, and feature distributions.
  6. visualize_som_interactive: Build the interactive HTML SOM report.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.utils.logging import get_logger, setup_logger

logger = get_logger("v1_mnist.pipeline.runner")

PIPELINE_STAGES = [
    "prepare_data",
    "train_baseline",
    "extract_embeddings",
    "train_som",
    "visualize_som",
    "visualize_som_interactive",
]


def run_stage(
    stage: str,
    cnn_config: Optional[str | Path] = None,
    som_config: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    overwrite: bool = False,
    smoke_test: bool = False,
    grid_size: Optional[str] = None,
) -> Any:
    """Execute a single pipeline stage by name."""
    logger.info(">>> Running stage: %s <<<", stage)

    if stage == "prepare_data":
        from src.v1_mnist.component.pipeline.prepare_data import prepare_data
        return prepare_data(config_path=cnn_config)

    elif stage == "train_baseline":
        from src.v1_mnist.component.pipeline.train_baseline import train_baseline
        return train_baseline(config_path=cnn_config)

    elif stage == "extract_embeddings":
        from src.v1_mnist.component.pipeline.extract_embeddings import extract_embeddings
        if checkpoint is None:
            checkpoint = (
                PROJECT_ROOT
                / "outputs"
                / "v1_mnist"
                / "cnn_baseline"
                / "checkpoints"
                / "lenet5_best.pth"
            )
        return extract_embeddings(
            checkpoint_path=checkpoint,
            config_path=cnn_config,
        )

    elif stage == "train_som":
        from src.v1_mnist.component.pipeline.train_som import train_som_pipeline
        cfg = som_config or "src/v1_mnist/component/configs/som.yaml"
        return train_som_pipeline(
            config_path=cfg,
            run_id=run_id,
            smoke_test=smoke_test,
            overwrite=overwrite,
            grid_size=grid_size,
        )

    elif stage == "visualize_som":
        from src.v1_mnist.component.pipeline.visualize_som import main as run_visualize_som
        # Handled through internal main or direct invocation
        cfg = som_config or "src/v1_mnist/component/configs/som.yaml"
        sys.argv = [
            "visualize_som.py",
            "--config",
            str(cfg),
        ]
        if run_id:
            sys.argv.extend(["--model-name", run_id])
        return run_visualize_som()

    elif stage == "visualize_som_interactive":
        from src.v1_mnist.component.pipeline.visualize_som_interactive import (
            main as run_visualize_som_interactive,
        )
        cfg = som_config or "src/v1_mnist/component/configs/som.yaml"
        sys.argv = [
            "visualize_som_interactive.py",
            "--config",
            str(cfg),
        ]
        if run_id:
            sys.argv.extend(["--model-name", run_id])
        return run_visualize_som_interactive()

    else:
        raise ValueError(
            f"Unknown pipeline stage '{stage}'. Permitted stages: {PIPELINE_STAGES}"
        )


def run_pipeline(
    stages: Optional[Sequence[str]] = None,
    cnn_config: Optional[str | Path] = None,
    som_config: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    overwrite: bool = False,
    smoke_test: bool = False,
    grid_size: Optional[str] = None,
) -> dict[str, Any]:
    """Execute multiple stages in sequence.

    Args:
        stages: Ordered list of stage names to execute, or None for all.
    """
    if stages is None:
        stages = PIPELINE_STAGES

    results: dict[str, Any] = {}
    for stage in stages:
        results[stage] = run_stage(
            stage=stage,
            cnn_config=cnn_config,
            som_config=som_config,
            checkpoint=checkpoint,
            run_id=run_id,
            overwrite=overwrite,
            smoke_test=smoke_test,
            grid_size=grid_size,
        )

    logger.info("Pipeline execution completed for stages: %s", list(stages))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential pipeline runner for MNIST v1 experiments."
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=PIPELINE_STAGES + ["all"],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--cnn-config",
        type=str,
        default="src/v1_mnist/component/configs/cnn_baseline.yaml",
        help="Path to CNN baseline configuration.",
    )
    parser.add_argument(
        "--som-config",
        type=str,
        default="src/v1_mnist/component/configs/som.yaml",
        help="Path to SOM configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained CNN checkpoint (for extract_embeddings).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for SOM training or visualization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SOM artifacts.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short smoke test for SOM training.",
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default=None,
        help="Override SOM grid dimensions (e.g. 15x15).",
    )

    args = parser.parse_args()

    setup_logger("v1_mnist.pipeline.runner")

    if args.stage == "all":
        run_pipeline(
            stages=PIPELINE_STAGES,
            cnn_config=args.cnn_config,
            som_config=args.som_config,
            checkpoint=args.checkpoint,
            run_id=args.run_id,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            grid_size=args.grid_size,
        )
    else:
        run_stage(
            stage=args.stage,
            cnn_config=args.cnn_config,
            som_config=args.som_config,
            checkpoint=args.checkpoint,
            run_id=args.run_id,
            overwrite=args.overwrite,
            smoke_test=args.smoke_test,
            grid_size=args.grid_size,
        )


if __name__ == "__main__":
    main()
