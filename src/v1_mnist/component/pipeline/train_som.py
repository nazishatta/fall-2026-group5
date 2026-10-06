from __future__ import annotations

"""MNIST v1 pipeline: train and evaluate NNSOM on frozen LeNet-5 embeddings.

All generated artifacts are written below outputs/v1_mnist/som through the
paths declared in som.yaml.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.som.data import load_som_data, resolve_embeddings_dir
from src.v1_mnist.component.som.metrics import evaluate_som_quality
from src.v1_mnist.component.som.trainer import (
    get_nnsom_backend,
    train_som,
    train_som_with_history,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NNSOM on frozen LeNet-5 MNIST embeddings."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/v1_mnist/component/configs/som.yaml",
        help="Path to SOM configuration.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Unique identifier for a full scientific SOM run. "
            "Required unless --smoke-test is used."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small local integration test instead of full experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model and metrics if --run-id was already executed.",
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default=None,
        metavar="HxW",
        help=(
            "Override SOM grid dimensions from config (e.g. --grid-size 15x15). "
            "Format: HEIGHTxWIDTH."
        ),
    )
    return parser.parse_args()


def validate_run_id(run_id: str) -> str:
    """Validate run_id string format."""
    if not run_id:
        raise ValueError("run_id must be a non-empty string.")

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError(
            "run_id may contain only letters, numbers, '.', '_', "
            "and '-', and must begin with a letter or number."
        )

    return run_id


def preflight_scientific_outputs(
    run_id: str,
    logs_dir: Path,
    models_dir: Path,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Verify that output paths do not exist, or allow overwrite if requested."""
    metrics_path = logs_dir / f"{run_id}_metrics.json"
    model_path = models_dir / run_id

    existing_paths = [path for path in (metrics_path, model_path) if path.exists()]

    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Scientific run_id '{run_id}' already has existing artifact(s): {existing}. "
            "Use a new --run-id or pass --overwrite to replace them."
        )

    return metrics_path, model_path


def deterministic_subset(
    x: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Extract a reproducible random subset of samples."""
    if n_samples >= len(x):
        return x

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=n_samples, replace=False)
    return x[indices]


def save_qe_history(
    history: Any,
    run_id: str,
    logs_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, str]:
    """Save epoch-wise QE history as JSON, CSV, SVG, and PDF."""
    svg_convergence_dir = figures_dir / "svg" / "convergence"
    pdf_convergence_dir = figures_dir / "pdf" / "convergence"
    svg_convergence_dir.mkdir(parents=True, exist_ok=True)
    pdf_convergence_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = logs_dir / f"{run_id}_qe_history.json"
    csv_path = tables_dir / f"{run_id}_qe_history.csv"
    figure_svg = svg_convergence_dir / f"{run_id}_qe_curve.svg"
    figure_pdf = pdf_convergence_dir / f"{run_id}_qe_curve.pdf"

    payload = history.to_dict()
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "quantization_error", "neighborhood_radius"])
        writer.writerows(
            zip(
                history.epoch,
                history.quantization_error,
                history.neighborhood_radius,
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.quantization_error, marker="o", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NNSOM-style quantization error")
    ax.set_title("SOM Quantization Error Across Training")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(figure_svg, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "figure_svg": str(figure_svg),
        "figure_pdf": str(figure_pdf),
    }


def train_som_pipeline(
    config_path: str | Path = "src/v1_mnist/component/configs/som.yaml",
    run_id: Optional[str] = None,
    smoke_test: bool = False,
    overwrite: bool = False,
    grid_size: Optional[str] = None,
) -> dict[str, Any]:
    """Execute the SOM training pipeline with full logging and metric export."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    config = load_config(str(cfg_path))

    if grid_size is not None:
        try:
            h_str, w_str = grid_size.lower().split("x")
            grid_override_h = int(h_str)
            grid_override_w = int(w_str)
        except ValueError:
            raise ValueError(f"--grid-size must be in HxW format, got: '{grid_size}'")
        if grid_override_h <= 0 or grid_override_w <= 0:
            raise ValueError(f"--grid-size dimensions must be positive, got: {grid_size}")
        config.som.grid_height = grid_override_h
        config.som.grid_width = grid_override_w

    if smoke_test:
        effective_run_id = "smoke_test"
    else:
        if run_id is None:
            raise ValueError("Full scientific SOM runs require --run-id.")
        effective_run_id = validate_run_id(run_id)

    requested_backend = str(config.som.backend).lower()
    if requested_backend in ("numpy",):
        requested_backend = "cpu"
    if requested_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("som.backend must be one of: 'auto', 'cpu', or 'gpu'.")

    if config.preprocessing.fit_split != "train":
        raise ValueError("Leakage protection violation: preprocessing.fit_split must be 'train'.")

    if config.evaluation.use_test_for_selection:
        raise ValueError("Leakage protection violation: test data cannot be used for selection.")

    def resolve_path(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    embeddings_dir = resolve_embeddings_dir(config.paths.embeddings_dir, REPO_ROOT)
    output_dir = resolve_path(config.paths.output_dir)
    models_dir = resolve_path(config.paths.som_models_dir)
    logs_dir = resolve_path(config.paths.logs_dir)
    figures_dir = resolve_path(config.paths.figures_dir)
    tables_dir = resolve_path(config.paths.tables_dir)

    for directory in (output_dir, models_dir, logs_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Initialize structured logging to console and per-run log file
    log_file = logs_dir / f"{effective_run_id}.log"
    logger = setup_logger("train_som", log_file=log_file)

    if smoke_test:
        metrics_path = logs_dir / "som_smoke_test_metrics.json"
        model_path = None
    else:
        metrics_path, model_path = preflight_scientific_outputs(
            run_id=effective_run_id,
            logs_dir=logs_dir,
            models_dir=models_dir,
            overwrite=overwrite,
        )

    logger.info("=" * 70)
    logger.info("MNIST v1 NNSOM Training - Run ID: %s", effective_run_id)
    logger.info("=" * 70)
    logger.info("Config:           %s", cfg_path)
    logger.info("Embedding source: %s", embeddings_dir)
    logger.info("SOM output:       %s", output_dir)
    logger.info("Log file:         %s", log_file)

    data = load_som_data(embeddings_dir=embeddings_dir, expected_feature_dim=84)
    logger.info("Validated LeNet-5 embeddings:")
    logger.info("  Train: %s", data.train.features.shape)
    logger.info("  Val:   %s", data.val.features.shape)
    logger.info("  Test:  %s", data.test.features.shape)

    x_train = data.train.features
    x_val = data.val.features

    grid_height = int(config.som.grid_height)
    grid_width = int(config.som.grid_width)
    init_neighborhood = int(config.som.init_neighborhood)
    epochs = int(config.som.epochs)
    steps = int(config.som.steps)
    seed = int(config.som.random_seed)
    track_history = bool(config.som.track_qe_history)
    history_every = int(config.som.qe_history_every)

    if smoke_test:
        logger.info("SMOKE TEST MODE: using small subset")
        x_train = deterministic_subset(x_train, n_samples=2000, seed=seed)
        x_val = deterministic_subset(x_val, n_samples=500, seed=seed + 1)
        grid_height = 5
        grid_width = 5
        epochs = 2
        steps = 10

    logger.info("Training configuration:")
    logger.info("  Grid:              %dx%d", grid_height, grid_width)
    logger.info("  Neurons:           %d", grid_height * grid_width)
    logger.info("  init_neighborhood: %d", init_neighborhood)
    logger.info("  Epochs:            %d", epochs)
    logger.info("  Steps:             %d", steps)
    logger.info("  Seed:              %d", seed)
    logger.info("  QE history:        %s", track_history)

    if track_history:
        som, training_history = train_som_with_history(
            x_train=x_train,
            grid_height=grid_height,
            grid_width=grid_width,
            init_neighborhood=init_neighborhood,
            epochs=epochs,
            steps=steps,
            norm_func=data.scaler.transform,
            seed=seed,
            history_every=history_every,
        )
    else:
        som = train_som(
            x_train=x_train,
            grid_height=grid_height,
            grid_width=grid_width,
            init_neighborhood=init_neighborhood,
            epochs=epochs,
            steps=steps,
            norm_func=data.scaler.transform,
            seed=seed,
            backend=requested_backend,
        )
        training_history = None

    actual_backend = get_nnsom_backend(som)
    logger.info("Requested SOM backend: %s | Actual backend: %s", requested_backend, actual_backend)
    logger.info("SOM training completed.")

    logger.info("Evaluating training split...")
    train_metrics = evaluate_som_quality(som, x_train)
    logger.info("Evaluating validation split...")
    val_metrics = evaluate_som_quality(som, x_val)

    history_artifacts = None
    if training_history is not None:
        history_artifacts = save_qe_history(
            history=training_history,
            run_id=effective_run_id,
            logs_dir=logs_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )

    metrics: dict[str, Any] = {
        "experiment": {
            "run_id": effective_run_id,
            "run": config.run.name,
            "stage": config.run.stage,
            "input_run": config.run.input_run,
            "smoke_test": bool(smoke_test),
        },
        "som": {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_neurons": grid_height * grid_width,
            "init_neighborhood": init_neighborhood,
            "epochs": epochs,
            "steps": steps,
            "requested_backend": requested_backend,
            "actual_backend": (
                f"cpu (NumPy loop; NNSOM class: {actual_backend})"
                if track_history
                else actual_backend
            ),
            "training_implementation": (
                "train_som_with_history: NumPy batch SOM, winners recomputed every epoch"
                if track_history
                else "NNSOM native som.train (winners from initial weights)"
            ),
            "random_seed": seed,
            "track_qe_history": track_history,
            "qe_history_every": history_every,
        },
        "preprocessing": {
            "scaler": config.preprocessing.scaler,
            "feature_range": list(config.preprocessing.feature_range),
            "fit_split": config.preprocessing.fit_split,
        },
        "data": {
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "held_out_test_shape": list(data.test.features.shape),
        },
        "train_metrics": train_metrics.to_dict(),
        "validation_metrics": val_metrics.to_dict(),
        "qe_history_artifacts": history_artifacts,
        "test_evaluated": False,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("SOM quality metrics:")
    logger.info("  Train QE:                  %.6f", train_metrics.quantization_error)
    logger.info("  Train TE 1st-order (%%):    %.4f", train_metrics.topological_error_1st_order_pct)
    logger.info("  Train TE 1st+2nd (%%):      %.4f", train_metrics.topological_error_1st_2nd_order_pct)
    logger.info("  Train occupancy:           %.4f", train_metrics.occupancy_rate)
    logger.info("  Val QE:                    %.6f", val_metrics.quantization_error)
    logger.info("  Val TE 1st-order (%%):      %.4f", val_metrics.topological_error_1st_order_pct)
    logger.info("  Val TE 1st+2nd (%%):        %.4f", val_metrics.topological_error_1st_2nd_order_pct)
    logger.info("  Val occupancy:             %.4f", val_metrics.occupancy_rate)
    logger.info("Metrics written to: %s", metrics_path)

    if not smoke_test:
        if model_path is None:
            raise RuntimeError("Scientific model path was not initialized.")
        som.save_pickle(effective_run_id, str(models_dir) + os.sep)
        if not model_path.is_file():
            raise RuntimeError(f"NNSOM model save failed: {model_path}")
        logger.info("SOM model saved to: %s", model_path)

    logger.info("Test split was NOT evaluated. SOM pipeline completed successfully.")
    return metrics


def main() -> None:
    args = parse_args()
    train_som_pipeline(
        config_path=args.config,
        run_id=args.run_id,
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
