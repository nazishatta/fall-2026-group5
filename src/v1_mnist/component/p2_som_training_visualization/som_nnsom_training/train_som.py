from __future__ import annotations

"""Train and evaluate NNSOM on frozen CNN embeddings (v1: LeNet-5 on MNIST; dataset and model facts from the config).

All generated artifacts are written below outputs/v1_mnist/p2_som_training_visualization/som_nnsom_training through the
paths declared in som.yaml.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v1_mnist.component.utils.dataset_info import dataset_info
from src.v1_mnist.component.utils.som_data import load_som_data, resolve_embeddings_dir
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.quality_metrics import evaluate_som_quality
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer import (
    get_nnsom_backend,
    train_som,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.run_records import (
    validate_run_id,
    sha256_file,
    append_run_manifest,
    preflight_scientific_outputs,
    deterministic_subset,
    save_qe_history,
)
from src.v1_mnist.component.utils.config import load_config
from src.v1_mnist.component.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NNSOM on frozen CNN embeddings (dataset and model from the config)."
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


def train_som_pipeline(
    config_path: str | Path = "src/v1_mnist/component/configs/som.yaml",
    run_id: Optional[str] = None,
    smoke_test: bool = False,
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

    # Exactly the configured folder; stops with a clear error if it is missing.
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
        )

    logger.info("=" * 70)
    logger.info("%s NNSOM Training - Run ID: %s", dataset_info(config).display_name, effective_run_id)
    logger.info("=" * 70)
    logger.info("Config:           %s", cfg_path)
    logger.info("Embedding source: %s", embeddings_dir)
    logger.info("SOM output:       %s", output_dir)
    logger.info("Log file:         %s", log_file)

    data = load_som_data(embeddings_dir=embeddings_dir, expected_feature_dim=dataset_info(config).feature_dim)
    logger.info("Validated CNN embeddings:")
    logger.info("  Train: %s", data.train.features.shape)
    logger.info("  Val:   %s", data.val.features.shape)
    logger.info("  Test:  %s", data.test.features.shape)

    x_train = data.train.features
    x_val = data.val.features

    # Record (and optionally enforce) the exact embeddings used for this run.
    train_features_sha256 = sha256_file(Path(embeddings_dir) / "train_features.npy")
    val_features_sha256 = sha256_file(Path(embeddings_dir) / "val_features.npy")
    logger.info("  train_features.npy SHA-256: %s", train_features_sha256)
    logger.info("  val_features.npy SHA-256:   %s", val_features_sha256)
    expected_inputs = getattr(config, "expected_inputs", None)
    if expected_inputs is not None:
        for name, observed in (
            ("train_features_sha256", train_features_sha256),
            ("val_features_sha256", val_features_sha256),
        ):
            expected = getattr(expected_inputs, name, None)
            if expected and observed != expected:
                raise RuntimeError(
                    f"Embedding check failed for {name}: expected {expected}, got {observed}. "
                    "These are not the frozen embeddings used by the grid study."
                )

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
            "embeddings_dir": str(embeddings_dir),
            "train_features_sha256": train_features_sha256,
            "val_features_sha256": val_features_sha256,
        },
        "train_metrics": train_metrics.to_dict(),
        "validation_metrics": val_metrics.to_dict(),
        "qe_history_artifacts": history_artifacts,
        "test_evaluated": False,
    }

    if not smoke_test:
        if model_path is None:
            raise RuntimeError("Scientific model path was not initialized.")
        som.save_pickle(effective_run_id, str(models_dir) + os.sep)
        if not model_path.is_file():
            raise RuntimeError(f"NNSOM model save failed: {model_path}")
        model_sha256 = sha256_file(model_path)
        metrics["model"] = {"path": str(model_path), "sha256": model_sha256}
        logger.info("SOM model saved to: %s", model_path)
        logger.info("SOM model SHA-256: %s", model_sha256)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if not smoke_test:
        append_run_manifest(
            logs_dir / "run_manifest.csv",
            {
                "run_id": effective_run_id,
                "config": str(cfg_path.name),
                "grid": f"{grid_height}x{grid_width}",
                "init_neighborhood": init_neighborhood,
                "epochs": epochs,
                "seed": seed,
                "training_implementation": metrics["som"]["training_implementation"],
                "train_features_sha256": train_features_sha256,
                "model_sha256": model_sha256,
                "val_qe": val_metrics.quantization_error,
                "val_te1_pct": val_metrics.topological_error_1st_order_pct,
                "val_te12_pct": val_metrics.topological_error_1st_2nd_order_pct,
                "val_occupancy": val_metrics.occupancy_rate,
                "metrics_file": metrics_path.name,
            },
        )

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

    logger.info("Test split was NOT evaluated. SOM pipeline completed successfully.")
    return metrics


def main() -> None:
    args = parse_args()
    train_som_pipeline(
        config_path=args.config,
        run_id=args.run_id,
        smoke_test=args.smoke_test,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
