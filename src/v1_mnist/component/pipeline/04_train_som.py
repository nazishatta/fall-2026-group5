"""MNIST v1 pipeline: train and evaluate NNSOM on frozen LeNet-5 embeddings.

All generated artifacts are written below ``outputs/v1_mnist/som`` through the
paths declared in ``som.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train NNSOM on frozen LeNet-5 MNIST embeddings."
        )
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
        help=(
            "Run a small local integration test instead of the "
            "full scientific experiment."
        ),
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
            "Override the SOM grid dimensions from the config file. "
            "Format: HEIGHTxWIDTH, e.g. --grid-size 10x10 or --grid-size 15x15. "
            "Useful for optional grid-size experiments."
        ),
    )

    return parser.parse_args()


def validate_run_id(run_id: str) -> str:
    if not run_id:
        raise ValueError("run_id must be a non-empty string.")

    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        run_id,
    ) is None:
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
) -> tuple[Path, Path]:
    metrics_path = (
        logs_dir
        / f"{run_id}_metrics.json"
    )

    model_path = models_dir / run_id

    existing_paths = [
        path
        for path in (
            metrics_path,
            model_path,
        )
        if path.exists()
    ]

    if existing_paths and not overwrite:
        existing = ", ".join(
            str(path)
            for path in existing_paths
        )

        raise FileExistsError(
            f"Scientific run_id '{run_id}' already has "
            f"existing artifact(s): {existing}. "
            "Use a new --run-id or pass --overwrite to replace them."
        )

    return metrics_path, model_path



def deterministic_subset(
    x: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    if n_samples >= len(x):
        return x

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(x),
        size=n_samples,
        replace=False,
    )

    return x[indices]


def save_qe_history(
    history,
    run_id: str,
    logs_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, str]:
    """Save epoch-wise QE history as JSON, CSV, and PNG."""

    convergence_dir = (
        figures_dir
        / "convergence"
    )

    convergence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        logs_dir
        / f"{run_id}_qe_history.json"
    )

    csv_path = (
        tables_dir
        / f"{run_id}_qe_history.csv"
    )

    figure_path = (
        convergence_dir
        / f"{run_id}_qe_curve.png"
    )

    payload = history.to_dict()

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "quantization_error",
                "neighborhood_radius",
            ]
        )

        writer.writerows(
            zip(
                history.epoch,
                history.quantization_error,
                history.neighborhood_radius,
            )
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history.epoch,
        history.quantization_error,
        marker="o",
        markersize=2,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "NNSOM-style quantization error"
    )

    ax.set_title(
        "SOM Quantization Error Across Training"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = (REPO_ROOT / config_path).resolve()

    config = load_config(str(config_path))

    # --grid-size overrides the YAML values (e.g. --grid-size 10x10)
    if args.grid_size is not None:
        try:
            h_str, w_str = args.grid_size.lower().split("x")
            grid_override_h = int(h_str)
            grid_override_w = int(w_str)
        except ValueError:
            raise ValueError(
                f"--grid-size must be in HxW format (e.g. 10x10), got: '{args.grid_size}'"
            )
        if grid_override_h <= 0 or grid_override_w <= 0:
            raise ValueError(
                f"--grid-size dimensions must be positive integers, got: {args.grid_size}"
            )
        original_h = int(config.som.grid_height)
        original_w = int(config.som.grid_width)
        config.som.grid_height = grid_override_h
        config.som.grid_width = grid_override_w
        print(
            f"[grid-size override] {grid_override_h}x{grid_override_w} "
            f"(config default was {original_h}x{original_w})"
        )

    if args.smoke_test:
        run_id = "smoke_test"
    else:
        if args.run_id is None:
            raise ValueError(
                "Full scientific SOM runs require --run-id."
            )

        run_id = validate_run_id(
            args.run_id
        )

    requested_backend = str(
        config.som.backend
    ).lower()

    if requested_backend not in {
        "auto",
        "cpu",
        "gpu",
    }:
        raise ValueError(
            "som.backend must be one of: "
            "'auto', 'cpu', or 'gpu'."
        )

    if (
        config.preprocessing.fit_split
        != "train"
    ):
        raise ValueError(
            "Leakage protection violation: "
            "preprocessing.fit_split must be 'train'."
        )

    if (
        config.evaluation.use_test_for_selection
    ):
        raise ValueError(
            "Leakage protection violation: "
            "test data cannot be used for selection."
        )

    def resolve_path(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    embeddings_dir = resolve_embeddings_dir(
        config.paths.embeddings_dir,
        REPO_ROOT,
    )

    output_dir = resolve_path(config.paths.output_dir)
    models_dir = resolve_path(config.paths.som_models_dir)
    logs_dir = resolve_path(config.paths.logs_dir)
    figures_dir = resolve_path(config.paths.figures_dir)
    tables_dir = resolve_path(config.paths.tables_dir)

    for directory in (
        output_dir,
        models_dir,
        logs_dir,
        figures_dir,
        tables_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if args.smoke_test:
        metrics_path = (
            logs_dir
            / "som_smoke_test_metrics.json"
        )
        model_path = None
    else:
        (
            metrics_path,
            model_path,
        ) = preflight_scientific_outputs(
            run_id=run_id,
            logs_dir=logs_dir,
            models_dir=models_dir,
            overwrite=args.overwrite,
        )

    print("=" * 70)
    print("MNIST v1 NNSOM Training")
    print("=" * 70)

    print(
        f"Config:           {args.config}"
    )

    print(
        f"Embedding source: {embeddings_dir}"
    )

    print(
        f"SOM output:       {output_dir}"
    )

    data = load_som_data(
        embeddings_dir=embeddings_dir,
        expected_feature_dim=84,
    )

    print("\nValidated LeNet-5 embeddings:")
    print(
        f"  Train: {data.train.features.shape}"
    )
    print(
        f"  Val:   {data.val.features.shape}"
    )
    print(
        f"  Test:  {data.test.features.shape}"
    )

    x_train = data.train.features
    x_val = data.val.features

    grid_height = int(
        config.som.grid_height
    )
    grid_width = int(
        config.som.grid_width
    )
    init_neighborhood = int(
        config.som.init_neighborhood
    )
    epochs = int(
        config.som.epochs
    )
    steps = int(
        config.som.steps
    )
    seed = int(
        config.som.random_seed
    )

    track_history = bool(
        config.som.track_qe_history
    )

    history_every = int(
        config.som.qe_history_every
    )

    if args.smoke_test:
        print("\nSMOKE TEST MODE")
        print(
            "These values are NOT scientific results."
        )

        x_train = deterministic_subset(
            x_train,
            n_samples=2000,
            seed=seed,
        )

        x_val = deterministic_subset(
            x_val,
            n_samples=500,
            seed=seed + 1,
        )

        grid_height = 5
        grid_width = 5
        epochs = 2
        steps = 10

    print("\nTraining configuration:")
    print(
        f"  Grid:              "
        f"{grid_height}x{grid_width}"
    )
    print(
        f"  Neurons:           "
        f"{grid_height * grid_width}"
    )
    print(
        f"  init_neighborhood: "
        f"{init_neighborhood}"
    )
    print(
        f"  Epochs:            {epochs}"
    )
    print(
        f"  Steps:             {steps}"
    )
    print(
        f"  Seed:              {seed}"
    )
    print(
        f"  QE history:        {track_history}"
    )

    if track_history:
        (
            som,
            training_history,
        ) = train_som_with_history(
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

    actual_backend = get_nnsom_backend(
        som
    )

    print(
        f"Requested SOM backend: "
        f"{requested_backend}"
    )

    print(
        f"Actual NNSOM backend: "
        f"{actual_backend}"
    )

    print("\nSOM training completed.")

    print(
        "\nEvaluating training split..."
    )

    train_metrics = evaluate_som_quality(
        som,
        x_train,
    )

    print(
        "Evaluating validation split..."
    )

    val_metrics = evaluate_som_quality(
        som,
        x_val,
    )

    history_artifacts = None

    if training_history is not None:
        history_artifacts = save_qe_history(
            history=training_history,
            run_id=run_id,
            logs_dir=logs_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )

    metrics = {
        "experiment": {
            "run_id": run_id,
            "run": config.run.name,
            "stage": config.run.stage,
            "input_run": config.run.input_run,
            "smoke_test": bool(
                args.smoke_test
            ),
        },
        "som": {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_neurons": (
                grid_height
                * grid_width
            ),
            "init_neighborhood": (
                init_neighborhood
            ),
            "epochs": epochs,
            "steps": steps,
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "random_seed": seed,
            "track_qe_history": (
                track_history
            ),
            "qe_history_every": (
                history_every
            ),
        },
        "preprocessing": {
            "scaler": (
                config.preprocessing.scaler
            ),
            "feature_range": list(
                config.preprocessing.feature_range
            ),
            "fit_split": (
                config.preprocessing.fit_split
            ),
        },
        "data": {
            "train_shape": list(
                x_train.shape
            ),
            "validation_shape": list(
                x_val.shape
            ),
            "held_out_test_shape": list(
                data.test.features.shape
            ),
        },
        "train_metrics": (
            train_metrics.to_dict()
        ),
        "validation_metrics": (
            val_metrics.to_dict()
        ),
        "qe_history_artifacts": (
            history_artifacts
        ),
        "test_evaluated": False,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    print("\nSOM quality metrics:")

    print(
        "  Train QE:                  "
        f"{train_metrics.quantization_error:.6f}"
    )

    print(
        "  Train TE 1st-order (%):    "
        f"{train_metrics.topological_error_1st_order_pct:.4f}"
    )

    print(
        "  Train TE 1st+2nd (%):      "
        f"{train_metrics.topological_error_1st_2nd_order_pct:.4f}"
    )

    print(
        "  Train occupancy:           "
        f"{train_metrics.occupancy_rate:.4f}"
    )

    print(
        "  Val QE:                    "
        f"{val_metrics.quantization_error:.6f}"
    )

    print(
        "  Val TE 1st-order (%):      "
        f"{val_metrics.topological_error_1st_order_pct:.4f}"
    )

    print(
        "  Val TE 1st+2nd (%):        "
        f"{val_metrics.topological_error_1st_2nd_order_pct:.4f}"
    )

    print(
        "  Val occupancy:             "
        f"{val_metrics.occupancy_rate:.4f}"
    )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    if (
        not args.smoke_test
    ):
        if model_path is None:
            raise RuntimeError(
                "Scientific model path was not "
                "initialized."
            )

        som.save_pickle(
            run_id,
            str(models_dir) + os.sep,
        )

        if not model_path.is_file():
            raise RuntimeError(
                "NNSOM model save failed: "
                f"{model_path}"
            )

        print(
            f"SOM model saved to: {model_path}"
        )

    print("\nTest split was NOT evaluated.")
    print(
        "It remains reserved for final "
        "unbiased evaluation."
    )

    print(
        "\nSOM pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
