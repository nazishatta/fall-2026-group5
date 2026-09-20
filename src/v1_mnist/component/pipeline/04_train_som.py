"""Week 3 pipeline: train and evaluate NNSOM on frozen Week 2 embeddings."""

import argparse
import json
import sys
import os
import re
from pathlib import Path

import numpy as np


# Allow imports from repository root when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.component.som.data import load_som_data
from src.component.som.metrics import evaluate_som_quality
from src.component.som.trainer import train_som
from src.component.utils.config import load_config


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train Week 3 NNSOM on frozen Week 2 MNIST embeddings."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/component/configs/week_3_som.yaml",
        help="Path to Week 3 SOM configuration.",
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
            "full Week 3 scientific experiment."
        ),
    )

    return parser.parse_args()


def validate_run_id(run_id: str) -> str:
    """Validate a filesystem-safe scientific experiment identifier."""

    if not run_id:
        raise ValueError("run_id must be a non-empty string.")

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError(
            "run_id may contain only letters, numbers, '.', '_', and '-', "
            "and must begin with a letter or number."
        )

    return run_id


def preflight_scientific_outputs(
    run_id: str,
    logs_dir: Path,
    models_dir: Path,
) -> tuple[Path, Path]:
    """Reject duplicate scientific run IDs before expensive SOM training."""

    metrics_path = logs_dir / f"{run_id}_metrics.json"
    model_path = models_dir / run_id

    existing_paths = [
        path
        for path in (metrics_path, model_path)
        if path.exists()
    ]

    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)

        raise FileExistsError(
            f"Scientific run_id '{run_id}' already has existing "
            f"artifact(s): {existing}. Use a new --run-id instead of "
            f"overwriting an existing experiment."
        )

    return metrics_path, model_path


def deterministic_subset(
    x: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic random subset without replacement."""

    if n_samples >= len(x):
        return x

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(x),
        size=n_samples,
        replace=False,
    )

    return x[indices]


def main():
    """Run Week 3 NNSOM training and development evaluation."""

    args = parse_args()

    config = load_config(args.config)

    if args.smoke_test:
        run_id = "smoke_test"
    else:
        if args.run_id is None:
            raise ValueError(
                "Full scientific SOM runs require --run-id so artifacts "
                "cannot silently overwrite previous experiments."
            )

        run_id = validate_run_id(args.run_id)

    # ------------------------------------------------------------------
    # Scientific and implementation safeguards
    # ------------------------------------------------------------------

    if config.som.backend != "numpy":
        raise ValueError(
            "Current Week 3 implementation supports only "
            "backend='numpy' via NNSOM.plots.SOMPlots."
        )

    if config.preprocessing.fit_split != "train":
        raise ValueError(
            "Leakage protection violation: preprocessing.fit_split "
            "must be 'train'."
        )

    if config.evaluation.use_test_for_selection:
        raise ValueError(
            "Leakage protection violation: test data cannot be used "
            "for SOM hyperparameter selection."
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    input_run = config.run.input_run

    embeddings_dir = (
        REPO_ROOT
        / "outputs"
        / input_run
        / "embeddings"
        / "mnist"
    )

    output_dir = Path(config.paths.output_dir)
    models_dir = Path(config.paths.som_models_dir)
    logs_dir = Path(config.paths.logs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Artifact preflight
    # ------------------------------------------------------------------

    if args.smoke_test:
        metrics_path = logs_dir / "som_smoke_test_metrics.json"
        model_path = None
    else:
        metrics_path, model_path = preflight_scientific_outputs(
            run_id=run_id,
            logs_dir=logs_dir,
            models_dir=models_dir,
        )

    # ------------------------------------------------------------------
    # Run information
    # ------------------------------------------------------------------

    print("=" * 70)
    print("Week 3 NNSOM Training")
    print("=" * 70)

    print(f"Config:           {args.config}")
    print(f"Embedding source: {embeddings_dir}")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Load and validate frozen Week 2 embeddings
    # ------------------------------------------------------------------

    data = load_som_data(
        embeddings_dir=embeddings_dir,
        expected_feature_dim=84,
    )

    print("\nValidated Week 2 embeddings:")
    print(f"  Train: {data.train.features.shape}")
    print(f"  Val:   {data.val.features.shape}")
    print(f"  Test:  {data.test.features.shape}")
    print(
        "  Scaler fitted on training features only: "
        f"{data.scaler.n_features_in_} dimensions"
    )

    x_train = data.train.features
    x_val = data.val.features

    # ------------------------------------------------------------------
    # SOM configuration
    # ------------------------------------------------------------------

    grid_height = int(config.som.grid_height)
    grid_width = int(config.som.grid_width)
    init_neighborhood = int(config.som.init_neighborhood)
    epochs = int(config.som.epochs)
    steps = int(config.som.steps)
    seed = int(config.som.random_seed)

    # ------------------------------------------------------------------
    # Smoke-test configuration
    # ------------------------------------------------------------------

    if args.smoke_test:
        print("\nSMOKE TEST MODE")
        print("These values are NOT scientific experiment results.")

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

        print(f"  Train subset: {x_train.shape}")
        print(f"  Val subset:   {x_val.shape}")
        print(f"  Grid:         {grid_height}x{grid_width}")
        print(f"  Epochs:       {epochs}")
        print(f"  Steps:        {steps}")

    # ------------------------------------------------------------------
    # Train SOM
    # ------------------------------------------------------------------

    print("\nTraining configuration:")
    print(f"  Grid:              {grid_height}x{grid_width}")
    print(f"  Neurons:           {grid_height * grid_width}")
    print(f"  init_neighborhood: {init_neighborhood}")
    print(f"  Epochs:            {epochs}")
    print(f"  Steps:             {steps}")
    print(f"  Seed:              {seed}")

    som = train_som(
        x_train=x_train,
        grid_height=grid_height,
        grid_width=grid_width,
        init_neighborhood=init_neighborhood,
        epochs=epochs,
        steps=steps,
        norm_func=data.scaler.transform,
        seed=seed,
    )

    print("\nSOM training completed.")

    # ------------------------------------------------------------------
    # Development evaluation
    # ------------------------------------------------------------------

    print("\nEvaluating training split...")

    train_metrics = evaluate_som_quality(
        som,
        x_train,
    )

    print("Evaluating validation split...")

    val_metrics = evaluate_som_quality(
        som,
        x_val,
    )

    # ------------------------------------------------------------------
    # Machine-readable experiment record
    # ------------------------------------------------------------------

    metrics = {
        "experiment": {
            "run_id": run_id,
            "run": config.run.name,
            "stage": config.run.stage,
            "input_run": config.run.input_run,
            "smoke_test": bool(args.smoke_test),
        },
        "som": {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_neurons": grid_height * grid_width,
            "init_neighborhood": init_neighborhood,
            "epochs": epochs,
            "steps": steps,
            "backend": config.som.backend,
            "random_seed": seed,
        },
        "preprocessing": {
            "scaler": config.preprocessing.scaler,
            "feature_range": list(
                config.preprocessing.feature_range
            ),
            "fit_split": config.preprocessing.fit_split,
        },
        "data": {
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_val.shape),
            "held_out_test_shape": list(
                data.test.features.shape
            ),
        },
        "train_metrics": train_metrics.to_dict(),
        "validation_metrics": val_metrics.to_dict(),
        "test_evaluated": False,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Console metric summary
    # ------------------------------------------------------------------

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
        "  Train occupied neurons:    "
        f"{train_metrics.occupied_neurons}/"
        f"{train_metrics.total_neurons}"
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
        "  Val occupied neurons:      "
        f"{val_metrics.occupied_neurons}/"
        f"{val_metrics.total_neurons}"
    )

    print(f"\nMetrics saved to: {metrics_path}")

    # ------------------------------------------------------------------
    # Save scientific SOM model
    # ------------------------------------------------------------------

    # Smoke-test models are intentionally not saved as experiment models.
    if not args.smoke_test:
        model_name = run_id

        if model_path is None:
            raise RuntimeError(
                "Scientific model path was not initialized during preflight."
            )

        som.save_pickle(
            model_name,
            str(models_dir) + os.sep,
        )

        if not model_path.is_file():
            raise RuntimeError(
                f"NNSOM model save failed: expected artifact not found at "
                f"{model_path}"
            )

        print(
            f"SOM model saved to: {model_path}"
        )

    # ------------------------------------------------------------------
    # Test-set protection
    # ------------------------------------------------------------------

    print("\nTest split was NOT evaluated.")
    print(
        "It remains reserved for final unbiased evaluation "
        "and was not used for SOM model selection."
    )

    print("\nWeek 3 SOM pipeline completed successfully.")


if __name__ == "__main__":
    main()