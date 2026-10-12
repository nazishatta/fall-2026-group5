from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.p1_model_training_baseline.data_pipeline.mnist_dataset import get_dataloaders
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.evaluation import (
    evaluate_model,
    print_evaluation_summary,
)
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.lenet5 import LeNet5
from src.v1_mnist.component.utils.config import get_device, load_config, set_seed
from src.v1_mnist.component.utils.logging import setup_logger
from src.v1_mnist.component.utils.stats import wilson_interval
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.cnn_plots import (
    plot_confusion_matrix,
    plot_training_curves,
)
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extract_embeddings import (
    sha256_file,
)
from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.run_ids import (
    check_new_training_run,
    cnn_run_paths,
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def train_baseline(
    config_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Train the LeNet-5 CNN baseline model on MNIST.

    Args:
        config_path: Path to configuration file.

    Returns:
        Evaluation results dictionary.
    """
    import torch
    import torch.nn as nn

    if config_path is None:
        config_path = (
            PROJECT_ROOT
            / "src"
            / "v1_mnist"
            / "component"
            / "configs"
            / "cnn_baseline.yaml"
        )

    config = load_config(str(config_path))

    # Never overwrite a run: stop before writing anything if this run ID has results.
    run_paths = cnn_run_paths(config)
    check_new_training_run(run_paths)

    set_seed(config.training.seed)
    device = get_device(config)

    logger = setup_logger("train_baseline", log_file=run_paths.train_log)

    logger.info("Run ID: %s (seed %d)", run_paths.run_id, config.training.seed)
    logger.info("Using device: %s", device)
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(config)

    model = LeNet5(num_classes=config.dataset.num_classes).to(device)
    logger.info(
        "Initialized LeNet-5 CNN (penultimate layer: %s, dim: %d)",
        model.feature_layer_name,
        model.feature_dim,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.step_size,
        gamma=config.training.gamma,
    )

    epochs = config.training.epochs
    best_val_acc = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_model_path = run_paths.checkpoint
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting baseline training for %d epochs...", epochs)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        scheduler.step()

        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f",
            epoch + 1,
            epochs,
            epoch_train_loss,
            epoch_train_acc,
            epoch_val_loss,
            epoch_val_acc,
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model checkpoint to %s", best_model_path)

    logger.info("Training completed. Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_results = evaluate_model(model, test_loader, device)
    print_evaluation_summary(test_results, "test")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        run_paths.training_curves.with_suffix(".pdf"),
    )
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        [str(i) for i in range(10)],
        "Test Set Confusion Matrix",
        run_paths.confusion_matrix.with_suffix(".pdf"),
    )

    confusion = test_results["confusion_matrix"]
    test_total = int(confusion.sum())
    test_correct = int(confusion.trace())
    ci_low, ci_high = wilson_interval(test_correct, test_total)
    logger.info("Test accuracy %.4f (%d/%d), 95%% Wilson interval [%.4f, %.4f]",
                test_results["accuracy"], test_correct, test_total, ci_low, ci_high)

    eval_output = {
        "run_id": run_paths.run_id,
        "accuracy": test_results["accuracy"],
        "test_images": test_total,
        "test_correct": test_correct,
        "accuracy_wilson_95": [ci_low, ci_high],
        "classification_report": test_results["classification_report_dict"],
    }
    test_eval_path = run_paths.test_metrics
    test_eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)
    logger.info("Test results saved to %s", test_eval_path)

    config_file = Path(config_path)
    run_record = {
        "run_id": run_paths.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": _relative(config_file),
        "config_sha256": sha256_file(config_file),
        "seed": config.training.seed,
        "epochs": epochs,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_results["accuracy"],
        "checkpoint": _relative(best_model_path),
        "checkpoint_sha256": sha256_file(best_model_path),
        "device": str(device),
        "library_versions": {name: _package_version(name) for name in ("torch", "torchvision", "numpy")},
        "git_commit": _git_commit(),
    }
    with open(run_paths.run_record, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=4)
    logger.info("Run record (seed, config and checkpoint fingerprints) saved to %s", run_paths.run_record)
    return eval_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeNet-5 Baseline on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            PROJECT_ROOT
            / "src"
            / "v1_mnist"
            / "component"
            / "configs"
            / "cnn_baseline.yaml"
        ),
        help="Path to config file (default: cnn_baseline.yaml)",
    )
    args = parser.parse_args()
    train_baseline(args.config)


if __name__ == "__main__":
    main()
