from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v1_mnist.component.data.mnist_dataset import get_dataloaders
from src.v1_mnist.component.evaluation.metrics import (
    evaluate_model,
    print_evaluation_summary,
)
from src.v1_mnist.component.models.lenet5 import LeNet5
from src.v1_mnist.component.utils.config import get_device, load_config, set_seed
from src.v1_mnist.component.utils.logging import get_logger, setup_logger
from src.v1_mnist.component.visualization.plots import (
    plot_confusion_matrix,
    plot_training_curves,
)


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
    set_seed(config.training.seed)
    device = get_device(config)

    output_dir = Path(getattr(config.paths, "output_dir", "./outputs/v1_mnist/cnn_baseline"))
    logs_dir = Path(getattr(config.paths, "logs_dir", output_dir / "logs"))
    log_file = logs_dir / "train_baseline.log"
    logger = setup_logger("train_baseline", log_file=log_file)

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

    checkpoints_dir = Path(getattr(config.paths, "checkpoints_dir", output_dir / "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoints_dir / "lenet5_best.pth"

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

    figures_dir = Path(getattr(config.paths, "figures_dir", output_dir / "figures"))
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        figures_dir / "training_curves.svg",
    )
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        [str(i) for i in range(10)],
        "Test Set Confusion Matrix",
        figures_dir / "test_confusion_matrix.svg",
    )

    eval_output = {
        "accuracy": test_results["accuracy"],
        "classification_report": test_results["classification_report_dict"],
    }
    metrics_dir = Path(getattr(config.paths, "metrics_dir", output_dir / "metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_eval_path = metrics_dir / "test_evaluation.json"
    with open(test_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=4)

    logger.info("Test results saved to %s", test_eval_path)
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
