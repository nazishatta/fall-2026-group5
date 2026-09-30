from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

from src.v1_mnist.component.utils.logging import get_logger

logger = get_logger("v1_mnist.evaluation")


def evaluate_model(
    model: Any,
    dataloader: Any,
    device: Any,
) -> dict[str, Any]:
    """Evaluate PyTorch classification model on given DataLoader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    acc = float(np.mean(all_preds_arr == all_labels_arr))
    clf_report_str = classification_report(all_labels_arr, all_preds_arr, zero_division=0)
    clf_report_dict = classification_report(
        all_labels_arr, all_preds_arr, output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels_arr, all_preds_arr)

    return {
        "accuracy": acc,
        "classification_report_str": clf_report_str,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix": conf_matrix,
        "predictions": all_preds_arr,
        "true_labels": all_labels_arr,
    }


def print_evaluation_summary(
    eval_results: Mapping[str, Any],
    split_name: str,
) -> None:
    """Log formatted classification accuracy and report."""
    logger.info("--- Evaluation Summary: %s ---", split_name.upper())
    logger.info("Accuracy: %.4f", eval_results["accuracy"])
    logger.info("Classification Report:\n%s", eval_results["classification_report_str"])
