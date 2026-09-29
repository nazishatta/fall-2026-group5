from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm


class FeatureExtractor:
    """Capture one module's output while retaining sample-level metadata."""

    def __init__(self, model: torch.nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self.features: list[np.ndarray] = []
        modules = dict(self.model.named_modules())
        if self.layer_name not in modules:
            raise KeyError(
                f"Layer '{self.layer_name}' not found. "
                f"Available layers: {list(modules.keys())}"
            )
        self._layer = modules[self.layer_name]

    def extract(
        self,
        dataloader: Any,
        device: torch.device,
        return_metadata: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Extract aligned features, labels, predictions, IDs, confidence, correctness."""
        self.model.eval()
        self.features = []
        labels_all: list[np.ndarray] = []
        predictions_all: list[np.ndarray] = []
        sample_ids_all: list[np.ndarray] = []
        confidence_all: list[np.ndarray] = []

        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.features.append(output.detach().cpu().numpy())

        hook_handle = self._layer.register_forward_hook(hook)
        generated_offset = 0

        try:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader,
                    desc=f"Extracting [{self.layer_name}]",
                    leave=False,
                ):
                    if len(batch) == 3:
                        inputs, labels, sample_ids = batch
                        sample_ids_np = sample_ids.cpu().numpy()
                    elif len(batch) == 2:
                        inputs, labels = batch
                        sample_ids_np = np.arange(
                            generated_offset,
                            generated_offset + len(labels),
                            dtype=np.int64,
                        )
                    else:
                        raise ValueError(
                            "Expected each batch to contain (inputs, labels) or "
                            "(inputs, labels, sample_ids)"
                        )

                    inputs = inputs.to(device)
                    logits = self.model(inputs)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predictions = probabilities.max(dim=1)

                    labels_all.append(labels.cpu().numpy())
                    predictions_all.append(predictions.cpu().numpy())
                    confidence_all.append(confidence.cpu().numpy())
                    sample_ids_all.append(sample_ids_np)
                    generated_offset += len(labels)
        finally:
            hook_handle.remove()

        if not self.features:
            raise RuntimeError(
                f"No features captured from '{self.layer_name}'; "
                "ensure the dataloader is not empty"
            )

        features = np.concatenate(self.features, axis=0)
        labels = np.concatenate(labels_all).astype(np.int64, copy=False)
        predictions = np.concatenate(predictions_all).astype(np.int64, copy=False)
        sample_ids = np.concatenate(sample_ids_all).astype(np.int64, copy=False)
        confidence = np.concatenate(confidence_all).astype(np.float32, copy=False)
        correct = predictions == labels

        row_counts = {
            len(features),
            len(labels),
            len(predictions),
            len(sample_ids),
            len(confidence),
            len(correct),
        }
        if len(row_counts) != 1:
            raise RuntimeError("Extracted arrays are not row-aligned")

        if return_metadata:
            return features, labels, predictions, sample_ids, confidence, correct
        return features, labels, predictions


def save_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    split_name: str,
    output_dir: str | Path,
    sample_ids: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    correct: Optional[np.ndarray] = None,
) -> dict[str, Path]:
    """Save extracted arrays into numpy binary format."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": features,
        "labels": labels,
        "preds": preds,
        "sample_ids": sample_ids,
        "confidence": confidence,
        "correct": correct,
    }
    saved_paths: dict[str, Path] = {}
    for suffix, array in arrays.items():
        if array is not None:
            p = out_dir / f"{split_name}_{suffix}.npy"
            np.save(str(p), array)
            saved_paths[suffix] = p
    return saved_paths


def load_embeddings(
    split_name: str,
    input_dir: str | Path,
    include_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Load extracted embeddings and labels."""
    in_dir = Path(input_dir)
    features = np.load(str(in_dir / f"{split_name}_features.npy"))
    labels = np.load(str(in_dir / f"{split_name}_labels.npy"))
    predictions = np.load(str(in_dir / f"{split_name}_preds.npy"))
    if not include_metadata:
        return features, labels, predictions

    sample_ids = np.load(str(in_dir / f"{split_name}_sample_ids.npy"))
    confidence = np.load(str(in_dir / f"{split_name}_confidence.npy"))
    correct = np.load(str(in_dir / f"{split_name}_correct.npy"))
    return features, labels, predictions, sample_ids, confidence, correct
