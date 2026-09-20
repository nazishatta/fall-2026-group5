"""Data loading and validation utilities for Week 3 SOM experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class EmbeddingSplit:
    """Validated embedding artifacts for one dataset split."""

    features: np.ndarray
    labels: np.ndarray
    preds: np.ndarray
    sample_ids: np.ndarray
    correct: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class SOMDataBundle:
    """Validated train/validation/test embeddings and train-fitted scaler."""

    train: EmbeddingSplit
    val: EmbeddingSplit
    test: EmbeddingSplit
    scaler: MinMaxScaler


def _load_split(
    embeddings_dir: Path,
    split: str,
) -> EmbeddingSplit:
    """Load all embedding artifacts associated with one split."""

    files = {
        "features": embeddings_dir / f"{split}_features.npy",
        "labels": embeddings_dir / f"{split}_labels.npy",
        "preds": embeddings_dir / f"{split}_preds.npy",
        "sample_ids": embeddings_dir / f"{split}_sample_ids.npy",
        "correct": embeddings_dir / f"{split}_correct.npy",
        "confidence": embeddings_dir / f"{split}_confidence.npy",
    }

    missing = [
        str(path)
        for path in files.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Week 2 embedding artifacts:\n"
            + "\n".join(missing)
        )

    return EmbeddingSplit(
        features=np.load(files["features"]),
        labels=np.load(files["labels"]),
        preds=np.load(files["preds"]),
        sample_ids=np.load(files["sample_ids"]),
        correct=np.load(files["correct"]),
        confidence=np.load(files["confidence"]),
    )


def _validate_split(
    split_name: str,
    data: EmbeddingSplit,
    expected_feature_dim: int | None = None,
) -> None:
    """Validate shape, alignment, uniqueness and numerical integrity."""

    if data.features.ndim != 2:
        raise ValueError(
            f"{split_name}: features must be 2D "
            f"(samples, features), got {data.features.shape}"
        )

    n_samples, feature_dim = data.features.shape

    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"{split_name}: expected feature dimension "
            f"{expected_feature_dim}, got {feature_dim}"
        )

    arrays = {
        "labels": data.labels,
        "preds": data.preds,
        "sample_ids": data.sample_ids,
        "correct": data.correct,
        "confidence": data.confidence,
    }

    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(
                f"{split_name}: {name} must be 1D, "
                f"got shape {array.shape}"
            )

        if len(array) != n_samples:
            raise ValueError(
                f"{split_name}: {name} has {len(array)} rows "
                f"but features contain {n_samples}"
            )

    if not np.isfinite(data.features).all():
        raise ValueError(
            f"{split_name}: features contain NaN or infinite values"
        )

    if not np.isfinite(data.confidence).all():
        raise ValueError(
            f"{split_name}: confidence contains NaN or infinite values"
        )

    if len(np.unique(data.sample_ids)) != n_samples:
        raise ValueError(
            f"{split_name}: sample_ids are not unique"
        )

    expected_correct = data.preds == data.labels

    if not np.array_equal(
        data.correct.astype(bool),
        expected_correct,
    ):
        raise ValueError(
            f"{split_name}: stored correctness flags do not match "
            "predictions and labels"
        )


def load_som_data(
    embeddings_dir: str | Path,
    expected_feature_dim: int = 84,
) -> SOMDataBundle:
    """
    Load validated Week 2 embeddings and fit preprocessing on training only.

    The scaler is fitted exclusively on training embeddings to prevent
    validation/test leakage.
    """

    embeddings_dir = Path(embeddings_dir)

    train = _load_split(embeddings_dir, "train")
    val = _load_split(embeddings_dir, "val")
    test = _load_split(embeddings_dir, "test")

    _validate_split(
        "train",
        train,
        expected_feature_dim=expected_feature_dim,
    )
    _validate_split(
        "val",
        val,
        expected_feature_dim=expected_feature_dim,
    )
    _validate_split(
        "test",
        test,
        expected_feature_dim=expected_feature_dim,
    )

    if not (
        train.features.shape[1]
        == val.features.shape[1]
        == test.features.shape[1]
    ):
        raise ValueError(
            "Feature dimensions differ across train/val/test splits"
        )

    scaler = MinMaxScaler(
        feature_range=(-1.0, 1.0)
    )

    # Critical leakage rule:
    # fit ONLY on training embeddings.
    scaler.fit(train.features)

    return SOMDataBundle(
        train=train,
        val=val,
        test=test,
        scaler=scaler,
    )