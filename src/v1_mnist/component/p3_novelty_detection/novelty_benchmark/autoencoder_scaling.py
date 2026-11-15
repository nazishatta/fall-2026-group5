"""Controlled check of what drives the autoencoder and Isolation Forest AUROC on SVHN.

Varies one thing at a time (training size and input scaling) and adds a
length-only control score. Pure numpy/scikit-learn: the autoencoder is passed in as
a function, so these helpers run and are tested without PyTorch.

Every score follows the project convention: higher means more novel. Nothing here
reverses a score after seeing SVHN results.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.unified_novelty_metrics import (
    evaluate,
    subset_fingerprint,
    subset_indices,
)

SCALINGS = ("raw", "standardized")
MIN_SEEDS = 5
AutoencoderFitter = Callable[
    [NDArray[np.float32], NDArray[np.float32], dict[str, Any]],
    tuple[NDArray[np.float64], list[float]],
]


def check_design(sizes: list[int], scalings: list[str], seeds: list[int], train_rows: int) -> None:
    """Reject designs that are unsafe, duplicated, or too small to report spread."""
    if not sizes or len(set(sizes)) != len(sizes) or any(not 0 < size <= train_rows for size in sizes):
        raise ValueError("Training sizes must be unique and between 1 and the training set size")
    if not scalings or len(set(scalings)) != len(scalings) or any(item not in SCALINGS for item in scalings):
        raise ValueError(f"Scalings must be unique and drawn from {SCALINGS}")
    if len(seeds) < MIN_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(f"At least {MIN_SEEDS} unique seeds are required")


def prepare_inputs(
    subset: NDArray[np.floating],
    joined: NDArray[np.floating],
    scaling: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return (training, evaluation) inputs; the scaler is fitted on the training rows only."""
    if scaling not in SCALINGS:
        raise ValueError(f"Unknown scaling: {scaling}")
    train = np.asarray(subset, dtype=np.float64)
    both = np.asarray(joined, dtype=np.float64)
    if train.ndim != 2 or both.ndim != 2 or train.shape[1] != both.shape[1]:
        raise ValueError("Expected 2D arrays with the same number of columns")
    if scaling == "standardized":
        scaler = StandardScaler().fit(train)
        train, both = scaler.transform(train), scaler.transform(both)
    return train.astype(np.float32), both.astype(np.float32)


def forest_scores(
    subset: NDArray[np.floating],
    joined: NDArray[np.floating],
    settings: dict[str, Any],
    seed: int,
) -> NDArray[np.float64]:
    """Isolation Forest novelty scores on standardized input, as in the baseline runs.

    Trees split one column at a time, so scaling does not change which points
    isolate quickly; it is applied only to match the earlier runs.
    """
    scaler = StandardScaler().fit(np.asarray(subset, dtype=np.float64))
    forest = IsolationForest(
        n_estimators=int(settings["n_estimators"]),
        max_samples=int(settings["max_samples"]),
        contamination=float(settings["contamination"]),
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(scaler.transform(np.asarray(subset, dtype=np.float64)))
    return np.asarray(-forest.score_samples(scaler.transform(np.asarray(joined, dtype=np.float64))), dtype=np.float64)


def length_scores(features: NDArray[np.floating]) -> NDArray[np.float64]:
    """Embedding length (L2 norm). Control score: higher length counts as more novel."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite 2D array")
    return np.linalg.norm(values, axis=1)


def metric_row(
    method: str, size: int, scaling: str, seed: int,
    scores: NDArray[np.float64], id_count: int, target_tpr: float,
) -> dict[str, Any]:
    """Evaluate one detector on the joined [MNIST validation; SVHN] scores."""
    if len(scores) <= id_count:
        raise ValueError("Scores must cover the ID and OOD samples")
    return {
        "method": method, "training_size": size, "scaling": scaling, "seed": seed,
        **evaluate(scores[:id_count], scores[id_count:], target_tpr),
    }


def evaluate_conditions(
    train: NDArray[np.floating],
    joined: NDArray[np.floating],
    id_count: int,
    config: dict[str, Any],
    fit_autoencoder: AutoencoderFitter,
) -> list[dict[str, Any]]:
    """Fit the autoencoder in every size x scaling x seed cell and the forest per size x seed."""
    sizes = [int(item) for item in config["training_sizes"]]
    scalings = list(config["scalings"])
    seeds = [int(item) for item in config["seeds"]]
    check_design(sizes, scalings, seeds, len(train))
    target = float(config["target_tpr"])
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for size in sizes:
            indices = subset_indices(len(train), size, seed)
            subset = np.asarray(train)[indices]
            fingerprint = subset_fingerprint(indices)
            forest = forest_scores(subset, joined, config["isolation_forest"], seed)
            rows.append({**metric_row("isolation_forest", size, "standardized", seed, forest, id_count, target),
                         "training_subset_sha256": fingerprint})
            for scaling in scalings:
                train_in, joined_in = prepare_inputs(subset, joined, scaling)
                scores, losses = fit_autoencoder(
                    train_in, joined_in, {"seed": seed, "autoencoder": config["autoencoder"]}
                )
                rows.append({
                    **metric_row("autoencoder", size, scaling, seed, scores, id_count, target),
                    "training_subset_sha256": fingerprint,
                    "initial_training_loss": float(losses[0]),
                    "final_training_loss": float(losses[-1]),
                })
    return rows


def summarize_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean, sample SD and per-seed AUROC for every method x size x scaling cell."""
    groups: dict[tuple[str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["training_size"], row["scaling"])].append((row["seed"], row["auroc"]))
    summary = []
    for (method, size, scaling), pairs in sorted(groups.items()):
        pairs.sort()
        values = np.array([auroc for _, auroc in pairs], dtype=np.float64)
        if len(values) < MIN_SEEDS:
            raise ValueError("At least five seeds are required per cell")
        summary.append({
            "method": method, "training_size": size, "scaling": scaling,
            "seeds": [seed for seed, _ in pairs], "auroc_mean": float(values.mean()),
            "auroc_sample_sd": float(values.std(ddof=1)), "auroc_per_seed": values.tolist(),
        })
    return summary
