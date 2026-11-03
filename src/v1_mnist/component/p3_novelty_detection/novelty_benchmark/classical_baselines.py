"""MNIST classical novelty baselines on CNN embeddings."""

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def fit_classical_baselines(
    train_features: NDArray[np.float32],
    validation_features: NDArray[np.float32],
    config: dict,
) -> dict[str, NDArray[np.float64]]:
    """Fit on training embeddings and score validation samples."""
    train = np.asarray(train_features, dtype=np.float64)
    validation = np.asarray(validation_features, dtype=np.float64)

    if train.ndim != 2 or validation.ndim != 2:
        raise ValueError("Expected 2D feature arrays")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("Feature dimensions must match")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Features contain non-finite values")

    seed = int(config["seed"])
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    validation_scaled = scaler.transform(validation)

    forest_config = config["isolation_forest"]
    forest = IsolationForest(
        n_estimators=int(forest_config["n_estimators"]),
        max_samples=int(forest_config["max_samples"]),
        contamination=float(forest_config["contamination"]),
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(train_scaled)

    svm_config = config["one_class_svm"]
    rng = np.random.default_rng(seed)
    sample_count = min(int(svm_config["max_training_samples"]), len(train_scaled))
    selected = rng.choice(len(train_scaled), size=sample_count, replace=False)

    svm = OneClassSVM(
        kernel=str(svm_config["kernel"]),
        nu=float(svm_config["nu"]),
        gamma=svm_config["gamma"],
    )
    svm.fit(train_scaled[selected])

    return {
        "isolation_forest": np.asarray(
            -forest.score_samples(validation_scaled), dtype=np.float64
        ),
        "one_class_svm": np.asarray(
            -svm.decision_function(validation_scaled), dtype=np.float64
        ),
    }
