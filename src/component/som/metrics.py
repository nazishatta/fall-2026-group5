"""Quality metrics for Week 3 NNSOM experiments."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SOMQualityMetrics:
    """Serializable summary of SOM quality for one data split."""

    n_samples: int
    quantization_error: float
    topological_error_first: float
    topological_error_first_second: float
    occupied_neurons: int
    empty_neurons: int
    total_neurons: int
    occupancy_rate: float
    mean_hits_per_occupied_neuron: float
    max_hits_per_neuron: int

    def to_dict(self) -> dict:
        """Convert metrics to a plain dictionary."""
        return asdict(self)


def evaluate_som_quality(
    som,
    x: np.ndarray,
) -> SOMQualityMetrics:
    """
    Evaluate a trained NNSOM model on one embedding split.

    NNSOM quantization_error() expects the cluster_distances returned
    by cluster_data(), not the raw feature matrix.

    NNSOM topological_error() returns two values.
    """

    if x.ndim != 2:
        raise ValueError(
            f"SOM evaluation data must be 2D, got shape {x.shape}"
        )

    (
        _clusters,
        cluster_distances,
        _max_cluster_distances,
        cluster_sizes,
    ) = som.cluster_data(x)

    sizes = np.asarray(cluster_sizes, dtype=np.int64)

    assigned_samples = int(sizes.sum())

    if assigned_samples != x.shape[0]:
        raise RuntimeError(
            "NNSOM cluster assignment count mismatch: "
            f"expected {x.shape[0]}, got {assigned_samples}"
        )

    quantization_error = float(
        som.quantization_error(cluster_distances)
    )

    (
        topological_error_first,
        topological_error_first_second,
    ) = som.topological_error(x)

    total_neurons = int(som.numNeurons)
    occupied_neurons = int(np.count_nonzero(sizes))
    empty_neurons = total_neurons - occupied_neurons

    occupancy_rate = (
        occupied_neurons / total_neurons
        if total_neurons > 0
        else 0.0
    )

    mean_hits = (
        assigned_samples / occupied_neurons
        if occupied_neurons > 0
        else 0.0
    )

    max_hits = int(sizes.max()) if sizes.size > 0 else 0

    return SOMQualityMetrics(
        n_samples=int(x.shape[0]),
        quantization_error=quantization_error,
        topological_error_first=float(
            topological_error_first
        ),
        topological_error_first_second=float(
            topological_error_first_second
        ),
        occupied_neurons=occupied_neurons,
        empty_neurons=empty_neurons,
        total_neurons=total_neurons,
        occupancy_rate=float(occupancy_rate),
        mean_hits_per_occupied_neuron=float(mean_hits),
        max_hits_per_neuron=max_hits,
    )
