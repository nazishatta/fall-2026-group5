"""Quality metrics for Week 3 NNSOM experiments."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class SOMQualityMetrics:
    """Serializable summary of SOM quality for one data split."""

    n_samples: int
    quantization_error: float
    topological_error_1st_order_pct: float
    topological_error_1st_2nd_order_pct: float
    occupied_neurons: int
    empty_neurons: int
    total_neurons: int
    occupancy_rate: float
    mean_hits_per_occupied_neuron: float
    max_hits_per_neuron: int

    def to_dict(self) -> dict:
        """Convert metrics to a plain dictionary."""
        return asdict(self)


def calculate_topological_error_numpy(
    som,
    x: np.ndarray,
) -> tuple[float, float]:
    """Calculate NNSOM topological error safely on CPU.

    This reproduces the same topological-error definition used by
    NNSOM's NumPy SOM implementation while allowing the SOM itself
    to be trained with SOMGpu/CuPy.
    """

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D data, got shape {x.shape}"
        )

    # Apply the same preprocessing used during SOM training.
    if som.norm_func is not None:
        x_scaled = som.norm_func(x)
    else:
        x_scaled = x

    weights = np.asarray(som.w, dtype=np.float64)

    neuron_dist = np.asarray(som.neuron_dist, dtype=np.float64)

    # weights:   (neurons, features)
    # x_scaled:  (samples, features)
    # result:    (neurons, samples)
    distance_matrix = cdist(weights, x_scaled, metric="euclidean")

    # Find the two closest SOM neurons for every sample.
    sorted_neurons = np.argsort(distance_matrix, axis=0)

    first_bmu = sorted_neurons[0, :]
    second_bmu = sorted_neurons[1, :]

    # Distance between first and second BMU in SOM topology.
    top_dist = neuron_dist[first_bmu, second_bmu]

    # NNSOM first-order TE: percentage with topological distance > 1.1
    topological_error_1st = 100.0 * np.mean(top_dist > 1.1)

    # NNSOM first + second-order TE: percentage with topological distance > 2.1
    topological_error_1st_2nd = 100.0 * np.mean(top_dist > 2.1)

    return (
        float(topological_error_1st),
        float(topological_error_1st_2nd),
    )


def evaluate_som_quality(
    som,
    x: np.ndarray,
) -> SOMQualityMetrics:
    """
    Evaluate a trained NNSOM model on one embedding split.

    Notes
    -----
    NNSOM quantization_error() expects the cluster_distances object
    returned by cluster_data(), not the raw feature matrix.

    Topological error is calculated with a NumPy/SciPy fallback
    because the current NNSOM GPU implementation mixes CuPy and
    NumPy during topological-error evaluation.
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
    ) = calculate_topological_error_numpy(
        som,
        x,
    )

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

    max_hits = (
        int(sizes.max())
        if sizes.size > 0
        else 0
    )

    return SOMQualityMetrics(
        n_samples=int(x.shape[0]),
        quantization_error=quantization_error,
        topological_error_1st_order_pct=float(
            topological_error_first
        ),
        topological_error_1st_2nd_order_pct=float(
            topological_error_first_second
        ),
        occupied_neurons=occupied_neurons,
        empty_neurons=empty_neurons,
        total_neurons=total_neurons,
        occupancy_rate=float(occupancy_rate),
        mean_hits_per_occupied_neuron=float(mean_hits),
        max_hits_per_neuron=max_hits,
    )
