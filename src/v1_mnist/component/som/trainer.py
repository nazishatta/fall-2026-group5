"""Training helpers for Week 3 NNSOM experiments.

The standard ``train_som`` wrapper delegates training to NNSOM unchanged.

``train_som_with_history`` mirrors the NumPy batch-training loop used by
NNSOM 1.8.x, while recording NNSOM-style quantization error after each epoch.
It exists because NNSOM's public ``train`` method does not expose an
epoch callback/history object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from NNSOM.plots import SOMPlots
from NNSOM.som import SOM

try:
    from NNSOM.som_gpu import SOMGpu
except ImportError:
    SOMGpu = None
from scipy.spatial.distance import cdist

from src.v1_mnist.component.som.memory_safe_init import economy_svd_for_nnsom_init


def create_som(
    dimensions,
    backend: str = "auto",
):
    """Create the requested NNSOM implementation."""

    backend = str(backend).lower()

    if backend == "auto":
        # NNSOM itself decides:
        # CuPy available -> SOMGpu
        # otherwise      -> SOM
        # SOMPlots also provides the plotting API.
        return SOMPlots(
            dimensions=dimensions
        )

    if backend == "cpu":
        return SOM(
            dimensions=dimensions
        )

    if backend == "gpu":
        if SOMGpu is None:
            raise RuntimeError(
                "GPU backend was requested, but SOMGpu/CuPy "
                "is not available in this environment."
            )

        return SOMGpu(
            dimensions=dimensions
        )

    raise ValueError(
        "som.backend must be one of: "
        "'auto', 'cpu', or 'gpu'."
    )


def get_nnsom_backend(
    som,
) -> str:
    """Report the NNSOM implementation currently being used."""

    class_names = {
        cls.__name__
        for cls in type(som).__mro__
    }

    if "SOMGpu" in class_names:
        return "gpu"

    if "SOM" in class_names:
        return "cpu"

    return "unknown"


@dataclass(frozen=True)
class SOMTrainingHistory:
    """Serializable convergence history for one SOM training run."""

    epoch: list[int]
    quantization_error: list[float]
    neighborhood_radius: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _nnsom_style_quantization_error(
    distance_matrix: np.ndarray,
) -> float:
    """Reproduce NNSOM's quantization_error(cluster_distances) definition.

    NNSOM computes the mean BMU distance inside each neuron, uses zero for
    empty neurons, then averages those per-neuron means.
    """

    if distance_matrix.ndim != 2:
        raise ValueError(
            "distance_matrix must have shape (neurons, samples)."
        )

    num_neurons = distance_matrix.shape[0]
    winners = np.argmin(distance_matrix, axis=0)

    per_neuron_mean = np.zeros(
        num_neurons,
        dtype=np.float64,
    )

    for neuron_id in range(num_neurons):
        sample_mask = winners == neuron_id

        if np.any(sample_mask):
            per_neuron_mean[neuron_id] = float(
                np.mean(
                    distance_matrix[
                        neuron_id,
                        sample_mask,
                    ]
                )
            )

    return float(np.mean(per_neuron_mean))


def _one_hot_winners(
    winners: np.ndarray,
    num_neurons: int,
) -> np.ndarray:
    """Create the neuron x sample winner matrix used by NNSOM."""

    outputs = np.zeros(
        (num_neurons, winners.size),
        dtype=np.float64,
    )

    outputs[
        winners,
        np.arange(winners.size),
    ] = 1.0

    return outputs


def train_som(
    x_train,
    grid_height,
    grid_width,
    init_neighborhood,
    epochs,
    steps,
    norm_func,
    seed,
    backend="auto",
):
    """Train NNSOM using auto, explicit CPU, or explicit GPU mode."""

    np.random.seed(seed)

    som = create_som(
        dimensions=(
            grid_height,
            grid_width,
        ),
        backend=backend,
    )

    with economy_svd_for_nnsom_init():
        som.init_w(
            x_train,
            norm_func=norm_func,
        )

    som.train(
        x_train,
        init_neighborhood=init_neighborhood,
        epochs=epochs,
        steps=steps,
        norm_func=norm_func,
    )

    return som


def train_som_with_history(
    x_train,
    grid_height,
    grid_width,
    init_neighborhood,
    epochs,
    steps,
    norm_func,
    seed,
    history_every=1,
):
    """Optional CPU diagnostic for epoch-wise QE history.

    This function mirrors the NumPy NNSOM batch-training loop in order
    to record epoch-wise quantization error.

    It is not used for the main saved SOM. The main pipeline uses
    NNSOM's native SOM / SOMGpu / SOMPlots training implementation.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if history_every <= 0:
        raise ValueError("history_every must be positive.")

    np.random.seed(seed)

    som = SOMPlots(
        dimensions=(grid_height, grid_width)
    )

    with economy_svd_for_nnsom_init():
        som.init_w(
            x_train,
            norm_func=norm_func,
        )

    # NNSOM converts (samples, features) to (features, samples)
    # through som.normalize().
    x_internal = som.normalize(
        x_train,
        norm_func=norm_func,
    )

    weights = np.asarray(
        som.w,
        dtype=np.float64,
    ).copy()

    num_neurons = weights.shape[0]
    num_samples = x_internal.shape[1]

    if x_internal.shape[0] != weights.shape[1]:
        raise RuntimeError(
            "Normalized feature dimension does not match SOM weights."
        )

    history_epoch = []
    history_qe = []
    history_radius = []

    # This is the same distance computation NNSOM's sim_som() performs.
    # Reusing it for the next epoch lets us record QE without doubling
    # the dominant distance-computation cost.
    distance_matrix = cdist(
        weights,
        x_internal.T,
        metric="euclidean",
    )

    step = 0

    # Optional baseline before the first update.
    history_epoch.append(0)
    history_qe.append(
        _nnsom_style_quantization_error(
            distance_matrix
        )
    )
    history_radius.append(
        float(init_neighborhood)
    )

    for epoch_index in range(1, epochs + 1):
        winners = np.argmin(
            distance_matrix,
            axis=0,
        )

        outputs = _one_hot_winners(
            winners,
            num_neurons,
        )

        neighborhood_radius = (
            1
            + (init_neighborhood - 1)
            * (1 - step / steps)
        )

        neighborhood = (
            som.neuron_dist
            <= neighborhood_radius
        )

        # Match NNSOM's stochastic 90% winner retention.
        outputs = outputs * (
            np.random.rand(
                num_neurons,
                num_samples,
            )
            < 0.90
        )

        neighborhood_outputs = (
            np.matmul(
                neighborhood,
                outputs,
            )
            + outputs
        )

        winner_counts = np.sum(
            neighborhood_outputs,
            axis=1,
        )

        loser_index = np.squeeze(
            np.asarray(
                winner_counts == 0
            )
        )

        winner_counts[
            loser_index
        ] = 1

        winner_counts = np.expand_dims(
            winner_counts,
            axis=1,
        )

        normalized_outputs = (
            neighborhood_outputs
            / np.repeat(
                winner_counts,
                num_samples,
                axis=1,
            )
        )

        new_weights = np.matmul(
            normalized_outputs,
            x_internal.T,
        )

        delta = new_weights - weights
        delta[loser_index] = 0
        weights = weights + np.asarray(delta)

        step += 1

        # These distances are also what the next epoch needs.
        distance_matrix = cdist(
            weights,
            x_internal.T,
            metric="euclidean",
        )

        if (
            epoch_index % history_every == 0
            or epoch_index == epochs
        ):
            history_epoch.append(
                epoch_index
            )

            history_qe.append(
                _nnsom_style_quantization_error(
                    distance_matrix
                )
            )

            history_radius.append(
                float(neighborhood_radius)
            )

        if epoch_index % 50 == 0:
            print(
                "Tracked NNSOM training epoch "
                f"{epoch_index}/{epochs}"
            )

    som.w = weights

    final_winners = np.argmin(
        distance_matrix,
        axis=0,
    )

    som.outputs = _one_hot_winners(
        final_winners,
        num_neurons,
    )

    som.sim_flag = False
    som.norm_func = norm_func

    history = SOMTrainingHistory(
        epoch=history_epoch,
        quantization_error=history_qe,
        neighborhood_radius=history_radius,
    )

    return som, history
