"""Training helpers for Week 3 NNSOM experiments.

``train_som`` delegates training to NNSOM's native ``som.train``.
NOTE: NNSOM 1.8.x picks winners from the *initial* weights in every epoch,
because ``sim_som`` reads ``self.w``, which is only updated after the last
epoch.

``train_som_with_history`` (the training with a quantization-error history, used
for the saved models) is in trainer_history.py, split from this file on 2026-10-10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.v1_mnist.component.utils.som_types import SOMModel

from typing import Any

import numpy as np
from NNSOM.plots import SOMPlots
from NNSOM.som import SOM

try:
    from NNSOM.som_gpu import SOMGpu
except ImportError:
    SOMGpu = None

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.memory_safe_init import economy_svd_for_nnsom_init


def create_som(
    dimensions: tuple[int, int],
    backend: str = "auto",
) -> SOMModel:
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
    som: SOMModel,
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


def train_som(
    x_train: np.ndarray,
    grid_height: int,
    grid_width: int,
    init_neighborhood: int,
    epochs: int,
    steps: int,
    norm_func: Any,
    seed: int,
    backend: str = "auto",
) -> Any:
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
