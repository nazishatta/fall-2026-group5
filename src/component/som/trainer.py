import numpy as np
from NNSOM.plots import SOMPlots
from src.component.som.memory_safe_init import economy_svd_for_nnsom_init


def train_som(
    x_train,
    grid_height,
    grid_width,
    init_neighborhood,
    epochs,
    steps,
    norm_func,
    seed,
):
    """Train a reproducible NNSOM model on training embeddings only."""

    np.random.seed(seed)

    som = SOMPlots(
        dimensions=(grid_height, grid_width)
    )

    with economy_svd_for_nnsom_init():
        som.init_w(
            x_train,
            norm_func=norm_func
        )

    som.train(
        x_train,
        init_neighborhood=init_neighborhood,
        epochs=epochs,
        steps=steps,
        norm_func=norm_func
    )

    return som