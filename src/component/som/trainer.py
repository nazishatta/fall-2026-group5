import numpy as np
from NNSOM.plots import SOMPlots


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