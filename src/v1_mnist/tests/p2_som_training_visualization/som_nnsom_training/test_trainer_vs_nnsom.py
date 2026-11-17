"""Check the history trainer against NNSOM's native train().

The two are meant to be identical only after epoch 1. From epoch 2 they
differ on purpose: NNSOM 1.8.x picks winners from the initial weights,
while train_som_with_history recomputes them from the updated weights.

The reference starts from the trainer's own initialization (SOMPlots with the
economy SVD). With CuPy installed SOMPlots initializes on the GPU, whose SVD
can flip singular-vector signs relative to NumPy's, so NNSOM's CPU init_w
would start from different (mirrored) weights.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from NNSOM.plots import SOMPlots
from NNSOM.som import SOM

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.memory_safe_init import economy_svd_for_nnsom_init
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import train_som_with_history


def test_matches_nnsom_after_one_epoch() -> None:
    X = np.random.default_rng(0).normal(size=(300, 20))
    norm = MinMaxScaler((-1, 1)).fit(X).transform
    mine, hist = train_som_with_history(X, 4, 4, 3, 1, 10, norm, seed=42)
    init = SOMPlots((4, 4))
    with economy_svd_for_nnsom_init():
        init.init_w(X, norm_func=norm)
    ref = SOM((4, 4))
    ref.w = np.asarray(init.w).copy()
    np.random.seed(42)  # same dropout mask as the trainer (its init draws nothing from np.random)
    ref.train(X, init_neighborhood=3, epochs=1, steps=10, norm_func=norm)
    assert np.allclose(np.asarray(mine.w), ref.w, atol=1e-10)
    assert len(hist.quantization_error) == 2  # baseline + 1 epoch
