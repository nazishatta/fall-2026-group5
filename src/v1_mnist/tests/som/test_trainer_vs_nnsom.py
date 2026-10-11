"""Check the history trainer against NNSOM's native train().

The two are meant to be identical only after epoch 1. From epoch 2 they
differ on purpose: NNSOM 1.8.x picks winners from the initial weights,
while train_som_with_history recomputes them from the updated weights.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from NNSOM.som import SOM

from src.v1_mnist.component.som.trainer import train_som_with_history


def test_matches_nnsom_after_one_epoch():
    X = np.random.default_rng(0).normal(size=(300, 20))
    norm = MinMaxScaler((-1, 1)).fit(X).transform
    mine, hist = train_som_with_history(X, 4, 4, 3, 1, 10, norm, seed=42)
    np.random.seed(42)
    ref = SOM((4, 4))
    ref.init_w(X, norm_func=norm)
    ref.train(X, init_neighborhood=3, epochs=1, steps=10, norm_func=norm)
    assert np.allclose(np.asarray(mine.w), ref.w, atol=1e-10)
    assert len(hist.quantization_error) == 2  # baseline + 1 epoch
