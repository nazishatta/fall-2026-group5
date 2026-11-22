"""The fast trainer is the project trainer's algorithm (pre-registration, section 3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.grid_comparison.fast_trainer import (
    nnsom_qe,
    train_som_fast,
    winners,
)
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.trainer_history import (
    train_som_with_history,
)


def _toy() -> tuple[np.ndarray, MinMaxScaler]:
    x = np.random.default_rng(7).normal(size=(400, 6))
    return x, MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x)


def test_exact_mode_gives_the_project_trainers_map() -> None:
    x, scaler = _toy()
    fast, _ = train_som_fast(x, 4, 4, 3, 6, 6, scaler.transform, seed=11, rng_mode="exact")
    project, _ = train_som_with_history(x_train=x, grid_height=4, grid_width=4, init_neighborhood=3, epochs=6,
                                        steps=6, norm_func=scaler.transform, seed=11)
    np.testing.assert_allclose(np.asarray(fast.w), np.asarray(project.w), rtol=0, atol=1e-9)


def test_fast_mode_is_repeatable_and_uses_its_own_random_stream() -> None:
    x, scaler = _toy()
    a, _ = train_som_fast(x, 4, 4, 3, 6, 6, scaler.transform, seed=3, rng_mode="fast")
    b, _ = train_som_fast(x, 4, 4, 3, 6, 6, scaler.transform, seed=3, rng_mode="fast")
    exact, _ = train_som_fast(x, 4, 4, 3, 6, 6, scaler.transform, seed=3, rng_mode="exact")
    np.testing.assert_array_equal(np.asarray(a.w), np.asarray(b.w))
    assert not np.allclose(np.asarray(a.w), np.asarray(exact.w))


def test_history_has_one_entry_per_epoch_plus_the_start() -> None:
    x, scaler = _toy()
    _, history = train_som_fast(x, 3, 3, 2, 5, 5, scaler.transform, seed=1)
    assert len(history["quantization_error"]) == 6 and history["neighborhood_radius"][0] == 2.0


def test_unknown_random_mode_stops() -> None:
    x, scaler = _toy()
    with pytest.raises(ValueError, match="rng_mode"):
        train_som_fast(x, 3, 3, 2, 2, 2, scaler.transform, seed=1, rng_mode="gpu")


def test_winners_and_qe_match_a_direct_computation() -> None:
    rng = np.random.default_rng(0)
    w, x = rng.normal(size=(5, 3)), rng.normal(size=(40, 3))
    win, dist = winners(w, x, (x * x).sum(1))
    direct = np.linalg.norm(x[:, None, :] - w[None, :, :], axis=2)
    np.testing.assert_array_equal(win, direct.argmin(1))
    np.testing.assert_allclose(dist, direct.min(1), atol=1e-12)
    per = [direct.min(1)[win == k].mean() if (win == k).any() else 0.0 for k in range(5)]
    assert nnsom_qe(dist, win, 5) == pytest.approx(np.mean(per))
