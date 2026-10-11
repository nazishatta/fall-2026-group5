"""The data split seed comes only from the config (no default seed in the code)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.v1_mnist.component.utils.config import require_seed


def test_seed_is_read_from_the_config() -> None:
    config = SimpleNamespace(training=SimpleNamespace(seed=7))
    assert require_seed(config) == 7


@pytest.mark.parametrize(
    "config",
    [SimpleNamespace(), SimpleNamespace(training=SimpleNamespace(epochs=1))],
    ids=["no training section", "no seed"],
)
def test_missing_seed_stops_with_a_clear_error(config: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="training.seed"):
        require_seed(config)
