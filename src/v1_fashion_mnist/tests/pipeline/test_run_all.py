"""The Fashion-MNIST runner calls the shared runner with the Fashion-MNIST configs."""

from __future__ import annotations

from typing import Sequence

import pytest

from src.v1_fashion_mnist.component.pipeline import run_all


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_main(argv: Sequence[str] | None = None) -> None:
        calls.append(list(argv or []))

    monkeypatch.setattr(run_all.shared_run_all, "main", fake_main)
    return calls


def test_the_fashion_mnist_config_folder_is_added(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    run_all.main(["--from", "data_pipeline", "--dry-run"])
    assert calls == [["--config-dir", run_all.CONFIG_DIR, "--from", "data_pipeline", "--dry-run"]]


def test_a_given_config_folder_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    run_all.main(["--config-dir", "elsewhere"])
    assert calls == [["--config-dir", "elsewhere"]]
