"""Safety tests for the frozen QE paired-comparison runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.pipeline.p3_novelty_detection import (
    run_qe_energy_knn_paired as runner,
)


def test_resolve_path_rejects_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured paths must stay inside the repository."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="outside repository"):
        runner.resolve_path("../outside")


def test_run_rejects_invalid_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsafe run IDs must fail before accessing scores."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"run_id": "../old_run"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid run ID"):
        runner.run(config_path)


def test_run_does_not_overwrite_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing output run must never be reused."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    destination = tmp_path / "outputs" / "existing_run"
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"run_id": "existing_run", "output_root": "outputs"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Existing run"):
        runner.run(config_path)

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
