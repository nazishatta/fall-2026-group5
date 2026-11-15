"""Run provenance: a first run starts, a changed input stops, an unrecorded folder is refused."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.run_state import (
    start_run,
    verify_run,
)


def make_settings(tmp_path: Path) -> SimpleNamespace:
    files = {name: tmp_path / name for name in ("protocol.md", "config.yaml", "retr.yaml", "som.yaml", "ck.pth")}
    for path in files.values():
        path.write_text(path.name)
    out = tmp_path / "run"
    return SimpleNamespace(
        run_id="run", output_dir=out, inputs_path=out / "metadata" / "inputs.json", arms=("standard",),
        seeds=(1,), protocol_path=files["protocol.md"], config_path=files["config.yaml"],
        retraining=SimpleNamespace(
            config_path=files["retr.yaml"], analysis=SimpleNamespace(config_path=files["som.yaml"]),
            checkpoint_path=lambda arm, seed: files["ck.pth"],
        ),
    )


def test_first_run_starts_even_with_a_log_folder(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    (settings.output_dir / "logs").mkdir(parents=True)
    (settings.output_dir / "logs" / "embeddings.log").write_text("x")
    start_run(settings)  # type: ignore[arg-type]
    assert settings.inputs_path.is_file()
    start_run(settings)  # type: ignore[arg-type]  # continuing is allowed


def test_changed_checkpoint_stops_the_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    start_run(settings)  # type: ignore[arg-type]
    (tmp_path / "ck.pth").write_text("different")
    with pytest.raises(RuntimeError, match="Inputs changed"):
        verify_run(settings)  # type: ignore[arg-type]


def test_folder_without_record_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    (settings.output_dir / "tables").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="provenance"):
        start_run(settings)  # type: ignore[arg-type]
