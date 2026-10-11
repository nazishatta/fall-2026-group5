"""CNN run IDs: file names carry the run ID, and existing runs are never overwritten."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.run_ids import (
    check_new_embeddings_dir,
    check_new_training_run,
    cnn_run_id,
    cnn_run_paths,
)
from src.v1_mnist.component.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[5]
CNN_CONFIG = REPO_ROOT / "src/v1_mnist/component/configs/cnn_baseline.yaml"


def make_config(tmp_path: Path, run_id: str | None = "lenet5_test") -> SimpleNamespace:
    out = tmp_path / "cnn_baseline_training"
    paths = SimpleNamespace(
        output_dir=str(out),
        checkpoints_dir=str(out / "checkpoints"),
        figures_dir=str(out / "figures"),
        metrics_dir=str(out / "metrics"),
        logs_dir=str(out / "logs"),
        embeddings_dir=str(out / "embeddings"),
    )
    run = SimpleNamespace(name="cnn_baseline") if run_id is None else SimpleNamespace(name="cnn_baseline", id=run_id)
    return SimpleNamespace(run=run, paths=paths)


def test_config_run_id_keeps_existing_checkpoint_name() -> None:
    """The current run's ID must map to the checkpoint every SOM was built from."""
    config = load_config(str(CNN_CONFIG))
    paths = cnn_run_paths(config)
    assert paths.run_id == "lenet5_best"
    assert paths.checkpoint.name == "lenet5_best.pth"


def test_file_names_carry_run_id(tmp_path: Path) -> None:
    paths = cnn_run_paths(make_config(tmp_path, "lenet5_s7_v2"))
    assert paths.checkpoint.name == "lenet5_s7_v2.pth"
    assert paths.test_metrics.name == "lenet5_s7_v2_test_evaluation.json"
    assert paths.run_record.name == "lenet5_s7_v2_run.json"
    assert paths.train_log.name == "lenet5_s7_v2_train_baseline.log"
    assert paths.training_curves.name == "lenet5_s7_v2_training_curves"


@pytest.mark.parametrize("bad", [None, "", "7lenet", "lenet 5", "lenet5.v2", "../x"])
def test_missing_or_bad_run_id_is_refused(tmp_path: Path, bad: str | None) -> None:
    config = make_config(tmp_path, bad)
    with pytest.raises(ValueError):
        cnn_run_id(config)


def test_new_run_id_passes(tmp_path: Path) -> None:
    check_new_training_run(cnn_run_paths(make_config(tmp_path)))


@pytest.mark.parametrize("which", ["checkpoint", "test_metrics", "run_record", "train_log"])
def test_existing_run_output_is_refused(tmp_path: Path, which: str) -> None:
    paths = cnn_run_paths(make_config(tmp_path))
    target = getattr(paths, which)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old run")
    with pytest.raises(FileExistsError, match="new run.id"):
        check_new_training_run(paths)
    assert target.read_text() == "old run"


def test_existing_run_figure_is_refused(tmp_path: Path) -> None:
    paths = cnn_run_paths(make_config(tmp_path))
    pdf = paths.training_curves.parent / "pdf" / f"{paths.training_curves.name}.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    with pytest.raises(FileExistsError):
        check_new_training_run(paths)


def test_embeddings_folder_with_embeddings_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "embeddings"
    check_new_embeddings_dir(folder)            # missing folder: fine
    folder.mkdir()
    check_new_embeddings_dir(folder)            # empty folder: fine
    (folder / "train_features.npy").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="new folder"):
        check_new_embeddings_dir(folder)
