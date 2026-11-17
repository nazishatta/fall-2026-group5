from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.run_preflight import (
    ensure_not_done,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
    DEFAULT_CONFIG,
    load_baseline_settings,
    resolve_path,
)

CODE_ROOT = Path(__file__).resolve().parents[6]


def _embeddings_available() -> bool:
    """True when the DVC-tracked embeddings named in the configs are in this checkout."""
    baseline_cfg = yaml.safe_load(resolve_path(CODE_ROOT, DEFAULT_CONFIG).read_text(encoding="utf-8"))
    som_cfg_path = resolve_path(CODE_ROOT, str(baseline_cfg["inputs"]["som_config"]))
    som_cfg = yaml.safe_load(som_cfg_path.read_text(encoding="utf-8"))
    embeddings_dir = resolve_path(CODE_ROOT, str(som_cfg["paths"]["embeddings_dir"]))
    return (embeddings_dir / "train_features.npy").is_file()


def test_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    ensure_not_done(results)  # not there yet: fine
    results.write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_not_done(results)


@pytest.mark.skipif(
    not _embeddings_available(),
    reason="DVC embeddings (train_features.npy) unavailable in this checkout; run `dvc pull`",
)
def test_settings_come_from_the_config_and_the_som_config() -> None:
    settings, analysis = load_baseline_settings(None, CODE_ROOT)
    assert settings.num_cells == analysis.grid_height * analysis.grid_width
    assert settings.run_id.startswith("error_pattern_baselines_som_") and f"{analysis.grid_height}x{analysis.grid_width}" in settings.run_id
    assert "week" not in settings.run_id.lower() and "rq" not in settings.run_id.lower()
    assert settings.robustness.n_splits > 0 and settings.output_dir.name == settings.run_id
