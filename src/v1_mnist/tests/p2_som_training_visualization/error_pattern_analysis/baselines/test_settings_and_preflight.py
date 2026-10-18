from __future__ import annotations

from pathlib import Path

import pytest

from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.run_preflight import (
    ensure_not_done,
)
from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
    load_baseline_settings,
)

CODE_ROOT = Path(__file__).resolve().parents[6]


def test_finished_run_is_never_overwritten(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    ensure_not_done(results)  # not there yet: fine
    results.write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_not_done(results)


def test_settings_come_from_the_config_and_the_som_config() -> None:
    settings, analysis = load_baseline_settings(None, CODE_ROOT)
    assert settings.num_cells == analysis.grid_height * analysis.grid_width
    assert settings.run_id.startswith("error_pattern_baselines_") and analysis.selected_model in settings.run_id
    assert "week" not in settings.run_id.lower() and "rq" not in settings.run_id.lower()
    assert settings.robustness.n_splits > 0 and settings.output_dir.name == settings.run_id
