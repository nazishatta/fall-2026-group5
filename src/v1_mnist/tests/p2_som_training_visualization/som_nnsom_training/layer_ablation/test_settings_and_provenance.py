"""Feature-layer ablation: the real config, the run ID, the checks on layers and seeds, and the run record."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation import provenance
from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
    DEFAULT_CONFIG,
    cell_name,
    load_ablation_settings,
)

CODE_ROOT = Path(__file__).resolve().parents[6]


def test_real_config_gives_the_protocol_settings() -> None:
    s = load_ablation_settings(DEFAULT_CONFIG, CODE_ROOT)
    assert s.layers == ("fc1", "fc2")
    assert s.som_seeds == (1, 2, 3, 4, 5)
    assert (s.recipe.grid_height, s.recipe.grid_width, s.recipe.init_neighborhood, s.recipe.epochs) == (15, 15, 11, 250)
    assert s.run_id == "feature_layer_ablation_som15x15_nb11_ep250_seeds5"
    assert s.min_hotspots == 5 and s.hotspot_fraction == 0.05 and s.detection_repeats == 200
    assert s.protocol_path.is_file()
    assert cell_name("fc1", 3) == "fc1_seed3"


def _config_with(tmp_path: Path, **changes: object) -> Path:
    raw = yaml.safe_load((CODE_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))
    raw.update(changes)
    path = tmp_path / "ablation.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("changes", [{"layers": ["fc2", "fc1"]}, {"layers": ["fc1", "fc3"]},
                                     {"som_seeds": [1]}, {"som_seeds": [1, 42]}])
def test_bad_layers_or_seeds_are_refused(tmp_path: Path, changes: dict) -> None:
    with pytest.raises(ValueError):
        load_ablation_settings(_config_with(tmp_path, **changes), CODE_ROOT)


def test_run_record_is_written_once_and_changes_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = dataclasses.replace(load_ablation_settings(DEFAULT_CONFIG, CODE_ROOT), output_dir=tmp_path / "run")
    fake = {"run_id": s.run_id, "protocol_sha256": "a"}
    monkeypatch.setattr(provenance, "current_inputs", lambda _s: dict(fake))
    provenance.start_run(s)
    assert s.inputs_path.is_file()
    provenance.start_run(s)                      # unchanged inputs: continues
    fake["protocol_sha256"] = "b"
    with pytest.raises(RuntimeError, match="changed"):
        provenance.verify_run(s)


def test_existing_folder_without_record_is_refused(tmp_path: Path) -> None:
    s = dataclasses.replace(load_ablation_settings(DEFAULT_CONFIG, CODE_ROOT), output_dir=tmp_path / "run")
    (s.output_dir / "metrics").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without a provenance record"):
        provenance.start_run(s)
